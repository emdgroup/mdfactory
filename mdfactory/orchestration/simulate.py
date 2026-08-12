# ABOUTME: Main dispatcher for Parsl-based GROMACS simulation orchestration
# ABOUTME: Handles checkpoint detection, dry-run, and progress monitoring
"""GROMACS simulation orchestration via Parsl.

Provides :func:`run_simulations`, the main entry point for orchestrating
GROMACS MD simulations via Parsl. Handles checkpoint detection, dry-run mode,
and progress monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from loguru import logger

from .apps import get_grompp_app, get_mdrun_app
from .config import get_stage_config_or_none
from .session import parsl_session
from .stages import (
    STAGE_BY_NAME,
    STAGE_REGISTRY,
    StageSpec,
    run_stage,
)

# Import MDAnalysis at module level for testability
try:
    import MDAnalysis as mda
except ImportError:
    mda = None

if TYPE_CHECKING:
    from parsl import AppFuture

    from .config import ExecutorConfig
    from .progress import StageProgressTracker


class _StageState(TypedDict):
    """Checkpoint detection result for a single stage."""

    status: str  # "complete" | "partial" | "not_started"
    cpt_file: "Path | None"
    restart: bool


def _has_restart_pair(cpt_file: Path, tpr_file: Path) -> bool:
    """Return True if both checkpoint and TPR files exist (restartable state)."""
    return cpt_file.exists() and tpr_file.exists()


def _bash_result_to_dict(raw: object, sim_hash: str) -> dict:
    """Normalise a bash_app result to a ``{"hash": ..., "status": ...}`` dict."""
    if isinstance(raw, dict):
        return raw
    return {"hash": sim_hash, "status": "success"}


def run_simulations(
    sim_paths: list[Path],
    config: "ExecutorConfig",
    *,
    stages: list[str] | None = None,
    wait: bool = True,
    dry_run: bool = False,
    clean: bool = False,
    checkpoint_mode: str = "auto",
    max_rescue: int = 3,
) -> list[dict]:
    """Orchestrate GROMACS simulations via Parsl.

    Parameters
    ----------
    sim_paths : list[Path]
        Simulation directories (must contain system.pdb, topology.top, *.mdp).
    config : ExecutorConfig
        Parsl executor configuration (local or SLURM).
    stages : list[str], optional
        Stages to run. Default: ["EM", "NVT", "NPT", "Production"].
    wait : bool
        Wait for completion (default: True).
    dry_run : bool
        Preview plan without executing (default: False).
    clean : bool
        Remove simulation outputs before running (default: False).
        Respects ``stages`` filter — only files belonging to requested
        stages are deleted.  Combined with ``dry_run``, previews what
        would be deleted without acting.
    checkpoint_mode : str
        - "auto": Skip stages with valid outputs (default)
        - "skip": Never re-run completed stages
        - "force": Overwrite all stages
    max_rescue : int
        Maximum number of rescue tiers for physics failures in EM/NVT/NPT
        stages. Each tier halves the step size and doubles nsteps. Set to 0
        to disable rescue. Default: 3.

    Returns
    -------
    list[dict]
        Results with status, errors, timing (or futures if wait=False).

    """
    valid_stages = [s.name for s in STAGE_REGISTRY]
    valid_modes = ("auto", "skip", "force")

    if stages is None:
        stages = list(valid_stages)
    else:
        invalid = [s for s in stages if s not in valid_stages]
        if invalid:
            raise ValueError(f"Invalid stage(s): {invalid}. Valid stages: {valid_stages}")

    if checkpoint_mode not in valid_modes:
        raise ValueError(
            f"Invalid checkpoint_mode: {checkpoint_mode!r}. Valid: {list(valid_modes)}"
        )

    # 1. Filter: skip directories whose build did not complete.
    # Skipped dirs are collected as result entries so callers can account for
    # all requested simulations (succeeded + failed + skipped == requested).
    skipped_results: list[dict] = []
    ready_paths: list[Path] = []
    for sim_dir in sim_paths:
        missing = _missing_build_files(sim_dir, stages)
        if missing:
            logger.warning(f"Skipping {sim_dir.name}: build incomplete, missing: {missing}")
            skipped_results.append(
                {
                    "hash": sim_dir.name,
                    "status": "skipped",
                    "directory": str(sim_dir),
                    "reason": f"build incomplete, missing: {missing}",
                }
            )
        else:
            ready_paths.append(sim_dir)

    n_skipped = len(skipped_results)
    if n_skipped:
        logger.info(
            f"Skipped {n_skipped} simulation(s) with incomplete builds; "
            f"{len(ready_paths)} ready to simulate."
        )
    if not ready_paths:
        logger.warning("No simulation directories have complete builds. Nothing to run.")
        return skipped_results
    sim_paths = ready_paths

    # 1b. Clean outputs (if requested) before checkpoint detection.
    if clean:
        for sim_dir in sim_paths:
            deleted = clean_simulation_outputs(sim_dir, stages, dry_run=dry_run)
            if dry_run and deleted:
                logger.info(
                    f"Would delete {len(deleted)} file(s) from {sim_dir.name}: "
                    + ", ".join(p.name for p in deleted)
                )
        if dry_run:
            # After previewing deletions, still show the dry-run execution plan
            # (which will report all stages as needed since nothing was deleted).
            pass

    # 2. Checkpoint detection (includes restart info for -cpi -append support)
    work_plan = []
    for sim_dir in sim_paths:
        stage_items = _detect_needed_stages_with_restart_info(sim_dir, stages, checkpoint_mode)
        needed_stages = [item["stage"] for item in stage_items]
        stage_restarts = {
            item["stage"]: str(item["cpt_file"])
            for item in stage_items
            if item["restart"] and item["cpt_file"] is not None
        }
        work_plan.append(
            {
                "sim_dir": sim_dir,
                "hash": sim_dir.name,
                "stages": needed_stages,
                "stage_restarts": stage_restarts,
            }
        )

    logger.info(f"Prepared work plan for {len(work_plan)} simulation(s)")

    # 3. Dry-run mode
    if dry_run:
        return skipped_results + _log_dry_run_plan(work_plan, config)

    # 4. Validate prerequisites before Parsl session
    for item in work_plan:
        if item["stages"]:
            # Validate first stage has required inputs
            _validate_stage_prerequisites(item["sim_dir"], item["stages"][0])

    # 5. Filter active items (stages still needed)
    active_items = []
    for item in work_plan:
        if not item["stages"]:
            logger.info(f"Skipping {item['hash']} (all stages complete)")
            continue
        active_items.append(item)

    if not active_items:
        logger.info("All simulations already complete.")
        return skipped_results

    # 6. Parsl session
    with parsl_session(config) as session:
        grompp_app = get_grompp_app()
        mdrun_app = get_mdrun_app()

        # wait=False: legacy path returning raw futures (no progress display)
        if not wait:
            from concurrent.futures import Future as StdFuture
            from concurrent.futures import ThreadPoolExecutor

            futures: list[tuple[str, object]] = []

            def _run_pipeline_nowait(item):
                return _execute_stage_list(
                    item["sim_dir"],
                    item["stages"],
                    grompp_app,
                    mdrun_app,
                    stage_restarts=item.get("stage_restarts"),
                    config=config,
                    max_rescue=max_rescue,
                )

            with ThreadPoolExecutor() as pool:
                thread_futs = [
                    (item["hash"], pool.submit(_run_pipeline_nowait, item))
                    for item in active_items
                ]
                for h, tf in thread_futs:
                    try:
                        futures.append((h, tf.result()))
                    except Exception as exc:
                        logger.error(f"Pipeline failed for {h}: {exc}")
                        failed_fut = StdFuture()
                        failed_fut.set_exception(exc)
                        futures.append((h, failed_fut))

            logger.info(f"Submitted {len(futures)} simulation(s)")
            logger.warning(
                "Returning raw AppFuture objects — futures resolve to None (bash_app exit), "
                "not status dicts.  Caller must call parsl.clear() when done."
            )
            session.detach()
            return [fut for _, fut in futures]

        # 7. wait=True: tracked execution with per-stage progress display
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from .errors import _describe_failure
        from .progress import StageProgressTracker, display_stage_progress

        all_hashes = [item["hash"] for item in active_items]
        tracker = StageProgressTracker(stages=stages, sim_hashes=all_hashes)

        # Pre-mark stages already completed by checkpoint detection
        for item in active_items:
            for stage in stages:
                if stage not in item["stages"]:
                    tracker.mark_succeeded(stage, item["hash"])

        def _run_pipeline(item):
            sim_hash = item["hash"]
            try:
                final_future = _execute_stage_list(
                    item["sim_dir"],
                    item["stages"],
                    grompp_app,
                    mdrun_app,
                    stage_restarts=item.get("stage_restarts"),
                    config=config,
                    max_rescue=max_rescue,
                    tracker=tracker,
                )
                # Wait for the terminal future (Production) to collect its result
                if final_future is not None:
                    try:
                        raw = final_future.result()
                        result = _bash_result_to_dict(raw, sim_hash)
                    except Exception as exc:
                        failure_type, error_detail = _describe_failure(exc)
                        result = {
                            "hash": sim_hash,
                            "status": "failed",
                            "error": error_detail,
                            "failure_type": failure_type,
                            "error_detail": error_detail,
                        }
                else:
                    result = {"hash": sim_hash, "status": "success"}
                tracker.store_result(sim_hash, result)
            except Exception as exc:
                failure_type, error_detail = _describe_failure(exc)
                tracker.store_result(sim_hash, {
                    "hash": sim_hash,
                    "status": "failed",
                    "error": error_detail,
                    "failure_type": failure_type,
                    "error_detail": error_detail,
                })

        logger.info(f"Submitted {len(active_items)} simulation(s)")

        # Launch worker threads without blocking the main thread
        pool = ThreadPoolExecutor()
        thread_futures = [pool.submit(_run_pipeline, item) for item in active_items]
        pool.shutdown(wait=False)

        # Main thread: display progress until all stages complete
        display_stage_progress(tracker)

        # Ensure all worker threads have finished (no-op in normal usage
        # since display blocks until all_done, but needed for test mocks)
        from concurrent.futures import wait as futures_wait

        futures_wait(thread_futures)

        # Collect results
        run_results = tracker.collect_results()
        return skipped_results + run_results


def _missing_build_files(sim_dir: Path, stages: list[str]) -> list[str]:
    """Return names of required build output files that are absent from *sim_dir*."""
    required = ["system.pdb", "topology.top"] + [STAGE_BY_NAME[s].mdp_file for s in stages]
    return [f for f in required if not (sim_dir / f).exists()]


def clean_simulation_outputs(
    sim_dir: Path,
    stages: list[str],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove simulation outputs for the given stages, preserving build inputs.

    Derives deletable files from :data:`STAGE_BY_NAME` (tpr, cpt, log, edr,
    gro, trr, xtc) plus ``mdout.mdp``, rescue-tier MDPs, and GROMACS backup
    files (``#*#``).

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stages : list[str]
        Stage names whose outputs should be removed.
    dry_run : bool
        If ``True``, collect and return the list of files that *would* be
        deleted without actually removing them.

    Returns
    -------
    list[Path]
        Paths that were deleted (or would be deleted in dry-run mode).

    """
    to_delete: list[Path] = []

    for stage_name in stages:
        spec = STAGE_BY_NAME[stage_name]
        deffnm = spec.deffnm

        # Named output files: tpr, cpt, log, edr
        for ext in ("tpr", "cpt", "log", "edr"):
            to_delete.append(sim_dir / f"{deffnm}.{ext}")

        # Structure output (EM/NVT/NPT)
        if spec.gro_out:
            to_delete.append(sim_dir / spec.gro_out)

        # Trajectory outputs (Production)
        for traj in spec.traj_files:
            to_delete.append(sim_dir / traj)

        # Rescue-tier MDPs (e.g. em_rescue_t1.mdp, em_rescue_t2.mdp)
        mdp_stem = spec.mdp_file.rsplit(".", 1)[0]
        to_delete.extend(sim_dir.glob(f"{mdp_stem}_rescue_t*.mdp"))

        # GROMACS backup files (#deffnm.*#)
        to_delete.extend(sim_dir.glob(f"#{deffnm}.*#"))

    # mdout.mdp — grompp output, always regenerated
    to_delete.append(sim_dir / "mdout.mdp")

    # Deduplicate (glob results may overlap with named files) and filter to
    # files that actually exist.
    seen: set[Path] = set()
    existing: list[Path] = []
    for p in to_delete:
        if p not in seen and p.exists():
            seen.add(p)
            existing.append(p)

    if not dry_run:
        for p in existing:
            p.unlink()
            logger.debug(f"Deleted {p}")
        if existing:
            logger.info(f"Cleaned {len(existing)} file(s) from {sim_dir.name}")

    return existing



def _detect_skip_mode_state(output_file: Path, cpt_file: Path, tpr_file: Path) -> _StageState:
    """Return stage state for 'skip' mode: complete if output exists, partial if cpt+tpr exist."""
    if output_file.exists():
        return {"status": "complete", "cpt_file": None, "restart": False}
    if _has_restart_pair(cpt_file, tpr_file):
        return {"status": "partial", "cpt_file": cpt_file, "restart": True}
    return {"status": "not_started", "cpt_file": None, "restart": False}


def _detect_production_output_state(
    sim_dir: Path,
    cpt_file: Path,
    tpr_file: Path,
    traj_files: "tuple[str, ...]",
) -> _StageState:
    """Check Production trajectory completeness; restart from cpt if incomplete."""
    expected_frames = _extract_expected_frames_from_mdp(sim_dir, "Production")
    for traj_file in traj_files:
        if _validate_trajectory_complete(sim_dir, traj_file, expected_frames):
            return {"status": "complete", "cpt_file": None, "restart": False}
    if _has_restart_pair(cpt_file, tpr_file):
        return {"status": "partial", "cpt_file": cpt_file, "restart": True}
    # All trajectory files incomplete and no checkpoint to restart from
    return {"status": "partial", "cpt_file": None, "restart": False}


def _detect_skip_stage_state(
    sim_dir: Path,
    spec: "StageSpec",
    cpt_file: Path,
    tpr_file: Path,
) -> _StageState:
    """Return stage state for 'skip' mode: file-existence only, no integrity checks."""
    if spec.traj_files:
        # Mirror auto mode: complete if ANY trajectory file exists (XTC or TRR).
        if any((sim_dir / tf).exists() for tf in spec.traj_files):
            return {"status": "complete", "cpt_file": None, "restart": False}
        if _has_restart_pair(cpt_file, tpr_file):
            return {"status": "partial", "cpt_file": cpt_file, "restart": True}
        return {"status": "not_started", "cpt_file": None, "restart": False}
    return _detect_skip_mode_state(sim_dir / spec.gro_out, cpt_file, tpr_file)


def _detect_auto_output_state(
    sim_dir: Path,
    stage: str,
    spec: "StageSpec",
    cpt_file: Path,
    tpr_file: Path,
    prereq_cpt: "Path | None",
) -> _StageState:
    """Return stage state for 'auto' mode; validates prerequisite integrity before accepting."""
    # Workflow integrity: prerequisite checkpoint must exist for the output to
    # be trustworthy (e.g. npt.gro is only valid if nvt.cpt is present).
    if prereq_cpt and not prereq_cpt.exists():
        return {"status": "not_started", "cpt_file": None, "restart": False}
    if stage == "Production":
        return _detect_production_output_state(sim_dir, cpt_file, tpr_file, spec.traj_files)
    return {"status": "complete", "cpt_file": None, "restart": False}


def _detect_stage_state(sim_dir: Path, stage: str, mode: str = "auto") -> _StageState:
    """Detect completion or partial progress of a stage given checkpoint *mode*."""
    spec = STAGE_BY_NAME[stage]
    cpt_file = sim_dir / spec.cpt_file
    tpr_file = sim_dir / spec.tpr_file

    if mode == "skip":
        return _detect_skip_stage_state(sim_dir, spec, cpt_file, tpr_file)

    # For trajectory stages (Production) any accepted file counts as output.
    # For structure stages (EM/NVT/NPT) the single gro_out is the output.
    if spec.traj_files:
        output_exists = any(
            (sim_dir / tf).exists() and (sim_dir / tf).stat().st_size > 0 for tf in spec.traj_files
        )
    else:
        gro_out_file = sim_dir / spec.gro_out
        output_exists = gro_out_file.exists() and gro_out_file.stat().st_size > 0

    # Prerequisite checkpoint (for validating workflow integrity in auto mode)
    prereq_cpt = sim_dir / spec.prereq_cpt if spec.prereq_cpt else None

    if output_exists:
        return _detect_auto_output_state(sim_dir, stage, spec, cpt_file, tpr_file, prereq_cpt)

    # Check partial progress (checkpoint exists, output doesn't).
    if _has_restart_pair(cpt_file, tpr_file):
        if spec.traj_files:
            # Trajectory stage (Production): stale checkpoint without a
            # trajectory file cannot use -append — GROMACS would crash.
            # Treat as not_started so grompp+mdrun run from scratch.
            return {"status": "not_started", "cpt_file": None, "restart": False}
        return {"status": "partial", "cpt_file": cpt_file, "restart": True}

    return {"status": "not_started", "cpt_file": None, "restart": False}


def _detect_needed_stages(sim_dir: Path, stages: list[str], mode: str) -> list[str]:
    """Return stage names that still need to run (discards restart metadata)."""
    return [
        item["stage"] for item in _detect_needed_stages_with_restart_info(sim_dir, stages, mode)
    ]


def _detect_needed_stages_with_restart_info(
    sim_dir: Path, stages: list[str], mode: str
) -> list[dict]:
    """Return stage work items ``[{"stage", "restart", "cpt_file"}, ...]`` for incomplete stages."""
    if mode == "force":
        return [{"stage": s, "restart": False, "cpt_file": None} for s in stages]

    needed = []
    for stage in stages:
        state = _detect_stage_state(sim_dir, stage, mode)

        if state["status"] == "complete":
            continue  # Skip completed stages
        elif state["status"] == "partial" and state["restart"]:
            # Can resume from checkpoint
            needed.append(
                {
                    "stage": stage,
                    "restart": True,
                    "cpt_file": state["cpt_file"],
                }
            )
        else:
            # Not started or can't restart - run from beginning
            needed.append(
                {
                    "stage": stage,
                    "restart": False,
                    "cpt_file": None,
                }
            )

    return needed


def _execute_stage_list(
    sim_dir: Path,
    stages: list[str],
    grompp_app,
    mdrun_app,
    stage_restarts: "dict[str, str] | None" = None,
    config: "ExecutorConfig | None" = None,
    max_rescue: int = 3,
    tracker: "StageProgressTracker | None" = None,
) -> "AppFuture | None":
    """Chain and submit stages in dependency order, with rescue retry and progress tracking."""
    if not stages:
        return None

    restarts = stage_restarts or {}
    sim_hash = sim_dir.name

    # Validate that stages are in dependency order (derived from STAGE_REGISTRY — single source)
    stage_order = [s.name for s in STAGE_REGISTRY]
    stage_indices = {name: idx for idx, name in enumerate(stage_order)}

    prev_idx = -1
    for stage in stages:
        if stage not in stage_indices:
            raise ValueError(f"Unknown stage: {stage}. Valid: {stage_order}")

        curr_idx = stage_indices[stage]
        if curr_idx <= prev_idx:
            raise ValueError(f"Stages must be in dependency order: {stage_order}. Got: {stages}")
        prev_idx = curr_idx

    from .mdp import RESCUE_ELIGIBLE_STAGES
    from .rescue import execute_stage_with_rescue

    prev_future = None
    for i, stage in enumerate(stages):
        cpt_file = restarts.get(stage, "")

        if tracker is not None:
            tracker.mark_running(stage, sim_hash)

        try:
            if max_rescue > 0 and stage in RESCUE_ELIGIBLE_STAGES and not cpt_file:
                prev_future = execute_stage_with_rescue(
                    sim_dir,
                    stage,
                    prev_future,
                    grompp_app,
                    mdrun_app,
                    max_rescue=max_rescue,
                    config=config,
                )
                # Rescue stages block — if we get here, it succeeded
                if tracker is not None:
                    tracker.mark_succeeded(stage, sim_hash)
            else:
                stage_cfg = get_stage_config_or_none(config, stage)
                cfg_kwarg = {"stage_config": stage_cfg} if stage_cfg is not None else {}
                prev_future = run_stage(
                    STAGE_BY_NAME[stage],
                    sim_dir,
                    prev_future,
                    grompp_app,
                    mdrun_app,
                    restart_from_cpt=cpt_file,
                    **cfg_kwarg,
                )
                # Non-rescue stages return immediately — track via callback
                if tracker is not None:
                    _s, _h, _t = stage, sim_hash, tracker

                    def _on_done(fut, s=_s, h=_h, t=_t):
                        if fut.exception() is not None:
                            t.mark_failed(s, h)
                        else:
                            t.mark_succeeded(s, h)

                    prev_future.add_done_callback(_on_done)

        except Exception:
            if tracker is not None:
                tracker.mark_failed(stage, sim_hash)
                for skip_stage in stages[i + 1 :]:
                    tracker.mark_skipped(skip_stage, sim_hash)
            raise

    return prev_future


def _validate_stage_prerequisites(sim_dir: Path, first_stage: str) -> None:
    """Fail fast if prerequisite files for *first_stage* are missing."""
    # Prerequisites: files needed BEFORE this stage can start.
    # Derived from StageSpec: the input .gro file and, when present, the
    # prerequisite checkpoint (which is also the grompp -t velocity input).
    spec = STAGE_BY_NAME[first_stage]
    required: list[Path] = []
    if spec.gro_in and spec.gro_in != "system.pdb":
        required.append(sim_dir / spec.gro_in)
    if spec.prereq_cpt:
        required.append(sim_dir / spec.prereq_cpt)
    missing = [f.name for f in required if not f.exists()]

    if missing:
        raise FileNotFoundError(
            f"Cannot start {first_stage} in {sim_dir.name}: "
            f"missing prerequisite files: {missing}\n\n"
            f"Resolution:\n"
            f"  1. Run earlier stages first:\n"
            f"     mdfactory simulate {sim_dir} --stages EM NVT\n"
            f"  2. Or force overwrite all:\n"
            f"     mdfactory simulate {sim_dir} --checkpoint force"
        )


def _validate_trajectory_complete(
    sim_dir: Path, traj_file: str, expected_frames: int | None = None
) -> bool:
    """Return True if *traj_file* exists, is readable via MDAnalysis, and has enough frames."""
    traj_path = sim_dir / traj_file

    if not traj_path.exists():
        return False

    # Quick check: empty file
    if traj_path.stat().st_size == 0:
        return False

    # Try to read with MDAnalysis
    try:
        if mda is None:
            raise ImportError("MDAnalysis not available")

        # Find structure file for topology
        structure_file = find_structure_file(sim_dir)
        if not structure_file:
            logger.warning(f"No structure file found in {sim_dir}, skipping frame check")
            # Cannot validate frames without a topology — treat as incomplete so
            # the caller can decide (partial restart will regenerate if needed).
            return False

        # Load trajectory and count frames
        u = mda.Universe(str(structure_file), str(traj_path))
        num_frames = len(u.trajectory)

        logger.debug(f"{traj_file}: {num_frames} frames")

        if expected_frames is not None:
            return num_frames >= expected_frames
        else:
            # If no expectation, just check it's readable and non-trivial
            return num_frames > 0

    except Exception as e:
        logger.warning(f"Trajectory validation failed for {traj_file}: {e}")
        # A parse failure means the file is corrupt or truncated — not complete.
        # Return False so the caller triggers a partial restart rather than
        # silently skipping the stage on a bad trajectory.
        return False


#: Candidate structure files checked by :func:`find_structure_file` in
#: priority order (most-equilibrated first), followed by the raw input.
#: Derived from :data:`~mdfactory.orchestration.stages.STAGE_REGISTRY` at
#: import time — add a new stage to the registry and this list stays in sync
#: automatically without any further edits to this module.
_STRUCTURE_CANDIDATES: list[str] = [
    spec.gro_out for spec in reversed(STAGE_REGISTRY) if spec.gro_out
] + ["system.pdb"]


def find_structure_file(sim_dir: Path) -> Path | None:
    """Find the best available structure file in a simulation directory.

    Checks candidates in GROMACS output priority order so that the most
    equilibrated coordinates are used when available.  This is the canonical
    priority list shared by trajectory validation (this module) and benchmark
    pre-processing (:mod:`mdfactory.performance.benchmark`).

    The candidate list is derived from
    :data:`~mdfactory.orchestration.stages.STAGE_REGISTRY` at import time
    (see :data:`_STRUCTURE_CANDIDATES`), so adding a new stage with a
    ``gro_out`` field automatically extends the search without modifying this
    function.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.

    Returns
    -------
    Path or None
        Path to the first existing structure file, or ``None`` if none of the
        candidates are found.

    Notes
    -----
    Priority order (highest to lowest):

    1. Most-recently-added stage's ``.gro`` (most equilibrated)
    2. …earlier stages in reverse registry order…
    3. ``system.pdb`` — raw starting structure

    """
    for candidate in _STRUCTURE_CANDIDATES:
        path = sim_dir / candidate
        if path.exists():
            return path
    return None


def _extract_expected_frames_from_mdp(sim_dir: Path, stage: str) -> int | None:
    """Extract expected frame count from MDP file (nsteps / output_frequency)."""
    from .mdp import get_mdp_value, parse_mdp

    mdp_path = sim_dir / STAGE_BY_NAME.get(stage, STAGE_BY_NAME["Production"]).mdp_file
    if not mdp_path.exists():
        return None
    try:
        parsed = parse_mdp(mdp_path)
        nsteps_val = get_mdp_value(parsed, "nsteps")
        # XTC frequency preferred; fall back to TRR
        nstxout_val = get_mdp_value(parsed, "nstxout_compressed") or get_mdp_value(
            parsed, "nstxout"
        )
        if nsteps_val and nstxout_val:
            nsteps = int(nsteps_val)
            nstxout = int(nstxout_val)
            if nstxout > 0:
                return nsteps // nstxout
    except Exception as e:
        logger.debug(f"Could not parse MDP file: {e}")
    return None


def _log_dry_run_plan(work_plan: list[dict], config: "ExecutorConfig") -> list[dict]:
    """Log the dry-run work plan with resolved gmx commands per stage."""
    from .apps import _build_grompp_script, _build_mdrun_script
    from .stages import _extract_resource_hints

    logger.info("=" * 60)
    logger.info("DRY RUN - No jobs will be submitted")
    logger.info("=" * 60)

    for item in work_plan:
        logger.info(f"Simulation: {item['hash']}")
        logger.info(f"  Directory: {item['sim_dir']}")
        if item["stages"]:
            restarts = item.get("stage_restarts", {})
            stage_parts = []
            for s in item["stages"]:
                label = f"{s} (resume from {restarts[s]})" if s in restarts else s
                stage_parts.append(label)
            logger.info(f"  Stages: {', '.join(stage_parts)}")

            # Resolve and display actual commands per stage
            work_dir = str(Path(item["sim_dir"]).resolve())
            stage_config = get_stage_config_or_none(config, item["stages"][0])
            hints = _extract_resource_hints(stage_config)

            for stage_name in item["stages"]:
                spec = STAGE_BY_NAME[stage_name]
                restart_cpt = restarts.get(stage_name, "")

                logger.info(f"  [{spec.name}]")

                if restart_cpt:
                    # Restart: skip grompp, resume mdrun from checkpoint
                    logger.info(f"    grompp: skipped (restart from {restart_cpt})")
                else:
                    grompp_script = _build_grompp_script(
                        mdp_file=spec.mdp_file,
                        gro_file=spec.gro_in,
                        top_file="topology.top",
                        tpr_file=spec.tpr_file,
                        work_dir=work_dir,
                        ref_file=spec.ref_file or "",
                        cpt_file=spec.prereq_cpt or "",
                        maxwarn=spec.maxwarn,
                    )
                    _log_resolved_script(grompp_script, label="grompp")

                mdrun_script = _build_mdrun_script(
                    deffnm=spec.deffnm,
                    work_dir=work_dir,
                    restart_from_cpt=restart_cpt,
                    ntasks=hints.ntasks,
                    disable_gpu=hints.disable_gpu,
                    pme_gpu=spec.supports_pme_gpu,
                    gro_out=spec.gro_out,
                    traj_files=spec.traj_files,
                    gmx_binary=hints.gmx_binary,
                )
                _log_resolved_script(mdrun_script, label="mdrun")
        else:
            logger.info("  Stages: None (all complete)")

    logger.info("=" * 60)
    logger.info(f"Executor: {config.provider}")
    if hasattr(config, "account"):
        logger.info(f"SLURM Account: {config.account}")
        logger.info(f"SLURM Partition: {config.partition}")

    return work_plan


def _log_resolved_script(script: str, *, label: str) -> None:
    """Extract and log the ``$GMX_BIN`` command line from a generated bash script."""
    # Extract the actual gmx command line(s) — lines starting with $GMX_BIN
    # or containing "gmx" invocations, skipping boilerplate (set, cd, echo, if)
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("$GMX_BIN"):
            # Remove trailing shell continuation characters for display
            cmd = stripped.rstrip("\\").strip()
            logger.info(f"    {label}: {cmd}")
            return

    # Fallback: show the label with an indicator that resolution failed
    logger.info(f"    {label}: <could not resolve command>")
