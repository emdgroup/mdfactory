# ABOUTME: Main dispatcher for Parsl-based GROMACS simulation orchestration
# ABOUTME: Handles dry-run logging, stage execution, and progress monitoring
"""GROMACS simulation orchestration via Parsl.

Provides :func:`run_simulations`, the main entry point for orchestrating
GROMACS MD simulations via Parsl. Handles stage execution, dry-run mode,
and progress monitoring.  Checkpoint detection lives in
:mod:`.checkpoint`; trajectory validation in :mod:`.trajectory`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .apps import get_grompp_app, get_mdrun_app
from .checkpoint import _detect_needed_stages_with_restart_info
from .config import get_stage_config_or_none
from .execution import _execute_stage_list, _validate_stage_prerequisites
from .session import parsl_session
from .stages import STAGE_BY_NAME, STAGE_REGISTRY

if TYPE_CHECKING:
    from .config import ExecutorConfig


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
        Results with status, errors, and timing per simulation.

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

    # 6. Parsl session: tracked execution with per-stage progress display
    import threading

    from .errors import _describe_failure
    from .progress import StageProgressTracker, display_stage_progress

    with parsl_session(config):
        grompp_app = get_grompp_app()
        mdrun_app = get_mdrun_app()

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
                tracker.store_result(
                    sim_hash,
                    {
                        "hash": sim_hash,
                        "status": "failed",
                        "error": error_detail,
                        "failure_type": failure_type,
                        "error_detail": error_detail,
                    },
                )

        logger.info(f"Submitted {len(active_items)} simulation(s)")

        # Launch daemon threads — they won't block process exit on Ctrl+C
        workers = []
        for item in active_items:
            t = threading.Thread(target=_run_pipeline, args=(item,), daemon=True)
            t.start()
            workers.append(t)

        # Main thread: display progress until all stages complete
        display_stage_progress(tracker)

        # Ensure all worker threads have stored results before collecting
        for t in workers:
            t.join()

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
