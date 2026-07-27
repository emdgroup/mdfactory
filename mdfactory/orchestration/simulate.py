# ABOUTME: Main dispatcher for Parsl-based GROMACS simulation orchestration
# ABOUTME: Handles checkpoint detection, dry-run, and progress monitoring
"""GROMACS simulation orchestration via Parsl.

Provides :func:`run_simulations`, the main entry point for orchestrating
GROMACS MD simulations via Parsl. Handles checkpoint detection, dry-run mode,
and progress monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .apps import get_grompp_app, get_mdrun_app
from .build import _wait_with_progress
from .session import parsl_session
from .stages import (
    STAGE_BY_NAME,
    STAGE_REGISTRY,
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


def _bash_result_to_dict(raw: object, sim_hash: str) -> dict:
    """Normalise a bash_app result to a status dict.

    Parsl ``@bash_app`` futures resolve to ``None`` or an integer exit code
    on success — neither of which satisfies the ``dict`` contract expected by
    :func:`_report_simulation_results`.  This transform converts any non-dict
    value to ``{"hash": sim_hash, "status": "success"}``.  Dict values (e.g.
    from ``@python_app`` callers) are passed through unchanged.

    Parameters
    ----------
    raw : object
        Raw value from ``AppFuture.result()``.
    sim_hash : str
        Display identifier for the simulation (used as ``hash`` key).

    Returns
    -------
    dict
        Result dict with at minimum ``hash`` and ``status`` keys.

    """
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
    checkpoint_mode: str = "auto",
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
    checkpoint_mode : str
        - "auto": Skip stages with valid outputs (default)
        - "skip": Never re-run completed stages
        - "force": Overwrite all stages

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

    # 1. Filter: skip directories whose build did not complete
    ready_paths = []
    for sim_dir in sim_paths:
        missing = _missing_build_files(sim_dir, stages)
        if missing:
            logger.warning(f"Skipping {sim_dir.name}: build incomplete, missing: {missing}")
        else:
            ready_paths.append(sim_dir)

    n_skipped = len(sim_paths) - len(ready_paths)
    if n_skipped:
        logger.info(
            f"Skipped {n_skipped} simulation(s) with incomplete builds; "
            f"{len(ready_paths)} ready to simulate."
        )
    if not ready_paths:
        logger.warning("No simulation directories have complete builds. Nothing to run.")
        return []
    sim_paths = ready_paths

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
        return _log_dry_run_plan(work_plan, config)

    # 4. Validate prerequisites before Parsl session
    for item in work_plan:
        if item["stages"]:
            # Validate first stage has required inputs
            _validate_stage_prerequisites(item["sim_dir"], item["stages"][0])

    # 5. Parsl session
    with parsl_session(config) as session:
        grompp_app = get_grompp_app()
        mdrun_app = get_mdrun_app()

        # 6. Submit all pipelines (embarrassingly parallel)
        futures = []
        for item in work_plan:
            if not item["stages"]:
                logger.info(f"Skipping {item['hash']} (all stages complete)")
                continue

            # Execute only the needed stages with explicit control
            pipeline_fut = _execute_stage_list(
                item["sim_dir"],
                item["stages"],
                grompp_app,
                mdrun_app,
                stage_restarts=item.get("stage_restarts"),
                config=config,
            )
            futures.append((item["hash"], pipeline_fut))

        logger.info(f"Submitted {len(futures)} simulation(s)")

        if not wait:
            logger.warning(
                "Returning raw AppFuture objects — futures resolve to None (bash_app exit), "
                "not status dicts.  Caller must call parsl.clear() when done."
            )
            session.detach()
            return [fut for _, fut in futures]

        # 7. Wait with progress UI (reuse from build.py)
        hashes = [h for h, _ in futures]
        parsl_futures = [fut for _, fut in futures]
        return _wait_with_progress(
            parsl_futures,
            hashes=hashes,
            label="Simulations",
            result_transform=_bash_result_to_dict,
        )


def _missing_build_files(sim_dir: Path, stages: list[str]) -> list[str]:
    """Return the names of required build output files that are absent.

    Checks for ``system.pdb``, ``topology.top``, and the MDP file for every
    requested stage.  Returns an empty list when the build is complete.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stages : list[str]
        Stages to run (determines which MDP files are required).

    Returns
    -------
    list[str]
        Names of missing files, or ``[]`` when the build is complete.

    """
    required = ["system.pdb", "topology.top"] + [STAGE_BY_NAME[s].mdp_file for s in stages]
    return [f for f in required if not (sim_dir / f).exists()]


def _validate_simulation_dir(sim_dir: Path, stages: list[str]) -> None:
    """Raise if any required build output files are missing.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stages : list[str]
        Stages to run.

    Raises
    ------
    FileNotFoundError
        If required files are missing.

    """
    missing = _missing_build_files(sim_dir, stages)
    if missing:
        raise FileNotFoundError(f"Missing files in {sim_dir}: {missing}")


def _detect_skip_mode_state(output_file: Path, cpt_file: Path, tpr_file: Path) -> dict:
    """Return stage state dict for 'skip' checkpoint mode (no integrity checks).

    Parameters
    ----------
    output_file : Path
        Primary output file (gro or trajectory).
    cpt_file : Path
        Stage checkpoint file.
    tpr_file : Path
        Stage TPR run-input file.

    Returns
    -------
    dict
        Stage state dict with ``status``, ``cpt_file``, and ``restart`` keys.

    """
    if output_file.exists():
        return {"status": "complete", "cpt_file": None, "restart": False}
    if cpt_file.exists() and tpr_file.exists():
        return {"status": "partial", "cpt_file": cpt_file, "restart": True}
    return {"status": "not_started", "cpt_file": None, "restart": False}


def _detect_production_output_state(
    sim_dir: Path,
    cpt_file: Path,
    tpr_file: Path,
    traj_files: "tuple[str, ...]",
) -> dict:
    """Return stage state dict for Production when trajectory output exists.

    Validates frame count; falls back to partial restart if every trajectory
    file is incomplete and a checkpoint/TPR pair exists.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory (must contain prod.mdp).
    cpt_file : Path
        Production checkpoint file.
    tpr_file : Path
        Production TPR file.
    traj_files : tuple of str
        Accepted trajectory output filenames from :attr:`StageSpec.traj_files`
        (e.g. ``("prod.xtc", "prod.trr")``).  The stage is marked *complete*
        when **any** of these passes :func:`_validate_trajectory_complete`,
        covering both XTC-only and TRR-only MDP configurations.

    Returns
    -------
    dict
        Stage state dict with ``status``, ``cpt_file``, and ``restart`` keys.

    """
    expected_frames = _extract_expected_frames_from_mdp(sim_dir, "Production")
    for traj_file in traj_files:
        if _validate_trajectory_complete(sim_dir, traj_file, expected_frames):
            return {"status": "complete", "cpt_file": None, "restart": False}
    if cpt_file.exists() and tpr_file.exists():
        return {"status": "partial", "cpt_file": cpt_file, "restart": True}
    # All trajectory files incomplete and no checkpoint to restart from
    return {"status": "partial", "cpt_file": None, "restart": False}


def _detect_stage_state(sim_dir: Path, stage: str, mode: str = "auto") -> dict:
    """Detect completion or partial progress of a stage.

    In "auto" mode, validates workflow integrity by checking both output AND prerequisite
    checkpoint files. This ensures scientific correctness - we can't trust npt.gro if
    nvt.cpt is missing, as we can't verify NPT used correct initial conditions.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stage : str
        Stage name.
    mode : str
        Checkpoint mode: "auto", "skip", or "force".

    Returns
    -------
    dict
        {"status": "complete"|"partial"|"not_started", "cpt_file": Path|None}

    """
    spec = STAGE_BY_NAME[stage]
    cpt_file = sim_dir / spec.cpt_file
    tpr_file = sim_dir / spec.tpr_file

    # For trajectory stages (Production) any accepted file counts as output.
    # For structure stages (EM/NVT/NPT) the single gro_out is the output.
    if spec.traj_files:
        output_exists = any(
            (sim_dir / tf).exists() and (sim_dir / tf).stat().st_size > 0 for tf in spec.traj_files
        )
    else:
        gro_out_file = sim_dir / spec.gro_out
        output_exists = gro_out_file.exists() and gro_out_file.stat().st_size > 0

    # For "skip" mode, just check if output file exists (even if empty).
    # Don't validate workflow integrity - trust the user.
    if mode == "skip":
        # Use first traj file or gro_out as the representative output for skip mode.
        primary_output = spec.traj_files[0] if spec.traj_files else spec.gro_out
        return _detect_skip_mode_state(sim_dir / primary_output, cpt_file, tpr_file)

    # Prerequisite checkpoint (for validating workflow integrity in auto mode)
    prereq_cpt = sim_dir / spec.prereq_cpt if spec.prereq_cpt else None

    # For "auto" mode, validate workflow integrity by checking prerequisite checkpoints.
    if output_exists:
        # Check workflow integrity: prerequisite checkpoint must exist.
        if prereq_cpt and not prereq_cpt.exists():
            # Output exists but prerequisite checkpoint missing - workflow integrity broken.
            # Can't trust this output; need to re-run from earlier stage.
            return {"status": "not_started", "cpt_file": None, "restart": False}
        if stage == "Production":
            return _detect_production_output_state(sim_dir, cpt_file, tpr_file, spec.traj_files)
        # Non-trajectory stages: output exists + prerequisite valid = complete.
        return {"status": "complete", "cpt_file": None, "restart": False}

    # Check partial progress (checkpoint exists, output doesn't).
    if cpt_file.exists() and tpr_file.exists():
        return {"status": "partial", "cpt_file": cpt_file, "restart": True}

    return {"status": "not_started", "cpt_file": None, "restart": False}


def _detect_needed_stages(sim_dir: Path, stages: list[str], mode: str) -> list[str]:
    """Determine which stages need to run based on checkpoint mode.

    Thin wrapper around :func:`_detect_needed_stages_with_restart_info` that
    returns only stage names, discarding restart metadata.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stages : list[str]
        Requested stages.
    mode : str
        Checkpoint mode: "auto", "skip", or "force".

    Returns
    -------
    list[str]
        Stages that need to run.

    """
    return [
        item["stage"] for item in _detect_needed_stages_with_restart_info(sim_dir, stages, mode)
    ]


def _detect_needed_stages_with_restart_info(
    sim_dir: Path, stages: list[str], mode: str
) -> list[dict]:
    """Determine which stages need to run with checkpoint restart information.

    Extended version that returns restart info for -cpi -append support (Phase 2).

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stages : list[str]
        Requested stages.
    mode : str
        Checkpoint mode: "auto", "skip", or "force".

    Returns
    -------
    list[dict]
        List of stage work items: [{"stage": "NPT", "restart": True, "cpt_file": Path}, ...]

    """
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
) -> "AppFuture | None":
    """Execute a list of stages with automatic dependency chaining.

    Explicitly calls stage functions based on the needed_stages list,
    validating stage ordering and chaining dependencies via inputs=[].

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stages : list[str]
        Ordered stages to execute (e.g., ["EM", "NVT"], ["NPT", "Production"]).
    grompp_app : callable
        Grompp app from get_grompp_app().
    mdrun_app : callable
        Mdrun app from get_mdrun_app().
    stage_restarts : dict[str, str] or None, optional
        Maps stage name → absolute path of checkpoint file to resume from.
        When a stage has an entry here, grompp is skipped and mdrun is called
        with ``-cpi <file> -append``.  Stages absent from the dict (or when
        the dict is ``None``) run normally from scratch.
    config : ExecutorConfig or None, optional
        Executor configuration.  When a :class:`SlurmExecutorConfig` is
        provided, per-stage resource overrides are applied via
        :meth:`~mdfactory.orchestration.config.SlurmExecutorConfig.get_stage_config`
        before each stage function call.

    Returns
    -------
    AppFuture or None
        Final future from last stage, or None if no stages to run.

    Raises
    ------
    ValueError
        If stage dependencies are not met or stages are out of order.

    """
    if not stages:
        return None

    restarts = stage_restarts or {}

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

    # Execute stages sequentially, chaining futures through run_stage.
    # STAGE_REGISTRY is the single dispatch source: a new stage added there
    # becomes immediately executable here without further edits to this module.
    # prev_future=None on the first iteration is handled correctly by run_stage
    # for all stages (including non-EM checkpoint resumes).
    prev_future = None
    for stage in stages:
        cpt_file = restarts.get(stage, "")
        # Resolve per-stage resource overrides when a SlurmExecutorConfig is given.
        # Only inject stage_config kwarg when non-None to avoid polluting mock
        # assertions in tests that don't exercise per-stage config.
        stage_cfg = config.get_stage_config(stage) if hasattr(config, "get_stage_config") else None
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

    return prev_future


def _validate_stage_prerequisites(sim_dir: Path, first_stage: str) -> None:
    """Validate that all prerequisite files exist for the first stage.

    Fails fast on login node with clear error message before SLURM submission.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    first_stage : str
        First stage to run (e.g., "NPT" when resuming from checkpoint).

    Raises
    ------
    FileNotFoundError
        If required input files are missing. Error message includes
        actionable guidance with fix commands.

    """
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
    """Check if trajectory file is complete and readable.

    Uses MDAnalysis to count frames. Falls back to size heuristic if MDAnalysis fails.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    traj_file : str
        Trajectory filename (e.g., "prod.xtc").
    expected_frames : int, optional
        Expected number of frames. If None, checks readability only.

    Returns
    -------
    bool
        True if trajectory is complete and readable.

    """
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
            logger.warning(f"No structure file found in {sim_dir}, using size check fallback")
            # Heuristic: typical prod.xtc is > 10 MB for any reasonable system
            return traj_path.stat().st_size > 10_000_000

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
        # Fallback: size-based heuristic
        # Typical prod.xtc is > 10 MB for any reasonable system
        return traj_path.stat().st_size > 10_000_000


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
    """Extract expected frame count from MDP file.

    Reads nsteps and nstxout-compressed from MDP to compute expected frames.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stage : str
        Stage name (EM, NVT, NPT, Production).

    Returns
    -------
    int or None
        Expected number of frames, or None if cannot be determined.

    """
    mdp_path = sim_dir / STAGE_BY_NAME.get(stage, STAGE_BY_NAME["Production"]).mdp_file

    if not mdp_path.exists():
        return None

    try:
        content = mdp_path.read_text()
        nsteps = None
        nstxout = None

        for raw_line in content.split("\n"):
            line = raw_line.split(";")[0].strip()  # Remove comments
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key == "nsteps":
                    nsteps = int(val)
                elif key == "nstxout-compressed":
                    nstxout = int(val)

        if nsteps and nstxout:
            return nsteps // nstxout
    except Exception as e:
        logger.debug(f"Could not parse MDP file: {e}")

    return None


def _log_dry_run_plan(work_plan: list[dict], config: "ExecutorConfig") -> list[dict]:
    """Log what would be executed without running.

    Parameters
    ----------
    work_plan : list[dict]
        Work plan with sim_dir, hash, and stages.
    config : ExecutorConfig
        Executor configuration.

    Returns
    -------
    list[dict]
        The work plan (for testing).

    """
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
        else:
            logger.info("  Stages: None (all complete)")

    logger.info("=" * 60)
    logger.info(f"Executor: {config.provider}")
    if hasattr(config, "account"):
        logger.info(f"SLURM Account: {config.account}")
        logger.info(f"SLURM Partition: {config.partition}")

    return work_plan
