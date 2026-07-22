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
from .stages import run_full_pipeline

if TYPE_CHECKING:
    from .config import ExecutorConfig


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
    if stages is None:
        stages = ["EM", "NVT", "NPT", "Production"]

    # 1. Validate inputs
    for sim_dir in sim_paths:
        _validate_simulation_dir(sim_dir, stages)

    # 2. Checkpoint detection
    work_plan = []
    for sim_dir in sim_paths:
        needed_stages = _detect_needed_stages(sim_dir, stages, checkpoint_mode)
        work_plan.append({
            "sim_dir": sim_dir,
            "hash": sim_dir.name,
            "stages": needed_stages,
        })

    logger.info(f"Prepared work plan for {len(work_plan)} simulation(s)")

    # 3. Dry-run mode
    if dry_run:
        return _log_dry_run_plan(work_plan, config)

    # 4. Parsl session
    with parsl_session(config) as session:
        grompp_app = get_grompp_app()
        mdrun_app = get_mdrun_app()

        # 5. Submit all pipelines (embarrassingly parallel)
        futures = []
        for item in work_plan:
            if not item["stages"]:
                logger.info(f"Skipping {item['hash']} (all stages complete)")
                continue

            # For now, always run full pipeline
            # TODO: Support partial stage execution
            pipeline_fut = run_full_pipeline(item["sim_dir"], grompp_app, mdrun_app)
            futures.append((item["hash"], pipeline_fut))

        logger.info(f"Submitted {len(futures)} simulation(s)")

        if not wait:
            session.detach()
            return [fut for _, fut in futures]

        # 6. Wait with progress UI (reuse from build.py)
        hashes = [h for h, _ in futures]
        parsl_futures = [fut for _, fut in futures]
        return _wait_with_progress(parsl_futures, hashes=hashes, label="Simulations")


def _validate_simulation_dir(sim_dir: Path, stages: list[str]):
    """Ensure required files exist before submission.

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
    required = ["system.pdb", "topology.top"]
    mdp_map = {
        "EM": "em.mdp",
        "NVT": "nvt.mdp",
        "NPT": "npt.mdp",
        "Production": "md.mdp",
    }
    for stage in stages:
        required.append(mdp_map[stage])

    missing = [f for f in required if not (sim_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {sim_dir}: {missing}")


def _detect_needed_stages(sim_dir: Path, stages: list[str], mode: str) -> list[str]:
    """Determine which stages need to run based on checkpoint mode.

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
    if mode == "force":
        return stages

    stage_outputs = {
        "EM": sim_dir / "min.gro",
        "NVT": sim_dir / "nvt.gro",
        "NPT": sim_dir / "npt.gro",
        "Production": sim_dir / "prod.xtc",
    }

    needed = []
    for stage in stages:
        output_file = stage_outputs[stage]

        if mode == "skip" and output_file.exists():
            continue

        # Auto mode: check file exists AND has content
        if mode == "auto" and output_file.exists() and output_file.stat().st_size > 0:
            continue

        needed.append(stage)

    return needed


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
        stages_str = ", ".join(item["stages"]) if item["stages"] else "None (all complete)"
        logger.info(f"  Stages: {stages_str}")

    logger.info("=" * 60)
    logger.info(f"Executor: {config.provider}")
    if hasattr(config, "account"):
        logger.info(f"SLURM Account: {config.account}")
        logger.info(f"SLURM Partition: {config.partition}")

    return work_plan
