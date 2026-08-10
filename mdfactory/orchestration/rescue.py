# ABOUTME: Rescue retry loop for GROMACS physics failures
# ABOUTME: Retries failed stages with halved MDP parameters (binary division)
"""Adaptive rescue retry for GROMACS simulation stages.

When a stage fails due to a physics error (overlapping atoms, exploding box,
constraint failures), the rescue loop modifies MDP parameters using binary
division (halving step sizes, doubling nsteps) and re-submits the stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from .errors import FailureType, classify_failure
from .mdp import apply_rescue_tier
from .stages import STAGE_BY_NAME, run_stage

if TYPE_CHECKING:
    from parsl import AppFuture

    from .config import ExecutorConfig


def _clean_partial_outputs(sim_dir: Path, stage: str) -> None:
    """Remove partial outputs from a failed stage so it can be re-run cleanly.

    Removes the ``.tpr``, ``.cpt``, ``.gro`` output, ``.log``, ``.edr``,
    and trajectory files for the given stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stage : str
        Stage name.

    """
    spec = STAGE_BY_NAME[stage]
    candidates = [
        spec.tpr_file,
        spec.cpt_file,
        spec.gro_out,
        f"{spec.deffnm}.log",
        f"{spec.deffnm}.edr",
    ]
    candidates.extend(spec.traj_files)

    for fname in candidates:
        if not fname:
            continue
        f = sim_dir / fname
        if f.exists():
            f.unlink()
            logger.debug(f"Removed partial output: {f}")


def execute_stage_with_rescue(
    sim_dir: Path,
    stage: str,
    prev_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    *,
    max_rescue: int = 3,
    config: "ExecutorConfig | None" = None,
) -> "AppFuture":
    """Execute a single stage with adaptive rescue on physics failures.

    Submits the stage and waits for its result.  If the stage fails with
    a physics error and rescue tiers remain, modifies the MDP parameters
    (binary division), cleans partial outputs, and re-submits.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stage : str
        Stage name (must be in :data:`RESCUE_ELIGIBLE_STAGES`).
    prev_future : AppFuture or None
        Dependency future from the preceding stage.
    grompp_app : callable
        Result of :func:`~mdfactory.orchestration.apps.get_grompp_app`.
    mdrun_app : callable
        Result of :func:`~mdfactory.orchestration.apps.get_mdrun_app`.
    max_rescue : int
        Maximum number of rescue tiers to attempt.
    config : ExecutorConfig or None, optional
        Executor configuration for per-stage resource hints.

    Returns
    -------
    AppFuture
        A completed future (already resolved — the result has been waited on).

    Raises
    ------
    Exception
        Re-raises the stage failure if rescue is exhausted or the failure
        is not physics-related.

    """
    spec = STAGE_BY_NAME[stage]
    stage_cfg = config.get_stage_config(stage) if hasattr(config, "get_stage_config") else None
    cfg_kwarg: dict[str, Any] = {"stage_config": stage_cfg} if stage_cfg is not None else {}

    for tier in range(max_rescue + 1):
        mdp_override = None
        if tier > 0:
            # Apply rescue tier modifications
            rescue_mdp = apply_rescue_tier(sim_dir, stage, tier)
            mdp_override = rescue_mdp.name
            logger.warning(
                f"RESCUE tier {tier}/{max_rescue} for {sim_dir.name}/{stage}: using {mdp_override}"
            )

        future = run_stage(
            spec,
            sim_dir,
            prev_future,
            grompp_app,
            mdrun_app,
            mdp_override=mdp_override,
            **cfg_kwarg,
        )

        try:
            future.result()
            if tier > 0:
                logger.warning(f"RESCUE succeeded for {sim_dir.name}/{stage} at tier {tier}")
            return future
        except Exception as exc:
            failure_type = classify_failure(exc, sim_dir, stage)

            if failure_type != FailureType.PHYSICS:
                logger.debug(
                    f"{sim_dir.name}/{stage}: non-physics failure "
                    f"({failure_type.value}), not rescuable"
                )
                raise

            if tier >= max_rescue:
                logger.error(
                    f"RESCUE exhausted for {sim_dir.name}/{stage} after {max_rescue} tier(s)"
                )
                raise

            logger.warning(
                f"{sim_dir.name}/{stage}: physics failure detected, "
                f"will attempt rescue tier {tier + 1}"
            )
            _clean_partial_outputs(sim_dir, stage)

    # Unreachable — the loop always returns or raises
    raise RuntimeError("Rescue loop exited unexpectedly")  # pragma: no cover
