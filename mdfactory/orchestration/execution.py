# ABOUTME: Stage execution mechanics for GROMACS simulation pipelines
# ABOUTME: Chains stages in dependency order with rescue retry and progress tracking
"""Stage execution for simulation pipelines.

Provides :func:`_execute_stage_list` (stage chaining with rescue retry
and progress tracking) and :func:`_validate_stage_prerequisites`
(fail-fast prerequisite check before Parsl submission).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import get_stage_config_or_none
from .stages import STAGE_BY_NAME, STAGE_REGISTRY, run_stage

if TYPE_CHECKING:
    from parsl import AppFuture

    from .config import ExecutorConfig
    from .progress import StageProgressTracker


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
