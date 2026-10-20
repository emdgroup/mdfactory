# ABOUTME: Checkpoint detection logic for GROMACS simulation stage progression
# ABOUTME: Determines which stages need to run based on mode (auto/skip/force) and existing outputs
"""Checkpoint detection for simulation stage progression.

Determines which pipeline stages still need to run based on the checkpoint
*mode* (``auto`` / ``skip`` / ``force``) and existing output files.  The
core entry point is :func:`_detect_needed_stages_with_restart_info`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from .stages import STAGE_BY_NAME, StageSpec
from .trajectory import _extract_expected_frames_from_mdp, _validate_trajectory_complete


class _StageState(TypedDict):
    """Checkpoint detection result for a single stage."""

    status: str  # "complete" | "partial" | "not_started"
    cpt_file: "Path | None"
    restart: bool


def _has_restart_pair(cpt_file: Path, tpr_file: Path) -> bool:
    """Return True if both checkpoint and TPR files exist (restartable state)."""
    return cpt_file.exists() and tpr_file.exists()


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
            continue
        needed.append({"stage": stage, "restart": state["restart"], "cpt_file": state["cpt_file"]})

    return needed
