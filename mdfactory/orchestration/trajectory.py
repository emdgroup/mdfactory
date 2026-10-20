# ABOUTME: Trajectory validation and structure file discovery for GROMACS simulations
# ABOUTME: Checks frame counts via MDAnalysis and locates best-available structure files
"""Trajectory validation and structure file discovery.

Provides :func:`find_structure_file` (shared utility for locating the best
available structure file) and trajectory completeness checking used by the
checkpoint detection layer.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from .stages import STAGE_BY_NAME, STAGE_REGISTRY

# Import MDAnalysis at module level for testability
try:
    import MDAnalysis as mda
except ImportError:
    mda = None


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


def _validate_trajectory_complete(
    sim_dir: Path, traj_file: str, expected_frames: int | None = None
) -> bool:
    """Return True if *traj_file* exists, is readable via MDAnalysis, and has enough frames."""
    traj_path = sim_dir / traj_file

    if not traj_path.exists() or traj_path.stat().st_size == 0:
        return False

    if mda is None:
        logger.warning(f"Trajectory validation skipped for {traj_file}: MDAnalysis not available")
        return False

    structure_file = find_structure_file(sim_dir)
    if not structure_file:
        # Cannot validate frames without a topology — treat as incomplete so
        # the caller can decide (partial restart will regenerate if needed).
        logger.warning(f"No structure file found in {sim_dir}, skipping frame check")
        return False

    try:
        u = mda.Universe(str(structure_file), str(traj_path))
        num_frames = len(u.trajectory)
    except Exception as e:
        # Corrupt or truncated trajectory — trigger partial restart rather than
        # silently skipping the stage.
        logger.warning(f"Trajectory validation failed for {traj_file}: {e}")
        return False

    logger.debug(f"{traj_file}: {num_frames} frames")

    if expected_frames is not None:
        return num_frames >= expected_frames
    return num_frames > 0


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
