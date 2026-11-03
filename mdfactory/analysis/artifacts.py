# ABOUTME: Artifact producers and registry for simulation artifacts
# ABOUTME: Creates derivative files (e.g., last-frame PDB) from completed simulations
"""Artifact producers and registry for simulation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .bilayer.artifacts import render_bilayer_movie, render_bilayer_snapshot
from .bilayer.conformational_density import conformational_density_map

if TYPE_CHECKING:
    from .simulation import Simulation


def create_last_frame_pdb_artifact(
    simulation: Simulation,
    filename: str = "last_frame.pdb",
    **_kwargs,
) -> Path:
    """Write the last frame of the trajectory to a PDB file.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance owning the artifact
    filename : str
        Output filename

    Returns
    -------
    Path
        Path to the created PDB file

    """

    output_path = simulation.path / filename
    universe = simulation.universe
    universe.trajectory[-1]

    # MDAnalysis Universe.atoms is never None for a valid universe
    universe.atoms.write(str(output_path))
    return output_path


ARTIFACT_REGISTRY: dict[str, dict[str, Callable]] = {
    "bilayer": {
        "last_frame_pdb": create_last_frame_pdb_artifact,
        "bilayer_snapshot": render_bilayer_snapshot,
        "bilayer_movie": render_bilayer_movie,
        "conformational_density": conformational_density_map,
    },
    "mixedbox": {
        "last_frame_pdb": create_last_frame_pdb_artifact,
    },
    "proteinbox": {
        "last_frame_pdb": create_last_frame_pdb_artifact,
    },
    "protein_mixedbox": {
        "last_frame_pdb": create_last_frame_pdb_artifact,
    },
}
