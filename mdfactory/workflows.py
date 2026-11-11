# ABOUTME: High-level build workflow entry points
# ABOUTME: Dispatches builds from dict, YAML, or BuildInput objects
"""High-level build workflow entry points."""

from pathlib import Path
from typing import Any

from .build import (
    build_bilayer,
    build_lnp,
    build_mixedbox,
    build_protein_mixedbox,
    build_proteinbox,
)
from .models.input import BuildInput, write_build_record

DISPATCH_BUILD = {
    "mixedbox": build_mixedbox,
    "bilayer": build_bilayer,
    "lnp": build_lnp,
    "proteinbox": build_proteinbox,
    "protein_mixedbox": build_protein_mixedbox,
}


def run_build_from_dict(inp_dict: dict[Any, Any] | BuildInput):
    """Dispatch a build from a dict or BuildInput object in the working directory.

    The build record ``<hash>.yaml`` (the input with its resolved simulation
    settings) is written first, then the system and its run schedule.

    Parameters
    ----------
    inp_dict : dict or BuildInput
        Build specification as a raw dict or validated BuildInput

    """
    if isinstance(inp_dict, dict):
        build_model = BuildInput(**inp_dict)
    elif isinstance(inp_dict, BuildInput):
        build_model = inp_dict.model_copy(deep=True)
    else:
        raise TypeError("Input must either be dict or BuildInput.")

    build_function = DISPATCH_BUILD.get(build_model.simulation_type, None)
    if build_function is None:
        raise NotImplementedError(
            f"Build for simulation type {build_model.simulation_type} not yet implemented."
        )
    write_build_record(build_model, Path.cwd())
    build_function(build_model)
