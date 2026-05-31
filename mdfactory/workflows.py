# ABOUTME: High-level build workflow entry points
# ABOUTME: Dispatches builds from dict, YAML, or BuildInput objects
"""High-level build workflow entry points."""

from pathlib import Path
from typing import Any

from .build import build_bilayer, build_lnp, build_mixedbox, build_proteinbox
from .models.input import BuildInput
from .utils.utilities import load_yaml_file

DISPATCH_BUILD = {
    "mixedbox": build_mixedbox,
    "bilayer": build_bilayer,
    "lnp": build_lnp,
    "proteinbox": build_proteinbox,
}


def run_build_from_dict(inp_dict: dict[Any, Any] | BuildInput):
    """Dispatch a build from a dict or BuildInput object.

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
    build_function(build_model)


def run_build_from_file(fname: Path):
    """Load a YAML build specification and dispatch the build.

    Parameters
    ----------
    fname : Path
        Path to the YAML build specification file

    """
    dct = load_yaml_file(fname)
    run_build_from_dict(dct)
