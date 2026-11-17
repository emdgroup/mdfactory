# ABOUTME: GROMACS run schedule rendered from plain dicts: stage bases plus per-type/scheme overlays
# ABOUTME: Turns SimulationSettings into exact step counts, applies overrides, writes mdp files
"""GROMACS run schedule (EM, NVT, NPT, production) rendered from plain dicts.

Each stage starts from a base dict. The simulation type adds the coupling
geometry, the force-field family's nonbonded scheme adds the nonbonded block,
the :class:`~mdfactory.models.simulation_settings.SimulationSettings` add the
state point, step counts and output intervals, and the per-stage GROMACS
overrides are applied last. A protein's selection restraints then append their
``-DPOSRES_<NAME>`` flags to the production ``define`` and, for COM-distance
restraints, the production pull block. grompp is the validator of the result.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from mdfactory import __version__
from mdfactory.models.parametrization import PROTEIN_SIMULATION_TYPES
from mdfactory.models.simulation_settings import (
    STAGES,
    MdpValue,
    SimulationSettings,
    normalize_mdp_key,
    to_steps,
)

if TYPE_CHECKING:
    from mdfactory.models.input import BuildInput

#: mdp file per stage. The orchestration reads them by these names
#: (``StageSpec.mdp_file``).
STAGE_FILES: dict[str, str] = {
    "minimization": "em.mdp",
    "nvt": "nvt.mdp",
    "npt": "npt.mdp",
    "production": "md.mdp",
}

#: Index file of a build with COM-distance restraints (default groups plus the
#: pull groups). The stages pass it to grompp (``-n``) when it exists.
INDEX_FILE = "index.ndx"

#: Nonbonded block per scheme, identical in all four stages.
#: charmm: CHARMM36/CGenFF force switch 1.0 -> 1.2 nm.
#: sage: SMIRNOFF 0.9 nm cutoff with a 0.1 nm potential switch (OpenFF Sage;
#: also used for Amber ff14SB + Sage builds).
NONBONDED: dict[str, dict[str, MdpValue]] = {
    "charmm": {
        "coulombtype": "PME",
        "rcoulomb": 1.2,
        "vdwtype": "cutoff",
        "vdw-modifier": "Force-switch",
        "rvdw": 1.2,
        "rvdw-switch": 1.0,
    },
    "sage": {
        "coulombtype": "PME",
        "rcoulomb": 0.9,
        "vdwtype": "cutoff",
        "vdw-modifier": "Potential-switch",
        "rvdw": 0.9,
        "rvdw-switch": 0.8,
    },
}

#: Long-range dispersion correction per scheme and simulation type. CHARMM
#: lipids and proteins were parametrized without it (and it is wrong for
#: bilayer interfaces); Sage always uses it.
DISPCORR: dict[str, dict[str, str]] = {
    "charmm": {
        "mixedbox": "EnerPres",
        "lnp": "EnerPres",
        "bilayer": "no",
        "proteinbox": "no",
        "protein_mixedbox": "no",
    },
    "sage": {
        "mixedbox": "EnerPres",
        "lnp": "EnerPres",
        "bilayer": "EnerPres",
        "proteinbox": "EnerPres",
        "protein_mixedbox": "EnerPres",
    },
}

_BERENDSEN_ISOTROPIC = {"pcoupl": "Berendsen", "pcoupltype": "isotropic", "tau-p": 2.0}
_BERENDSEN_SEMIISOTROPIC = {"pcoupl": "Berendsen", "pcoupltype": "semi-isotropic", "tau-p": 5.0}
_PARRINELLO_RAHMAN_ISOTROPIC = {
    "pcoupl": "Parrinello-Rahman",
    "pcoupltype": "isotropic",
    "tau-p": 5.0,
}

#: Coupling per simulation type: thermostat groups and the NPT/production barostats.
COUPLING: dict[str, dict[str, Any]] = {
    "mixedbox": {
        "tc-grps": ["System"],
        "npt": _BERENDSEN_ISOTROPIC,
        "production": _BERENDSEN_ISOTROPIC,
    },
    "bilayer": {
        "tc-grps": ["System"],
        "npt": _BERENDSEN_SEMIISOTROPIC,
        "production": _BERENDSEN_SEMIISOTROPIC,
    },
    "proteinbox": {
        "tc-grps": ["Protein", "Non-Protein"],
        "npt": _BERENDSEN_ISOTROPIC,
        "production": _PARRINELLO_RAHMAN_ISOTROPIC,
    },
}
COUPLING["lnp"] = COUPLING["mixedbox"]
COUPLING["protein_mixedbox"] = COUPLING["proteinbox"]

COMPRESSIBILITY = "4.5e-5"  # water, bar^-1
TAU_T = 0.1  # ps


def render_stage(
    stage: str,
    *,
    simulation_type: str,
    nonbonded_scheme: str,
    settings: SimulationSettings,
    restraint_macros: tuple[str, ...] = (),
    pull_coordinates: tuple[PullCoordinate, ...] = (),
) -> dict[str, MdpValue | list[MdpValue]]:
    """Return the mdp parameters of one stage.

    Parameters
    ----------
    stage : str
        One of :data:`~mdfactory.models.simulation_settings.STAGES`.
    simulation_type : str
        The build's simulation type.
    nonbonded_scheme : str
        The force-field family's nonbonded scheme (``charmm`` or ``sage``).
    settings : SimulationSettings
        Resolved simulation settings.
    restraint_macros : tuple of str, optional
        ``POSRES_<NAME>`` macros of the protein's selection restraints. The
        production ``define`` gets a ``-D<macro>`` flag for each, appended after
        the overrides, so a user ``define`` keeps them. Other stages ignore them.
    pull_coordinates : tuple of PullCoordinate, optional
        COM-distance restraints. The production stage gets their pull block,
        after the overrides (the ``pull`` keys are build-owned while COM
        restraints exist). Other stages ignore them.

    Returns
    -------
    dict
        Ordered mapping of mdp key to value (a list for multi-valued keys).

    """
    nonbonded = {
        **NONBONDED[nonbonded_scheme],
        "DispCorr": DISPCORR[nonbonded_scheme][simulation_type],
    }
    timestep = settings.timestep

    if stage == "minimization":
        params: dict[str, Any] = {
            "integrator": "steep",
            "emtol": settings.minimization.force_tolerance,
            "emstep": 0.01,
            "nsteps": settings.minimization.max_steps,
            "nstlist": 1,
            "cutoff-scheme": "Verlet",
            "pbc": "xyz",
            **nonbonded,
        }
    else:
        coupling = COUPLING[simulation_type]
        groups = coupling["tc-grps"]
        params = {}
        if stage in ("nvt", "npt"):
            params["define"] = "-DPOSRES"
        params["integrator"] = "md"
        if stage == "production":
            length_ps = settings.production.length * 1000.0
            interval = to_steps(settings.production.output_interval, timestep)
            params["nsteps"] = to_steps(length_ps, timestep)
            params["dt"] = timestep / 1000.0
            params.update(
                {
                    "nstxout": 0,
                    "nstvout": 0,
                    "nstfout": 0,
                    "nstenergy": interval,
                    "nstlog": interval,
                    "nstxout-compressed": interval,
                    "compressed-x-grps": "System",
                }
            )
        else:
            section = getattr(settings, stage)
            interval = to_steps(section.output_interval, timestep)
            params["nsteps"] = to_steps(section.length, timestep)
            params["dt"] = timestep / 1000.0
            params.update(
                {
                    "nstxout": interval,
                    "nstvout": interval,
                    "nstfout": 0,
                    "nstenergy": interval,
                    "nstlog": interval,
                }
            )
        params.update(
            {
                "continuation": "no" if stage == "nvt" else "yes",
                "constraint-algorithm": "lincs",
                "constraints": "h-bonds",
                "lincs-iter": 1,
                "lincs-order": 4,
                "cutoff-scheme": "Verlet",
                "nstlist": 10,
                "pbc": "xyz",
                **nonbonded,
                "pme-order": 4,
                "fourierspacing": 0.16,
                "tcoupl": "V-rescale",
                "tc-grps": groups,
                "tau-t": [TAU_T] * len(groups),
                "ref-t": [settings.temperature] * len(groups),
            }
        )
        if stage == "nvt":
            params["pcoupl"] = "no"
        else:
            barostat = coupling[stage]
            n_pressure = 2 if barostat["pcoupltype"] == "semi-isotropic" else 1
            params.update(barostat)
            params["ref-p"] = [settings.pressure] * n_pressure
            params["compressibility"] = [COMPRESSIBILITY] * n_pressure
        if stage == "production":
            params["refcoord-scaling"] = "com"
        if stage == "nvt":
            params.update({"gen-vel": "yes", "gen-temp": settings.temperature, "gen-seed": -1})
        else:
            params["gen-vel"] = "no"

    params = _apply_overrides(params, getattr(settings.gromacs_overrides, stage))
    if stage == "production" and restraint_macros:
        params = _append_defines(params, restraint_macros)
    if stage == "production" and pull_coordinates:
        interval = to_steps(settings.production.output_interval, timestep)
        params = {**params, **_pull_block(pull_coordinates, interval)}
    return params


def _apply_overrides(params: dict[str, Any], overrides: dict[str, MdpValue]) -> dict[str, Any]:
    """Apply raw overrides: replace the value of an existing key, append new keys."""
    existing = {normalize_mdp_key(key): key for key in params}
    result = dict(params)
    for key, value in overrides.items():
        result[existing.get(normalize_mdp_key(key), key)] = value
    return result


def _append_defines(params: dict[str, Any], macros: tuple[str, ...]) -> dict[str, Any]:
    """Append ``-D<macro>`` flags to ``define``; a missing ``define`` is added first."""
    key = next((k for k in params if normalize_mdp_key(k) == "define"), None)
    flags = [f"-D{macro}" for macro in macros]
    if key is None:
        return {"define": " ".join(flags), **params}
    current = format_mdp_value(params[key]).strip()
    return {**params, key: " ".join([current, *flags] if current else flags)}


class PullCoordinate(NamedTuple):
    """One COM-distance restraint as a GROMACS pull coordinate (geometry ``distance``)."""

    #: Index group names of the two selections.
    groups: tuple[str, str]
    #: 1-based periodic reference atom of each group.
    pbcatoms: tuple[int, int]
    #: ``umbrella`` or ``flat-bottom``.
    potential: str
    #: Reference distance in nm (``pull-coord-init``).
    init: float
    #: Force constant in kJ mol^-1 nm^-2.
    k: float


def _pull_block(coordinates: tuple[PullCoordinate, ...], interval: int) -> dict[str, Any]:
    """Return the pull-code mdp keys of the COM-distance restraints."""
    block: dict[str, Any] = {
        "pull": "yes",
        "pull-ngroups": 2 * len(coordinates),
        "pull-ncoords": len(coordinates),
        "pull-pbc-ref-prev-step-com": "yes",
        "pull-nstxout": interval,
        "pull-nstfout": interval,
    }
    for i, coordinate in enumerate(coordinates):
        for group, name, atom in zip(
            (2 * i + 1, 2 * i + 2), coordinate.groups, coordinate.pbcatoms
        ):
            block[f"pull-group{group}-name"] = name
            block[f"pull-group{group}-pbcatom"] = atom
    for i, coordinate in enumerate(coordinates, 1):
        block.update(
            {
                f"pull-coord{i}-type": coordinate.potential,
                f"pull-coord{i}-geometry": "distance",
                f"pull-coord{i}-groups": [2 * i - 1, 2 * i],
                f"pull-coord{i}-dim": ["Y", "Y", "Y"],
                f"pull-coord{i}-start": "no",
                f"pull-coord{i}-init": coordinate.init,
                f"pull-coord{i}-k": coordinate.k,
            }
        )
    return block


def _pull_coordinates(restraints, directory: Path) -> tuple[PullCoordinate, ...]:
    """Return the pull coordinates of COM-distance restraints measured in *directory*."""
    from mdfactory.setup.restraints import read_com_restraints, reference_distance

    measured = read_com_restraints(directory)
    coordinates = []
    for restraint in restraints:
        entry = measured.get(restraint.name)
        if entry is None:
            raise ValueError(
                f"COM-distance restraint '{restraint.name}' is not in "
                f"{directory}/com_restraints.json; rebuild the system."
            )
        coordinates.append(
            PullCoordinate(
                groups=tuple(entry["groups"]),
                pbcatoms=tuple(entry["pbcatoms"]),
                potential="umbrella" if restraint.width is None else "flat-bottom",
                init=reference_distance(restraint, entry["measured_distance"]),
                k=restraint.k,
            )
        )
    return tuple(coordinates)


def render_run_schedule(
    build_input: "BuildInput", directory: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return the mdp parameters of every stage of *build_input*.

    Parameters
    ----------
    build_input : BuildInput
        The build input; its simulation settings are resolved.
    directory : Path, optional
        Build directory (default: the working directory). Only read for
        COM-distance restraints, whose measured distances it holds
        (``com_restraints.json``).

    Returns
    -------
    dict
        Mapping of stage name to its mdp parameters.

    """
    macros: tuple[str, ...] = ()
    pull: tuple[PullCoordinate, ...] = ()
    if build_input.simulation_type in PROTEIN_SIMULATION_TYPES:
        protein = build_input.system.protein
        macros = tuple(restraint.macro for restraint in protein.position_restraints)
        if protein.com_restraints:
            directory = Path.cwd() if directory is None else Path(directory)
            pull = _pull_coordinates(protein.com_restraints, directory)
    return {
        stage: render_stage(
            stage,
            simulation_type=build_input.simulation_type,
            nonbonded_scheme=build_input.parametrization.nonbonded_scheme,
            settings=build_input.simulation_settings,
            restraint_macros=macros,
            pull_coordinates=pull,
        )
        for stage in STAGES
    }


def format_mdp_value(value: Any) -> str:
    """Format a value as GROMACS expects it.

    Integers are plain integers, floats are plain decimals (never exponent
    notation, so ``dt`` and ``emstep`` keep a decimal point), booleans are
    ``yes``/``no`` and lists are space separated.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(Decimal(repr(value)), "f")
    if isinstance(value, (list, tuple)):
        return " ".join(format_mdp_value(item) for item in value)
    return str(value)


def format_mdp(params: dict[str, Any], header: list[str]) -> str:
    """Return the text of an mdp file: comment header, then ``key = value`` lines."""
    width = max(len(key) for key in params)
    lines = [f"; {line}" for line in header]
    lines += [f"{key:<{width}} = {format_mdp_value(value)}" for key, value in params.items()]
    return "\n".join(lines) + "\n"


def _header(build_input: "BuildInput", stage: str) -> list[str]:
    settings = build_input.simulation_settings
    return [
        f"{STAGE_FILES[stage]}: {stage} stage, generated by mdfactory {__version__}",
        f"simulation type {build_input.simulation_type}, "
        f"nonbonded scheme {build_input.parametrization.nonbonded_scheme}",
        f"state point: {settings.temperature:g} K, {settings.pressure:g} bar",
    ]


def write_mdp_files(
    build_input: "BuildInput",
    directory: Path | None = None,
    stages: tuple[str, ...] = STAGES,
) -> dict[str, Path]:
    """Write the run schedule's mdp files, overwriting existing ones.

    Parameters
    ----------
    build_input : BuildInput
        The build input; its simulation settings are resolved.
    directory : Path, optional
        Target directory (default: the working directory).
    stages : tuple of str, optional
        Stages to write (default: all four).

    Returns
    -------
    dict
        Mapping of stage name to the written file.

    """
    directory = Path.cwd() if directory is None else Path(directory)
    schedule = render_run_schedule(build_input, directory)
    written = {}
    for stage in stages:
        path = directory / STAGE_FILES[stage]
        path.write_text(format_mdp(schedule[stage], _header(build_input, stage)))
        written[stage] = path
    return written
