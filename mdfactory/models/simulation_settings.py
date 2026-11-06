# ABOUTME: SimulationSettings contract: how a build is run (state point, lengths, outputs)
# ABOUTME: Complements BuildInput; only temperature and pressure belong to the run identity
"""Simulation settings: how a build is run.

:class:`~mdfactory.models.input.BuildInput` describes *what* is simulated;
:class:`SimulationSettings` describes *how* it is run, in physical units. The
GROMACS run schedule (``em.mdp``, ``nvt.mdp``, ``npt.mdp``, ``md.mdp``) is
rendered from both at build time.

Identity rule
-------------
Only the state point (:attr:`SimulationSettings.temperature` and
:attr:`SimulationSettings.pressure`) enters the build hash. A different
temperature is a different data point; a longer production run, another
timestep, other output intervals, minimization limits or GROMACS overrides
keep the hash. Every new field must be classified in
:attr:`SimulationSettings.IDENTITY_FIELDS` (a test pins the set).
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Optional, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

#: Stage names, in run order. They name the sections of
#: :class:`GromacsOverrides` and the rendered stages of the run schedule.
STAGES: tuple[str, ...] = ("minimization", "nvt", "npt", "production")

#: Default production length per simulation type, in ns.
DEFAULT_PRODUCTION_LENGTH_NS: dict[str, float] = {
    "mixedbox": 100.0,
    "lnp": 100.0,
    "bilayer": 1000.0,
    "proteinbox": 100.0,
    "protein_mixedbox": 100.0,
}

#: Largest supported timestep in fs (H-bond constraints, no hydrogen mass repartitioning).
MAX_TIMESTEP_FS = 2.0

#: mdp keys set from typed settings fields. ``nstxtcout`` is GROMACS' old
#: name of ``nstxout-compressed``.
SETTINGS_OWNED_MDP_KEYS: frozenset[str] = frozenset(
    {
        "nsteps",
        "dt",
        "emtol",
        "ref-t",
        "gen-temp",
        "ref-p",
        "nstxout",
        "nstvout",
        "nstfout",
        "nstenergy",
        "nstlog",
        "nstxout-compressed",
        "nstxtcout",
    }
)

#: mdp keys derived from the build: the force-field family's nonbonded block
#: and the coupling geometry of the simulation type.
BUILD_OWNED_MDP_KEYS: frozenset[str] = frozenset(
    {
        "coulombtype",
        "rcoulomb",
        "vdwtype",
        "rvdw",
        "vdw-modifier",
        "rvdw-switch",
        "DispCorr",
        "tc-grps",
        "pcoupltype",
        "compressibility",
    }
)

MdpValue = bool | int | float | str


def normalize_mdp_key(key: str) -> str:
    """Return the form GROMACS matches mdp keys by: lowercase, without ``-`` and ``_``.

    Parameters
    ----------
    key : str
        An mdp parameter name, e.g. ``ref_t`` or ``Ref-T``.

    Returns
    -------
    str
        The normalized name, e.g. ``reft``.

    """
    return key.lower().replace("-", "").replace("_", "")


_OWNED_NORMALIZED = frozenset(
    normalize_mdp_key(key) for key in SETTINGS_OWNED_MDP_KEYS | BUILD_OWNED_MDP_KEYS
)


def to_steps(duration_ps: float, timestep_fs: float, what: str = "duration") -> int:
    """Convert a duration into an exact number of MD steps.

    Parameters
    ----------
    duration_ps : float
        Duration in ps.
    timestep_fs : float
        Timestep in fs.
    what : str, optional
        Name of the quantity, used in the error message.

    Returns
    -------
    int
        Number of steps.

    Raises
    ------
    ValueError
        If the duration is not a positive integer multiple of the timestep.

    """
    steps = duration_ps * 1000.0 / timestep_fs
    rounded = round(steps)
    if rounded < 1 or not math.isclose(steps, rounded, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"{what} ({duration_ps:g} ps) must be a positive integer multiple of the "
            f"timestep ({timestep_fs:g} fs)."
        )
    return rounded


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MinimizationSettings(_Section):
    """Energy-minimization limits (steepest descent)."""

    max_steps: PositiveInt = Field(50_000, description="Maximum number of minimization steps.")
    force_tolerance: PositiveFloat = Field(
        1000.0, description="Stop when the maximum force is below this value, in kJ mol⁻¹ nm⁻¹."
    )


class EquilibrationSettings(_Section):
    """Length and output interval of an equilibration stage (NVT or NPT)."""

    length: PositiveFloat = Field(100.0, description="Stage length in ps.")
    output_interval: PositiveFloat = Field(
        1.0, description="Interval of coordinates, velocities, energies and log output in ps."
    )


class ProductionSettings(_Section):
    """Length and output interval of the production stage."""

    length: Optional[PositiveFloat] = Field(
        None,
        description=(
            "Production length in ns. Omitted: the simulation type's default "
            "(1000 ns for bilayer, 100 ns otherwise)."
        ),
    )
    output_interval: PositiveFloat = Field(
        10.0, description="Interval of compressed coordinates, energies and log output in ps."
    )


StageOverrides = dict[str, MdpValue]


class GromacsOverrides(_Section):
    """Raw GROMACS mdp keys per stage, applied after everything else.

    Keys set from typed settings (:data:`SETTINGS_OWNED_MDP_KEYS`) or derived
    from the build (:data:`BUILD_OWNED_MDP_KEYS`) are rejected. GROMACS
    ignores case, ``-`` and ``_`` in key names, and so does the check.
    """

    minimization: StageOverrides = Field(default_factory=dict)
    nvt: StageOverrides = Field(default_factory=dict)
    npt: StageOverrides = Field(default_factory=dict)
    production: StageOverrides = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_keys(self) -> Self:
        for stage in STAGES:
            seen: dict[str, str] = {}
            for key, value in getattr(self, stage).items():
                if not key or any(ch.isspace() or ch in "=;" for ch in key):
                    raise ValueError(f"gromacs_overrides.{stage}: invalid mdp key {key!r}.")
                normalized = normalize_mdp_key(key)
                if normalized in _OWNED_NORMALIZED:
                    raise ValueError(
                        f"gromacs_overrides.{stage}: '{key}' is set by mdfactory (from a "
                        "typed setting or from the build) and cannot be overridden."
                    )
                if normalized in seen:
                    raise ValueError(
                        f"gromacs_overrides.{stage}: '{seen[normalized]}' and '{key}' name "
                        "the same mdp key."
                    )
                seen[normalized] = key
                if isinstance(value, str) and (";" in value or "\n" in value):
                    raise ValueError(
                        f"gromacs_overrides.{stage}.{key}: value {value!r} must not contain "
                        "';' or a line break."
                    )
        return self


class SimulationSettings(BaseModel):
    """How a build is run: state point, timestep, stage lengths and outputs.

    All fields are optional. Omitted fields take defaults; the production
    length depends on the simulation type and is filled by :meth:`resolve`.
    Only :attr:`IDENTITY_FIELDS` (temperature and pressure) enter the build
    hash.

    Attributes
    ----------
    temperature : float
        Reference temperature in K, for every stage's thermostat and the NVT
        velocity generation. Part of the identity.
    pressure : float
        Reference pressure in bar, for NPT and production (both values for
        semi-isotropic coupling). Part of the identity.
    timestep : float
        Integration timestep in fs, at most 2 fs.
    minimization : MinimizationSettings
        Energy-minimization limits.
    nvt, npt : EquilibrationSettings
        Equilibration lengths and output intervals in ps.
    production : ProductionSettings
        Production length in ns and output interval in ps.
    gromacs_overrides : GromacsOverrides
        Raw mdp keys per stage, applied last.

    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Fields that define the run identity (enter the build hash).
    IDENTITY_FIELDS: ClassVar[frozenset[str]] = frozenset({"temperature", "pressure"})

    temperature: PositiveFloat = Field(300.0, description="Reference temperature in K.")
    pressure: PositiveFloat = Field(1.0, description="Reference pressure in bar.")
    timestep: float = Field(
        2.0, gt=0.0, le=MAX_TIMESTEP_FS, description="Integration timestep in fs (at most 2)."
    )
    minimization: MinimizationSettings = Field(default_factory=MinimizationSettings)
    nvt: EquilibrationSettings = Field(default_factory=EquilibrationSettings)
    npt: EquilibrationSettings = Field(default_factory=EquilibrationSettings)
    production: ProductionSettings = Field(default_factory=ProductionSettings)
    gromacs_overrides: GromacsOverrides = Field(default_factory=GromacsOverrides)

    @model_validator(mode="after")
    def _check_step_multiples(self) -> Self:
        for name in ("nvt", "npt"):
            section: EquilibrationSettings = getattr(self, name)
            length = to_steps(section.length, self.timestep, f"{name}.length")
            interval = to_steps(section.output_interval, self.timestep, f"{name}.output_interval")
            if interval > length:
                raise ValueError(f"{name}.output_interval must not exceed {name}.length.")
        interval = to_steps(
            self.production.output_interval, self.timestep, "production.output_interval"
        )
        if self.production.length is not None:
            length = to_steps(self.production.length * 1000.0, self.timestep, "production.length")
            if length % interval:
                raise ValueError(
                    f"production.length ({self.production.length:g} ns) must be a multiple "
                    f"of production.output_interval ({self.production.output_interval:g} ps), "
                    "so the expected frame count is exact."
                )
        return self

    def resolve(self, simulation_type: str) -> Self:
        """Return a copy with the per-type defaults filled in.

        Parameters
        ----------
        simulation_type : str
            The build's simulation type.

        Returns
        -------
        SimulationSettings
            Settings with every field set.

        """
        data = self.model_dump()
        if data["production"]["length"] is None:
            data["production"]["length"] = DEFAULT_PRODUCTION_LENGTH_NS[simulation_type]
        return type(self).model_validate(data)

    def identity(self) -> dict[str, Any]:
        """Return the fields that enter the build hash."""
        return {name: getattr(self, name) for name in sorted(self.IDENTITY_FIELDS)}


def _as_mapping(settings: "SimulationSettings | Mapping[str, Any] | None") -> dict[str, Any]:
    if settings is None:
        return {}
    if isinstance(settings, SimulationSettings):
        # Only the fields the caller set, so defaults never mask a lower layer.
        return settings.model_dump(mode="json", exclude_unset=True)
    if isinstance(settings, Mapping):
        return dict(settings)
    raise TypeError(f"Expected SimulationSettings or a mapping, got {type(settings).__name__}.")


def merge_simulation_settings(
    base: "SimulationSettings | Mapping[str, Any] | None",
    update: "SimulationSettings | Mapping[str, Any] | None",
) -> dict[str, Any]:
    """Merge two partial settings mappings; *update* wins.

    Sections (``nvt``, ``production``, ...) merge field by field. GROMACS
    overrides merge key by key per stage; a key in *update* replaces every
    spelling of the same key in *base* (``tau_p`` replaces ``tau-p``).

    Parameters
    ----------
    base, update : SimulationSettings, mapping or None
        Partial settings. A :class:`SimulationSettings` contributes only the
        fields that were set explicitly.

    Returns
    -------
    dict
        The merged raw mapping (not validated).

    """
    merged = _as_mapping(base)
    for key, value in _as_mapping(update).items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            if key == "gromacs_overrides":
                merged[key] = _merge_overrides(current, value)
            else:
                merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def _merge_overrides(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = {stage: dict(keys) for stage, keys in base.items()}
    for stage, keys in update.items():
        if not isinstance(keys, Mapping) or not isinstance(merged.get(stage), Mapping):
            merged[stage] = keys
            continue
        replaced = {normalize_mdp_key(k) for k in keys}
        kept = {k: v for k, v in merged[stage].items() if normalize_mdp_key(k) not in replaced}
        merged[stage] = {**kept, **keys}
    return merged


def load_simulation_settings(path: Path) -> dict[str, Any]:
    """Load a simulation-settings (protocol) YAML file.

    The file holds the settings mapping itself (the content of a
    ``simulation_settings:`` section). It is validated here so a bad file
    fails before any build starts; the raw mapping is returned so it can be
    merged under per-input settings.

    Parameters
    ----------
    path : Path
        YAML file.

    Returns
    -------
    dict
        The settings mapping.

    Raises
    ------
    ValueError
        If the file is not a mapping or does not validate.

    """
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Simulation settings file {path} must contain a mapping.")
    SimulationSettings.model_validate(data)
    return data
