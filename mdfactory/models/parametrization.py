# ABOUTME: Pydantic models for per-component force-field specs and parametrization results
# ABOUTME: Defines the charmm/amber/openff families and GROMACS parameter set storage
"""Pydantic models for per-component force-field specs and parametrization results.

A build names one force-field *family* (``charmm``, ``amber`` or ``openff``).
Each family describes its components (protein, small molecules, water, ions)
by force field and gives every component a default, which a run may override.
"""

import json
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Optional, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    FilePath,
    Tag,
    field_validator,
    model_validator,
)

SimulationType = Literal["mixedbox", "bilayer", "lnp", "proteinbox", "protein_mixedbox"]
#: Nonbonded block of the generated run schedule (see ``mdfactory.run_schedules.gromacs``):
#: ``charmm`` = PME, force switch 1.0 -> 1.2 nm; ``sage`` = PME, 0.9 nm cutoff with a
#: potential switch from 0.8 nm. Adding a scheme means adding its nonbonded overlay.
NonbondedScheme = Literal["charmm", "sage"]

PROTEIN_SIMULATION_TYPES = frozenset({"proteinbox", "protein_mixedbox"})
# Simulation types that parametrize small molecules (a proteinbox has none).
SMALL_MOLECULE_SIMULATION_TYPES = frozenset({"mixedbox", "bilayer", "lnp", "protein_mixedbox"})

# The CHARMM proteinbox pipeline solvates with 3-site spc216 coordinates and uses
# CHARMM Force-switch LJ in every mdp, so only these water models are supported.
SUPPORTED_CHARMM_WATER_MODELS = frozenset({"tip3p", "spc", "spce"})

DEFAULT_OPENFF_FORCEFIELD = "openff-2.2.0.offxml"
DEFAULT_NAGL_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"


def validate_absolute_path(path: Path):
    if not path.is_absolute():
        raise ValueError(f"Path must be absolute path: {path}")
    return path


AbsoluteFilePath = Annotated[FilePath, AfterValidator(validate_absolute_path)]


class CgenffSmallMolecules(BaseModel):
    """CGenFF small-molecule parameters (uses the SILCSBIO installation from config.ini)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ClassVar[str] = "cgenff"

    forcefield: Literal["cgenff"] = Field("cgenff", description="Small-molecule force field.")

    def describe(self) -> str:
        """Return a one-line description of the small-molecule force field."""
        return self.forcefield


class OpenFFSmallMolecules(BaseModel):
    """OpenFF (SMIRNOFF) small-molecule parameters, e.g. Sage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ClassVar[str] = "openff"

    forcefield: str = Field(
        DEFAULT_OPENFF_FORCEFIELD,
        description="OpenFF force field file (e.g. openff-2.2.0.offxml).",
    )
    charge_method: str = Field(
        DEFAULT_NAGL_MODEL,
        description=(
            "Partial charge method: a NAGL model file (fast, neural network; 'nagl' "
            "selects the default model), or 'am1bcc'/'am1bccelf10' (slower, standard)."
        ),
    )

    @field_validator("forcefield")
    @classmethod
    def _check_offxml(cls, value: str) -> str:
        if not value.endswith(".offxml"):
            raise ValueError(
                f"OpenFF force field '{value}' must be an .offxml file name, "
                f"e.g. '{DEFAULT_OPENFF_FORCEFIELD}'."
            )
        return value

    @property
    def resolved_charge_method(self) -> str:
        """Return the charge method with the ``nagl`` alias resolved to the default model."""
        if self.charge_method.lower() == "nagl":
            return DEFAULT_NAGL_MODEL
        return self.charge_method

    def describe(self) -> str:
        """Return a one-line description of the force field and charge method."""
        return f"{self.forcefield} (charges: {self.resolved_charge_method})"


def _reject_offxml_water(value: Optional[str]) -> Optional[str]:
    if value is not None and value.endswith(".offxml"):
        raise ValueError(
            f"water_model is a water model name, not a file: use "
            f"'{value.removesuffix('.offxml')}' instead of '{value}'."
        )
    return value


WaterModelName = Annotated[Optional[str], AfterValidator(_reject_offxml_water)]


class _FamilySpec(BaseModel):
    """Shared behavior of the force-field family specs.

    Subclasses set the class variables below and implement family-specific
    validation. Components default to None and are filled in per simulation
    type by :meth:`resolve`; components that do not apply stay None.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_types: ClassVar[frozenset[str]]
    default_protein: ClassVar[Optional[str]]
    default_water_model: ClassVar[str]
    small_molecules_model: ClassVar[type[BaseModel]]

    family: str
    protein: Optional[str] = None
    small_molecules: Optional[BaseModel] = None
    water_model: WaterModelName = None

    @property
    def nonbonded_scheme(self) -> NonbondedScheme:
        """Return the nonbonded scheme of the family's generated run schedule.

        It selects the nonbonded block (cutoffs, switching, dispersion
        correction) of every stage, for every simulation type.
        """
        raise NotImplementedError

    @property
    def ions(self) -> Optional[str]:
        """Return the force field that provides the ion parameters."""
        raise NotImplementedError

    def components(self) -> dict[str, Optional[str]]:
        """Return a description of each component (None where it does not apply)."""
        return {
            "protein": self.protein,
            "small_molecules": (
                self.small_molecules.describe() if self.small_molecules is not None else None
            ),
            "water": self.water_model,
            "ions": self.ions,
        }

    def summary(self) -> str:
        """Return the family and its components as one line."""
        parts = [f"{name}={value}" for name, value in self.components().items() if value]
        return f"{self.family} ({', '.join(parts)})"

    def resolve(self, simulation_type: str) -> Self:
        """Return a copy with family defaults filled in for *simulation_type*.

        Raises
        ------
        ValueError
            If the family or one of the given components does not apply to the
            simulation type, or the combination is not supported.

        """
        if simulation_type not in self.simulation_types:
            allowed = sorted(
                family
                for family, spec in FAMILY_SPECS.items()
                if simulation_type in spec.simulation_types
            )
            raise ValueError(
                f"Parametrization family '{self.family}' is not valid for simulation type "
                f"'{simulation_type}'. Allowed families: {allowed}."
            )
        has_protein = simulation_type in PROTEIN_SIMULATION_TYPES
        has_small_molecules = simulation_type in SMALL_MOLECULE_SIMULATION_TYPES
        if self.protein is not None and not has_protein:
            raise ValueError(
                f"parametrization.protein does not apply to simulation type "
                f"'{simulation_type}'; it is only used by {sorted(PROTEIN_SIMULATION_TYPES)}."
            )
        if self.small_molecules is not None and not has_small_molecules:
            raise ValueError(
                f"parametrization.small_molecules does not apply to simulation type "
                f"'{simulation_type}', which has no small molecules."
            )
        data: dict[str, Any] = {
            "family": self.family,
            "protein": (self.protein or self.default_protein) if has_protein else None,
            "small_molecules": (
                (self.small_molecules or self.small_molecules_model()).model_dump()
                if has_small_molecules
                else None
            ),
            "water_model": self.water_model or self.default_water_model,
        }
        resolved = type(self).model_validate(data)
        resolved._check_simulation_type(simulation_type)
        return resolved

    def _check_simulation_type(self, simulation_type: str) -> None:
        """Reject resolved components that *simulation_type* cannot honor."""


class CharmmParametrization(_FamilySpec):
    """CHARMM family: CHARMM36m protein, CGenFF small molecules, CHARMM TIP3P and ions.

    The run schedules use CHARMM Force-switch LJ (cutoff, not LJ-PME), and the
    protein builds solvate with 3-site spc216 coordinates, so a non-CHARMM
    protein force field, an LJ-PME variant or a non-3-site water model would
    silently mismatch the mdp settings.
    """

    simulation_types: ClassVar[frozenset[str]] = frozenset(
        {"mixedbox", "bilayer", "lnp", "proteinbox", "protein_mixedbox"}
    )
    default_protein: ClassVar[Optional[str]] = "charmm36m"
    default_water_model: ClassVar[str] = "tip3p"
    small_molecules_model: ClassVar[type[BaseModel]] = CgenffSmallMolecules

    family: Literal["charmm"] = Field("charmm", description="Force-field family.")
    protein: Optional[str] = Field(
        None, description="Protein force field as recognized by gmx pdb2gmx (default charmm36m)."
    )
    small_molecules: Optional[CgenffSmallMolecules] = Field(
        None, description="Small-molecule force field (CGenFF)."
    )
    water_model: WaterModelName = Field(None, description="Water model name (default tip3p).")

    @property
    def nonbonded_scheme(self) -> NonbondedScheme:
        return "charmm"

    @property
    def ions(self) -> Optional[str]:
        # Ions come from the pdb2gmx protein force field, or from CGenFF's charmm36.ff.
        return self.protein or "charmm36"

    @model_validator(mode="after")
    def _check_charmm_components(self) -> Self:
        if self.protein is not None:
            if not self.protein.lower().startswith("charmm"):
                raise ValueError(
                    f"Protein force field '{self.protein}' is not supported by the charmm "
                    "family. The run schedule uses CHARMM Force-switch LJ settings, so only "
                    "CHARMM force fields (e.g. charmm36m, charmm36, charmm27) are supported."
                )
            if "ljpme" in self.protein.lower():
                raise ValueError(
                    f"Protein force field '{self.protein}' is not supported. The run schedule "
                    "uses cutoff/Force-switch LJ, which is incompatible with an LJ-PME force "
                    "field. Use the standard CHARMM variant (e.g. charmm36m)."
                )
        if (
            self.water_model is not None
            and self.water_model.lower() not in SUPPORTED_CHARMM_WATER_MODELS
        ):
            raise ValueError(
                f"Water model '{self.water_model}' is not supported by the charmm family. "
                "Solvation uses 3-site spc216 coordinates, so only 3-site water models are "
                f"supported: {sorted(SUPPORTED_CHARMM_WATER_MODELS)}."
            )
        return self

    def _check_simulation_type(self, simulation_type: str) -> None:
        if simulation_type == "protein_mixedbox":
            if self.protein != "charmm36m":
                raise ValueError(
                    "protein_mixedbox currently supports only the registered 'charmm36m' "
                    "bundle, which pins a compatible CHARMM36m + CGenFF parameter family."
                )
            if self.water_model != "tip3p":
                raise ValueError(
                    "protein_mixedbox currently supports only 'tip3p', whose native atom "
                    "and residue names are normalized during packing."
                )
        elif simulation_type != "proteinbox" and self.water_model != "tip3p":
            raise ValueError(
                f"Water model '{self.water_model}' is not supported for '{simulation_type}' "
                "with the charmm family: CGenFF builds use the TIP3P water of its charmm36.ff."
            )


class AmberParametrization(_FamilySpec):
    """Amber family: ff14SB protein, OpenFF Sage small molecules, TIP3P and JC ions.

    The protein is built with gmx pdb2gmx and the ``amber14sb.ff`` force field that
    ships with GROMACS 2026 or later. Native TIP3P water and Joung-Cheatham ions
    come from the same force-field directory, and SMILES solutes are parametrized
    with OpenFF Sage, so no CGenFF/SILCSBIO installation is needed.
    """

    simulation_types: ClassVar[frozenset[str]] = PROTEIN_SIMULATION_TYPES
    default_protein: ClassVar[Optional[str]] = "amber14sb"
    default_water_model: ClassVar[str] = "tip3p"
    small_molecules_model: ClassVar[type[BaseModel]] = OpenFFSmallMolecules

    family: Literal["amber"] = Field("amber", description="Force-field family.")
    protein: Optional[str] = Field(
        None, description="Protein force field as recognized by gmx pdb2gmx (only amber14sb)."
    )
    small_molecules: Optional[OpenFFSmallMolecules] = Field(
        None, description="Small-molecule force field (OpenFF Sage)."
    )
    water_model: WaterModelName = Field(None, description="Water model name (only tip3p).")

    @property
    def nonbonded_scheme(self) -> NonbondedScheme:
        return "sage"

    @property
    def ions(self) -> Optional[str]:
        # Joung-Cheatham ions_<water_model>.itp from the protein force field.
        return self.protein

    @model_validator(mode="after")
    def _check_amber_components(self) -> Self:
        if self.protein is not None and self.protein != "amber14sb":
            raise ValueError(
                f"Protein force field '{self.protein}' is not supported by the amber "
                "family. Only 'amber14sb' (bundled with GROMACS >= 2026) is supported; "
                "its run schedule uses Sage-style nonbonded settings."
            )
        if self.water_model is not None and self.water_model != "tip3p":
            raise ValueError(
                f"Water model '{self.water_model}' is not supported by the amber family. "
                "Only 'tip3p' is supported: solvation uses 3-site spc216 coordinates and "
                "ff14SB is paired with TIP3P."
            )
        return self


class OpenFFParametrization(_FamilySpec):
    """OpenFF family: Sage small molecules with OpenFF water and ions (no protein)."""

    simulation_types: ClassVar[frozenset[str]] = frozenset({"mixedbox", "bilayer", "lnp"})
    default_protein: ClassVar[Optional[str]] = None
    default_water_model: ClassVar[str] = "opc3"
    small_molecules_model: ClassVar[type[BaseModel]] = OpenFFSmallMolecules

    family: Literal["openff"] = Field("openff", description="Force-field family.")
    protein: Optional[str] = Field(
        None, description="Not supported: the openff family has no protein force field."
    )
    small_molecules: Optional[OpenFFSmallMolecules] = Field(
        None, description="Small-molecule force field (OpenFF Sage)."
    )
    water_model: WaterModelName = Field(
        None,
        description="OpenFF water model name, loaded as <name>.offxml (default opc3).",
    )

    @property
    def nonbonded_scheme(self) -> NonbondedScheme:
        return "sage"

    @property
    def ions(self) -> Optional[str]:
        # Ion parameters come from the small-molecule .offxml.
        return self.small_molecules.forcefield if self.small_molecules is not None else None

    @model_validator(mode="after")
    def _check_no_protein(self) -> Self:
        if self.protein is not None:
            raise ValueError(
                "The openff family has no protein force field; use family 'amber' "
                "for protein systems."
            )
        return self


FAMILY_SPECS: dict[str, type[_FamilySpec]] = {
    "charmm": CharmmParametrization,
    "amber": AmberParametrization,
    "openff": OpenFFParametrization,
}


def _expand_family_shorthand(value: Any) -> Any:
    """Accept ``parametrization: <family>`` as shorthand for ``{family: <family>}``."""
    if isinstance(value, str):
        return {"family": value}
    return value


ParametrizationSpec = Annotated[
    Annotated[CharmmParametrization, Tag("charmm")]
    | Annotated[AmberParametrization, Tag("amber")]
    | Annotated[OpenFFParametrization, Tag("openff")],
    Discriminator("family"),
    BeforeValidator(_expand_family_shorthand),
]


class GromacsSingleMoleculeParameterSet(BaseModel):
    """Store GROMACS topology and parameter file paths for a single molecule.

    ``topology_merge`` tells the topology writers how to combine the set with
    others: ``cgenff`` sets share atom types with the CHARMM force field,
    ``smirnoff`` sets carry private atom types and their own ``[ defaults ]``,
    and ``native`` sets reuse a force-field directory's water/ion topology.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    moleculetype: str  # = hash
    smiles: str
    component: Literal["small_molecule", "water", "ion"]
    forcefield: str = Field(..., description="Force field (or water file) that produced the set.")
    charge_method: Optional[str] = Field(
        None, description="Resolved partial charge method (OpenFF small molecules only)."
    )
    water_model: Optional[str] = Field(
        None, description="Water model the set belongs to (water, water-specific ions)."
    )
    topology_merge: Literal["cgenff", "smirnoff", "native"]
    itp: AbsoluteFilePath
    parameter_itp: Optional[AbsoluteFilePath]
    forcefield_itp: Optional[AbsoluteFilePath]

    def to_data_row(self):
        """Serialize the parameter set to a flat dictionary for tabular storage."""
        model_json_str = self.model_dump_json()
        # TODO: model for standard data row
        return {
            "hash": self.moleculetype,
            "smiles": self.smiles,
            "engine": "gromacs",
            "forcefield": self.forcefield,
            "parameter_data": model_json_str,
            "parameter_data_type": "GromacsSingleMoleculeParameterSet",
        }

    @classmethod
    def from_data_row(cls, data_row):
        """Create a GromacsSingleMoleculeParameterSet from a data row.
        The data row should contain the keys: hash, smiles, engine, forcefield,
            parameter_data, parameter_data_type.
        """
        if data_row["parameter_data_type"] != "GromacsSingleMoleculeParameterSet":
            raise ValueError("Invalid parameter_data_type for GromacsSingleMoleculeParameterSet.")
        data = json.loads(data_row["parameter_data"])
        return cls(**data)


class GromacsProteinParameterSet(BaseModel):
    """Store GROMACS topology output from pdb2gmx for a protein system."""

    model_config = ConfigDict(extra="forbid")

    topology_file: Path
    structure_file: Path
    topology_include_files: list[Path]
    chains: list[str]
    forcefield: str
    water_model: str
    total_charge: int
