# ABOUTME: Pydantic models for force field parametrization configuration and results
# ABOUTME: Defines CGenFF/SMIRNOFF/CHARMM/Amber configs and GROMACS parameter set storage
"""Pydantic models for force field parametrization configuration and results."""

import json
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    FilePath,
    Tag,
    model_validator,
)

# The CHARMM protein pipelines solvate with 3-site spc216 coordinates and use
# CHARMM Force-switch LJ in every mdp, so only these combinations are supported.
SUPPORTED_WATER_MODELS = frozenset({"tip3p", "spc", "spce"})


def validate_absolute_path(path: Path):
    if not path.is_absolute():
        raise ValueError(f"Path must be absolute path: {path}")
    return path


AbsoluteFilePath = Annotated[FilePath, AfterValidator(validate_absolute_path)]


class SmirnoffConfig(BaseModel):
    """Configuration for SMIRNOFF parametrization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["smirnoff"] = Field("smirnoff", description="Config type discriminator.")
    forcefield: str = Field(
        "openff-2.2.0.offxml",
        description="OpenFF forcefield file (e.g., openff-2.2.0.offxml).",
    )
    water_model: str = Field(
        "opc3.offxml",
        description="Water model (e.g., opc3.offxml, tip3p_fb.offxml, tip4p_fb.offxml).",
    )
    charge_method: str = Field(
        "openff-gnn-am1bcc-0.1.0-rc.3.pt",
        description=(
            "Partial charge assignment method. NAGL models are fast (neural network). "
            "'am1bcc' or 'am1bccelf10' are slower but standard."
        ),
    )


class CgenffConfig(BaseModel):
    """Configuration for CGenFF parametrization (currently uses global config)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["cgenff"] = Field("cgenff", description="Config type discriminator.")
    # CGenFF uses SILCSBIODIR from config.ini, no additional settings needed


class _CharmmFamilyConfig(BaseModel):
    """Shared fields and CHARMM force-field/water validation for protein configs.

    The run schedules use CHARMM Force-switch LJ (cutoff, not LJ-PME) and solvate
    with 3-site spc216 coordinates, so a non-CHARMM force field, an LJ-PME variant,
    or a non-3-site water model would silently mismatch the mdp settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str  # narrowed to a Literal discriminator by subclasses (kept first in field order)
    forcefield: str = Field(
        "charmm36m", description="Force field name as recognized by gmx pdb2gmx."
    )
    water_model: str = Field("tip3p", description="Water model name as recognized by gmx pdb2gmx.")
    ignore_hydrogens: bool = Field(
        True, description="Ignore hydrogens in input PDB (regenerate with pdb2gmx)."
    )
    merge_all: bool = Field(False, description="Merge all chains into a single moleculetype.")

    @model_validator(mode="after")
    def _check_charmm_forcefield_and_water(self):
        """Reject force fields and water models the CHARMM run schedules cannot honor."""
        if not self.forcefield.lower().startswith("charmm"):
            raise ValueError(
                f"Force field '{self.forcefield}' is not supported. The run schedule uses "
                "CHARMM Force-switch LJ settings, so only CHARMM force fields (e.g. "
                "charmm36m, charmm36, charmm27) are supported."
            )
        if "ljpme" in self.forcefield.lower():
            raise ValueError(
                f"Force field '{self.forcefield}' is not supported. The run schedule uses "
                "cutoff/Force-switch LJ, which is incompatible with an LJ-PME force "
                "field. Use the standard CHARMM variant (e.g. charmm36m)."
            )
        if self.water_model.lower() not in SUPPORTED_WATER_MODELS:
            raise ValueError(
                f"Water model '{self.water_model}' is not supported. Solvation uses 3-site "
                "spc216 coordinates, so only 3-site water models are supported: "
                f"{sorted(SUPPORTED_WATER_MODELS)}."
            )
        return self


class Pdb2gmxConfig(_CharmmFamilyConfig):
    """Configuration for pdb2gmx protein parametrization."""

    type: Literal["pdb2gmx"] = Field("pdb2gmx", description="Config type discriminator.")


class CharmmConfig(_CharmmFamilyConfig):
    """Configuration for CHARMM parametrization of a protein packed with small molecules.

    Protein atoms are built with gmx pdb2gmx and ordinary SMILES solutes with
    CGenFF under one compatible CHARMM/CGenFF force-field family; native water and
    supported CHARMM ions skip CGenFF.
    """

    type: Literal["charmm"] = Field("charmm", description="Config type discriminator.")

    @model_validator(mode="after")
    def _check_protein_mixedbox_bundle(self) -> "CharmmConfig":
        """Reject force fields and water models the protein_mixedbox pipeline cannot honor."""
        if self.forcefield != "charmm36m":
            raise ValueError(
                "protein_mixedbox currently supports only the registered 'charmm36m' "
                "bundle, which pins a compatible CHARMM36m + CGenFF parameter family."
            )
        if self.water_model != "tip3p":
            raise ValueError(
                "protein_mixedbox currently supports only 'tip3p', whose native atom "
                "and residue names are normalized during packing."
            )
        return self


class AmberConfig(BaseModel):
    """Configuration for the open-source Amber ff14SB + OpenFF Sage parametrization.

    The protein is built with gmx pdb2gmx and the ``amber14sb.ff`` force field that
    ships with GROMACS 2026 or later. Native TIP3P water and Joung-Cheatham ions
    come from the same force-field directory. SMILES solutes are parametrized with
    OpenFF Sage through SMIRNOFF, so no CGenFF/SILCSBIO installation is needed.

    Used by both ``proteinbox`` and ``protein_mixedbox``. A ``proteinbox`` has no
    small molecules, so it accepts the Sage fields and ignores them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["amber"] = Field("amber", description="Config type discriminator.")
    forcefield: str = Field(
        "amber14sb",
        description="Protein force field as recognized by gmx pdb2gmx (only amber14sb).",
    )
    water_model: str = Field(
        "tip3p", description="Water model as recognized by gmx pdb2gmx (only tip3p)."
    )
    ignore_hydrogens: bool = Field(
        True, description="Ignore hydrogens in input PDB (regenerate with pdb2gmx)."
    )
    merge_all: bool = Field(False, description="Merge all chains into a single moleculetype.")
    sage_forcefield: str = Field(
        SmirnoffConfig.model_fields["forcefield"].default,
        description="OpenFF force field file for small-molecule solutes.",
    )
    charge_method: str = Field(
        SmirnoffConfig.model_fields["charge_method"].default,
        description="Partial charge assignment method for small-molecule solutes.",
    )

    @model_validator(mode="after")
    def _check_amber_forcefield_and_water(self) -> "AmberConfig":
        """Reject protein force fields and water models other than amber14sb + tip3p."""
        if self.forcefield != "amber14sb":
            raise ValueError(
                f"Force field '{self.forcefield}' is not supported for the amber "
                "parametrization. Only 'amber14sb' (bundled with GROMACS >= 2026) is "
                "supported; its run schedule uses Sage-style nonbonded settings."
            )
        if self.water_model != "tip3p":
            raise ValueError(
                f"Water model '{self.water_model}' is not supported for the amber "
                "parametrization. Only 'tip3p' is supported: solvation uses 3-site "
                "spc216 coordinates and ff14SB is paired with TIP3P."
            )
        return self

    def smirnoff_config(self) -> SmirnoffConfig:
        """Return the SMIRNOFF configuration used for the small-molecule solutes."""
        return SmirnoffConfig(forcefield=self.sage_forcefield, charge_method=self.charge_method)


ParametrizationConfig = Annotated[
    Annotated[SmirnoffConfig, Tag("smirnoff")]
    | Annotated[CgenffConfig, Tag("cgenff")]
    | Annotated[Pdb2gmxConfig, Tag("pdb2gmx")]
    | Annotated[CharmmConfig, Tag("charmm")]
    | Annotated[AmberConfig, Tag("amber")],
    Discriminator("type"),
]


class GromacsSingleMoleculeParameterSet(BaseModel):
    """Store GROMACS topology and parameter file paths for a single molecule."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    moleculetype: str  # = hash
    smiles: str
    parametrization: Literal["cgenff", "smirnoff"]
    parametrization_config: ParametrizationConfig
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
            "parametrization": self.parametrization,
            "parameter_data": model_json_str,
            "parameter_data_type": "GromacsSingleMoleculeParameterSet",
        }

    @classmethod
    def from_data_row(cls, data_row):
        """Create a GromacsSingleMoleculeParameterSet from a data row.
        The data row should contain the keys: hash, smiles, engine, parametrization,
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
