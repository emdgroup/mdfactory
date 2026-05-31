# ABOUTME: Pydantic models for force field parametrization configuration and results
# ABOUTME: Defines CGenFF/SMIRNOFF configs and GROMACS parameter set storage
"""Pydantic models for force field parametrization configuration and results."""

import json
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Discriminator, Field, FilePath, Tag


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


class Pdb2gmxConfig(BaseModel):
    """Configuration for pdb2gmx protein parametrization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["pdb2gmx"] = Field("pdb2gmx", description="Config type discriminator.")
    forcefield: str = Field(
        "charmm36m", description="Force field name as recognized by gmx pdb2gmx."
    )
    water_model: str = Field(
        "tip3p", description="Water model name as recognized by gmx pdb2gmx."
    )
    ignh: bool = Field(
        True, description="Ignore hydrogens in input PDB (regenerate with pdb2gmx)."
    )
    merge_all: bool = Field(
        False, description="Merge all chains into a single moleculetype."
    )


ParametrizationConfig = Annotated[
    Annotated[SmirnoffConfig, Tag("smirnoff")]
    | Annotated[CgenffConfig, Tag("cgenff")]
    | Annotated[Pdb2gmxConfig, Tag("pdb2gmx")],
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
    position_restraint_file: Path
    forcefield: str
    water_model: str
    total_charge: int
