# ABOUTME: BuildInput model representing a complete simulation build specification
# ABOUTME: Handles hashing, serialization, and type-dispatched composition validation
"""BuildInput model representing a complete simulation build specification."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .composition import BilayerComposition, LNPComposition, MixedBoxComposition
from .parametrization import CgenffConfig, ParametrizationConfig, SmirnoffConfig

# Map system_type to the corresponding model class
# TODO: StrEnum?
type_mapping = {
    "mixedbox": MixedBoxComposition,
    "bilayer": BilayerComposition,
    "lnp": LNPComposition,
}


class BuildInput(BaseModel):
    """Represent a complete simulation build specification with composition and parametrization."""

    simulation_type: Literal["mixedbox", "bilayer", "lnp"]
    system: MixedBoxComposition | BilayerComposition | LNPComposition
    parametrization: Literal["cgenff", "smirnoff"] = Field(
        "cgenff", description="Parametrization to use."
    )
    parametrization_config: ParametrizationConfig | None = Field(
        None, description="Parametrization-specific configuration. If None, uses defaults."
    )
    engine: Literal["gromacs"] = Field("gromacs", description="MD engine.")

    @property
    def hash(self):
        """Return a SHA-1 hash of the full JSON representation."""
        json_repr = self.model_dump_json()
        return hashlib.sha1(json_repr.encode("UTF-8")).hexdigest().upper()

    @property
    def metadata(self) -> dict[str, Any]:
        """Generate metadata dict with species composition.

        Extracts species counts, fractions, and system-specific parameters
        into a flattened structure suitable for analysis metadata storage.

        Returns
        -------
        dict
            Metadata including hash, simulation_type, species_composition,
            system_specific parameters, and full BuildInput JSON.

        """
        # TODO: refactor species composition extraction to avoid duplication
        # TODO: needs to be refactored if more simulation types are added
        # TODO: needs to be refactored if species structure changes (for proteins, RNA, etc.)
        species_composition = []
        for species in self.system.species:
            species_composition.append(
                {
                    "resname": species.resname,
                    "count": species.count,
                    "fraction": species.fraction,
                }
            )

        system_specific = {}
        if self.simulation_type == "bilayer":
            system_specific["z_padding"] = self.system.z_padding
            system_specific["monolayer"] = self.system.monolayer
            system_specific["ionization"] = (
                self.system.ionization.model_dump() if self.system.ionization is not None else None
            )
        elif self.simulation_type == "mixedbox":
            system_specific["target_density"] = self.system.target_density
            system_specific["ionization"] = (
                self.system.ionization.model_dump() if self.system.ionization is not None else None
            )

        return {
            "hash": self.hash,
            "simulation_type": self.simulation_type,
            "engine": self.engine,
            "parametrization": self.parametrization,
            "total_count": self.system.total_count,
            "species_composition": species_composition,
            "system_specific": system_specific,
            "build_input_json": self.model_dump_json(),
        }

    def to_data_row(self) -> dict[str, Any]:
        """Convert the BuildInput instance to a flat dictionary suitable
        for CSV output."""
        row = {
            "hash": self.hash,
            "engine": self.engine,
            "parametrization": self.parametrization,
            "simulation_type": self.simulation_type,
            "input_data": self.model_dump_json(),
            "input_data_type": "BuildInput",
        }
        return row

    @classmethod
    def from_data_row(cls, row: dict[str, Any]) -> "BuildInput":
        """Reconstruct a BuildInput from a data row dictionary.

        Parameters
        ----------
        row : dict[str, Any]
            Dictionary containing an ``input_data`` key with a JSON string
            or dictionary of BuildInput fields.

        Returns
        -------
        BuildInput
            Deserialized instance.

        """
        if "input_data" not in row:
            raise ValueError("Row must contain 'input_data' key with JSON string.")
        input_data = row["input_data"]
        if isinstance(input_data, str):
            input_dict = json.loads(input_data)
        elif isinstance(input_data, dict):
            input_dict = input_data
        else:
            raise ValueError("'input_data' must be a JSON string or a dictionary.")
        return cls(**input_dict)

    @model_validator(mode="before")
    @classmethod
    def cast_system_based_on_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict) and "simulation_type" in data and "system" in data:
            system_type = data["simulation_type"]
            system_data = data["system"]

            if system_type in type_mapping:
                # Cast the system data to the appropriate model
                model_class = type_mapping[system_type]
                if isinstance(system_data, dict):
                    data["system"] = model_class(**system_data)
                elif not isinstance(system_data, model_class):
                    # If it's already a model instance but wrong type, convert it
                    if hasattr(system_data, "model_dump"):
                        data["system"] = model_class(**system_data.model_dump())
                    else:
                        data["system"] = model_class(**dict(system_data))

        return data

    @model_validator(mode="after")
    def validate_system(self) -> "BuildInput":
        if not isinstance(self.system, type_mapping[self.simulation_type]):
            raise ValueError(f"System type does not match simulation type: {type(self.system)}")

        return self

    @model_validator(mode="after")
    def set_default_parametrization_config(self) -> "BuildInput":
        """Set default config based on parametrization type if not provided."""
        if self.parametrization_config is None:
            if self.parametrization == "smirnoff":
                object.__setattr__(self, "parametrization_config", SmirnoffConfig())
            elif self.parametrization == "cgenff":
                object.__setattr__(self, "parametrization_config", CgenffConfig())
        return self
