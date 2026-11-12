# ABOUTME: BuildInput model representing a complete simulation build specification
# ABOUTME: Handles hashing, serialization, and type-dispatched composition validation
"""BuildInput model representing a complete simulation build specification."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .composition import (
    BilayerComposition,
    LNPComposition,
    MixedBoxComposition,
    ProteinBoxComposition,
    ProteinMixedBoxComposition,
)
from .parametrization import PROTEIN_SIMULATION_TYPES, ParametrizationSpec, SimulationType
from .simulation_settings import SimulationSettings

# Map system_type to the corresponding model class
# TODO: StrEnum?
type_mapping = {
    "mixedbox": MixedBoxComposition,
    "bilayer": BilayerComposition,
    "lnp": LNPComposition,
    "proteinbox": ProteinBoxComposition,
    "protein_mixedbox": ProteinMixedBoxComposition,
}


def protein_anchor(build_input: Any) -> dict[str, Any] | None:
    """Return the species entry (resname, count, fraction) of a protein_mixedbox's protein.

    The protein anchors a protein_mixedbox but is not part of ``system.species``,
    which holds the packed solution only. Returns None for every other type.
    """
    if build_input.simulation_type != "protein_mixedbox":
        return None
    protein = build_input.system.protein
    return {"resname": protein.resname, "count": protein.count, "fraction": protein.fraction}


class BuildInput(BaseModel):
    """Represent a complete simulation build specification with composition and parametrization."""

    # Unknown keys are rejected so a misspelled or removed key is never dropped silently.
    model_config = ConfigDict(extra="forbid")

    simulation_type: SimulationType
    system: (
        MixedBoxComposition
        | BilayerComposition
        | LNPComposition
        | ProteinBoxComposition
        | ProteinMixedBoxComposition
    )
    parametrization: ParametrizationSpec = Field(
        ...,
        description=(
            "Force-field family (charmm, amber or openff) with optional per-component "
            "overrides (protein, small_molecules, water_model). A bare family name "
            "takes every component's family default."
        ),
    )
    engine: Literal["gromacs"] = Field("gromacs", description="MD engine.")
    simulation_settings: SimulationSettings = Field(
        default_factory=SimulationSettings,
        description=(
            "How the system is run: temperature, pressure, timestep, stage lengths, "
            "output intervals and GROMACS overrides. Omitted fields take per-type "
            "defaults; only temperature and pressure enter the hash."
        ),
    )

    @property
    def hash(self):
        """Return a SHA-1 hash of the build identity.

        The identity is the full input except the simulation settings outside
        :attr:`SimulationSettings.IDENTITY_FIELDS`: a different temperature or
        pressure gives a new hash, a longer production run does not.
        """
        non_identity = set(SimulationSettings.model_fields) - SimulationSettings.IDENTITY_FIELDS
        json_repr = self.model_dump_json(exclude={"simulation_settings": non_identity})
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
                    "concentration": getattr(species, "concentration", None),
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
        elif self.simulation_type == "proteinbox":
            system_specific["padding"] = self.system.padding
            system_specific["ionization"] = self.system.ionization.model_dump()
            system_specific["pdb_path"] = str(self.system.protein.pdb_path)
        elif self.simulation_type == "protein_mixedbox":
            system_specific["padding"] = self.system.padding
            system_specific["sizing"] = self.system.sizing.model_dump()
            system_specific["concentration_volume_basis"] = self.system.concentration_volume_basis
            system_specific["partial_specific_volume"] = self.system.partial_specific_volume
            system_specific["ionization"] = self.system.ionization.model_dump()
            system_specific["pdb_path"] = str(self.system.protein.pdb_path)
            # The protein anchors the box but is not part of system.species
            # (packing is solution-only); list it as a species so metadata
            # matches proteinbox.
            species_composition.insert(0, {**protein_anchor(self), "concentration": None})

        return {
            "hash": self.hash,
            "simulation_type": self.simulation_type,
            "engine": self.engine,
            "parametrization": self.parametrization.family,
            "parametrization_components": self.parametrization.components(),
            "temperature": self.simulation_settings.temperature,
            "pressure": self.simulation_settings.pressure,
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
            "parametrization": self.parametrization.family,
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
    def resolve_parametrization(self) -> "BuildInput":
        """Fill family defaults for the simulation type and reject invalid components."""
        resolved = self.parametrization.resolve(self.simulation_type)
        object.__setattr__(self, "parametrization", resolved)
        return self

    @model_validator(mode="after")
    def resolve_simulation_settings(self) -> "BuildInput":
        """Fill the per-type simulation-settings defaults (e.g. the production length)."""
        resolved = self.simulation_settings.resolve(self.simulation_type)
        object.__setattr__(self, "simulation_settings", resolved)
        return self

    @model_validator(mode="after")
    def validate_protein_chain_config(self) -> "BuildInput":
        """Validate protein chain grouping against pdb2gmx molecule handling.

        merge_all fuses every chain into a single ``Protein`` moleculetype, which
        contradicts protein.chains declaring subunits to build separately. Catching
        this here avoids a confusing chain-mismatch failure deep inside pdb2gmx.
        Compression additionally requires a multi-chain complex to be merged so
        the barostat cannot scale chain centers independently.
        """
        if self.simulation_type not in PROTEIN_SIMULATION_TYPES:
            return self
        protein = self.system.protein
        if (
            self.simulation_type == "protein_mixedbox"
            and len(protein.chains) > 1
            and not protein.merge_all
        ):
            raise ValueError(
                "A multi-chain protein_mixedbox must be one pdb2gmx moleculetype so "
                "the barostat moves the complete protein as a rigid unit. Set "
                "system.protein.merge_all=true and omit protein.chains."
            )
        if protein.merge_all and protein.chains:
            raise ValueError(
                "protein.chains declares subunits to build as separate moleculetypes, but "
                "system.protein.merge_all merges all chains into one. Set "
                "merge_all=false to keep per-chain subunits, or drop protein.chains to merge."
            )
        return self


def write_build_record(model: BuildInput, directory: Path) -> Path:
    """Write ``<hash>.yaml``, the build directory's record of its BuildInput.

    The record holds the full input, including the resolved simulation
    settings, so the directory documents the protocol its mdp files follow.

    Parameters
    ----------
    model : BuildInput
        The build input.
    directory : Path
        Build directory.

    Returns
    -------
    Path
        The written file.

    """
    path = Path(directory) / f"{model.hash}.yaml"
    # mode="json" so Path and tuple fields (e.g. proteinbox pdb_path and
    # disulfide_bonds) serialize to plain str/list that yaml.safe_dump can write.
    # sort_keys=False keeps the order of dict fields (e.g. LNP input_fractions),
    # which the hash depends on, so the record reloads with the same hash.
    path.write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))
    return path


def check_unique_hashes(models: Sequence[BuildInput], labels: Sequence[str] | None = None) -> None:
    """Reject inputs that share a hash, i.e. would build into the same directory.

    Only temperature and pressure distinguish otherwise equal systems, so
    inputs that differ only in other simulation settings are duplicates.

    Parameters
    ----------
    models : sequence of BuildInput
        The inputs of one build call.
    labels : sequence of str, optional
        Names of the inputs for the error message (default ``input <i>``).

    Raises
    ------
    ValueError
        If two inputs have the same hash.

    """
    labels = list(labels) if labels is not None else [f"input {i}" for i in range(len(models))]
    first: dict[str, int] = {}
    problems = []
    for index, model in enumerate(models):
        key = model.hash
        if key not in first:
            first[key] = index
            continue
        other = models[first[key]]
        if other.simulation_settings == model.simulation_settings:
            reason = "they are identical"
        else:
            reason = (
                "they differ only in simulation settings outside the identity; "
                "only temperature and pressure define a different system"
            )
        problems.append(f"{labels[first[key]]} and {labels[index]} (hash {key}): {reason}")
    if problems:
        raise ValueError("Duplicate build inputs:\n  " + "\n  ".join(problems))
