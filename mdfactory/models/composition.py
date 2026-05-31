# ABOUTME: Pydantic models for system composition (mixedbox, bilayer, LNP, proteinbox)
# ABOUTME: Defines species counts, ionization, and composition validation
"""Pydantic models for system composition (mixedbox, bilayer, LNP, proteinbox)."""

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .species import LipidSpecies, ProteinSpecies, SingleMoleculeSpecies, Species


def distribute_counts(fractions: list[float], total: int) -> list[int]:
    """Distribute total into integer counts preserving the sum exactly.

    Uses floor + largest fractional part method to ensure the sum of
    returned counts equals total exactly, while respecting proportions.

    Parameters
    ----------
    fractions : list[float]
        Fractions for each species (must sum to 1.0).
    total : int
        Total count to distribute.

    Returns
    -------
    list[int]
        Integer counts that sum to exactly `total`.

    """
    raw = [f * total for f in fractions]
    floors = [int(np.floor(r)) for r in raw]
    remainders = [r - f for r, f in zip(raw, floors)]

    # Distribute the remaining counts to species with largest fractional parts
    deficit = total - sum(floors)
    # Get indices sorted by remainder (descending)
    indices_by_remainder = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)

    for i in indices_by_remainder[:deficit]:
        floors[i] += 1

    return floors


class IonizationConfig(BaseModel):
    """Configuration for ionization of the system."""

    model_config = ConfigDict(extra="forbid")
    neutralize: bool = Field(
        True, description="Whether to add ions to neutralize the system charge."
    )
    concentration: float = Field(
        0.15, description="Target salt concentration in mol/L (M).", ge=0.0
    )
    min_distance: float = Field(
        5.0,
        description=(
            "Minimum distance (in Å) from non-water atoms for water molecules to be "
            "considered for replacement."
        ),
        ge=0.0,
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")


class SystemComposition(BaseModel):
    """Define a system composition as a list of species with fractional or absolute counts."""

    model_config = ConfigDict(extra="forbid")

    species: list[Species]
    total_count: Optional[int] = Field(None, description="Total number of molecules", ge=0)

    @model_validator(mode="after")
    def validate_system(self) -> "SystemComposition":
        fractions_used = all(s.fraction is not None for s in self.species)
        counts_used = all(s.count is not None for s in self.species)

        if not fractions_used and not counts_used:
            raise ValueError("Use fractions or counts for all species consistently.")

        if counts_used:
            sum_of_counts = sum(s.count for s in self.species if s.count is not None)
            self.total_count = sum_of_counts
            for s in self.species:
                s.fraction = s.count / sum_of_counts
        elif fractions_used:
            fraction_sum = sum(s.fraction for s in self.species if s.fraction is not None)
            if abs(fraction_sum - 1.0) > 1e-9:
                raise ValueError("Sum of fractions must be exactly 1.0.")
            if self.total_count is None:
                raise ValueError("If fractions are used, 'total_count' must be provided.")
            for s in self.species:
                s.count = int(self.total_count * s.fraction)

        # if a species has zero count, remove it from the model
        self.species = [s for s in self.species if s.fraction > 0 and s.count > 0]

        return self

    @property
    def charge(self) -> int:
        """Return the total system charge summed over all species."""
        total_charge = sum(
            s.count * s.charge
            for s in self.species
            if isinstance(s, SingleMoleculeSpecies) and s.charge is not None
        )
        return total_charge


class MixedBoxComposition(SystemComposition):
    """Define a mixed-box system composition with density and ionization settings."""

    species: list[SingleMoleculeSpecies]
    target_density: float = Field(1.0, description="Packing density of the box in g/cm^3.")
    ionization: IonizationConfig = Field(
        default_factory=IonizationConfig, description="Configuration for ionization."
    )

    # @field_validator("ionization")
    # @classmethod
    # def validate_ionization(cls, v: Any):
    #     if v == None or v == "None":
    #         return None
    #     elif v == "auto":
    #         return IonizationConfig()
    #     return v


class BilayerComposition(SystemComposition):
    """Define a lipid bilayer composition with z-padding and ionization settings."""

    species: list[LipidSpecies]
    z_padding: float = Field(
        20.0, description="Z-direction box padding above and below the bilayer in A.", ge=0.0
    )
    monolayer: bool = Field(False, description="Whether to just build a monolayer.")
    ionization: IonizationConfig = Field(
        default_factory=IonizationConfig, description="Configuration for ionization."
    )

    @model_validator(mode="after")
    def validate_bilayer(self) -> "BilayerComposition":
        if not self.monolayer:
            # check that we have an even number of lipids
            for spec in self.species:
                if spec.count % 2 != 0:
                    raise ValueError(
                        f"All lipid species for bilayer must have an even count."
                        f" Found {spec.count} for {spec.resname}."
                    )
        return self


class CoreComposition(BaseModel):
    """Configuration for the LNP core (internal) sphere."""

    model_config = ConfigDict(extra="forbid")

    species: list[SingleMoleculeSpecies]  # Must use fractions, not counts
    target_density: float = Field(0.95, description="Packing density in g/cm^3.", gt=0)
    replication_factor: int = Field(
        1, description="Number of times to replicate unit cell in each dimension."
    )
    input_fractions: dict[str, float] = Field(
        default_factory=dict,
        description="Original input fractions by resname (preserved for reporting).",
    )

    @model_validator(mode="after")
    def validate_fractions(self) -> "CoreComposition":
        """Ensure all species use fractions and they sum to 1.0."""
        if not all(s.fraction is not None for s in self.species):
            raise ValueError("Core species must specify fractions, not counts.")
        fraction_sum = sum(s.fraction for s in self.species)
        if abs(fraction_sum - 1.0) > 1e-9:
            raise ValueError(f"Sum of core fractions must be exactly 1.0, got {fraction_sum}.")
        # Store original input fractions before any modifications
        if not self.input_fractions:
            self.input_fractions = {s.resname: s.fraction for s in self.species}
        return self

    def calculate_counts(self, radius: float, min_count: int = 5) -> None:
        """Calculate molecule counts for a unit cell that will be replicated.

        We pack only 1/n³ of the required molecules (where n = 2, 3, 4, ...)
        and then replicate the compressed box n times in each dimension.
        This dramatically speeds up packmol for large systems.

        The smallest n is chosen such that all species have at least `min_count`
        molecules in the unit cell.

        Parameters
        ----------
        radius : float
            Core radius in Angstroms.
        min_count : int
            Minimum count per species to ensure representation in unit cell.

        """
        V_angstrom3 = (4 / 3) * np.pi * radius**3
        V_cm3 = V_angstrom3 * 1e-24  # Å³ to cm³
        mass_g = self.target_density * V_cm3
        mass_dalton = mass_g * 6.022e23  # g to Daltons

        # Weighted average mass
        fractions = np.array([s.fraction for s in self.species])
        masses = np.array([s.mass for s in self.species])  # Daltons
        avg_mass = fractions @ masses

        total_count = int(np.ceil(mass_dalton / avg_mass))

        # Try progressively smaller unit cells (n³ replication)
        # Volume scales as n³, so we try n = 4, 3, 2, 1
        fractions = [s.fraction for s in self.species]
        for n in [4, 3, 2, 1]:
            unit_cell_count = int(np.ceil(total_count / (n**3)))
            counts = distribute_counts(fractions, unit_cell_count)
            if all(c >= min_count for c in counts):
                self.replication_factor = n
                for s, c in zip(self.species, counts):
                    s.count = c
                return

        # Fallback: no replication (should not reach here for reasonable systems)
        self.replication_factor = 1
        counts = distribute_counts(fractions, total_count)
        for s, c in zip(self.species, counts):
            s.count = c


class ShellComposition(BaseModel):
    """Configuration for the LNP shell (external) monolayer."""

    model_config = ConfigDict(extra="forbid")

    species: list[LipidSpecies]  # Must use fractions, not counts
    z0: float = Field(10.0, description="Pivotal plane offset for shell projection in Angstroms.")
    area_per_lipid: float = Field(65.0, description="Average area per lipid in Å².", gt=0)
    input_fractions: dict[str, float] = Field(
        default_factory=dict,
        description="Original input fractions by resname (preserved for reporting).",
    )

    @model_validator(mode="after")
    def validate_fractions(self) -> "ShellComposition":
        """Ensure all species use fractions and they sum to 1.0."""
        if not all(s.fraction is not None for s in self.species):
            raise ValueError("Shell species must specify fractions, not counts.")
        fraction_sum = sum(s.fraction for s in self.species)
        if abs(fraction_sum - 1.0) > 1e-9:
            raise ValueError(f"Sum of shell fractions must be exactly 1.0, got {fraction_sum}.")
        # Store original input fractions before any modifications
        if not self.input_fractions:
            self.input_fractions = {s.resname: s.fraction for s in self.species}
        return self

    def calculate_counts(self, core_radius: float, min_count: int = 5) -> None:
        """Calculate lipid counts for the monolayer patch to equilibrate.

        We use the smallest patch size that ensures all species have at least
        `min_count` lipids, trying fractions [1/8, 1/4, 1/2, 1] of the total.
        This is more economical than equilibrating all lipids, since the
        patch will be replicated to cover the sphere surface.

        Parameters
        ----------
        core_radius : float
            Core radius in Angstroms (shell wraps around this).
        min_count : int
            Minimum count per species to ensure representation after cropping.

        """
        surface_area = 4 * np.pi * core_radius**2
        total_lipids = int(np.ceil(surface_area / self.area_per_lipid))

        # Try progressively larger fractions until all species have count >= min_count
        fractions = [s.fraction for s in self.species]
        for scale in [1 / 8, 1 / 4, 1 / 2, 1]:
            patch_lipids = int(np.ceil(total_lipids * scale))
            counts = distribute_counts(fractions, patch_lipids)
            if all(c >= min_count for c in counts):
                for s, c in zip(self.species, counts):
                    s.count = c
                return

        # Fallback: use full count (should not reach here if fractions sum to 1)
        counts = distribute_counts(fractions, total_lipids)
        for s, c in zip(self.species, counts):
            s.count = c


class LNPComposition(BaseModel):
    """LNP system composition with separate core and shell layers.

    User specifies total radius and fractions; counts are calculated
    from molecular masses, density, and geometry.

    The core is built as a compressed mixed box shaped into a sphere.
    The shell is built from a monolayer projected onto the core surface.
    """

    model_config = ConfigDict(extra="forbid")

    radius: float = Field(..., description="Total LNP radius in Angstroms.", gt=0)
    shell_thickness: float = Field(
        28.0, description="Estimated shell thickness in Angstroms.", gt=0
    )
    core: CoreComposition
    shell: ShellComposition
    padding: float = Field(25.0, description="Solvation padding around LNP in Angstroms.", ge=0)
    ionization: IonizationConfig = Field(
        default_factory=IonizationConfig, description="Configuration for ionization."
    )

    @model_validator(mode="after")
    def calculate_all_counts(self) -> "LNPComposition":
        """Calculate counts for both core and shell based on total radius."""
        core_radius = self.radius - self.shell_thickness
        if core_radius <= 0:
            raise ValueError(
                f"Core radius ({core_radius}) must be positive. "
                f"Total radius ({self.radius}) too small for "
                f"shell thickness ({self.shell_thickness})."
            )

        self.core.calculate_counts(core_radius)
        self.shell.calculate_counts(core_radius)  # Shell wraps core
        return self

    @property
    def core_radius(self) -> float:
        """Computed core radius (total radius - shell thickness)."""
        return self.radius - self.shell_thickness

    @property
    def charge(self) -> int:
        """Total system charge from core and shell species."""
        total = 0
        for s in self.core.species:
            if s.charge is not None:
                total += s.count * s.charge
        for s in self.shell.species:
            if s.charge is not None:
                total += s.count * s.charge
        return total

    def get_unique_species_for_parametrization(self) -> list[Species]:
        """Get unique species by resname for parametrization (parameters are shared)."""
        species_map: dict[str, Species] = {}
        for spec in self.core.species:
            if spec.resname not in species_map:
                species_map[spec.resname] = spec.model_copy()
        for spec in self.shell.species:
            if spec.resname not in species_map:
                species_map[spec.resname] = spec.model_copy()
        return list(species_map.values())

    def get_species_with_counts(self) -> list[Species]:
        """Get all species with total counts for topology (sum core + shell)."""
        species_map: dict[str, Species] = {}
        for spec in self.core.species:
            key = spec.resname
            if key in species_map:
                species_map[key].count += spec.count
            else:
                species_map[key] = spec.model_copy()
        for spec in self.shell.species:
            key = spec.resname
            if key in species_map:
                species_map[key].count += spec.count
            else:
                species_map[key] = spec.model_copy()
        return list(species_map.values())


class ProteinBoxComposition(BaseModel):
    """Define a protein-in-waterbox system: one protein centered in a cubic box with ions."""

    model_config = ConfigDict(extra="forbid")

    protein: ProteinSpecies
    box_padding: float = Field(
        10.0, description="Minimum distance from protein to box edge in Angstroms.", ge=0.0
    )
    ionization: IonizationConfig = Field(
        default_factory=IonizationConfig, description="Configuration for ionization."
    )

    @property
    def species(self) -> list[ProteinSpecies]:
        """Return protein as a single-element species list for metadata compatibility."""
        return [self.protein]

    @property
    def total_count(self) -> int:
        """Return 1 (one protein) for metadata compatibility."""
        return 1

    @property
    def charge(self) -> int:
        """Protein charge is not pre-computable; return 0 as placeholder."""
        return 0
