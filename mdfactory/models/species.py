# ABOUTME: Pydantic models for molecular species (small molecules, lipids, proteins)
# ABOUTME: Provides SMILES-based identity, charge, and molecular object properties
"""Pydantic models for molecular species (small molecules, lipids, proteins)."""

import hashlib
import io
import warnings
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    Tag,
    model_validator,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from ..utils.chemistry_utilities import detect_lipid_parts_from_smiles_modified
from .validators import CanonicalIsomericSmiles, ResidueName


class Species(BaseModel):
    """Represent a molecular species with a fractional or absolute count in a system."""

    fraction: Optional[float] = None
    count: Optional[int] = None
    resname: ResidueName = Field(..., description="Residue name.")

    @model_validator(mode="after")
    def check_fraction_or_count(self) -> "Species":
        # if self.fraction is not None and self.count is not None:
        #     raise ValueError("Provide either fraction or 'count', not both.")
        if self.fraction is None and self.count is None:
            raise ValueError("Must provide 'fraction' or 'count'.")
        return self

    @property
    def charge(self) -> int:
        """Return the formal charge of the species."""
        raise NotImplementedError("Charge property not implemented for base Species class.")


class NativeSpecies(NamedTuple):
    """Naming of a water or ion that the force-field directory provides itself."""

    #: Residue name, which is also the GROMACS moleculetype name.
    resname: str
    #: Atom names in packing order (the same in CHARMM36m and ``amber14sb.ff``).
    atom_names: tuple[str, ...]


#: Native water and monatomic ions, keyed by SMILES.
NATIVE_SPECIES: dict[str, NativeSpecies] = {
    "O": NativeSpecies("SOL", ("OW", "HW1", "HW2")),
    "[Na+]": NativeSpecies("NA", ("NA",)),
    "[Cl-]": NativeSpecies("CL", ("CL",)),
}
WATER_RESNAME = NATIVE_SPECIES["O"].resname


class SingleMoleculeSpecies(Species):
    """Represent a single-molecule species identified by its canonical SMILES string."""

    smiles: CanonicalIsomericSmiles = Field(
        ..., description="SMILES string describing the molecule."
    )

    @property
    def hash(self) -> str:
        """Return a truncated SHA-1 hash of the SMILES string."""
        digits = 20
        return hashlib.sha1(self.smiles.encode("UTF-8")).hexdigest()[:digits].upper()

    @property
    def is_water(self):
        """Return True if the species is water."""
        return self.smiles == "O"

    @property
    def is_ion(self):
        """Return True if the species is a monatomic ion."""
        return (
            self.smiles in NATIVE_SPECIES
            and not self.is_water
            and self.rdkit_molecule.GetAtoms()[0].GetFormalCharge() != 0
        )

    @cached_property
    def rdkit_molecule(self) -> "Chem.rdchem.Mol":
        """Return an RDKit molecule with 3D coordinates from the SMILES string."""
        # check whether we have a lipid-like structure
        head_index = None
        tail_indices = []
        try:
            head_index, tail_indices, *_ = detect_lipid_parts_from_smiles_modified(
                self.smiles, head_search_radius=3
            )
        except Exception as e:
            warnings.warn(
                f"Attempted to build lipid, but it didn't work: {e}. Continuing with normal code..."
            )
        if head_index is not None and len(tail_indices):
            from ..utils.setup_utilities import generate_lipid_structure

            try:
                mol = generate_lipid_structure(
                    self.smiles, head_indices=[head_index], tail_indices=tail_indices
                )
                return mol
            except Exception as e:
                warnings.warn(
                    f"Attempted to build lipid, but it didn't work: {e}. "
                    "Continuing with normal code..."
                )

        # proceed as normal
        rdkit_mol = Chem.MolFromSmiles(self.smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        Chem.SanitizeMol(rdkit_mol)

        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42000
        ec = AllChem.EmbedMolecule(rdkit_mol, ps)

        if ec != 0:
            # Retry with random coordinate initialization
            ps.useRandomCoords = True
            ec = AllChem.EmbedMolecule(rdkit_mol, ps)

        if ec != 0:
            raise ValueError(
                f"RDKit could not generate a 3D conformer for SMILES '{self.smiles}'. "
                "Consider simplifying the molecule or providing coordinates manually."
            )

        AllChem.UFFOptimizeMolecule(rdkit_mol)
        return rdkit_mol

    @property
    def openff_molecule(self):
        """Return an OpenFF Molecule with residue name set."""
        from openff.toolkit.topology import Molecule

        ret = Molecule.from_rdkit(
            self.rdkit_molecule,
            allow_undefined_stereo=False,
        )
        ret.name = self.resname.upper()
        ret.properties["resname"] = self.resname.upper()
        return ret

    @property
    def universe(self):
        """Return an MDAnalysis Universe built from the molecule's PDB representation."""
        import MDAnalysis as mda

        pdb = Chem.MolToPDBBlock(self.rdkit_molecule)
        u = mda.Universe(io.StringIO(pdb), format="pdb")
        return u

    @property
    def charge(self) -> int:
        """Return the total formal charge of the molecule."""
        total_charge = sum(atom.GetFormalCharge() for atom in self.rdkit_molecule.GetAtoms())
        return total_charge

    @property
    def mass(self) -> float:
        """Return the molecular weight in Daltons."""
        ml = Chem.MolFromSmiles(self.smiles)
        return Descriptors.MolWt(ml)


class SolutionSpecies(SingleMoleculeSpecies):
    """Represent a solution molecule specified by an exact count or a molar concentration.

    Used for the molecules packed around a protein in a protein_mixedbox: water
    (expressed as ``smiles: O``), ions, and arbitrary SMILES solutes. Exactly one
    of ``count`` or ``concentration`` must be given; fractions are not supported,
    because the box packs an explicit number of each molecule or resolves a molar
    concentration to a count against a chosen volume basis.
    """

    count: Optional[int] = Field(None, description="Exact number of molecules to pack.", ge=0)
    concentration: Optional[float] = Field(
        None, description="Target molar concentration in mol/L (M).", ge=0.0
    )

    @model_validator(mode="after")
    def check_fraction_or_count(self) -> "SolutionSpecies":
        """Require exactly one of count or concentration and reject fractions.

        Shadows the base ``Species`` validator (same method name) so a species
        carrying only a concentration is not rejected for lacking a count.
        """
        if self.fraction is not None:
            raise ValueError(
                "SolutionSpecies does not support 'fraction'; provide 'count' or 'concentration'."
            )
        if (self.count is None) == (self.concentration is None):
            raise ValueError("SolutionSpecies requires exactly one of 'count' or 'concentration'.")
        native = NATIVE_SPECIES.get(self.smiles)
        if native is not None and self.resname.upper() != native.resname:
            raise ValueError(
                f"Native water/ion species {self.smiles!r} must use resname "
                f"'{native.resname}', not '{self.resname}'."
            )
        return self


class LipidSpecies(SingleMoleculeSpecies):
    """Represent a lipid species with head, tail, and branch atom annotations."""

    head_atoms: list[int] = Field(
        default_factory=list, description="0-based indices of head group atoms."
    )
    tail_atoms: list[int] = Field(
        default_factory=list, description="0-based indices of tail group atoms."
    )
    branch_atoms: list[int] = Field(
        default_factory=list, description="0-based indices of branch point atoms."
    )

    def model_post_init(self, _) -> None:
        """Detect head, tail, and branch atoms from SMILES if not provided."""
        if len(self.head_atoms) == 0 or len(self.tail_atoms) == 0 or len(self.branch_atoms) == 0:
            head_index, tail_indices, branch_indices = detect_lipid_parts_from_smiles_modified(
                self.smiles, head_search_radius=3
            )
            if head_index is None or len(tail_indices) == 0:
                raise ValueError("Could not determine head or tail atoms of this lipid.")
            self.head_atoms = [head_index]
            self.tail_atoms = tail_indices
            self.branch_atoms = branch_indices

    @cached_property
    def rdkit_molecule(self) -> "Chem.rdchem.Mol":
        """Return an RDKit molecule with lipid-specific 3D coordinate generation."""
        from ..utils.setup_utilities import generate_lipid_structure

        mol = generate_lipid_structure(
            self.smiles, head_indices=self.head_atoms, tail_indices=self.tail_atoms
        )
        return mol

    # @property
    # def universe(self) -> mda.Universe:
    #     from ..utils.setup_utilities import align_lipid_with_z_axis
    #     pdb = Chem.MolToPDBBlock(self.rdkit_molecule)
    #     u = mda.Universe(io.StringIO(pdb), format="pdb")

    #     u = align_lipid_with_z_axis(
    #         u, tail_atom_ids=self.tail_atoms, head_atom_ids=self.head_atoms, z_axis=[1, 0, 0]
    #     )
    #     return u


#: Restraint names whose ``POSRES_<NAME>`` macro pdb2gmx already uses.
_RESERVED_RESTRAINT_NAMES = {"WATER"}

#: Identifier of a restraint, unique within a protein (ignoring case).
RestraintName = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description=(
            "Identifier of the restraint, used for its file and group names. Unique "
            "within a protein, ignoring case."
        ),
    ),
]

#: MDAnalysis selection evaluated on the pdb2gmx structure.
Selection = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PositionRestraint(BaseModel):
    """A position restraint on the protein atoms matched by an MDAnalysis selection.

    The build writes the restraint as a ``[ position_restraints ]`` include
    guarded by ``#ifdef POSRES_<NAME>`` and adds ``-DPOSRES_<NAME>`` to the
    production ``define``. The atoms are restrained to their positions in the
    production starting structure.
    """

    model_config = ConfigDict(extra="forbid")

    name: RestraintName
    selection: Selection = Field(
        ...,
        description=(
            "MDAnalysis selection of the restrained atoms, evaluated on the pdb2gmx "
            "structure with the input chain IDs and residue numbers."
        ),
    )
    type: Literal["harmonic", "flat_bottom"] = Field(
        "harmonic",
        description=(
            "harmonic (GROMACS funct 1, isotropic) or flat_bottom (funct 2, a sphere "
            "of the given radius)."
        ),
    )
    k: float = Field(..., gt=0, description="Force constant in kJ mol^-1 nm^-2.")
    radius: Optional[float] = Field(
        None, gt=0, description="Flat-bottom sphere radius in nm (flat_bottom only)."
    )

    @model_validator(mode="after")
    def check_radius_and_name(self) -> "PositionRestraint":
        """Require ``radius`` for flat-bottom restraints only and reject reserved names."""
        if self.type == "flat_bottom" and self.radius is None:
            raise ValueError(f"Flat-bottom restraint '{self.name}' needs a 'radius' (nm).")
        if self.type == "harmonic" and self.radius is not None:
            raise ValueError(
                f"Harmonic restraint '{self.name}' takes no 'radius'; "
                "use type 'flat_bottom' for a flat-bottom sphere."
            )
        if self.name.upper() in _RESERVED_RESTRAINT_NAMES:
            raise ValueError(
                f"Restraint name '{self.name}' is reserved: pdb2gmx already defines "
                f"POSRES_{self.name.upper()}."
            )
        return self

    @property
    def macro(self) -> str:
        """Return the preprocessor macro that switches the restraint on."""
        return f"POSRES_{self.name.upper()}"


class ComDistanceRestraint(BaseModel):
    """A restraint on the distance between the centres of mass of two atom selections.

    The build writes both selections as index groups and adds a GROMACS pull
    coordinate (``geometry = distance``) to the production stage: an umbrella
    potential around the reference distance, or with ``width`` a flat-bottom
    potential that only acts beyond reference + ``width``.
    """

    model_config = ConfigDict(extra="forbid")

    name: RestraintName
    type: Literal["com_distance"]
    selections: tuple[Selection, Selection] = Field(
        ...,
        description=(
            "The two MDAnalysis selections whose centres of mass are restrained, "
            "evaluated like a position restraint's selection."
        ),
    )
    k: float = Field(
        ..., gt=0, description="Force constant on the COM distance in kJ mol^-1 nm^-2."
    )
    width: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Flat-bottom width in nm: no force while the distance stays below the "
            "reference distance + width. Omit for an umbrella (harmonic) potential."
        ),
    )
    distance: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Reference distance in nm. Default: the distance measured in the built system."
        ),
    )

    @property
    def group_names(self) -> tuple[str, str]:
        """Return the index group names of the two selections."""
        return f"COM_{self.name.upper()}_1", f"COM_{self.name.upper()}_2"


def _restraint_kind(value) -> str:
    """Return the union tag of a restraint: ``com`` for COM distances, else ``position``."""
    kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return "com" if kind == "com_distance" else "position"


#: One entry of ``ProteinSpecies.restraints``, told apart by ``type``.
Restraint = Annotated[
    Annotated[PositionRestraint, Tag("position")] | Annotated[ComDistanceRestraint, Tag("com")],
    Discriminator(_restraint_kind),
]


class ProteinSpecies(Species):
    """Represent a protein species identified by a PDB structure file."""

    # A protein species is exactly one copy: the builder produces a single structure
    # and metadata reports a total count of 1.
    count: Literal[1] = Field(1, description="Number of protein copies (always 1).")
    fraction: Literal[1.0] = Field(1.0, description="Fraction (always 1.0 for protein).")
    pdb_path: Path = Field(..., description="Path to the input PDB file.")
    chains: list[str] = Field(
        default_factory=list,
        description=(
            "Chain identifiers to build, e.g. ['A'] or ['H', 'L', 'Y']. Required for "
            "multi-chain structures and validated against the chains pdb2gmx produces; "
            "optional for a single-chain structure."
        ),
    )
    disulfide_bonds: list[tuple[str, str]] = Field(
        default_factory=list,
        description=(
            "Pairs of cysteine residues forming disulfide bonds, "
            "e.g. [('CYS6', 'CYS127'), ('CYS30', 'CYS115')]. Each residue may be "
            "chain-qualified and may include a PDB insertion code (e.g. 'A:CYS6' or "
            "'H:CYS100A'); a partially qualified reference matching multiple residues is "
            "rejected as ambiguous."
        ),
    )
    protonation_states: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Residue-specific protonation states, e.g. {'HIS15': 'HIE'}. Keys may be "
            "chain-qualified and may include a PDB insertion code (e.g. "
            "{'A:HIS15': 'HIE'} or {'H:HIS100A': 'HIE'}); a partially qualified key "
            "matching multiple residues is rejected as ambiguous. For CHARMM only "
            "3-character states are expressible (histidine tautomers, neutral lysine)."
        ),
    )
    ignore_hydrogens: bool = Field(
        True, description="Ignore hydrogens in the input PDB (pdb2gmx regenerates them)."
    )
    merge_all: bool = Field(
        False, description="Merge all chains into a single pdb2gmx moleculetype."
    )
    restraints: list[Restraint] = Field(
        default_factory=list,
        description=(
            "Selection-based restraints applied in the production stage: position "
            "restraints (harmonic, flat_bottom) and COM-distance restraints "
            "(com_distance). NVT and NPT keep pdb2gmx's heavy-atom restraints (-DPOSRES)."
        ),
    )

    @model_validator(mode="after")
    def check_unique_restraint_names(self) -> "ProteinSpecies":
        """Reject restraints whose names differ only in case (they share file and group names)."""
        seen: dict[str, str] = {}
        for restraint in self.restraints:
            previous = seen.get(restraint.name.upper())
            if previous is not None:
                raise ValueError(
                    f"Restraint names '{previous}' and '{restraint.name}' are not unique "
                    "(names are compared ignoring case)."
                )
            seen[restraint.name.upper()] = restraint.name
        return self

    @property
    def position_restraints(self) -> list[PositionRestraint]:
        """Return the harmonic and flat-bottom position restraints."""
        return [r for r in self.restraints if isinstance(r, PositionRestraint)]

    @property
    def com_restraints(self) -> list[ComDistanceRestraint]:
        """Return the COM-distance restraints."""
        return [r for r in self.restraints if isinstance(r, ComDistanceRestraint)]

    @property
    def charge(self) -> int | None:
        """Return None; protein charge comes from the pdb2gmx topology, not the species."""
        return None
