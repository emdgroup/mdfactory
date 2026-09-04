# ABOUTME: Pydantic models for molecular species (small molecules, lipids, proteins)
# ABOUTME: Provides SMILES-based identity, charge, and molecular object properties
"""Pydantic models for molecular species (small molecules, lipids, proteins)."""

import hashlib
import io
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator
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
        # TODO: expose/central location?
        allowed_ions = {"[Na+]", "[Cl-]"}
        ret = (
            self.smiles in allowed_ions and self.rdkit_molecule.GetAtoms()[0].GetFormalCharge() != 0
        )
        return ret

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
            from ..utils.setup_utilities import generate_lipid_structure  # noqa: PLC0415

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
        from openff.toolkit.topology import Molecule  # noqa: PLC0415

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
        import MDAnalysis as mda  # noqa: PLC0415

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
        from ..utils.setup_utilities import generate_lipid_structure  # noqa: PLC0415

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


class ProteinSpecies(Species):
    """Represent a protein species identified by a PDB structure file."""

    count: Optional[int] = Field(1, description="Number of protein copies (always 1).")
    fraction: Optional[float] = Field(1.0, description="Fraction (always 1.0 for protein).")
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

    @model_validator(mode="after")
    def check_single_copy(self) -> "ProteinSpecies":
        """Enforce that a protein species describes exactly one copy.

        The builder produces a single protein structure and metadata reports a
        total count of 1, so a count or fraction other than one would silently
        contradict what is built.
        """
        if self.count != 1:
            raise ValueError(
                f"ProteinSpecies represents one protein; count must be 1, got {self.count}."
            )
        if self.fraction != 1.0:
            raise ValueError(
                f"ProteinSpecies represents one protein; fraction must be 1.0, got {self.fraction}."
            )
        return self

    @property
    def charge(self) -> int | None:
        """Return None; protein charge comes from the pdb2gmx topology, not the species."""
        return None
