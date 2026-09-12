# ABOUTME: Tests for topology utility functions including CGenFF topology parsing,
# ABOUTME: reusable parameter extraction, mol2 generation from RDKit, and ITP merging.
"""Tests for topology utility functions including CGenFF topology parsing and mol2 generation."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from mdfactory.models.species import SingleMoleculeSpecies
from mdfactory.settings import settings
from mdfactory.utils.topology_utilities import (
    _atomtype_signature,
    _get_sybyl_atom_type,
    extract_reusable_parts_from_cgenff_gmx_top,
    merge_extra_parameter_itps,
    validate_cgenff_atomtype_compatibility,
    write_mol2_from_rdkit,
)
from mdfactory.utils.utilities import working_directory

cgenff_dir = settings.cgenff_dir
cgenff_available = cgenff_dir.exists() and (cgenff_dir / "cgenff" / "cgenff_batch.sh").exists()

MOL2_REFERENCES = Path(__file__).parent / "testfiles" / "mol2_references.json"


def test_atomtype_signature_ignores_inline_comment():
    """A trailing ';' comment does not change the physical-parameter signature."""
    plain = "CG331 6 12.011 0.0 A 0.36 0.28"
    commented = "CG331 6 12.011 0.0 A 0.36 0.28 ; sp3 carbon"
    assert _atomtype_signature(commented) == _atomtype_signature(plain)


def test_cgenff_atomtype_compatibility_ignores_inline_comment(tmp_path):
    """Matching atom types with a trailing comment on one side are still compatible."""
    ff_dir = tmp_path / "charmm36m.ff"
    ff_dir.mkdir()
    (ff_dir / "atomtypes.itp").write_text("[ atomtypes ]\nCG331 6 12.011 0.0 A 0.36 0.28\n")
    generated = tmp_path / "generated.itp"
    generated.write_text("[ atomtypes ]\nCG331 6 12.011 0.0 A 0.36 0.28 ; sp3 carbon\n")

    # Identical physical parameters, only a comment differs -> no conflict raised.
    validate_cgenff_atomtype_compatibility([generated], ff_dir)


def test_cgenff_atomtype_compatibility_rejects_conflict(tmp_path):
    ff_dir = tmp_path / "charmm36m.ff"
    ff_dir.mkdir()
    (ff_dir / "atomtypes.itp").write_text("[ atomtypes ]\nCG331 6 12.011 0.0 A 0.36 0.28\n")
    generated = tmp_path / "generated.itp"
    generated.write_text("[ atomtypes ]\nCG331 6 12.011 0.0 A 0.40 0.28\n")

    with pytest.raises(ValueError, match="conflicts with the selected CHARMM"):
        validate_cgenff_atomtype_compatibility([generated], ff_dir)


def _load_mol2_references():
    with open(MOL2_REFERENCES) as f:
        return json.load(f)


def _extract_atom_types(mol2_str):
    """Extract Sybyl atom types from a mol2 string."""
    types = []
    in_atoms = False
    for line in mol2_str.strip().split("\n"):
        if "@<TRIPOS>ATOM" in line:
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atoms = False
            continue
        if in_atoms and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                types.append(parts[5])
    return types


def _extract_bond_types(mol2_str):
    """Extract Tripos bond types from a mol2 string."""
    types = []
    in_bonds = False
    for line in mol2_str.strip().split("\n"):
        if "@<TRIPOS>BOND" in line:
            in_bonds = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_bonds = False
            continue
        if in_bonds and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                types.append(parts[3])
    return types


def _mol2_reference_ids(*, lipids_only=False, exclude_lipids=False):
    """Return list of (name, smiles, ref_mol2) from mol2_references.json."""
    refs = _load_mol2_references()
    return [
        (name, ref["smiles"], ref["mol2"])
        for name, ref in refs.items()
        if (not lipids_only or name.startswith("lipid_"))
        and (not exclude_lipids or not name.startswith("lipid_"))
    ]


_CORE_REFS = _mol2_reference_ids(exclude_lipids=True)
_LIPID_REFS = _mol2_reference_ids(lipids_only=True)


def _assert_mol2_matches_reference(tmp_path, name, smiles, ref_mol2):
    """Shared assertion logic for mol2 reference tests."""
    species = SingleMoleculeSpecies(smiles=smiles, resname="TST", count=1)
    mol2_file = tmp_path / f"{name}.mol2"
    write_mol2_from_rdkit(species.rdkit_molecule, mol2_file, title="XXX")

    with open(mol2_file) as f:
        new_mol2 = f.read()

    assert "@<TRIPOS>MOLECULE" in new_mol2
    assert "@<TRIPOS>ATOM" in new_mol2
    assert "@<TRIPOS>BOND" in new_mol2
    ref_atom_types = _extract_atom_types(ref_mol2)
    new_atom_types = _extract_atom_types(new_mol2)
    assert len(new_atom_types) == len(ref_atom_types), (
        f"atom count {len(new_atom_types)} != ref {len(ref_atom_types)}"
    )

    ref_bond_types = _extract_bond_types(ref_mol2)
    new_bond_types = _extract_bond_types(new_mol2)
    assert len(new_bond_types) == len(ref_bond_types), (
        f"bond count {len(new_bond_types)} != ref {len(ref_bond_types)}"
    )

    assert sorted(new_atom_types) == sorted(ref_atom_types), (
        f"atom type mismatch: {sorted(new_atom_types)} vs {sorted(ref_atom_types)}"
    )

    assert sorted(new_bond_types) == sorted(ref_bond_types), (
        f"bond type mismatch: {sorted(new_bond_types)} vs {sorted(ref_bond_types)}"
    )


@pytest.mark.parametrize("name,smiles,ref_mol2", _CORE_REFS, ids=[t[0] for t in _CORE_REFS])
def test_write_mol2_from_rdkit(tmp_path, name, smiles, ref_mol2):
    """Test mol2 writer against 30 hand-picked reference molecules."""
    _assert_mol2_matches_reference(tmp_path, name, smiles, ref_mol2)


@pytest.mark.slow
@pytest.mark.parametrize("name,smiles,ref_mol2", _LIPID_REFS, ids=[t[0] for t in _LIPID_REFS])
def test_write_mol2_from_rdkit_lipids(tmp_path, name, smiles, ref_mol2):
    """Test mol2 writer against ~100 diverse lipids. Use ``pytest -m slow -n auto``."""
    _assert_mol2_matches_reference(tmp_path, name, smiles, ref_mol2)


def test_write_mol2_from_rdkit_no_conformer(tmp_path):
    """Test that write_mol2_from_rdkit raises ValueError when molecule has no conformers."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    with pytest.raises(ValueError, match="must have at least one conformer"):
        write_mol2_from_rdkit(mol, tmp_path / "test.mol2")


def test_extract_reusable_parts_from_cgenff_gmx_top(tmp_path):
    itp = Path(__file__).parent / "testfiles" / "cgenff_output.itp"
    with open(itp, "r") as f:
        itp_str = f.read()
        assert "[ moleculetype ]" in itp_str
        assert "[ atoms ]" in itp_str
        assert "[ bonds ]" in itp_str
        assert "[ angles ]" in itp_str
        assert "[ dihedrals ]" in itp_str
        assert "[ bondtypes ]" in itp_str
        assert "[ angletypes ]" in itp_str
        assert "[ dihedraltypes ]" in itp_str

    main_itp, extra_params, sections = extract_reusable_parts_from_cgenff_gmx_top(
        itp, must_have_moleculetype=True
    )
    section_names = [s[0] for s in sections]
    assert section_names == ["bondtypes", "angletypes", "dihedraltypes", "dihedraltypes"]

    num_rows_per_section = []
    for s in sections:
        sl = s[1].strip().split("\n")
        sl = [
            ll
            for ll in sl
            if ll.strip() and not ll.strip().startswith(";") and not ll.strip().startswith("[")
        ]
        num_rows_per_section.append(len(sl))

    sums = defaultdict(int)
    for name, n in zip(section_names, num_rows_per_section):
        sums[name] += n

    extra_path = tmp_path / "extra.itp"
    with open(extra_path, "w") as f:
        f.write(extra_params)

    merged = merge_extra_parameter_itps([extra_path, extra_path])
    merged_path = tmp_path / "merged.itp"
    with open(merged_path, "w") as f:
        f.write(merged)
    *_, sections_merged = extract_reusable_parts_from_cgenff_gmx_top(
        merged_path, must_have_moleculetype=False
    )
    section_names_merged = [s[0] for s in sections_merged]
    assert section_names_merged == ["bondtypes", "angletypes", "dihedraltypes"]
    num_rows_per_section_merged = []
    for s in sections_merged:
        sl = s[1].strip().split("\n")
        sl = [
            ll
            for ll in sl
            if ll.strip() and not ll.strip().startswith(";") and not ll.strip().startswith("[")
        ]
        num_rows_per_section_merged.append(len(sl))

    for name, n in zip(section_names_merged, num_rows_per_section_merged):
        assert sums[name] == n


def test_count_contiguous_strings():
    from mdfactory.utils.topology_utilities import count_contiguous_strings

    lst = [1, 1, 2, 2, 2, 3, 3, 1, 1, 1]
    counts = count_contiguous_strings(lst)
    assert counts == [(1, 2), (2, 3), (3, 2), (1, 3)]

    lst = ["A", "A", "B", "B", "B", "C", "C", "A", "A", "A"]
    counts = count_contiguous_strings(lst)
    assert counts == [("A", 2), ("B", 3), ("C", 2), ("A", 3)]

    lst = []
    counts = count_contiguous_strings(lst)
    assert counts == []


@pytest.mark.skipif(not cgenff_available, reason="CGenFF not available")
def test_run_cgenff_to_gmx(tmp_path):
    from mdfactory.parametrize import run_cgenff_to_gmx

    species = SingleMoleculeSpecies(smiles="CC", resname="ETH", count=1)
    mol2_file = tmp_path / "XXX.mol2"
    write_mol2_from_rdkit(species.rdkit_molecule, mol2_file, title="XXX")

    with working_directory(tmp_path):
        run_cgenff_to_gmx(mol2_file)

    itp = tmp_path / "1_cgenff_batch" / "gromacs_output" / "XXX.itp"
    assert itp.is_file()

    with pytest.raises(FileNotFoundError):
        run_cgenff_to_gmx("file_does_not_exist")


def test_sybyl_atom_types_positional():
    """Test _get_sybyl_atom_type positionally on small hand-built molecules."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Ethanol: CCO
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    types = [_get_sybyl_atom_type(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())]
    assert types == ["C.3", "C.3", "O.3", "H", "H", "H", "H", "H", "H"]

    # Guanidine: N=C(N)N
    mol = Chem.AddHs(Chem.MolFromSmiles("N=C(N)N"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    types = [_get_sybyl_atom_type(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())]
    assert types == ["N.pl3", "C.cat", "N.pl3", "N.pl3", "H", "H", "H", "H", "H"]

    # Acetate fragment: [O-]C=O — both oxygens should be O.co2
    mol = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    types = [_get_sybyl_atom_type(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())]
    assert types[0] == "O.co2"  # charged oxygen
    assert types[2] == "O.co2"  # partner oxygen via resonance logic


def test_oco2_carboxylate_partner():
    """Test that both oxygens in a carboxylate get O.co2.

    The charged oxygen matches via formal charge, the neutral one via partner logic.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    # Atom 0: O with formal charge -1 → O.co2 directly
    o_charged = mol.GetAtomWithIdx(0)
    assert o_charged.GetFormalCharge() == -1
    assert _get_sybyl_atom_type(o_charged) == "O.co2"

    # Atom 2: neutral O bonded to C that also has O- → O.co2 via partner logic
    o_neutral = mol.GetAtomWithIdx(2)
    assert o_neutral.GetFormalCharge() == 0
    assert _get_sybyl_atom_type(o_neutral) == "O.co2"
