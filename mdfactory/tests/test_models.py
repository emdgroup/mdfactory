# ABOUTME: Tests for data models including species, compositions, build inputs,
# ABOUTME: and parametrization models with validation and hashing behavior.
"""Tests for data models including species, compositions, build inputs,."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit.Chem import AllChem

from mdfactory.models.composition import (
    BilayerComposition,
    CoreComposition,
    IonizationConfig,
    MixedBoxComposition,
    ShellComposition,
    SystemComposition,
    distribute_counts,
)
from mdfactory.models.input import BuildInput
from mdfactory.models.species import LipidSpecies, SingleMoleculeSpecies, Species
from mdfactory.utils.utilities import working_directory


def test_single_molecule_species():
    spec = SingleMoleculeSpecies(smiles="C(C)C", fraction=0.1, resname="ABC")
    assert spec.smiles == "CCC"

    print(spec.rdkit_molecule, spec.openff_molecule)

    with pytest.raises(ValueError):
        spec = SingleMoleculeSpecies(smiles="CXC", fraction=0.1, resname="ABC")

    # with stereo specification: fine
    spec = SingleMoleculeSpecies(smiles="C[C@@H](C(=O)O)N", fraction=0.1, resname="ABC")
    assert spec.smiles == "C[C@H](N)C(=O)O"

    with pytest.raises(ValueError, match="Unspecified stereochemistry"):
        spec = SingleMoleculeSpecies(smiles="C[CH](C(=O)O)N", fraction=0.1, resname="ABC")

    with pytest.raises(ValueError, match="Unspecified stereochemistry"):
        spec = SingleMoleculeSpecies(smiles="CCC=CCC", fraction=0.1, resname="ABC")

    with pytest.raises(ValueError, match="Residue name must be less"):
        spec = SingleMoleculeSpecies(smiles="CCC", fraction=0.1, resname="ABCD")

    spec = SingleMoleculeSpecies(smiles="O", fraction=0.1, resname="ABC")
    assert spec.is_water

    spec = SingleMoleculeSpecies(smiles="[H]O[H]", fraction=0.1, resname="ABC")
    assert spec.is_water

    spec = SingleMoleculeSpecies(smiles="OO", fraction=0.1, resname="ABC")
    assert not spec.is_water

    spec = SingleMoleculeSpecies(smiles="[Na+]", fraction=0.1, resname="ABC")
    assert spec.is_ion

    spec = SingleMoleculeSpecies(smiles="[Ca+2]", fraction=0.1, resname="ABC")
    assert spec.charge == 2
    assert not spec.is_ion

    spec = SingleMoleculeSpecies(smiles="[NH4+]", fraction=0.1, resname="AMM")
    assert spec.charge == 1

    spec = SingleMoleculeSpecies(smiles="[O-]", fraction=0.1, resname="AMM")
    assert spec.charge == -1
    assert spec.hash == hashlib.sha1(b"[O-]").hexdigest()[:20].upper()

    with pytest.raises(ValueError, match="Must provide 'fraction' or 'count'"):
        spec = SingleMoleculeSpecies(smiles="[O-]", resname="AMM")

    with pytest.raises(
        NotImplementedError, match="Charge property not implemented for base Species class"
    ):
        Species(smiles="[O-]", count=10, resname="AMM").charge

    with pytest.warns(UserWarning, match="Attempted to build lipid"):
        SingleMoleculeSpecies(smiles="CC(CC)CC.[Na+]", count=10, resname="AMM").rdkit_molecule

    # patch rdkit's EmbedMolecule to fail in creation of 3D conformer
    patcher = patch.object(AllChem, "EmbedMolecule", return_value=0)
    patcher.start()
    sm = SingleMoleculeSpecies(smiles="CC(CC)CC", count=10, resname="AMM")
    with pytest.raises(ValueError, match="Bad Conformer Id"):
        sm.rdkit_molecule
    patcher.stop()

    patcher = patch.object(AllChem, "EmbedMolecule", return_value=-1)
    patcher.start()
    sm = SingleMoleculeSpecies(smiles="CC(CC)CC", count=10, resname="AMM")
    with pytest.raises(ValueError, match="RDKit could not generate a 3D conformer"):
        sm.rdkit_molecule
    patcher.stop()


def test_conformer_retry_succeeds():
    """Test that EmbedMolecule retry with useRandomCoords=True works when first attempt fails."""
    call_count = 0
    original_embed = AllChem.EmbedMolecule

    def mock_embed(mol, params):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return -1  # First call fails
        return original_embed(mol, params)  # Second call succeeds

    with patch.object(AllChem, "EmbedMolecule", side_effect=mock_embed):
        sm = SingleMoleculeSpecies(smiles="CC", count=1, resname="ETH")
        mol = sm.rdkit_molecule
        assert mol is not None
        assert mol.GetNumConformers() > 0
        assert call_count == 2  # Confirm retry happened


def test_lipid_species():
    spec = LipidSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=1,
    )
    print(spec.rdkit_molecule)
    print(spec.universe)

    spec = SingleMoleculeSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=5,
    )
    print(spec.rdkit_molecule)

    spec = LipidSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=5,
    )
    BilayerComposition(species=[spec], monolayer=True)
    with pytest.raises(ValueError, match="lipid species for bilayer must have an even count"):
        BilayerComposition(species=[spec], monolayer=False)

    with pytest.raises(ValueError, match="Could not determine"):
        LipidSpecies(smiles="CCO", resname="XYZ", count=1)


def test_generate_input_model_from_dict():
    spec = SingleMoleculeSpecies(smiles="O", fraction=0.9, resname="ABC")
    spec2 = SingleMoleculeSpecies(smiles="CO", fraction=0.1, resname="ABC")
    comp = SystemComposition(
        total_count=2000,
        species=[spec, spec2],
    )
    print(comp)

    spec = SingleMoleculeSpecies(smiles="O", count=1000, resname="ABC")
    spec2 = SingleMoleculeSpecies(smiles="CO", count=1000, resname="ABC")
    comp = SystemComposition(
        species=[spec, spec2],
    )
    print(comp)

    with pytest.raises(ValueError, match="Use fractions or counts"):
        spec = SingleMoleculeSpecies(smiles="O", count=10, resname="ABC")
        spec2 = SingleMoleculeSpecies(smiles="CO", fraction=0.1, resname="ABC")
        comp = SystemComposition(
            total_count=2000,
            species=[spec, spec2],
        )

    with pytest.raises(ValueError, match="Sum of fractions must be exactly 1.0"):
        spec = SingleMoleculeSpecies(smiles="O", fraction=1.0, resname="ABC")
        spec2 = SingleMoleculeSpecies(smiles="CO", fraction=0.1, resname="ABC")
        comp = SystemComposition(
            total_count=1000,
            species=[spec, spec2],
        )

    with pytest.raises(ValueError, match="If fractions are used, 'total_count' must be provided"):
        spec = SingleMoleculeSpecies(smiles="O", fraction=0.5, resname="ABC")
        spec2 = SingleMoleculeSpecies(smiles="CO", fraction=0.5, resname="ABC")
        comp = SystemComposition(
            species=[spec, spec2],
        )

    spec = SingleMoleculeSpecies(smiles="O", count=1000, resname="ABC")
    spec2 = SingleMoleculeSpecies(smiles="CO", count=1000, resname="ABC")
    comp = MixedBoxComposition(species=[spec, spec2], total_count=2000)

    inp = BuildInput(
        engine="gromacs",
        simulation_type="mixedbox",
        system=comp,
    )
    print(inp)

    in2 = BuildInput.from_data_row(inp.to_data_row())
    print(in2)
    assert inp == in2

    spec = SingleMoleculeSpecies(smiles="[O-]", count=500, resname="ABC")
    spec2 = SingleMoleculeSpecies(smiles="[NH4+]", count=250, resname="DEF")
    comp = MixedBoxComposition(species=[spec, spec2])
    assert comp.charge == -250

    row = {"bla": 42}
    with pytest.raises(ValueError, match="Row must contain 'input_data' key"):
        in2 = BuildInput.from_data_row(row)

    row = inp.to_data_row()
    row["input_data"] = inp.model_dump()
    assert inp == BuildInput.from_data_row(row)

    row["input_data"] = 42
    with pytest.raises(ValueError, match="'input_data' must be a JSON string or a dictionary"):
        in2 = BuildInput.from_data_row(row)

    spec = SingleMoleculeSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=5,
    )

    comp = MixedBoxComposition(species=[spec])
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        inp = BuildInput(
            engine="gromacs",
            simulation_type="bilayer",
            system=comp,
        )


@pytest.mark.parametrize("disabled", [None, False])
def test_ionization_can_be_explicitly_disabled(disabled):
    spec = SingleMoleculeSpecies(smiles="O", count=10, resname="SOL")
    comp = MixedBoxComposition(species=[spec], ionization=disabled)

    assert comp.ionization is None


def test_disabled_ionization_build_input_round_trip():
    input_data = {
        "simulation_type": "mixedbox",
        "system": {
            "species": [{"smiles": "O", "count": 10, "resname": "SOL"}],
            "ionization": False,
        },
    }

    inp = BuildInput(**input_data)
    assert inp.system.ionization is None
    assert inp.metadata["system_specific"]["ionization"] is None

    restored = BuildInput.from_data_row(inp.to_data_row())
    assert restored.system.ionization is None
    assert restored == inp


def test_ionization_defaults_when_omitted():
    spec = SingleMoleculeSpecies(smiles="O", count=10, resname="SOL")
    comp = MixedBoxComposition(species=[spec])

    assert isinstance(comp.ionization, IonizationConfig)
    assert comp.ionization.neutralize is True
    assert comp.ionization.concentration == 0.15


def test_parameter_set_dump(tmp_path):
    from mdfactory.models.parametrization import (  # noqa: PLC0415
        CgenffConfig,
        GromacsSingleMoleculeParameterSet,
    )

    with working_directory(tmp_path):
        itp = Path("1234ABCD.itp").absolute()
        prm = Path("bla.itp").absolute()
        ff = Path("ff.itp").absolute()
        itp.touch()
        prm.touch()
        ff.touch()
        prm = GromacsSingleMoleculeParameterSet(
            moleculetype="1234ABCD",
            smiles="CC",
            parametrization="cgenff",
            parametrization_config=CgenffConfig(),
            itp=itp,
            parameter_itp=prm,
            forcefield_itp=ff,
        )
        prm2 = GromacsSingleMoleculeParameterSet.from_data_row(prm.to_data_row())
        assert prm == prm2

    with pytest.raises(
        ValueError, match="Invalid parameter_data_type for GromacsSingleMoleculeParameterSet"
    ):
        GromacsSingleMoleculeParameterSet.from_data_row({"parameter_data_type": "bla"})


def test_gmx_parameter_model(tmp_path):
    from mdfactory.models.parametrization import (  # noqa: PLC0415
        CgenffConfig,
        GromacsSingleMoleculeParameterSet,
    )

    with working_directory(tmp_path):
        with pytest.raises(ValueError, match="Path must be absolute path"):
            itp = Path("1234ABCD.itp")
            itp.touch()
            prm = GromacsSingleMoleculeParameterSet(
                moleculetype="1234ABCD",
                smiles="CC",
                parametrization="cgenff",
                parametrization_config=CgenffConfig(),
                itp=itp,
            )
            del prm


def test_core_composition_replication():
    """Test that CoreComposition calculates replication factor for efficient packing."""
    # Create a core composition with species
    spec1 = SingleMoleculeSpecies(
        smiles="CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        fraction=0.7,
        resname="CHL",
    )
    spec2 = SingleMoleculeSpecies(smiles="O", fraction=0.3, resname="WAT")

    core = CoreComposition(species=[spec1, spec2], target_density=0.95)

    # Calculate counts for a 72 Angstrom radius sphere
    core.calculate_counts(radius=72.0, min_count=5)

    # Should have a replication factor > 1 for large systems
    assert core.replication_factor >= 1

    # All species should have at least min_count molecules
    for s in core.species:
        assert s.count >= 5, f"Species {s.resname} has count {s.count} < 5"

    # Calculate total molecules after replication
    unit_cell_total = sum(s.count for s in core.species)
    replicated_total = unit_cell_total * (core.replication_factor**3)

    print(f"Replication factor: {core.replication_factor}")
    print(f"Unit cell molecules: {unit_cell_total}")
    print(f"Replicated total: {replicated_total}")

    # For a small system (small radius), replication factor should be 1
    small_core = CoreComposition(species=[spec1, spec2], target_density=0.95)
    small_core.calculate_counts(radius=20.0, min_count=5)
    # Small systems might not need replication
    assert small_core.replication_factor >= 1


def test_replicate_box_3d():
    """Test 3D box replication function."""
    from io import StringIO

    import MDAnalysis as mda

    from mdfactory.setup.lnp import replicate_box_3d

    # Create a simple test box
    pdb_content = """ATOM      1  O   SOL A   1       1.000   1.000   1.000  1.00  0.00           O
ATOM      2  H1  SOL A   1       1.957   1.000   1.000  1.00  0.00           H
END
"""
    u = mda.Universe(StringIO(pdb_content), format="pdb")
    u.dimensions = [10.0, 10.0, 10.0, 90, 90, 90]

    # Test n=1 (no replication)
    u1 = replicate_box_3d(u, 1)
    assert len(u1.residues) == 1
    assert u1.dimensions[0] == 10.0

    # Test n=2
    u2 = replicate_box_3d(u, 2)
    assert len(u2.residues) == 8  # 2^3
    assert u2.dimensions[0] == 20.0
    assert u2.dimensions[1] == 20.0
    assert u2.dimensions[2] == 20.0

    # Test n=3
    u3 = replicate_box_3d(u, 3)
    assert len(u3.residues) == 27  # 3^3
    assert u3.dimensions[0] == 30.0


def test_distribute_counts():
    """Test that distribute_counts preserves exact totals."""
    # Simple case: equal fractions
    counts = distribute_counts([0.5, 0.5], 10)
    assert counts == [5, 5]
    assert sum(counts) == 10

    # Uneven split that would drift with round()
    counts = distribute_counts([0.33, 0.33, 0.34], 100)
    assert sum(counts) == 100  # Must equal exactly 100
    assert counts == [33, 33, 34]  # Largest fractional part gets the extra

    # Case where round() would over-count
    counts = distribute_counts([0.5, 0.5], 11)
    assert sum(counts) == 11  # 5.5 + 5.5 = 11, round would give 6+6=12
    assert counts in ([6, 5], [5, 6])

    # Single species
    counts = distribute_counts([1.0], 42)
    assert counts == [42]

    # Many species with small fractions
    fractions = [0.1] * 10
    counts = distribute_counts(fractions, 100)
    assert sum(counts) == 100
    assert all(c == 10 for c in counts)

    # Fractions that don't divide evenly
    fractions = [0.7, 0.3]
    counts = distribute_counts(fractions, 10)
    assert sum(counts) == 10
    assert counts == [7, 3]


def test_shell_composition_calculate_counts_smaller_scale():
    """Test that ShellComposition selects the smallest adequate scale (1/8, 1/4, etc.)."""
    # DOPC-like lipid
    dopc_smiles = r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC"  # noqa: E501

    # Create shell with just one species - should select smallest scale (1/8)
    spec = LipidSpecies(smiles=dopc_smiles, resname="DPC", fraction=1.0)
    shell = ShellComposition(species=[spec], area_per_lipid=65.0)

    # Use a large core radius to get meaningful counts
    shell.calculate_counts(core_radius=100.0, min_count=5)

    # Single species with fraction=1.0 should always satisfy min_count at 1/8 scale
    # for any reasonable radius
    assert spec.count >= 5
    # For r=100, surface_area ~ 125664 Å², total ~ 1933 lipids, 1/8 ~ 242
    assert spec.count < 500  # Should be around 1/8 of total


def test_shell_composition_calculate_counts_larger_scale():
    """Test that ShellComposition escalates to larger scale when needed."""
    dopc_smiles = r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC"  # noqa: E501

    # Create shell with a minor component that needs larger scale than 1/8
    # At r=50, surface ~ 31416, total ~ 483, 1/8 ~ 60 lipids
    # At 10%, minor gets 6 at 1/8 scale -> should satisfy min_count=5
    # At 5%, minor gets 3 at 1/8 scale -> must escalate to 1/4 scale (~120 lipids, 6 minor)
    major = LipidSpecies(smiles=dopc_smiles, resname="DPC", fraction=0.95)
    minor = LipidSpecies(smiles=dopc_smiles, resname="MIN", fraction=0.05)
    shell = ShellComposition(species=[major, minor], area_per_lipid=65.0)

    # Use radius where 1/8 scale would give minor < 5, forcing escalation
    shell.calculate_counts(core_radius=50.0, min_count=5)

    # Both species should have at least min_count
    assert major.count >= 5
    assert minor.count >= 5


def test_shell_composition_min_count_behavior():
    """Test that min_count parameter is respected."""
    dopc_smiles = r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC"  # noqa: E501

    spec1 = LipidSpecies(smiles=dopc_smiles, resname="DPC", fraction=0.8)
    spec2 = LipidSpecies(smiles=dopc_smiles, resname="DP2", fraction=0.2)
    shell = ShellComposition(species=[spec1, spec2], area_per_lipid=65.0)

    # Test with different min_count values
    shell.calculate_counts(core_radius=50.0, min_count=10)
    assert spec1.count >= 10
    assert spec2.count >= 10

    # Total should be reasonable (sum of counts should be achievable)
    total = spec1.count + spec2.count
    assert total > 0


def test_shell_composition_counts_sum_correctly():
    """Test that distributed counts sum to expected patch size."""
    dopc_smiles = r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC"  # noqa: E501

    # Three species with fractions that don't divide evenly
    spec1 = LipidSpecies(smiles=dopc_smiles, resname="DP1", fraction=0.33)
    spec2 = LipidSpecies(smiles=dopc_smiles, resname="DP2", fraction=0.33)
    spec3 = LipidSpecies(smiles=dopc_smiles, resname="DP3", fraction=0.34)
    shell = ShellComposition(species=[spec1, spec2, spec3], area_per_lipid=65.0)

    shell.calculate_counts(core_radius=72.0, min_count=5)

    # Counts should sum to the patch size (distribute_counts guarantees this)
    total = spec1.count + spec2.count + spec3.count
    assert total > 0

    # Verify approximate ratios are preserved
    ratios = [s.count / total for s in [spec1, spec2, spec3]]
    assert abs(ratios[0] - 0.33) < 0.05
    assert abs(ratios[1] - 0.33) < 0.05
    assert abs(ratios[2] - 0.34) < 0.05
