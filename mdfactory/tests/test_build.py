# ABOUTME: Tests for bilayer and mixed-box system building, including structure
# ABOUTME: assembly, composition validation, and buildability checks.
"""Tests for bilayer and mixed-box system building, including structure."""

from pathlib import Path
from unittest.mock import patch

import MDAnalysis as mda

from mdfactory.build import build_bilayer, build_mixedbox, ionize_solvated_system
from mdfactory.check import check_bilayer_buildable
from mdfactory.models.composition import BilayerComposition, IonizationConfig, MixedBoxComposition
from mdfactory.models.input import BuildInput
from mdfactory.models.species import LipidSpecies, SingleMoleculeSpecies
from mdfactory.setup import solvation
from mdfactory.utils.setup_utilities import create_bilayer_from_model, create_mixed_box_universe
from mdfactory.utils.utilities import working_directory


def test_create_bilayer_from_model():
    spec = LipidSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=50,
    )
    comp = BilayerComposition(species=[spec], monolayer=False)
    bilayer = create_bilayer_from_model(comp)

    residues = bilayer.residues
    assert len(residues) == 50
    assert all(res.resname == "POC" for res in residues)

    comp = BilayerComposition(species=[spec], monolayer=True)
    monolayer = create_bilayer_from_model(comp)

    residues = monolayer.residues
    assert len(residues) == 50
    assert all(res.resname == "POC" for res in residues)

    check_bilayer_buildable(comp)


def test_create_mixedbox_universe():
    spec = SingleMoleculeSpecies(smiles="O", fraction=0.9, resname="SOL")
    spec2 = SingleMoleculeSpecies(smiles="CO", fraction=0.1, resname="MET")
    comp = MixedBoxComposition(
        total_count=200,
        species=[spec, spec2],
    )
    box, _ = create_mixed_box_universe(comp)
    residues = box.residues
    assert len(residues) == 200
    n_sol = sum(1 for res in residues if res.resname == "SOL")
    n_met = sum(1 for res in residues if res.resname == "MET")
    assert n_sol == 180
    assert n_met == 20


def test_ionize_solvated_system_skips_zero_ions():
    u = mda.Universe(Path(solvation.__file__).parent / "data" / "spc216.gro")
    ion_config = IonizationConfig(concentration=0.0, neutralize=False)

    with patch("mdfactory.build.ionize") as ionize:
        ionized, additional_species = ionize_solvated_system(ion_config, u, total_charge=0)

    assert ionized is u
    assert additional_species == []
    ionize.assert_not_called()


def test_build_mixedbox(tmp_path):
    spec = SingleMoleculeSpecies(smiles="O", fraction=0.9, resname="SOL")
    spec2 = SingleMoleculeSpecies(smiles="CO", fraction=0.1, resname="MET")
    comp = MixedBoxComposition(
        total_count=400,
        species=[spec, spec2],
    )
    inp = BuildInput(
        engine="gromacs",
        simulation_type="mixedbox",
        parametrization="cgenff",
        system=comp,
    )
    with working_directory(tmp_path):
        build_mixedbox(inp)
        system_pdb = tmp_path / "system.pdb"
        assert system_pdb.exists()
        u = mda.Universe(system_pdb)
        residues = u.residues
        assert len(residues) == 400
        n_sol = sum(1 for res in residues if res.resname == "SOL")
        n_met = sum(1 for res in residues if res.resname == "MET")
        n_ions = sum(1 for res in residues if res.resname in ["NA", "CL"])
        assert n_ions == 2  # Neutralizing ions added
        assert n_sol == 358
        assert n_met == 40


def test_build_bilayer(tmp_path):
    spec = LipidSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=32,
    )
    comp = BilayerComposition(species=[spec], monolayer=False)
    inp = BuildInput(
        engine="gromacs",
        simulation_type="bilayer",
        parametrization="cgenff",
        system=comp,
    )
    with working_directory(tmp_path):
        build_bilayer(inp)
        system_pdb = tmp_path / "system.pdb"
        assert system_pdb.exists()
        u = mda.Universe(system_pdb)
        residues = u.residues
        n_poc = sum(1 for res in residues if res.resname == "POC")
        n_ions = sum(1 for res in residues if res.resname in ["NA", "CL"])
        assert n_ions == 6
        assert n_poc == 32
