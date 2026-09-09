# ABOUTME: Tests for protein_mixedbox models, helpers, topology, and YAML/CSV validation
# ABOUTME: Covers SolutionSpecies, CharmmConfig, sizing, volume/count helpers, and re-imaging
"""Tests for protein_mixedbox models, helpers, topology, and YAML/CSV validation."""

import textwrap
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest
import yaml

from mdfactory import workflows
from mdfactory.build import (
    PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A,
    _check_no_protein_clashes,
    _reimage_protein_centered,
    ionize_solvated_system,
)
from mdfactory.models.composition import (
    CountDensitySizing,
    FixedBoxSizing,
    IonizationConfig,
    ProteinMixedBoxComposition,
)
from mdfactory.models.input import BuildInput
from mdfactory.models.parametrization import CharmmConfig
from mdfactory.models.species import ProteinSpecies, SolutionSpecies
from mdfactory.parametrize import generate_gromacs_topology_with_protein
from mdfactory.prepare import df_to_build_input_models
from mdfactory.run_schedules import RunScheduleManager
from mdfactory.utils.setup_utilities import (
    N_AVOGADRO,
    cubic_box_edge_for_density,
    protein_displaced_volume_a3,
    resolve_solution_counts,
)
from mdfactory.workflows import _resolve_proteinbox_pdb_path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "protein_mixedbox"


def _write_protein_pdb(path):
    """Write a one-atom PDB so ProteinSpecies has an existing (unvalidated) path."""
    path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")


def _write_pdb2gmx_topology(path):
    """Write a minimal pdb2gmx-style CHARMM topology for composite-topology tests.

    Mirrors the include layout generate_gromacs_topology_with_protein anchors to:
    a bundled force-field include first, the protein moleculetype, the water/ion
    force-field includes, a [ system ] title, then [ molecules ] with chain lines.
    """
    path.write_text(
        textwrap.dedent("""\
        [ defaults ]
        1 2 yes 0.5 0.83333

        #include "charmm36m.ff/forcefield.itp"

        [ moleculetype ]
        Protein_chain_A     3

        #include "charmm36m.ff/tip3p.itp"
        #include "charmm36m.ff/ions.itp"

        [ system ]
        Protein in water

        [ molecules ]
        ; Compound        #mols
        Protein_chain_A     1
        """)
    )


class TestSolutionSpecies:
    def test_count_only(self):
        spec = SolutionSpecies(smiles="O", resname="SOL", count=100)
        assert spec.count == 100
        assert spec.concentration is None

    def test_concentration_only(self):
        spec = SolutionSpecies(smiles="CCO", resname="ETH", concentration=1.0)
        assert spec.concentration == 1.0
        assert spec.count is None

    def test_rejects_both_count_and_concentration(self):
        with pytest.raises(ValueError, match="exactly one of 'count' or 'concentration'"):
            SolutionSpecies(smiles="O", resname="SOL", count=10, concentration=1.0)

    def test_rejects_neither_count_nor_concentration(self):
        with pytest.raises(ValueError, match="exactly one of 'count' or 'concentration'"):
            SolutionSpecies(smiles="O", resname="SOL")

    def test_rejects_fraction(self):
        with pytest.raises(ValueError, match="does not support 'fraction'"):
            SolutionSpecies(smiles="O", resname="SOL", fraction=0.5)


class TestCharmmConfig:
    def test_defaults(self):
        config = CharmmConfig()
        assert config.type == "charmm"
        assert config.forcefield == "charmm36m"
        assert config.water_model == "tip3p"
        assert config.ignore_hydrogens is True
        assert config.merge_all is False

    def test_frozen(self):
        config = CharmmConfig()
        with pytest.raises(Exception):
            config.forcefield = "other"

    def test_rejects_non_charmm_forcefield(self):
        with pytest.raises(ValueError, match="CHARMM"):
            CharmmConfig(forcefield="amber99sb-ildn")

    def test_rejects_non_three_site_water(self):
        with pytest.raises(ValueError, match="3-site"):
            CharmmConfig(water_model="tip4p")

    def test_rejects_ljpme_forcefield(self):
        with pytest.raises(ValueError, match="LJ-PME"):
            CharmmConfig(forcefield="charmm36m-ljpme")


class TestSizingConfigs:
    def test_fixed_box(self):
        sizing = FixedBoxSizing(box_size=72.0)
        assert sizing.type == "fixed_box"
        assert sizing.box_size == 72.0

    def test_fixed_box_rejects_non_positive(self):
        with pytest.raises(ValueError):
            FixedBoxSizing(box_size=0.0)

    def test_count_density_defaults(self):
        sizing = CountDensitySizing()
        assert sizing.type == "count_density"
        assert sizing.target_density == 1.0

    def test_count_density_rejects_non_positive(self):
        with pytest.raises(ValueError):
            CountDensitySizing(target_density=0.0)


class TestProteinMixedBoxComposition:
    def _protein(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        _write_protein_pdb(pdb)
        return ProteinSpecies(resname="LYZ", pdb_path=pdb)

    def test_fixed_box_with_concentration(self, tmp_path):
        comp = ProteinMixedBoxComposition(
            protein=self._protein(tmp_path),
            species=[
                SolutionSpecies(smiles="O", resname="SOL", concentration=55.0),
                SolutionSpecies(smiles="CCO", resname="ETH", concentration=1.0),
            ],
            sizing={"type": "fixed_box", "box_size": 72.0},
            padding=12.0,
        )
        assert isinstance(comp.sizing, FixedBoxSizing)
        assert comp.concentration_volume_basis == "protein_excluded"
        assert comp.partial_specific_volume == 0.73
        assert comp.relax_steps == 10000

    def test_count_density_with_counts(self, tmp_path):
        comp = ProteinMixedBoxComposition(
            protein=self._protein(tmp_path),
            species=[
                SolutionSpecies(smiles="O", resname="SOL", count=10000),
                SolutionSpecies(smiles="CCO", resname="ETH", count=200),
            ],
            sizing={"type": "count_density", "target_density": 1.0},
            padding=12.0,
        )
        assert isinstance(comp.sizing, CountDensitySizing)
        assert comp.total_count == 1 + 10000 + 200

    def test_fixed_box_rejects_padding_too_large(self, tmp_path):
        with pytest.raises(ValueError, match="must exceed 2 × padding"):
            ProteinMixedBoxComposition(
                protein=self._protein(tmp_path),
                species=[SolutionSpecies(smiles="O", resname="SOL", count=10)],
                sizing={"type": "fixed_box", "box_size": 20.0},
                padding=12.0,
            )

    def test_count_density_rejects_concentration_species(self, tmp_path):
        with pytest.raises(ValueError, match="count_density sizing derives the box from total mass"):
            ProteinMixedBoxComposition(
                protein=self._protein(tmp_path),
                species=[SolutionSpecies(smiles="O", resname="SOL", concentration=55.0)],
                sizing={"type": "count_density", "target_density": 1.0},
                padding=12.0,
            )

    def test_charge_counts_only_ions_with_counts(self, tmp_path):
        # A concentration-only species contributes no charge (count unknown until
        # build); a counted charged species does.
        comp = ProteinMixedBoxComposition(
            protein=self._protein(tmp_path),
            species=[
                SolutionSpecies(smiles="O", resname="SOL", concentration=55.0),
                SolutionSpecies(smiles="[Na+]", resname="NA", count=5),
            ],
            sizing={"type": "fixed_box", "box_size": 72.0},
            padding=12.0,
        )
        assert comp.charge == 5

    def test_total_count_none_with_concentration(self, tmp_path):
        comp = ProteinMixedBoxComposition(
            protein=self._protein(tmp_path),
            species=[SolutionSpecies(smiles="O", resname="SOL", concentration=55.0)],
            sizing={"type": "fixed_box", "box_size": 72.0},
            padding=12.0,
        )
        assert comp.total_count is None


class TestBuildInputProteinMixedBox:
    def test_yaml_roundtrip_fixed_box(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        _write_protein_pdb(pdb)
        yaml_str = f"""
simulation_type: protein_mixedbox
engine: gromacs
parametrization: charmm
parametrization_config:
  type: charmm
  forcefield: charmm36m
  water_model: tip3p
system:
  protein:
    resname: LYZ
    count: 1
    pdb_path: {pdb}
    chains: [A]
  species:
    - smiles: O
      resname: SOL
      concentration: 55.0
    - smiles: CCO
      resname: ETH
      concentration: 1.0
  sizing:
    type: fixed_box
    box_size: 72.0
  padding: 12.0
"""
        inp = BuildInput(**yaml.safe_load(yaml_str))
        assert inp.simulation_type == "protein_mixedbox"
        assert isinstance(inp.system, ProteinMixedBoxComposition)
        assert isinstance(inp.parametrization_config, CharmmConfig)
        assert isinstance(inp.system.sizing, FixedBoxSizing)

    def test_default_parametrization_config(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        _write_protein_pdb(pdb)
        data = {
            "simulation_type": "protein_mixedbox",
            "parametrization": "charmm",
            "system": {
                "protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)},
                "species": [{"smiles": "O", "resname": "SOL", "count": 10}],
                "sizing": {"type": "fixed_box", "box_size": 72.0},
                "padding": 12.0,
            },
        }
        inp = BuildInput(**data)
        assert isinstance(inp.parametrization_config, CharmmConfig)
        assert inp.parametrization_config.forcefield == "charmm36m"

    def test_rejects_cgenff_parametrization(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        _write_protein_pdb(pdb)
        data = {
            "simulation_type": "protein_mixedbox",
            "parametrization": "cgenff",
            "system": {
                "protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)},
                "species": [{"smiles": "O", "resname": "SOL", "count": 10}],
                "sizing": {"type": "fixed_box", "box_size": 72.0},
                "padding": 12.0,
            },
        }
        with pytest.raises(ValueError, match="not valid for simulation type"):
            BuildInput(**data)

    def test_rejects_mismatched_config_type(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        _write_protein_pdb(pdb)
        data = {
            "simulation_type": "protein_mixedbox",
            "parametrization": "charmm",
            "parametrization_config": {"type": "cgenff"},
            "system": {
                "protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)},
                "species": [{"smiles": "O", "resname": "SOL", "count": 10}],
                "sizing": {"type": "fixed_box", "box_size": 72.0},
                "padding": 12.0,
            },
        }
        with pytest.raises(ValueError, match="does not match"):
            BuildInput(**data)

    def test_rejects_merge_all_with_declared_chains(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        _write_protein_pdb(pdb)
        data = {
            "simulation_type": "protein_mixedbox",
            "parametrization": "charmm",
            "parametrization_config": {"type": "charmm", "merge_all": True},
            "system": {
                "protein": {"resname": "INS", "pdb_path": str(pdb), "chains": ["A", "B"]},
                "species": [{"smiles": "O", "resname": "SOL", "count": 10}],
                "sizing": {"type": "fixed_box", "box_size": 72.0},
                "padding": 12.0,
            },
        }
        with pytest.raises(ValueError, match="merge_all"):
            BuildInput(**data)


class TestExampleFiles:
    def test_fixed_box_yaml_loads(self):
        data = yaml.safe_load((EXAMPLES_DIR / "lysozyme_fixed_box.yaml").read_text())
        inp = BuildInput(**data)
        assert isinstance(inp.system, ProteinMixedBoxComposition)
        assert isinstance(inp.system.sizing, FixedBoxSizing)
        assert isinstance(inp.parametrization_config, CharmmConfig)

    def test_count_density_yaml_loads(self):
        data = yaml.safe_load((EXAMPLES_DIR / "lysozyme_count_density.yaml").read_text())
        inp = BuildInput(**data)
        assert isinstance(inp.system.sizing, CountDensitySizing)
        assert inp.system.total_count == 1 + 10000 + 200

    def test_csv_roundtrip(self):
        df = pd.read_csv(EXAMPLES_DIR / "protein_mixedbox.csv")
        models, errors = df_to_build_input_models(df)
        assert errors == {}
        assert len(models) == 1
        inp = models[0]
        assert inp.simulation_type == "protein_mixedbox"
        assert isinstance(inp.system, ProteinMixedBoxComposition)
        assert {s.resname for s in inp.system.species} == {"SOL", "ETH"}
        assert inp.system.protein.disulfide_bonds[0] == ("CYS6", "CYS127")
        assert inp.system.protein.protonation_states["HIS15"] == "HIE"


class TestVolumeAndCountHelpers:
    def test_protein_displaced_volume(self):
        # 1 g/mol at psv 1.0 mL/g displaces N_A^-1 cm^3 = 1e24/N_A Å^3.
        vol = protein_displaced_volume_a3(1.0, 1.0)
        assert vol == pytest.approx(1e24 / N_AVOGADRO)

    def test_cubic_box_edge_for_density(self):
        # A cube of edge L (Å) at density d holds mass = d * (L Å)^3.
        edge = 60.0
        volume_cm3 = (edge * 1e-8) ** 3
        mass_dalton = 1.0 * volume_cm3 * N_AVOGADRO  # density 1 g/cm^3
        assert cubic_box_edge_for_density(mass_dalton, 1.0) == pytest.approx(edge, rel=1e-6)

    def test_explicit_count_kept_regardless_of_basis(self):
        specs = [SolutionSpecies(smiles="CCO", resname="ETH", count=42)]
        counts = resolve_solution_counts(
            specs, box_volume_a3=1e5, protein_mass_dalton=1e4,
            partial_specific_volume=0.73, basis="protein_excluded"
        )
        assert counts == [42]

    def test_concentration_resolves_on_box_basis(self):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=1.0)]
        box_volume_a3 = 1e6
        counts = resolve_solution_counts(
            specs, box_volume_a3=box_volume_a3, protein_mass_dalton=0.0,
            partial_specific_volume=0.73, basis="box"
        )
        expected = int(round(1.0 * box_volume_a3 * 1e-27 * N_AVOGADRO))
        assert counts == [expected]

    def test_protein_excluded_basis_subtracts_displaced_volume(self):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=1.0)]
        box_volume_a3 = 1e6
        protein_mass = 1e5
        psv = 0.73
        counts = resolve_solution_counts(
            specs, box_volume_a3=box_volume_a3, protein_mass_dalton=protein_mass,
            partial_specific_volume=psv, basis="protein_excluded"
        )
        v_basis = box_volume_a3 - protein_displaced_volume_a3(protein_mass, psv)
        expected = int(round(1.0 * v_basis * 1e-27 * N_AVOGADRO))
        assert counts == [expected]
        # The excluded-volume count is strictly smaller than the whole-box count.
        box_counts = resolve_solution_counts(
            specs, box_volume_a3=box_volume_a3, protein_mass_dalton=protein_mass,
            partial_specific_volume=psv, basis="box"
        )
        assert counts[0] < box_counts[0]

    def test_positive_concentration_rounding_to_zero_raises(self):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=1e-9)]
        with pytest.raises(ValueError, match="resolves to zero"):
            resolve_solution_counts(
                specs, box_volume_a3=1e3, protein_mass_dalton=0.0,
                partial_specific_volume=0.73, basis="box"
            )

    def test_protein_larger_than_box_raises(self):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=1.0)]
        with pytest.raises(ValueError, match="non-positive"):
            resolve_solution_counts(
                specs, box_volume_a3=1e3, protein_mass_dalton=1e9,
                partial_specific_volume=0.73, basis="protein_excluded"
            )


class TestIonizeSolvatedSystem:
    def _water_universe(self, n_waters=10):
        n_atoms = n_waters * 3
        atom_resindex = np.repeat(np.arange(n_waters), 3)
        u = mda.Universe.empty(
            n_atoms, n_residues=n_waters, atom_resindex=atom_resindex, trajectory=True
        )
        u.add_TopologyAttr("resnames", ["SOL"] * n_waters)
        u.add_TopologyAttr("masses", [16.0, 1.0, 1.0] * n_waters)
        u.dimensions = [100.0, 100.0, 100.0, 90, 90, 90]
        u.atoms.positions = np.random.RandomState(0).uniform(0, 100, size=(n_atoms, 3))
        return u

    def test_salt_count_uses_solvent_volume_basis(self, monkeypatch):
        captured = {}

        def fake_ionize(u, num_na, num_cl, min_distance, seed):
            captured["num_na"] = num_na
            captured["num_cl"] = num_cl
            return u

        monkeypatch.setattr("mdfactory.build.ionize", fake_ionize)

        u = self._water_universe(n_waters=10)
        solvent_volume_a3 = 110_704.0
        config = IonizationConfig(neutralize=True, concentration=0.15)
        _, ion_species = ionize_solvated_system(
            config, u, total_charge=0, solvent_volume_a3=solvent_volume_a3
        )

        n_ions = int(round(0.15 * solvent_volume_a3 * 1e-27 * N_AVOGADRO))
        assert captured["num_na"] == n_ions
        assert captured["num_cl"] == n_ions
        # The water-count proxy would give a wildly different (tiny) count.
        proxy = int(np.ceil(0.15 * 10 / 55.55))
        assert n_ions != proxy
        assert [s.count for s in ion_species] == [n_ions, n_ions]
        assert [s.resname for s in ion_species] == ["NA", "CL"]

    def test_neutralization_adds_counterions(self, monkeypatch):
        captured = {}

        def fake_ionize(u, num_na, num_cl, min_distance, seed):
            captured["num_na"] = num_na
            captured["num_cl"] = num_cl
            return u

        monkeypatch.setattr("mdfactory.build.ionize", fake_ionize)

        u = self._water_universe(n_waters=10)
        solvent_volume_a3 = 110_704.0
        config = IonizationConfig(neutralize=True, concentration=0.15)
        ionize_solvated_system(config, u, total_charge=3, solvent_volume_a3=solvent_volume_a3)

        n_ions = int(round(0.15 * solvent_volume_a3 * 1e-27 * N_AVOGADRO))
        # A +3 system needs 3 extra Cl- to neutralize.
        assert captured["num_cl"] == n_ions + 3
        assert captured["num_na"] == n_ions

    def test_none_config_adds_no_ions(self):
        u = self._water_universe(n_waters=4)
        u_out, ion_species = ionize_solvated_system(None, u, total_charge=0)
        assert ion_species == []
        assert u_out is u


class TestReimageAndClash:
    def _system(self):
        # 5 protein atoms (residue 0), then 2 water residues of 3 atoms each.
        n_protein = 5
        n_atoms = n_protein + 6
        atom_resindex = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        u = mda.Universe.empty(
            n_atoms, n_residues=3, atom_resindex=atom_resindex, trajectory=True
        )
        u.add_TopologyAttr("resnames", ["LYZ", "SOL", "SOL"])
        u.add_TopologyAttr("masses", [12.0] * n_protein + [16.0, 1.0, 1.0, 16.0, 1.0, 1.0])
        u.dimensions = [30.0, 30.0, 30.0, 90, 90, 90]
        return u, n_protein

    def test_reimage_makes_protein_whole_and_centered(self):
        u, n_protein = self._system()
        L = 30.0
        reference = np.array(
            [[14.0, 15.0, 15.0], [16.0, 15.0, 15.0], [15.0, 16.0, 15.0],
             [15.0, 15.0, 16.0], [15.0, 14.0, 15.0]]
        )
        # A rigid translation pushes the protein across the +x boundary; wrapping
        # by box vector tears it apart, exactly as a barostat + restraint produces.
        placed = reference + np.array([13.0, 0.0, 0.0])
        torn = np.mod(placed, L)
        solution = np.array(
            [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [1.0, 1.5, 1.0],
             [29.0, 29.0, 29.0], [28.5, 29.0, 29.0], [29.0, 28.5, 29.0]]
        )
        u.atoms.positions = np.vstack([torn, solution])

        _reimage_protein_centered(u, n_protein, reference)

        protein_now = u.atoms.positions[:n_protein]
        shape_now = protein_now - protein_now.mean(axis=0)
        shape_ref = reference - reference.mean(axis=0)
        rmsd = np.sqrt(((shape_now - shape_ref) ** 2).sum(axis=1).mean())
        assert rmsd < 1e-4
        assert rmsd < PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A

        bbox_center = (protein_now.min(axis=0) + protein_now.max(axis=0)) / 2.0
        assert np.allclose(bbox_center, [L / 2, L / 2, L / 2], atol=1e-4)

    def test_clash_check_raises_on_overlap(self):
        u, n_protein = self._system()
        pos = np.zeros((u.atoms.n_atoms, 3))
        pos[:n_protein] = np.array(
            [[15.0, 15.0, 15.0], [16.0, 15.0, 15.0], [15.0, 16.0, 15.0],
             [15.0, 15.0, 16.0], [15.0, 14.0, 15.0]]
        )
        # First solution atom sits on top of a protein atom.
        pos[n_protein] = [15.0, 15.0, 15.1]
        pos[n_protein + 1 :] = np.array(
            [[2.0, 2.0, 2.0], [2.5, 2.0, 2.0], [3.0, 3.0, 3.0], [3.5, 3.0, 3.0], [4.0, 4.0, 4.0]]
        )
        u.atoms.positions = pos
        with pytest.raises(ValueError, match="closer than"):
            _check_no_protein_clashes(u, n_protein)

    def test_clash_check_passes_when_clear(self):
        u, n_protein = self._system()
        pos = np.zeros((u.atoms.n_atoms, 3))
        pos[:n_protein] = np.array(
            [[15.0, 15.0, 15.0], [16.0, 15.0, 15.0], [15.0, 16.0, 15.0],
             [15.0, 15.0, 16.0], [15.0, 14.0, 15.0]]
        )
        pos[n_protein:] = np.array(
            [[2.0, 2.0, 2.0], [2.5, 2.0, 2.0], [2.0, 2.5, 2.0],
             [5.0, 5.0, 5.0], [5.5, 5.0, 5.0], [5.0, 5.5, 5.0]]
        )
        u.atoms.positions = pos
        _check_no_protein_clashes(u, n_protein)  # must not raise


class TestCompositeTopology:
    def test_native_only_ordering_and_counts(self, tmp_path):
        top = tmp_path / "topol_protein.top"
        _write_pdb2gmx_topology(top)
        out = tmp_path / "topology.top"

        species = [
            SolutionSpecies(smiles="O", resname="SOL", count=100),
            SolutionSpecies(smiles="CCO", resname="ETH", count=5),
            SolutionSpecies(smiles="[Na+]", resname="NA", count=3),
            SolutionSpecies(smiles="[Cl-]", resname="CL", count=6),
        ]
        parameters = [
            SimpleNamespace(moleculetype="SOL", parameter_itp=None),
            SimpleNamespace(moleculetype="ETH0", parameter_itp=None),
            SimpleNamespace(moleculetype="SOD", parameter_itp=None),
            SimpleNamespace(moleculetype="CLA", parameter_itp=None),
        ]
        counts = [100, 5, 3, 6]

        generate_gromacs_topology_with_protein(
            top, tmp_path / "nonexistent.ff", species, parameters, counts,
            "protein_mixedbox", out_path=out
        )
        text = out.read_text()

        # Exactly one bundled force-field include.
        assert text.count('#include "charmm36m.ff/forcefield.itp"') == 1
        # System title replaced.
        assert "protein_mixedbox\n" in text
        assert "Protein in water" not in text

        molecules = text.split("[ molecules ]", 1)[1]
        mol_lines = [
            line.strip() for line in molecules.splitlines()
            if line.strip() and not line.strip().startswith(";")
        ]
        assert mol_lines[0].split() == ["Protein_chain_A", "1"]
        assert mol_lines[1].split()[:2] == ["SOL", "100"]
        assert mol_lines[2].split()[:2] == ["ETH0", "5"]
        assert mol_lines[3].split()[:2] == ["SOD", "3"]
        assert mol_lines[4].split()[:2] == ["CLA", "6"]

    def test_cgenff_small_molecule_merges_params_and_includes_itp(self, tmp_path):
        top = tmp_path / "topol_protein.top"
        _write_pdb2gmx_topology(top)
        out = tmp_path / "topology.top"

        # A force field that already defines CG331, so it is de-duplicated out of
        # the merged small-molecule parameters.
        ff_dir = tmp_path / "charmm36m.ff"
        ff_dir.mkdir()
        (ff_dir / "atomtypes.itp").write_text(
            "[ atomtypes ]\nCG331   6   12.011   0.000   A   0.36  0.28\n"
        )

        # The parameter files live in a separate source directory (as in the
        # parameter DB), so the topology function copies them next to topology.top.
        src_dir = tmp_path / "params"
        src_dir.mkdir()
        param_itp = src_dir / "MOL0_params.itp"
        param_itp.write_text(
            textwrap.dedent("""\
            [ defaults ]
            1 2 yes 0.5 0.83333

            [ atomtypes ]
            CG331   6   12.011   0.000   A   0.36  0.28
            NEWTYPE 7   14.007   0.000   A   0.32  0.20
            """)
        )
        mol_itp = src_dir / "MOL0.itp"
        mol_itp.write_text("[ moleculetype ]\nMOL0   3\n")

        species = [SolutionSpecies(smiles="CCO", resname="ETH", count=5)]
        parameters = [SimpleNamespace(moleculetype="MOL0", parameter_itp=param_itp, itp=mol_itp)]

        generate_gromacs_topology_with_protein(
            top, ff_dir, species, parameters, [5], "protein_mixedbox", out_path=out
        )
        text = out.read_text()

        assert text.count('#include "charmm36m.ff/forcefield.itp"') == 1
        assert '#include "extra_params.itp"' in text
        assert '#include "MOL0.itp"' in text
        assert (tmp_path / "MOL0.itp").is_file()

        merged = (tmp_path / "extra_params.itp").read_text()
        assert "NEWTYPE" in merged
        # CG331 is already in the force field, so it is excluded from the merge.
        assert "CG331" not in merged

        # The small-molecule include precedes the water include.
        assert text.index('#include "MOL0.itp"') < text.index('charmm36m.ff/tip3p.itp')

    def test_zero_count_species_skipped(self, tmp_path):
        top = tmp_path / "topol_protein.top"
        _write_pdb2gmx_topology(top)
        out = tmp_path / "topology.top"

        species = [
            SolutionSpecies(smiles="O", resname="SOL", count=100),
            SolutionSpecies(smiles="CCO", resname="ETH", count=0),
        ]
        parameters = [
            SimpleNamespace(moleculetype="SOL", parameter_itp=None),
            SimpleNamespace(moleculetype="ETH0", parameter_itp=None),
        ]
        generate_gromacs_topology_with_protein(
            top, tmp_path / "nonexistent.ff", species, parameters, [100, 0],
            "protein_mixedbox", out_path=out
        )
        molecules = out.read_text().split("[ molecules ]", 1)[1]
        assert "ETH0" not in molecules
        assert "SOL" in molecules


class TestPdbPathResolution:
    def test_relative_pdb_path_resolved_for_protein_mixedbox(self, tmp_path):
        base = tmp_path / "specs"
        base.mkdir()
        dct = {
            "simulation_type": "protein_mixedbox",
            "system": {"protein": {"pdb_path": "1aki.pdb"}},
        }
        _resolve_proteinbox_pdb_path(dct, base)
        assert dct["system"]["protein"]["pdb_path"] == str((base / "1aki.pdb").resolve())

    def test_run_build_from_file_resolves_relative_path(self, tmp_path, monkeypatch):
        pdb = tmp_path / "1aki.pdb"
        _write_protein_pdb(pdb)
        yaml_file = tmp_path / "build.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            simulation_type: protein_mixedbox
            parametrization: charmm
            system:
              protein:
                resname: LYZ
                count: 1
                pdb_path: 1aki.pdb
              species:
                - smiles: O
                  resname: SOL
                  count: 10
              sizing:
                type: fixed_box
                box_size: 72.0
              padding: 12.0
            """)
        )
        captured = {}
        monkeypatch.setattr(workflows, "run_build_from_dict", captured.update)
        workflows.run_build_from_file(yaml_file)
        assert captured["system"]["protein"]["pdb_path"] == str(pdb.resolve())


class TestRunSchedule:
    def test_protein_mixedbox_run_files_on_disk(self):
        manager = RunScheduleManager()
        paths = manager.get_all_run_file_paths(engine="gromacs", system_type="protein_mixedbox")
        assert set(paths) == {"em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"}
        for path in paths.values():
            assert path.is_file()
