# ABOUTME: Tests for protein_mixedbox models, helpers, topology, and YAML/CSV validation
# ABOUTME: Covers SolutionSpecies, CharmmConfig, sizing, volume/count helpers, and re-imaging
"""Tests for protein_mixedbox models, helpers, topology, and YAML/CSV validation."""

import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest
import yaml

from mdfactory.build import (
    PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A,
    _check_no_protein_clashes,
    _check_no_solution_clashes,
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
from mdfactory.parametrize import (
    generate_gromacs_topology_with_protein,
    parametrize_cgenff_gromacs,
)
from mdfactory.prepare import df_to_build_input_models
from mdfactory.run_schedules import RunScheduleManager
from mdfactory.settings import settings
from mdfactory.simulation.openmm_utils import (
    _add_protein_rmsd_restraint,
    simulate_compression_until_density_reached,
)
from mdfactory.utils.setup_utilities import (
    N_AVOGADRO,
    charmm_native_atom_names,
    create_protein_mixedbox_universe,
    cubic_box_edge_for_density,
    protein_displaced_volume_a3,
    resolve_solution_counts,
)

from .conftest import protein_input_data

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "protein_mixedbox"


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
    @pytest.mark.parametrize(
        ("kwargs", "count", "concentration"),
        [({"count": 100}, 100, None), ({"concentration": 1.0}, None, 1.0)],
    )
    def test_count_or_concentration(self, kwargs, count, concentration):
        spec = SolutionSpecies(smiles="CCO", resname="ETH", **kwargs)
        assert (spec.count, spec.concentration) == (count, concentration)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"count": 10, "concentration": 1.0}, "exactly one of 'count' or 'concentration'"),
            ({}, "exactly one of 'count' or 'concentration'"),
            ({"count": 10, "fraction": 0.5}, "does not support 'fraction'"),
            ({"count": -1}, ""),
        ],
    )
    def test_rejects_invalid_amounts(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SolutionSpecies(smiles="CCO", resname="ETH", **kwargs)

    @pytest.mark.parametrize(
        ("smiles", "wrong_resname"),
        [("O", "WAT"), ("[Na+]", "SOD"), ("[Cl-]", "CLA")],
    )
    def test_native_species_require_charmm_resname(self, smiles, wrong_resname):
        with pytest.raises(ValueError, match="Native CHARMM species"):
            SolutionSpecies(smiles=smiles, resname=wrong_resname, count=1)


class TestCharmmConfig:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"forcefield": "charmm27"}, "only the registered 'charmm36m'"),
            ({"water_model": "spce"}, "only 'tip3p'"),
        ],
    )
    def test_rejects_unpinned_choices(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            CharmmConfig(**overrides)


class TestSizingConfigs:
    @pytest.mark.parametrize(
        ("sizing_cls", "kwargs"),
        [(FixedBoxSizing, {"box_size": 0.0}), (CountDensitySizing, {"target_density": 0.0})],
    )
    def test_rejects_non_positive(self, sizing_cls, kwargs):
        with pytest.raises(ValueError):
            sizing_cls(**kwargs)


class TestProteinMixedBoxComposition:
    @pytest.fixture
    def protein(self, protein_pdb):
        return ProteinSpecies(resname="LYZ", pdb_path=protein_pdb)

    def test_fixed_box_with_concentration(self, protein):
        comp = ProteinMixedBoxComposition(
            protein=protein,
            species=[
                SolutionSpecies(smiles="O", resname="SOL", concentration=55.0),
                SolutionSpecies(smiles="CCO", resname="ETH", concentration=1.0),
            ],
            sizing={"type": "fixed_box", "box_size": 72.0},
            padding=12.0,
        )
        assert isinstance(comp.sizing, FixedBoxSizing)
        # Concentration-based species have no known count until build time.
        assert comp.total_count is None

    def test_count_density_with_counts(self, protein):
        comp = ProteinMixedBoxComposition(
            protein=protein,
            species=[
                SolutionSpecies(smiles="O", resname="SOL", count=10000),
                SolutionSpecies(smiles="CCO", resname="ETH", count=200),
            ],
            sizing={"type": "count_density", "target_density": 1.0},
            padding=12.0,
        )
        assert isinstance(comp.sizing, CountDensitySizing)
        assert comp.total_count == 1 + 10000 + 200

    def test_fixed_box_rejects_padding_too_large(self, protein):
        with pytest.raises(ValueError, match="must exceed 2 × padding"):
            ProteinMixedBoxComposition(
                protein=protein,
                species=[SolutionSpecies(smiles="O", resname="SOL", count=10)],
                sizing={"type": "fixed_box", "box_size": 20.0},
                padding=12.0,
            )

    def test_count_density_rejects_concentration_species(self, protein):
        with pytest.raises(
            ValueError, match="count_density sizing derives the box from total mass"
        ):
            ProteinMixedBoxComposition(
                protein=protein,
                species=[SolutionSpecies(smiles="O", resname="SOL", concentration=55.0)],
                sizing={"type": "count_density", "target_density": 1.0},
                padding=12.0,
            )


class TestProteinMixedBoxInput:
    """protein_mixedbox-specific BuildInput behavior (shared checks live in test_proteinbox)."""

    def test_metadata_includes_protein_and_requested_concentration(self, protein_pdb):
        data = protein_input_data("protein_mixedbox", protein_pdb)
        data["system"]["species"] = [{"smiles": "O", "resname": "SOL", "concentration": 55.0}]
        species_composition = BuildInput(**data).metadata["species_composition"]
        # The protein leads the composition; the packed solution follows.
        protein_entry = species_composition[0]
        assert protein_entry["resname"] == "LYZ"
        assert protein_entry["count"] == 1
        assert protein_entry["concentration"] is None
        assert species_composition[1]["resname"] == "SOL"
        assert species_composition[1]["concentration"] == 55.0

    def test_rejects_unmerged_multichain_compression(self, protein_pdb):
        data = protein_input_data(
            "protein_mixedbox", protein_pdb, protein={"resname": "INS", "chains": ["A", "B"]}
        )
        with pytest.raises(ValueError, match="multi-chain protein_mixedbox"):
            BuildInput(**data)


class TestExampleFiles:
    @pytest.mark.parametrize(
        ("filename", "sizing_cls", "total_count"),
        [
            ("lysozyme_fixed_box.yaml", FixedBoxSizing, None),
            ("lysozyme_count_density.yaml", CountDensitySizing, 1 + 10000 + 200),
        ],
    )
    def test_yaml_loads(self, filename, sizing_cls, total_count):
        inp = BuildInput(**yaml.safe_load((EXAMPLES_DIR / filename).read_text()))
        assert isinstance(inp.system, ProteinMixedBoxComposition)
        assert isinstance(inp.parametrization_config, CharmmConfig)
        assert isinstance(inp.system.sizing, sizing_cls)
        assert inp.system.total_count == total_count

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


def _resolve(specs, **overrides):
    """Call resolve_solution_counts with a whole-box basis unless overridden."""
    kwargs = {
        "box_volume_a3": 1e6,
        "protein_mass_dalton": 0.0,
        "partial_specific_volume": 0.73,
        "basis": "box",
    }
    kwargs.update(overrides)
    return resolve_solution_counts(specs, **kwargs)


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
        counts = _resolve(specs, protein_mass_dalton=1e4, basis="protein_excluded")
        assert counts == [42]

    def test_concentration_resolves_on_box_basis(self):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=1.0)]
        expected = int(round(1.0 * 1e6 * 1e-27 * N_AVOGADRO))
        assert _resolve(specs) == [expected]

    def test_protein_excluded_basis_subtracts_displaced_volume(self):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=1.0)]
        protein_mass, psv = 1e5, 0.73
        counts = _resolve(
            specs,
            protein_mass_dalton=protein_mass,
            partial_specific_volume=psv,
            basis="protein_excluded",
        )
        v_basis = 1e6 - protein_displaced_volume_a3(protein_mass, psv)
        assert counts == [int(round(1.0 * v_basis * 1e-27 * N_AVOGADRO))]
        # The excluded-volume count is strictly smaller than the whole-box count.
        assert counts[0] < _resolve(specs, protein_mass_dalton=protein_mass)[0]

    @pytest.mark.parametrize(
        ("concentration", "overrides", "match"),
        [
            (1e-9, {"box_volume_a3": 1e3}, "resolves to zero"),
            (
                1.0,
                {"box_volume_a3": 1e3, "protein_mass_dalton": 1e9, "basis": "protein_excluded"},
                "non-positive",
            ),
        ],
    )
    def test_unresolvable_concentration_raises(self, concentration, overrides, match):
        specs = [SolutionSpecies(smiles="O", resname="SOL", concentration=concentration)]
        with pytest.raises(ValueError, match=match):
            _resolve(specs, **overrides)


class TestIonizeSolvatedSystem:
    @pytest.fixture
    def captured(self, monkeypatch):
        """Replace the ionizer with a stub recording the requested ion counts."""
        counts = {}

        def fake_ionize(u, num_na, num_cl, min_distance, seed):
            counts.update(num_na=num_na, num_cl=num_cl)
            return u

        monkeypatch.setattr("mdfactory.build.ionize", fake_ionize)
        return counts

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

    @pytest.mark.parametrize("total_charge", [0, 3])
    def test_salt_count_uses_solvent_volume_and_neutralizes(self, captured, total_charge):
        solvent_volume_a3 = 110_704.0
        config = IonizationConfig(neutralize=True, concentration=0.15)
        _, ion_species = ionize_solvated_system(
            config,
            self._water_universe(n_waters=10),
            total_charge=total_charge,
            solvent_volume_a3=solvent_volume_a3,
        )

        # Salt is sized on the solvent volume (a water-count proxy would give a tiny count);
        # a +3 system needs 3 extra Cl- to neutralize.
        n_ions = int(round(0.15 * solvent_volume_a3 * 1e-27 * N_AVOGADRO))
        assert captured == {"num_na": n_ions, "num_cl": n_ions + total_charge}
        assert [s.resname for s in ion_species] == ["NA", "CL"]
        assert [s.count for s in ion_species] == [n_ions, n_ions + total_charge]

    def test_none_config_adds_no_ions(self):
        u = self._water_universe(n_waters=4)
        u_out, ion_species = ionize_solvated_system(None, u, total_charge=0)
        assert ion_species == []
        assert u_out is u


PROTEIN_XYZ = np.array(
    [
        [15.0, 15.0, 15.0],
        [16.0, 15.0, 15.0],
        [15.0, 16.0, 15.0],
        [15.0, 15.0, 16.0],
        [15.0, 14.0, 15.0],
    ]
)
CLEAR_SOLUTION_XYZ = np.array(
    [
        [2.0, 2.0, 2.0],
        [2.5, 2.0, 2.0],
        [2.0, 2.5, 2.0],
        [5.0, 5.0, 5.0],
        [5.5, 5.0, 5.0],
        [5.0, 5.5, 5.0],
    ]
)
# The first solution atom sits on top of a protein atom.
PROTEIN_OVERLAP_XYZ = np.vstack([[[15.0, 15.0, 15.1]], CLEAR_SOLUTION_XYZ[1:]])
# The first atom of the second water sits 0.1 Å from the first atom of the first water.
INTER_RESIDUE_CLASH_XYZ = np.array(
    [
        [2.0, 2.0, 2.0],
        [2.5, 2.0, 2.0],
        [2.0, 2.5, 2.0],
        [2.1, 2.0, 2.0],
        [8.0, 8.0, 8.0],
        [9.0, 9.0, 9.0],
    ]
)
# Atoms within a residue are close, but no atoms of different residues are.
INTRA_RESIDUE_CLOSE_XYZ = np.array(
    [
        [2.0, 2.0, 2.0],
        [2.1, 2.0, 2.0],
        [2.0, 2.1, 2.0],
        [8.0, 8.0, 8.0],
        [8.1, 8.0, 8.0],
        [8.0, 8.1, 8.0],
    ]
)


def _system(protein_xyz, solution_xyz):
    """Return a box with a 5-atom protein residue then two 3-atom waters, and n_protein."""
    n_protein = 5
    atom_resindex = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    u = mda.Universe.empty(11, n_residues=3, atom_resindex=atom_resindex, trajectory=True)
    u.add_TopologyAttr("resnames", ["LYZ", "SOL", "SOL"])
    u.add_TopologyAttr("masses", [12.0] * n_protein + [16.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    u.dimensions = [30.0, 30.0, 30.0, 90, 90, 90]
    u.atoms.positions = np.vstack([protein_xyz, solution_xyz])
    return u, n_protein


class TestReimageAndClash:
    def test_reimage_makes_protein_whole_and_centered(self):
        L = 30.0
        # A rigid translation pushes the protein across the +x boundary; wrapping
        # by box vector tears it apart, exactly as a barostat + restraint produces.
        torn = np.mod(PROTEIN_XYZ + np.array([13.0, 0.0, 0.0]), L)
        solution = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.5, 1.0, 1.0],
                [1.0, 1.5, 1.0],
                [29.0, 29.0, 29.0],
                [28.5, 29.0, 29.0],
                [29.0, 28.5, 29.0],
            ]
        )
        u, n_protein = _system(torn, solution)

        _reimage_protein_centered(u, n_protein, PROTEIN_XYZ)

        protein_now = u.atoms.positions[:n_protein]
        shape_now = protein_now - protein_now.mean(axis=0)
        shape_ref = PROTEIN_XYZ - PROTEIN_XYZ.mean(axis=0)
        rmsd = np.sqrt(((shape_now - shape_ref) ** 2).sum(axis=1).mean())
        assert rmsd < 1e-4
        assert rmsd < PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A

        bbox_center = (protein_now.min(axis=0) + protein_now.max(axis=0)) / 2.0
        assert np.allclose(bbox_center, [L / 2, L / 2, L / 2], atol=1e-4)

    @pytest.mark.parametrize(
        ("check", "solution_xyz", "match"),
        [
            (_check_no_protein_clashes, PROTEIN_OVERLAP_XYZ, "closer than"),
            (_check_no_protein_clashes, CLEAR_SOLUTION_XYZ, None),
            (_check_no_solution_clashes, INTER_RESIDUE_CLASH_XYZ, "solution-solution"),
            (_check_no_solution_clashes, INTRA_RESIDUE_CLOSE_XYZ, None),
        ],
        ids=["protein-overlap", "protein-clear", "solution-clash", "solution-same-residue"],
    )
    def test_clash_checks(self, check, solution_xyz, match):
        u, n_protein = _system(PROTEIN_XYZ, solution_xyz)
        with pytest.raises(ValueError, match=match) if match else nullcontext():
            check(u, n_protein)


class TestPackingAndRestraints:
    def test_native_charmm_atom_names(self):
        assert charmm_native_atom_names("O") == ["OW", "HW1", "HW2"]
        assert charmm_native_atom_names("[Na+]") == ["NA"]
        assert charmm_native_atom_names("[Cl-]") == ["CL"]
        assert charmm_native_atom_names("CCO") is None

    def test_native_parameters_come_from_selected_forcefield(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "parameter_store", tmp_path / "parameters")
        ff_dir = tmp_path / "selected.ff"
        ff_dir.mkdir()
        for filename in ("forcefield.itp", "tip3p.itp", "ions.itp"):
            (ff_dir / filename).touch()

        water = parametrize_cgenff_gromacs(
            SolutionSpecies(smiles="O", resname="SOL", count=1),
            native_forcefield_dir=ff_dir,
        )
        sodium = parametrize_cgenff_gromacs(
            SolutionSpecies(smiles="[Na+]", resname="NA", count=1),
            native_forcefield_dir=ff_dir,
        )
        assert water.itp == ff_dir / "tip3p.itp"
        assert sodium.itp == ff_dir / "ions.itp"
        assert water.forcefield_itp == ff_dir / "forcefield.itp"

    def test_packmol_reconstructs_native_residues_and_names(self, tmp_path, protein_pdb):
        from openff.interchange.components._packmol import _find_packmol

        if _find_packmol() is None:
            pytest.skip("Packmol is unavailable")

        species = [
            SolutionSpecies(smiles="O", resname="SOL", count=2),
            SolutionSpecies(smiles="[Na+]", resname="NA", count=1),
            SolutionSpecies(smiles="[Cl-]", resname="CL", count=1),
        ]
        universe = create_protein_mixedbox_universe(
            mda.Universe(protein_pdb),
            [spec.openff_molecule for spec in species],
            [2, 1, 1],
            [spec.resname for spec in species],
            30.0,
            tmp_path / "packmol",
            native_atom_names=[charmm_native_atom_names(spec.smiles) for spec in species],
        )
        assert list(universe.residues.resnames[1:]) == ["SOL", "SOL", "NA", "CL"]
        assert list(universe.residues[1].atoms.names) == ["OW", "HW1", "HW2"]
        assert list(universe.residues[3].atoms.names) == ["NA"]
        assert list(universe.residues[4].atoms.names) == ["CL"]

        # Validate the packed native coordinates against the same registered
        # CHARMM bundle used by protein_mixedbox. This catches atom/residue-name
        # drift at the real GROMACS boundary.
        import shutil
        import subprocess

        gmx = shutil.which("gmx")
        forcefields = sorted(settings.gromacs_forcefield_dir.glob("*cgenff-5.0.ff"))
        if gmx is None or not forcefields:
            pytest.skip("GROMACS or the registered CHARMM36m/CGenFF 5.0 bundle is unavailable")
        (tmp_path / "charmm36m.ff").symlink_to(forcefields[-1], target_is_directory=True)
        solution = mda.Merge(universe.atoms[1:])
        solution.dimensions = universe.dimensions
        solution.atoms.write(tmp_path / "solution.pdb")
        (tmp_path / "topology.top").write_text(
            textwrap.dedent("""\
            #include "charmm36m.ff/forcefield.itp"
            #include "charmm36m.ff/tip3p.itp"
            #include "charmm36m.ff/ions.itp"

            [ system ]
            Native solution

            [ molecules ]
            SOL 2
            NA  1
            CL  1
            """)
        )
        (tmp_path / "em.mdp").write_text(
            textwrap.dedent("""\
            integrator = steep
            nsteps = 0
            cutoff-scheme = Verlet
            coulombtype = Cut-off
            rlist = 1.0
            rcoulomb = 1.0
            rvdw = 1.0
            constraints = none
            """)
        )
        result = subprocess.run(
            [
                gmx,
                "grompp",
                "-f",
                "em.mdp",
                "-c",
                "solution.pdb",
                "-p",
                "topology.top",
                "-o",
                "topol.tpr",
                "-maxwarn",
                "0",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_rmsd_restraint_is_translation_invariant(self):
        import openmm as mm
        from openmm import unit

        positions = [
            mm.Vec3(0.0, 0.0, 0.0),
            mm.Vec3(0.1, 0.0, 0.0),
            mm.Vec3(0.0, 0.1, 0.0),
        ] * unit.nanometer
        system = mm.System()
        for _ in positions:
            system.addParticle(12.0)
        _add_protein_rmsd_restraint(system, positions, [0, 1, 2], 1000.0)
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        context = mm.Context(system, integrator)
        context.setPositions(positions)
        reference_energy = context.getState(getEnergy=True).getPotentialEnergy()
        translated = [p + mm.Vec3(1.0, 1.0, 1.0) * unit.nanometer for p in positions]
        context.setPositions(translated)
        translated_energy = context.getState(getEnergy=True).getPotentialEnergy()
        assert translated_energy.value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
            reference_energy.value_in_unit(unit.kilojoules_per_mole), abs=1e-8
        )

    @pytest.mark.parametrize(
        ("volumes", "expected_converged", "expected_steps"),
        [([12.0, 9.95, 9.5], True, 50), ([12.0, 11.0, 10.5], False, 75)],
    )
    def test_density_compression_reports_convergence(
        self, volumes, expected_converged, expected_steps
    ):
        import openmm as mm
        from openmm import unit

        class FakeSimulation:
            def __init__(self, sampled_volumes):
                self.context = self
                self.sampled_volumes = iter(sampled_volumes)
                self.edge = None

            def step(self, _steps):
                self.edge = next(self.sampled_volumes) ** (1.0 / 3.0)

            def getState(self):
                return self

            def getPeriodicBoxVectors(self):
                return (
                    mm.Vec3(self.edge, 0, 0) * unit.nanometer,
                    mm.Vec3(0, self.edge, 0) * unit.nanometer,
                    mm.Vec3(0, 0, self.edge) * unit.nanometer,
                )

        mass = 1000.0 * unit.dalton
        to_g_per_ml = unit.gram / unit.item / unit.milliliter
        sampled, completed_steps, converged = simulate_compression_until_density_reached(
            FakeSimulation(volumes),
            mass=mass,
            steps=75,
            target_density=(mass / (10.0 * unit.nanometer**3)).value_in_unit(to_g_per_ml),
        )
        assert converged is expected_converged
        assert completed_steps == expected_steps
        expected = (mass / (volumes[expected_steps // 25 - 1] * unit.nanometer**3)).value_in_unit(
            to_g_per_ml
        )
        assert sampled[-1] == pytest.approx(expected)


def _gen(tmp_path, species, parameters, counts, ff_dir=None):
    """Generate a composite topology from a pdb2gmx-style protein topology; return its text."""
    top = tmp_path / "topol_protein.top"
    _write_pdb2gmx_topology(top)
    out = tmp_path / "topology.top"
    generate_gromacs_topology_with_protein(
        top,
        ff_dir or tmp_path / "nonexistent.ff",
        species,
        parameters,
        counts,
        "protein_mixedbox",
        out_path=out,
    )
    return out.read_text()


class TestCompositeTopology:
    def test_native_only_ordering_and_counts(self, tmp_path):
        species = [
            SolutionSpecies(smiles="O", resname="SOL", count=100),
            SolutionSpecies(smiles="CCO", resname="ETH", count=5),
            SolutionSpecies(smiles="[Na+]", resname="NA", count=3),
            SolutionSpecies(smiles="[Cl-]", resname="CL", count=6),
        ]
        parameters = [
            SimpleNamespace(moleculetype=name, parameter_itp=None)
            for name in ("SOL", "ETH0", "SOD", "CLA")
        ]

        text = _gen(tmp_path, species, parameters, [100, 5, 3, 6])

        # Exactly one bundled force-field include.
        assert text.count('#include "charmm36m.ff/forcefield.itp"') == 1
        # System title replaced.
        assert "protein_mixedbox\n" in text
        assert "Protein in water" not in text

        molecules = text.split("[ molecules ]", 1)[1]
        mol_lines = [
            line.split()[:2]
            for line in molecules.splitlines()
            if line.strip() and not line.strip().startswith(";")
        ]
        assert mol_lines == [
            ["Protein_chain_A", "1"],
            ["SOL", "100"],
            ["ETH0", "5"],
            ["SOD", "3"],
            ["CLA", "6"],
        ]

    def test_cgenff_small_molecule_merges_params_and_includes_itp(self, tmp_path):
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

        text = _gen(tmp_path, species, parameters, [5], ff_dir=ff_dir)

        assert text.count('#include "charmm36m.ff/forcefield.itp"') == 1
        assert '#include "extra_params.itp"' in text
        assert '#include "MOL0.itp"' in text
        assert (tmp_path / "MOL0.itp").is_file()

        merged = (tmp_path / "extra_params.itp").read_text()
        assert "NEWTYPE" in merged
        # CG331 is already in the force field, so it is excluded from the merge.
        assert "CG331" not in merged

        # The small-molecule include precedes the water include.
        assert text.index('#include "MOL0.itp"') < text.index("charmm36m.ff/tip3p.itp")

    def test_zero_count_species_skipped(self, tmp_path):
        species = [
            SolutionSpecies(smiles="O", resname="SOL", count=100),
            SolutionSpecies(smiles="CCO", resname="ETH", count=0),
        ]
        parameters = [
            SimpleNamespace(moleculetype="SOL", parameter_itp=None),
            SimpleNamespace(moleculetype="ETH0", parameter_itp=None),
        ]
        text = _gen(tmp_path, species, parameters, [100, 0])
        molecules = text.split("[ molecules ]", 1)[1]
        assert "ETH0" not in molecules
        assert "SOL" in molecules


class TestRunSchedule:
    def test_protein_mixedbox_run_files_on_disk(self):
        manager = RunScheduleManager()
        paths = manager.get_all_run_file_paths(engine="gromacs", system_type="protein_mixedbox")
        assert set(paths) == {"em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"}
        for path in paths.values():
            assert path.is_file()
