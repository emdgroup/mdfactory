# ABOUTME: Tests for proteinbox models, topology parsing, and YAML validation
# ABOUTME: Covers ProteinSpecies, ProteinBoxComposition, Pdb2gmxConfig, and BuildInput
"""Tests for proteinbox models, topology parsing, and YAML validation."""

import textwrap
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import yaml

from mdfactory import workflows
from mdfactory.build import _center_in_cubic_box
from mdfactory.models.composition import ProteinBoxComposition
from mdfactory.models.input import BuildInput
from mdfactory.models.parametrization import Pdb2gmxConfig
from mdfactory.models.species import ProteinSpecies
from mdfactory.setup.protein import (
    _apply_protonation_states,
    _build_disulfide_prompt_input,
    _sum_charges_from_itp,
    bundle_forcefield_into_topology,
    check_forcefield_available,
    run_pdb2gmx,
    update_topology_molecules,
)
from mdfactory.workflows import _resolve_proteinbox_pdb_path


def _pdb_atom(serial, atom_name, resname, resid, x, y, z):
    return (
        f"ATOM  {serial:5d} {atom_name:<4s} {resname:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    )


def _patch_pdb2gmx_env(monkeypatch):
    """Stub the gmx binary, force-field, and environment lookups for run_pdb2gmx."""
    monkeypatch.setattr("mdfactory.setup.protein.check_gmx_available", lambda: Path("/mock/gmx"))
    monkeypatch.setattr("mdfactory.setup.protein.check_forcefield_available", lambda ff: None)
    monkeypatch.setattr(
        "mdfactory.setup.protein.get_gromacs_env", lambda: {"GMXLIB": "/mock/forcefields"}
    )


def _make_pdb2gmx_run(molecule_names, extra_files=()):
    """Build a fake subprocess.run writing a topology listing the given molecules."""

    def fake_run(cmd, cwd, text, input, capture_output, timeout, env=None, check=None):
        output_dir = Path(cwd)
        (output_dir / "processed.gro").write_text("mock gro\n")
        molecules = "".join(f"{name} 1\n" for name in molecule_names)
        (output_dir / "topol.top").write_text(f"[ molecules ]\n{molecules}")
        for name in extra_files:
            (output_dir / name).write_text(f"; {name}\n")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    return fake_run


class TestProteinSpecies:
    def test_basic_creation(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        spec = ProteinSpecies(resname="LYZ", pdb_path=pdb)
        assert spec.resname == "LYZ"
        assert spec.count == 1
        assert spec.fraction == 1.0

    def test_with_disulfides_and_protonation(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        spec = ProteinSpecies(
            resname="LYZ",
            pdb_path=pdb,
            disulfide_bonds=[("CYS6", "CYS127"), ("CYS30", "CYS115")],
            protonation_states={"HIS15": "HIE", "GLU35": "GLH"},
        )
        assert len(spec.disulfide_bonds) == 2
        assert spec.disulfide_bonds[0] == ("CYS6", "CYS127")
        assert spec.protonation_states["HIS15"] == "HIE"

    def test_charge_is_none(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        spec = ProteinSpecies(resname="LYZ", pdb_path=pdb)
        assert spec.charge is None

    def test_chains_default_empty_and_declared(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        assert ProteinSpecies(resname="LYZ", pdb_path=pdb).chains == []
        spec = ProteinSpecies(resname="LYZ", pdb_path=pdb, chains=["H", "L", "Y"])
        assert spec.chains == ["H", "L", "Y"]

    def test_resname_validation(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        with pytest.raises(ValueError, match="Residue name must be less"):
            ProteinSpecies(resname="TOOLONG", pdb_path=pdb)


class TestProteinBoxComposition:
    def test_basic_creation(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        comp = ProteinBoxComposition(
            protein=ProteinSpecies(resname="LYZ", pdb_path=pdb),
            padding=12.0,
        )
        assert comp.padding == 12.0
        assert comp.ionization.neutralize is True
        assert comp.ionization.concentration == 0.15

    def test_custom_ionization(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        comp = ProteinBoxComposition(
            protein=ProteinSpecies(resname="LYZ", pdb_path=pdb),
            ionization={"neutralize": True, "concentration": 0.10},
        )
        assert comp.ionization.concentration == 0.10


class TestPdb2gmxConfig:
    def test_defaults(self):
        config = Pdb2gmxConfig()
        assert config.forcefield == "charmm36m"
        assert config.water_model == "tip3p"
        assert config.ignore_hydrogens is True
        assert config.merge_all is False

    def test_custom(self):
        config = Pdb2gmxConfig(forcefield="charmm36", water_model="spce")
        assert config.forcefield == "charmm36"
        assert config.water_model == "spce"

    def test_frozen(self):
        config = Pdb2gmxConfig()
        with pytest.raises(Exception):
            config.forcefield = "other"

    def test_rejects_non_charmm_forcefield(self):
        with pytest.raises(ValueError, match="CHARMM"):
            Pdb2gmxConfig(forcefield="amber99sb-ildn")

    def test_rejects_non_three_site_water(self):
        with pytest.raises(ValueError, match="3-site"):
            Pdb2gmxConfig(water_model="tip4p")

    def test_rejects_ljpme_forcefield(self):
        with pytest.raises(ValueError, match="LJ-PME"):
            Pdb2gmxConfig(forcefield="charmm36m-ljpme")


class TestForceFieldCheck:
    def test_available_forcefield_passes(self, tmp_path, monkeypatch):
        (tmp_path / "charmm27.ff").mkdir()
        monkeypatch.setattr(
            "mdfactory.setup.protein._get_gmx_search_paths",
            lambda: [tmp_path],
        )
        check_forcefield_available("charmm27")

    def test_unavailable_forcefield_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mdfactory.setup.protein._get_gmx_search_paths",
            lambda: [tmp_path],
        )
        with pytest.raises(ValueError, match="not found"):
            check_forcefield_available("nonexistent_ff_xyz")

    def test_error_lists_available_forcefields(self, tmp_path, monkeypatch):
        (tmp_path / "charmm27.ff").mkdir()
        monkeypatch.setattr(
            "mdfactory.setup.protein._get_gmx_search_paths",
            lambda: [tmp_path],
        )
        with pytest.raises(ValueError, match="charmm27") as exc_info:
            check_forcefield_available("nonexistent_ff_xyz")
        assert "Available force fields" in str(exc_info.value)

    def test_registered_forcefield_check_does_not_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mdfactory.setup.protein._get_gmx_search_paths",
            lambda: [tmp_path],
        )
        monkeypatch.setattr(
            "mdfactory.setup.protein.download_forcefield",
            lambda name: pytest.fail("check_forcefield_available should not download"),
        )
        with pytest.raises(ValueError, match="Downloadable"):
            check_forcefield_available("charmm36m")


class TestBuildInputProteinbox:
    def test_yaml_roundtrip(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

        yaml_str = f"""
simulation_type: proteinbox
engine: gromacs
parametrization: pdb2gmx
parametrization_config:
  type: pdb2gmx
  forcefield: charmm36m
  water_model: tip3p
  ignore_hydrogens: true
system:
  protein:
    resname: LYZ
    count: 1
    pdb_path: {pdb}
    disulfide_bonds:
      - [CYS6, CYS127]
      - [CYS30, CYS115]
    protonation_states:
      HIS15: HIE
  padding: 12.0
  ionization:
    neutralize: true
    concentration: 0.15
"""
        data = yaml.safe_load(yaml_str)
        inp = BuildInput(**data)
        assert inp.simulation_type == "proteinbox"
        assert isinstance(inp.system, ProteinBoxComposition)
        assert isinstance(inp.parametrization_config, Pdb2gmxConfig)
        assert inp.parametrization == "pdb2gmx"

    def test_default_parametrization_config(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

        data = {
            "simulation_type": "proteinbox",
            "parametrization": "pdb2gmx",
            "system": {
                "protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)},
            },
        }
        inp = BuildInput(**data)
        assert isinstance(inp.parametrization_config, Pdb2gmxConfig)
        assert inp.parametrization_config.forcefield == "charmm36m"

    def test_rejects_cgenff_parametrization(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        data = {
            "simulation_type": "proteinbox",
            "parametrization": "cgenff",
            "system": {"protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)}},
        }
        with pytest.raises(ValueError, match="not valid for simulation type"):
            BuildInput(**data)

    def test_rejects_mismatched_config_type(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        data = {
            "simulation_type": "proteinbox",
            "parametrization": "pdb2gmx",
            "parametrization_config": {"type": "cgenff"},
            "system": {"protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)}},
        }
        with pytest.raises(ValueError, match="does not match"):
            BuildInput(**data)

    def test_rejects_merge_all_with_declared_chains(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        data = {
            "simulation_type": "proteinbox",
            "parametrization": "pdb2gmx",
            "parametrization_config": {"type": "pdb2gmx", "merge_all": True},
            "system": {
                "protein": {"resname": "INS", "pdb_path": str(pdb), "chains": ["A", "B"]},
            },
        }
        with pytest.raises(ValueError, match="merge_all"):
            BuildInput(**data)

    def test_merge_all_without_chains_allowed(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        data = {
            "simulation_type": "proteinbox",
            "parametrization": "pdb2gmx",
            "parametrization_config": {"type": "pdb2gmx", "merge_all": True},
            "system": {"protein": {"resname": "INS", "pdb_path": str(pdb)}},
        }
        inp = BuildInput(**data)
        assert inp.parametrization_config.merge_all is True
        assert inp.system.protein.chains == []


class TestTopologyParsing:
    def test_sum_charges_from_itp(self, tmp_path):
        itp = tmp_path / "protein.itp"
        itp.write_text(
            textwrap.dedent("""\
            [ moleculetype ]
            Protein   3

            [ atoms ]
            ;   nr  type  resnr residue  atom   cgnr   charge     mass
                 1    NH3    1    LYS      N      1    -0.300   14.007
                 2     CT    1    LYS     CA      2     0.210   12.011
                 3      C    1    LYS      C      3     0.510   12.011
                 4      O    1    LYS      O      4    -0.510   15.999

            [ bonds ]
            1 2
        """)
        )
        charge = _sum_charges_from_itp(itp)
        assert abs(charge - (-0.090)) < 1e-6

    def test_update_topology_molecules(self, tmp_path):
        top = tmp_path / "topol.top"
        top.write_text(
            textwrap.dedent("""\
            #include "charmm36m.ff/forcefield.itp"
            #include "protein.itp"

            [ system ]
            Protein in water

            [ molecules ]
            Protein_chain_A  1
        """)
        )
        update_topology_molecules(top, n_water=5000, num_na=10, num_cl=18)
        content = top.read_text()
        assert "SOL" in content
        assert "5000" in content
        assert "NA" in content
        assert "10" in content
        assert "CL" in content
        assert "18" in content

    def test_apply_protonation_states(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            textwrap.dedent("""\
            ATOM      1  N   HIS A  15       1.000   2.000   3.000  1.00  0.00
            ATOM      2  CA  HIS A  15       1.500   2.500   3.500  1.00  0.00
            ATOM      3  N   ALA A  16       2.000   3.000   4.000  1.00  0.00
            ATOM      4  N   GLU A  35       3.000   4.000   5.000  1.00  0.00
        """)
        )
        result = _apply_protonation_states(pdb, {"HIS15": "HIE", "GLU35": "GLH"}, tmp_path)
        content = result.read_text()
        # HIS at resid 15 should be renamed to HIE
        assert "HIE" in content
        # GLU at resid 35 should be renamed to GLH
        assert "GLH" in content
        # ALA at resid 16 should be unchanged
        assert "ALA" in content

    def test_apply_protonation_states_translates_charmm_histidine_alias(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "HIS", 15, 1.0, 2.0, 3.0))
        result = _apply_protonation_states(pdb, {"HIS15": "HIE"}, tmp_path, forcefield="charmm36m")
        assert "HSE" in result.read_text()

    def test_apply_protonation_states_rejects_four_character_names(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "GLU", 35, 1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match="cannot be written to a PDB"):
            _apply_protonation_states(pdb, {"GLU35": "GLUP"}, tmp_path)

    def test_apply_protonation_states_rejects_charmm_acid_protonation(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "GLU", 35, 1.0, 2.0, 3.0))
        # GLH aliases to CHARMM GLUP (4 chars), which cannot fit a PDB column.
        with pytest.raises(ValueError, match="cannot be written to a PDB"):
            _apply_protonation_states(pdb, {"GLU35": "GLH"}, tmp_path, forcefield="charmm36m")

    def test_apply_protonation_states_matches_existing_histidine_tautomer(self, tmp_path):
        # Residue 15 is already an HID tautomer; a "HIS15" key must still match it
        # instead of silently doing nothing.
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "HID", 15, 1.0, 2.0, 3.0))
        result = _apply_protonation_states(pdb, {"HIS15": "HIE"}, tmp_path, forcefield="charmm36m")
        content = result.read_text()
        assert "HSE" in content
        assert "HID" not in content

    def test_apply_protonation_states_raises_on_unmatched_override(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "HIS", 15, 1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match="matched no residue"):
            _apply_protonation_states(pdb, {"HIS99": "HIE"}, tmp_path)

    def test_build_disulfide_prompt_input_answers_exact_requested_pairs(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "".join(
                [
                    _pdb_atom(1, "SG", "CYS", 1, 0.0, 0.0, 0.0),
                    _pdb_atom(2, "SG", "CYS", 4, 2.0, 0.0, 0.0),
                    _pdb_atom(3, "SG", "CYS", 2, 10.0, 0.0, 0.0),
                    _pdb_atom(4, "SG", "CYS", 3, 12.0, 0.0, 0.0),
                ]
            )
        )
        assert _build_disulfide_prompt_input(pdb, [("CYS2", "CYS3")]) == "n\ny\n"

    def test_build_disulfide_prompt_input_rejects_missing_pair(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "".join(
                [
                    _pdb_atom(1, "SG", "CYS", 6, 0.0, 0.0, 0.0),
                    _pdb_atom(2, "SG", "CYS", 127, 8.0, 0.0, 0.0),
                ]
            )
        )
        with pytest.raises(ValueError, match="not detected as close CYS SG pairs"):
            _build_disulfide_prompt_input(pdb, [("CYS6", "CYS127")])

    def test_run_pdb2gmx_passes_disulfide_answers_to_subprocess(self, tmp_path, monkeypatch):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "".join(
                [
                    _pdb_atom(1, "SG", "CYS", 1, 0.0, 0.0, 0.0),
                    _pdb_atom(2, "SG", "CYS", 4, 2.0, 0.0, 0.0),
                    _pdb_atom(3, "SG", "CYS", 2, 10.0, 0.0, 0.0),
                    _pdb_atom(4, "SG", "CYS", 3, 12.0, 0.0, 0.0),
                ]
            )
        )
        calls = {}

        def fake_run(cmd, cwd, text, input, capture_output, timeout, env=None, check=None):
            calls["cmd"] = cmd
            calls["cwd"] = cwd
            calls["input"] = input
            calls["env"] = env
            calls["check"] = check
            output_dir = Path(cwd)
            (output_dir / "processed.gro").write_text("mock gro\n")
            (output_dir / "topol.top").write_text(
                "[ moleculetype ]\nProtein_chain_A 3\n[ atoms ]\n[ molecules ]\nProtein_chain_A 1\n"
            )
            (output_dir / "posre.itp").write_text("mock posre\n")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr("mdfactory.setup.protein.subprocess.run", fake_run)
        _patch_pdb2gmx_env(monkeypatch)

        run_pdb2gmx(
            pdb_path=pdb,
            config=Pdb2gmxConfig(forcefield="charmm36m", water_model="tip3p"),
            disulfide_bonds=[("CYS2", "CYS3")],
            protonation_states={},
            output_dir=tmp_path / "pdb2gmx_output",
        )

        assert "-ss" in calls["cmd"]
        assert calls["cmd"][:2] == ["/mock/gmx", "pdb2gmx"]
        assert calls["input"] == "n\ny\n"
        assert calls["cwd"] == str((tmp_path / "pdb2gmx_output").resolve())
        assert calls["env"] == {"GMXLIB": "/mock/forcefields"}
        assert calls["check"] is False

    def test_run_pdb2gmx_multichain_collects_chain_includes(self, tmp_path, monkeypatch):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "ALA", 1, 0.0, 0.0, 0.0))
        chains = ["H", "L", "Y"]
        extra = [f"topol_Protein_chain_{c}.itp" for c in chains] + [
            f"posre_Protein_chain_{c}.itp" for c in chains
        ]
        monkeypatch.setattr(
            "mdfactory.setup.protein.subprocess.run",
            _make_pdb2gmx_run([f"Protein_chain_{c}" for c in chains], extra_files=extra),
        )
        _patch_pdb2gmx_env(monkeypatch)

        params = run_pdb2gmx(
            pdb_path=pdb,
            config=Pdb2gmxConfig(forcefield="charmm36m", water_model="tip3p"),
            disulfide_bonds=[],
            protonation_states={},
            output_dir=tmp_path / "out",
            chains=chains,
        )

        assert params.chains == chains
        assert sorted(p.name for p in params.topology_include_files) == sorted(extra)

    def test_run_pdb2gmx_multichain_without_declaration_raises(self, tmp_path, monkeypatch):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "ALA", 1, 0.0, 0.0, 0.0))
        monkeypatch.setattr(
            "mdfactory.setup.protein.subprocess.run",
            _make_pdb2gmx_run(["Protein_chain_H", "Protein_chain_L"]),
        )
        _patch_pdb2gmx_env(monkeypatch)

        with pytest.raises(ValueError, match="none were declared"):
            run_pdb2gmx(
                pdb_path=pdb,
                config=Pdb2gmxConfig(forcefield="charmm36m", water_model="tip3p"),
                disulfide_bonds=[],
                protonation_states={},
                output_dir=tmp_path / "out",
            )

    def test_run_pdb2gmx_chain_mismatch_raises(self, tmp_path, monkeypatch):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(_pdb_atom(1, "N", "ALA", 1, 0.0, 0.0, 0.0))
        monkeypatch.setattr(
            "mdfactory.setup.protein.subprocess.run",
            _make_pdb2gmx_run(["Protein_chain_H", "Protein_chain_L", "Protein_chain_Y"]),
        )
        _patch_pdb2gmx_env(monkeypatch)

        with pytest.raises(ValueError, match="do not match"):
            run_pdb2gmx(
                pdb_path=pdb,
                config=Pdb2gmxConfig(forcefield="charmm36m", water_model="tip3p"),
                disulfide_bonds=[],
                protonation_states={},
                output_dir=tmp_path / "out",
                chains=["H", "L"],
            )


class TestBundleForcefield:
    def test_copies_ff_dir_and_rewrites_absolute_includes(self, tmp_path):
        ff_source = tmp_path / "ffsource" / "charmm36-feb2026_cgenff-5.0.ff"
        ff_source.mkdir(parents=True)
        (ff_source / "forcefield.itp").write_text("; forcefield\n")
        (ff_source / "tip3p.itp").write_text("; water\n")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        top = build_dir / "topology.top"
        top.write_text(
            textwrap.dedent(f"""\
            #include "{ff_source / "forcefield.itp"}"
            #include "protein.itp"
            #include "{ff_source / "tip3p.itp"}"
            #include "posre.itp"
            """)
        )

        ff_name = bundle_forcefield_into_topology(top)

        assert ff_name == "charmm36-feb2026_cgenff-5.0.ff"
        assert (build_dir / ff_name / "forcefield.itp").is_file()
        assert (build_dir / ff_name / "tip3p.itp").is_file()

        content = top.read_text()
        assert '#include "charmm36-feb2026_cgenff-5.0.ff/forcefield.itp"' in content
        assert '#include "charmm36-feb2026_cgenff-5.0.ff/tip3p.itp"' in content
        # Relative includes are left untouched.
        assert '#include "protein.itp"' in content
        assert '#include "posre.itp"' in content
        # No absolute paths remain.
        assert str(ff_source) not in content

    def test_copies_ff_dir_once_for_multiple_includes(self, tmp_path, monkeypatch):
        ff_source = tmp_path / "ffsource" / "charmm36-feb2026_cgenff-5.0.ff"
        ff_source.mkdir(parents=True)
        for name in ("forcefield.itp", "tip3p.itp", "ions.itp"):
            (ff_source / name).write_text(f"; {name}\n")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        top = build_dir / "topology.top"
        top.write_text(
            textwrap.dedent(f"""\
            #include "{ff_source / "forcefield.itp"}"
            #include "{ff_source / "tip3p.itp"}"
            #include "{ff_source / "ions.itp"}"
            """)
        )

        import mdfactory.setup.protein as protein_module

        calls = []
        real_copytree = protein_module.shutil.copytree

        def counting_copytree(src, dst, *args, **kwargs):
            calls.append((src, dst))
            return real_copytree(src, dst, *args, **kwargs)

        monkeypatch.setattr(protein_module.shutil, "copytree", counting_copytree)

        bundle_forcefield_into_topology(top)

        # Three includes into the same .ff directory, copied exactly once.
        assert len(calls) == 1
        content = top.read_text()
        assert '#include "charmm36-feb2026_cgenff-5.0.ff/ions.itp"' in content

    def test_no_absolute_ff_include_returns_none(self, tmp_path):
        top = tmp_path / "topology.top"
        top.write_text('#include "protein.itp"\n#include "posre.itp"\n')
        original = top.read_text()

        assert bundle_forcefield_into_topology(top) is None
        assert top.read_text() == original


class TestMetadata:
    def test_proteinbox_metadata(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        data = {
            "simulation_type": "proteinbox",
            "parametrization": "pdb2gmx",
            "system": {
                "protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)},
                "padding": 12.0,
            },
        }
        inp = BuildInput(**data)
        meta = inp.metadata
        assert meta["simulation_type"] == "proteinbox"
        assert meta["total_count"] == 1
        assert "padding" in meta["system_specific"]
        assert meta["system_specific"]["padding"] == 12.0


class TestPdbPathResolution:
    def test_relative_pdb_path_resolved_against_yaml_dir(self, tmp_path):
        base = tmp_path / "specs"
        base.mkdir()
        dct = {
            "simulation_type": "proteinbox",
            "system": {"protein": {"pdb_path": "1aki.pdb"}},
        }
        _resolve_proteinbox_pdb_path(dct, base)
        assert dct["system"]["protein"]["pdb_path"] == str((base / "1aki.pdb").resolve())

    def test_absolute_pdb_path_unchanged(self, tmp_path):
        abs_path = str((tmp_path / "abs.pdb").resolve())
        dct = {
            "simulation_type": "proteinbox",
            "system": {"protein": {"pdb_path": abs_path}},
        }
        _resolve_proteinbox_pdb_path(dct, tmp_path / "other")
        assert dct["system"]["protein"]["pdb_path"] == abs_path

    def test_non_proteinbox_untouched(self, tmp_path):
        dct = {"simulation_type": "mixedbox", "system": {"foo": "bar"}}
        _resolve_proteinbox_pdb_path(dct, tmp_path)
        assert dct == {"simulation_type": "mixedbox", "system": {"foo": "bar"}}

    def test_run_build_from_file_resolves_relative_path(self, tmp_path, monkeypatch):
        pdb = tmp_path / "1aki.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        yaml_file = tmp_path / "build.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            simulation_type: proteinbox
            parametrization: pdb2gmx
            system:
              protein:
                resname: LYZ
                count: 1
                pdb_path: 1aki.pdb
              padding: 12.0
            """)
        )
        captured = {}
        monkeypatch.setattr(workflows, "run_build_from_dict", captured.update)
        workflows.run_build_from_file(yaml_file)
        assert captured["system"]["protein"]["pdb_path"] == str(pdb.resolve())


class TestCenterInCubicBox:
    def test_padding_guaranteed_on_every_face(self):
        # Asymmetric molecule: long in x, short in y/z, offset from the origin.
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [40.0, 5.0, 2.0],
                [10.0, -3.0, 8.0],
            ],
            dtype=float,
        )
        u = mda.Universe.empty(len(positions), trajectory=True)
        u.atoms.positions = positions
        padding = 12.0
        box_size = _center_in_cubic_box(u, padding)

        pos = u.atoms.positions
        bbox_min = pos.min(axis=0)
        bbox_max = pos.max(axis=0)
        # Every face is at least `padding` from the protein.
        assert bbox_min.min() >= padding - 1e-3
        assert (box_size - bbox_max.max()) >= padding - 1e-3
        # The longest-extent axis sits exactly `padding` from both faces.
        long_axis = int(np.argmax(bbox_max - bbox_min))
        assert abs(bbox_min[long_axis] - padding) < 1e-3
        assert abs((box_size - bbox_max[long_axis]) - padding) < 1e-3
