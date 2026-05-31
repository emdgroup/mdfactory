# ABOUTME: Tests for proteinbox models, topology parsing, and YAML validation
# ABOUTME: Covers ProteinSpecies, ProteinBoxComposition, Pdb2gmxConfig, and BuildInput
"""Tests for proteinbox models, topology parsing, and YAML validation."""

import textwrap
from pathlib import Path

import pytest
import yaml

from mdfactory.models.composition import ProteinBoxComposition
from mdfactory.models.input import BuildInput
from mdfactory.models.parametrization import Pdb2gmxConfig
from mdfactory.models.species import ProteinSpecies
from mdfactory.setup.protein import (
    _apply_protonation_states,
    _build_disulfide_prompt_input,
    _sum_charges_from_itp,
    check_forcefield_available,
    run_pdb2gmx,
    update_topology_molecules,
)


def _pdb_atom(serial, atom_name, resname, resid, x, y, z):
    return (
        f"ATOM  {serial:5d} {atom_name:<4s} {resname:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    )


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
            disulfide_bonds=[(6, 127), (30, 115)],
            protonation_states={"HIS15": "HIE", "GLU35": "GLH"},
        )
        assert len(spec.disulfide_bonds) == 2
        assert spec.disulfide_bonds[0] == (6, 127)
        assert spec.protonation_states["HIS15"] == "HIE"

    def test_charge_not_precomputable(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        spec = ProteinSpecies(resname="LYZ", pdb_path=pdb)
        with pytest.raises(NotImplementedError, match="pdb2gmx"):
            _ = spec.charge

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
            box_padding=12.0,
        )
        assert comp.box_padding == 12.0
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
        assert config.ignh is True
        assert config.merge_all is False

    def test_custom(self):
        config = Pdb2gmxConfig(forcefield="amber99sb-ildn", water_model="spc")
        assert config.forcefield == "amber99sb-ildn"
        assert config.water_model == "spc"

    def test_frozen(self):
        config = Pdb2gmxConfig()
        with pytest.raises(Exception):
            config.forcefield = "other"


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
  ignh: true
system:
  protein:
    resname: LYZ
    count: 1
    pdb_path: {pdb}
    disulfide_bonds:
      - [6, 127]
      - [30, 115]
    protonation_states:
      HIS15: HIE
  box_padding: 12.0
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
        with pytest.raises(ValueError, match="cannot be written to PDB"):
            _apply_protonation_states(pdb, {"GLU35": "GLUP"}, tmp_path)

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
        assert _build_disulfide_prompt_input(pdb, [(2, 3)]) == "n\ny\n"

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
            _build_disulfide_prompt_input(pdb, [(6, 127)])

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
            (output_dir / "topol.top").write_text("[ atoms ]\n")
            (output_dir / "posre.itp").write_text("mock posre\n")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr("mdfactory.setup.protein.subprocess.run", fake_run)
        monkeypatch.setattr(
            "mdfactory.setup.protein.check_gmx_available", lambda: Path("/mock/gmx")
        )
        monkeypatch.setattr("mdfactory.setup.protein.check_forcefield_available", lambda ff: None)
        monkeypatch.setattr(
            "mdfactory.setup.protein.get_gromacs_env",
            lambda: {"GMXLIB": "/mock/forcefields"},
        )

        run_pdb2gmx(
            pdb_path=pdb,
            config=Pdb2gmxConfig(forcefield="amber99sb-ildn", water_model="tip3p"),
            disulfide_bonds=[(2, 3)],
            protonation_states={},
            output_dir=tmp_path / "pdb2gmx_output",
        )

        assert "-ss" in calls["cmd"]
        assert calls["cmd"][:2] == ["/mock/gmx", "pdb2gmx"]
        assert calls["input"] == "n\ny\n"
        assert calls["cwd"] == str((tmp_path / "pdb2gmx_output").resolve())
        assert calls["env"] == {"GMXLIB": "/mock/forcefields"}
        assert calls["check"] is False


class TestMetadata:
    def test_proteinbox_metadata(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        data = {
            "simulation_type": "proteinbox",
            "parametrization": "pdb2gmx",
            "system": {
                "protein": {"resname": "LYZ", "count": 1, "pdb_path": str(pdb)},
                "box_padding": 12.0,
            },
        }
        inp = BuildInput(**data)
        meta = inp.metadata
        assert meta["simulation_type"] == "proteinbox"
        assert meta["total_count"] == 1
        assert "box_padding" in meta["system_specific"]
        assert meta["system_specific"]["box_padding"] == 12.0
