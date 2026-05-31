# ABOUTME: Tests for proteinbox models, topology parsing, and YAML validation
# ABOUTME: Covers ProteinSpecies, ProteinBoxComposition, Pdb2gmxConfig, and BuildInput
"""Tests for proteinbox models, topology parsing, and YAML validation."""

import textwrap
from pathlib import Path

import pytest
import yaml

from mdfactory.models.composition import ProteinBoxComposition
from mdfactory.models.input import BuildInput
from mdfactory.models.parametrization import (
    GromacsProteinParameterSet,
    Pdb2gmxConfig,
)
from mdfactory.models.species import ProteinSpecies
from mdfactory.setup.protein import (
    _sum_charges_from_itp,
    extract_charge_from_topology,
    update_topology_molecules,
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
        itp.write_text(textwrap.dedent("""\
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
        """))
        charge = _sum_charges_from_itp(itp)
        assert abs(charge - (-0.090)) < 1e-6

    def test_update_topology_molecules(self, tmp_path):
        top = tmp_path / "topol.top"
        top.write_text(textwrap.dedent("""\
            #include "charmm36m.ff/forcefield.itp"
            #include "protein.itp"

            [ system ]
            Protein in water

            [ molecules ]
            Protein_chain_A  1
        """))
        update_topology_molecules(top, n_water=5000, num_na=10, num_cl=18)
        content = top.read_text()
        assert "SOL" in content
        assert "5000" in content
        assert "NA" in content
        assert "10" in content
        assert "CL" in content
        assert "18" in content
