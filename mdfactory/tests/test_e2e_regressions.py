# ABOUTME: Regression tests for build failures found in the protein_mixedbox end-to-end run:
# ABOUTME: compression overshoot rescaling, LNP leniency, CGenFF atom names, net-charge grompp.
"""Regression tests for the protein_mixedbox end-to-end findings."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from mdfactory.models.species import SingleMoleculeSpecies
from mdfactory.setup import protein
from mdfactory.simulation.openmm_utils import finish_compression, rescale_to_volume
from mdfactory.utils.setup_utilities import topology_atom_names
from mdfactory.utils.topology_utilities import read_itp_atoms

ETHANOL_ITP = """\
[ moleculetype ]
; Name            nrexcl
MOL               3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass
; residue   1 XXX rtp XXX  q +0.0
     1      CG331      1    XXX     C1      1     -0.271     12.011   ; qtot -0.271
     2      CG321      1    XXX     C2      2      0.052     12.011
     3      OG311      1    XXX     O3      3     -0.651    15.9994
     4       HGA3      1    XXX     H4      4       0.09      1.008
     5       HGA3      1    XXX     H5      5       0.09      1.008
     6       HGA3      1    XXX     H6      6       0.09      1.008
     7       HGA2      1    XXX     H7      7       0.09      1.008
     8       HGA2      1    XXX     H8      8       0.09      1.008
     9       HGP1      1    XXX     H9      9       0.42      1.008

[ bonds ]
    1     2     1
"""


def _three_dimer_context(edge_nm=3.0):
    """Return a context with three bonded dimers in a cubic box of the given edge."""
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(edge_nm, 0, 0), mm.Vec3(0, edge_nm, 0), mm.Vec3(0, 0, edge_nm)
    )
    bonds = mm.HarmonicBondForce()
    positions = []
    for i, center in enumerate(([0.5, 0.5, 0.5], [2.0, 1.0, 1.5], [1.0, 2.5, 2.0])):
        for offset in (-0.05, 0.05):
            system.addParticle(12.0)
            positions.append([center[0] + offset, center[1], center[2]])
        bonds.addBond(2 * i, 2 * i + 1, 0.1, 1000.0)
    system.addForce(bonds)
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(np.array(positions) * unit.nanometer)
    return context


def _positions(context):
    return (
        context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    )


def _volume(context):
    a, b, c = context.getState().getPeriodicBoxVectors()
    return (a[0] * b[1] * c[2]).value_in_unit(unit.nanometer**3)


class TestRescaleToVolume:
    def test_reaches_target_and_keeps_molecules_and_rigid_unit_intact(self):
        context = _three_dimer_context()
        before = _positions(context)
        target = 27.0 * 1.06

        rescale_to_volume(context, target, rigid_indices=[2, 3, 4, 5])

        after = _positions(context)
        assert _volume(context) == pytest.approx(target)
        for i in range(3):
            assert np.linalg.norm(after[2 * i] - after[2 * i + 1]) == pytest.approx(
                np.linalg.norm(before[2 * i] - before[2 * i + 1])
            )
        # dimers 1 and 2 form one rigid unit: their relative placement is unchanged
        assert after[4] - after[2] == pytest.approx(before[4] - before[2])
        # an independent molecule follows the box scaling
        assert np.linalg.norm(after[0] - after[2]) > np.linalg.norm(before[0] - before[2])

    def test_refuses_to_shrink(self):
        context = _three_dimer_context()
        with pytest.raises(ValueError, match="only expand"):
            rescale_to_volume(context, 20.0)


class TestFinishCompression:
    def test_corrects_volume_overshoot_within_tolerance(self):
        context = _three_dimer_context()  # 27 nm³, i.e. 4.5% below a 28.27 nm³ target
        overshoot = finish_compression(
            context,
            28.27,
            density_target=False,
            converged=True,
            require_convergence=True,
            tolerance=0.15,
        )
        assert overshoot == pytest.approx(1 - 27.0 / 28.27)
        assert _volume(context) == pytest.approx(28.27)

    def test_corrects_density_overshoot(self):
        context = _three_dimer_context()
        finish_compression(
            context,
            27.0 * 1.031,
            density_target=True,
            converged=True,
            require_convergence=True,
            tolerance=0.15,
        )
        assert _volume(context) == pytest.approx(27.0 * 1.031)

    def test_raises_beyond_tolerance(self):
        context = _three_dimer_context()
        with pytest.raises(RuntimeError, match="overshot the target volume"):
            finish_compression(
                context,
                40.0,
                density_target=False,
                converged=True,
                require_convergence=True,
                tolerance=0.15,
            )

    def test_unreached_target_raises_or_warns(self, capsys):
        context = _three_dimer_context()
        with pytest.raises(RuntimeError, match="did not reach"):
            finish_compression(
                context,
                20.0,
                density_target=True,
                converged=False,
                require_convergence=True,
                tolerance=0.15,
                not_reached_message="did not reach",
            )
        result = finish_compression(
            context,
            20.0,
            density_target=True,
            converged=False,
            require_convergence=False,
            tolerance=0.15,
            not_reached_message="did not reach",
        )
        assert result is None
        assert _volume(context) == pytest.approx(27.0)
        assert "WARNING" in capsys.readouterr().out


class TestTopologyAtomNames:
    def test_read_itp_atoms(self, tmp_path):
        itp = tmp_path / "mol.itp"
        itp.write_text(ETHANOL_ITP)
        atoms = read_itp_atoms(itp)
        assert [name for name, _ in atoms] == [
            "C1", "C2", "O3", "H4", "H5", "H6", "H7", "H8", "H9",
        ]  # fmt: skip
        assert atoms[2][1] == pytest.approx(15.9994)

    def test_cgenff_solute_uses_itp_names(self, tmp_path):
        itp = tmp_path / "mol.itp"
        itp.write_text(ETHANOL_ITP)
        species = SingleMoleculeSpecies(smiles="CCO", resname="ETH", count=1)
        params = SimpleNamespace(itp=itp, parameter_itp=tmp_path / "extra.itp")
        assert topology_atom_names(species, params) == [
            "C1", "C2", "O3", "H4", "H5", "H6", "H7", "H8", "H9",
        ]  # fmt: skip

    def test_mismatched_atom_order_raises(self, tmp_path):
        itp = tmp_path / "mol.itp"
        itp.write_text(ETHANOL_ITP.replace("12.011", "15.9994", 1))
        species = SingleMoleculeSpecies(smiles="CCO", resname="ETH", count=1)
        params = SimpleNamespace(itp=itp, parameter_itp=tmp_path / "extra.itp")
        with pytest.raises(ValueError, match="atom orders differ"):
            topology_atom_names(species, params)

    def test_native_and_parameterless_species(self, tmp_path):
        water = SingleMoleculeSpecies(smiles="O", resname="SOL", count=1)
        native = SimpleNamespace(itp=tmp_path / "tip3p.itp", parameter_itp=None)
        assert topology_atom_names(water, native) == ["OW", "HW1", "HW2"]
        ethanol = SingleMoleculeSpecies(smiles="CCO", resname="ETH", count=1)
        assert topology_atom_names(ethanol, native) is None


NET_CHARGE_OUTPUT = """\
NOTE 1 [file em.mdp]:
  something harmless

WARNING 1 [file topology.top, line 18492]:
  You are using Ewald electrostatics in a system with net charge. This can
  lead to severe artifacts, such as ions moving into regions with low
  dielectric, due to the uniform background charge.

There was 1 WARNING
"""

NAME_MISMATCH_OUTPUT = """\
WARNING 1 [file topology.top, line 18499]:
  1498 non-matching atom names
  atom names from topology.top will be used

WARNING 2 [file topology.top, line 18492]:
  You are using Ewald electrostatics in a system with net charge. This can
  lead to severe artifacts.

There were 2 WARNINGs
"""


class TestValidateWithGromppNetCharge:
    @pytest.fixture
    def fake_grompp(self, monkeypatch):
        """Mimic grompp: fail when the warnings in *output* exceed -maxwarn."""
        calls = []

        def install(output):
            n_warnings = output.count("\nWARNING ") + output.startswith("WARNING ")

            def fake_run(cmd, **_kwargs):
                maxwarn = int(cmd[cmd.index("-maxwarn") + 1])
                calls.append(maxwarn)
                code = 1 if n_warnings > maxwarn else 0
                return subprocess.CompletedProcess(cmd, code, stdout="", stderr=output)

            monkeypatch.setattr(protein.subprocess, "run", fake_run)
            monkeypatch.setattr(protein, "check_gmx_available", lambda: Path("/mock/gmx"))
            return calls

        return install

    def test_net_charge_warning_rejected_by_default(self, fake_grompp, tmp_path):
        calls = fake_grompp(NET_CHARGE_OUTPUT)
        with pytest.raises(RuntimeError, match="grompp validation failed"):
            protein.validate_with_grompp(Path("t.top"), Path("s.pdb"), Path("em.mdp"), tmp_path)
        assert calls == [0]

    def test_net_charge_warning_accepted_when_allowed(self, fake_grompp, tmp_path):
        calls = fake_grompp(NET_CHARGE_OUTPUT)
        protein.validate_with_grompp(
            Path("t.top"), Path("s.pdb"), Path("em.mdp"), tmp_path, allow_net_charge=True
        )
        # one grompp call only: a failing gmx_mpi call can abort an enclosing srun step
        assert calls == [1]

    def test_allowed_but_neutral_system_passes(self, fake_grompp, tmp_path):
        calls = fake_grompp("There were 0 warnings\n")
        protein.validate_with_grompp(
            Path("t.top"), Path("s.pdb"), Path("em.mdp"), tmp_path, allow_net_charge=True
        )
        assert calls == [1]

    def test_other_warning_rejected_even_within_maxwarn(self, fake_grompp, tmp_path):
        fake_grompp(NAME_MISMATCH_OUTPUT.split("\nWARNING 2")[0] + "\n")
        with pytest.raises(RuntimeError, match="other than the accepted net-charge"):
            protein.validate_with_grompp(
                Path("t.top"), Path("s.pdb"), Path("em.mdp"), tmp_path, allow_net_charge=True
            )

    def test_net_charge_plus_other_warning_fails(self, fake_grompp, tmp_path):
        fake_grompp(NAME_MISMATCH_OUTPUT)
        with pytest.raises(RuntimeError, match="grompp validation failed"):
            protein.validate_with_grompp(
                Path("t.top"), Path("s.pdb"), Path("em.mdp"), tmp_path, allow_net_charge=True
            )
