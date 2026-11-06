# ABOUTME: Opt-in real-GROMACS test for the Amber ff14SB + OpenFF Sage protein topology
# ABOUTME: Runs pdb2gmx amber14sb, a Sage solute composite topology, and grompp -maxwarn 0
"""Opt-in real-GROMACS test of the Amber ff14SB + OpenFF Sage protein route.

Skipped unless a GROMACS >= 2026 binary is available (configured or on PATH).
The module name contains ``build``, so ``pytest -k "not build"`` (CI) skips it.
"""

import shutil
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from mdfactory.build import _protein_solution_parametrizer, _stage_protein_topology
from mdfactory.models.parametrization import AmberConfig
from mdfactory.models.species import SolutionSpecies
from mdfactory.run_schedules import RunScheduleManager
from mdfactory.settings import settings
from mdfactory.setup import protein
from mdfactory.utils.setup_utilities import topology_atom_names

# Lys1-Val2-Phe3 heavy atoms from PDB 1AKI (hen egg-white lysozyme).
PEPTIDE_PDB = """\
ATOM      1  N   LYS A   1      35.365  22.342 -11.980  1.00 22.28           N
ATOM      2  CA  LYS A   1      35.892  21.073 -11.427  1.00 21.12           C
ATOM      3  C   LYS A   1      34.741  20.264 -10.844  1.00 16.85           C
ATOM      4  O   LYS A   1      33.945  20.813 -10.081  1.00 18.94           O
ATOM      5  CB  LYS A   1      36.872  21.435 -10.306  1.00 20.78           C
ATOM      6  CG  LYS A   1      37.453  20.248  -9.565  1.00 18.47           C
ATOM      7  CD  LYS A   1      38.688  20.649  -8.775  1.00 20.32           C
ATOM      8  CE  LYS A   1      39.057  19.508  -7.837  1.00 24.76           C
ATOM      9  NZ  LYS A   1      40.423  19.771  -7.299  1.00 28.27           N
ATOM     10  N   VAL A   2      34.739  18.961 -11.042  1.00 19.96           N
ATOM     11  CA  VAL A   2      33.903  17.998 -10.333  1.00 18.10           C
ATOM     12  C   VAL A   2      34.800  17.312  -9.294  1.00 19.39           C
ATOM     13  O   VAL A   2      35.759  16.605  -9.665  1.00 22.14           O
ATOM     14  CB  VAL A   2      33.140  17.034 -11.232  1.00 16.81           C
ATOM     15  CG1 VAL A   2      32.251  16.084 -10.434  1.00 21.92           C
ATOM     16  CG2 VAL A   2      32.294  17.714 -12.290  1.00 19.46           C
ATOM     17  N   PHE A   3      34.491  17.546  -8.038  1.00 19.89           N
ATOM     18  CA  PHE A   3      35.185  16.903  -6.918  1.00 17.43           C
ATOM     19  C   PHE A   3      34.742  15.441  -6.771  1.00 15.70           C
ATOM     20  O   PHE A   3      33.525  15.162  -6.862  1.00 18.52           O
ATOM     21  CB  PHE A   3      34.967  17.632  -5.594  1.00 17.94           C
ATOM     22  CG  PHE A   3      35.944  18.737  -5.375  1.00 16.78           C
ATOM     23  CD1 PHE A   3      35.666  20.050  -5.798  1.00 15.97           C
ATOM     24  CD2 PHE A   3      37.000  18.557  -4.473  1.00 19.95           C
ATOM     25  CE1 PHE A   3      36.577  21.076  -5.568  1.00 17.32           C
ATOM     26  CE2 PHE A   3      37.869  19.589  -4.157  1.00 17.65           C
ATOM     27  CZ  PHE A   3      37.636  20.873  -4.666  1.00 17.91           C
END
"""

BOX_A = 40.0


def _new_enough_gmx():
    try:
        gmx = protein.check_gmx_available()
        version, _ = protein.get_gmx_version(gmx)
    except RuntimeError:
        return None
    return gmx if version >= (2026, 0) else None


pytestmark = pytest.mark.skipif(_new_enough_gmx() is None, reason="needs GROMACS >= 2026")


def _solution_universe(species, names, offsets):
    """Place each species' RDKit conformer at an offset; use the topology atom names."""
    positions, atom_names, resindex, resnames = [], [], [], []
    for i, (spec, spec_names, offset) in enumerate(zip(species, names, offsets)):
        xyz = spec.rdkit_molecule.GetConformer().GetPositions()
        positions.append(xyz - xyz.mean(axis=0) + offset)
        atom_names += spec_names
        resindex += [i] * len(spec_names)
        resnames.append(spec.resname)
    u = mda.Universe.empty(
        len(atom_names), n_residues=len(species), atom_resindex=resindex, trajectory=True
    )
    u.add_TopologyAttr("names", atom_names)
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("resids", np.arange(1, len(species) + 1))
    u.atoms.positions = np.vstack(positions)
    return u


def test_amber_sage_composite_topology_passes_grompp_build(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "parameter_store", tmp_path / "parameters")
    pdb = tmp_path / "peptide.pdb"
    pdb.write_text(PEPTIDE_PDB)
    config = AmberConfig()

    params = protein.run_pdb2gmx(
        pdb_path=pdb,
        config=config,
        disulfide_bonds=[],
        protonation_states={},
        output_dir=tmp_path / "pdb2gmx",
    )
    ff_name = protein.bundle_forcefield_into_topology(params.topology_file)
    assert ff_name == "amber14sb.ff"
    ff_dir = params.topology_file.parent / ff_name

    _, parametrize = _protein_solution_parametrizer(config, ff_dir)
    species = [
        SolutionSpecies(smiles="CCO", resname="ETH", count=1),
        SolutionSpecies(smiles="O", resname="SOL", count=1),
    ]
    parameters = [parametrize(spec) for spec in species]
    assert parameters[0].parameter_itp is not None
    assert parameters[1].itp == ff_dir / "tip3p.itp"

    build_dir = tmp_path / "system"
    build_dir.mkdir()
    topology = _stage_protein_topology(params, ff_dir, build_dir, species, parameters, [1, 1])

    u_protein = mda.Universe(str(params.structure_file))
    u_protein.atoms.translate(np.full(3, BOX_A / 2) - u_protein.atoms.center_of_geometry())
    solution = _solution_universe(
        species,
        [topology_atom_names(spec, par) for spec, par in zip(species, parameters)],
        [np.array([5.0, 5.0, 5.0]), np.array([35.0, 35.0, 35.0])],
    )
    u = mda.Merge(u_protein.atoms, solution.atoms)
    u.dimensions = [BOX_A, BOX_A, BOX_A, 90, 90, 90]
    structure = build_dir / "system.gro"
    u.atoms.write(str(structure))
    # OpenMM's PDB reader drops atoms whose name repeats within a residue, so the
    # Sage solute must carry unique names for the OpenMM compression/relaxation.
    from openmm import app

    pdb_out = build_dir / "system.pdb"
    u.atoms.write(str(pdb_out))
    assert app.PDBFile(str(pdb_out)).topology.getNumAtoms() == u.atoms.n_atoms

    mdp = RunScheduleManager().get_all_run_file_paths(
        "gromacs", "protein_mixedbox", parametrization="amber"
    )["em.mdp"]
    shutil.copy(mdp, build_dir / "em.mdp")

    text = topology.read_text()
    for include in ("forcefield.itp", "tip3p.itp", "ions_tip3p.itp"):
        assert f'#include "amber14sb.ff/{include}"' in text
    assert f'#include "{parameters[0].itp.name}"' in text
    extra = (build_dir / "extra_params.itp").read_text()
    assert "[ defaults ]" not in extra and "ETH_0" in extra

    # The peptide carries a net charge (+1); accept only grompp's net-charge warning.
    protein.validate_with_grompp(
        topology, structure, build_dir / "em.mdp", build_dir, allow_net_charge=True
    )
    assert Path(build_dir / "topology.top").is_file()
