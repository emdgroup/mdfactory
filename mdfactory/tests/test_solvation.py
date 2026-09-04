# ABOUTME: Tests for solvation setup including water box creation, ion placement,
# ABOUTME: and solvent distance validation around solute molecules.
"""Tests for solvation setup including water box creation, ion placement,."""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis import distances

from mdfactory.build import ionize_solvated_system
from mdfactory.models.composition import IonizationConfig
from mdfactory.models.species import SingleMoleculeSpecies
from mdfactory.setup import solvation


@pytest.fixture
def u():
    u = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH").universe
    pos = u.atoms.positions.copy()
    dims = pos.max(axis=0) - pos.min(axis=0) + 20.0
    u.dimensions = [*dims, 90.0, 90.0, 90.0]
    # put the molecule in the center of the box
    u.atoms.translate(0.5 * (dims - (pos.max(axis=0) + pos.min(axis=0))))
    return u


def test_solvate_ionize(u):
    dims = u.dimensions[:3]
    u_solv = solvation.solvate(u)
    np.testing.assert_array_equal(u.dimensions, u_solv.dimensions)
    assert len(u_solv.select_atoms("water")) > 0
    print("Number of waters:", len(u_solv.select_atoms("water").residues))

    sel_outside = u_solv.select_atoms(
        f"water and same residue as (prop x >= {dims[0]} or prop y >= {dims[1]} or prop z >= {dims[2]})"  # noqa: E501
    )
    assert len(sel_outside) == 0

    # check prune in z
    u2 = mda.Merge(u.atoms)
    u2.atoms.translate([0, 0, 30])
    u = mda.Merge(u.atoms, u2.atoms)
    u.dimensions = [dims[0], dims[1], dims[2] + 50, 90.0, 90.0, 90.0]
    u_solv = solvation.solvate(u, prune_in_z=True)
    np.testing.assert_array_equal(u.dimensions, u_solv.dimensions)

    u_solv_without = solvation.solvate(u, prune_in_z=False)
    n_waters_with = len(u_solv.select_atoms("water"))
    n_waters_without = len(u_solv_without.select_atoms("water"))
    assert n_waters_with < n_waters_without
    assert n_waters_with > 0
    assert n_waters_without > 0

    u_before = mda.Merge(u_solv.atoms)
    u_ion = solvation.ionize(u_solv, num_na=2, num_cl=2)
    n_na = len(u_ion.select_atoms("resname NA"))
    n_cl = len(u_ion.select_atoms("resname CL"))
    assert (u_before.atoms.names == u_solv.atoms.names).all()
    assert n_na == 2
    assert n_cl == 2

    with pytest.raises(ValueError, match="Not enough water"):
        solvation.ionize(u, num_na=2, num_cl=2)

    with pytest.raises(ValueError, match="System box must be rectangular."):
        u_bad = u.copy()
        u_bad.dimensions = [*dims, 90.0, 91.0, 90.0]
        solvation.solvate(u_bad)


def test_solvate_remove_sphere(u):
    center = u.atoms.center_of_geometry()
    radius = 5.0
    u_solv = solvation.solvate(u, remove_sphere=(*center, radius))
    u_solv.atoms.write("tmp.pdb")
    np.testing.assert_array_equal(u.dimensions, u_solv.dimensions)
    assert len(u_solv.select_atoms("water")) > 0

    assert len(u_solv.select_atoms("not water")) == len(u.atoms)

    # check that no water is in the sphere
    water = u_solv.select_atoms("water")
    water_in_sphere = water.select_atoms(
        f"same residue as point {center[0]} {center[1]} {center[2]} {radius}"
    )
    assert len(water_in_sphere) == 0

    # check that we get an error if the sphere is too large
    # with pytest.raises(ValueError, match="Not enough water"):
    #     solvation.solvate(u, remove_sphere=(*center, 1000.0))


def test_remove_clashes():
    u = mda.Universe(Path(solvation.__file__).parent / "data" / "spc216.gro")
    mins = u.atoms.positions.min(axis=0)
    u.atoms.translate(-1.0 * mins)

    waters = u.select_atoms("water")
    n_waters = len(waters)

    waters.residues[0].atoms.positions = waters.residues[1].atoms.positions
    u_clean = solvation.remove_clashes(u, selection="water")
    n_waters_clean = len(u_clean.select_atoms("water"))

    assert n_waters_clean < n_waters
    assert n_waters_clean == n_waters - 6

    # check everything works at PBC boundary
    u = mda.Universe(Path(solvation.__file__).parent / "data" / "spc216.gro")
    mins = u.atoms.positions.min(axis=0)
    u.atoms.translate(-1.0 * mins)
    water = u.select_atoms("water")
    water_indices = water.atoms.indices
    at = water.atoms

    clash_position = water.residues[215].atoms.positions - np.array(
        [u.dimensions[0], u.dimensions[1], u.dimensions[2]]
    )
    water.residues[52].atoms.positions = clash_position

    u_removed = solvation.remove_clashes(u, selection="water")
    n_waters_removed = len(u_removed.select_atoms("water"))
    assert 53 not in u_removed.select_atoms("water").residues.resids
    assert 216 not in u_removed.select_atoms("water").residues.resids
    assert n_waters_removed == len(water) - 6

    pairs_no_pbc = distances.capped_distance(
        water.positions, water.positions, 0.5, box=None, return_distances=False
    )
    clashes_no_pbc = set()
    for pair in pairs_no_pbc:
        if pair[0] > pair[1] and at[pair[0]].residue != at[pair[1]].residue:
            um = water.select_atoms(
                f"same residue as index {water_indices[pair[0]]} or "
                f"same residue as index {water_indices[pair[1]]}"
            )
            clashes_no_pbc.add(tuple(sorted(um.atoms.residues.resids)))
    assert len(clashes_no_pbc) == 0


def test_ionize_solvated_system_skips_when_no_ions():
    """Skip ionization when no ions should be added."""
    u = mda.Universe(Path(solvation.__file__).parent / "data" / "spc216.gro")
    n_atoms_before = len(u.atoms)

    ion_config = IonizationConfig(concentration=0.0, neutralize=False)
    u_out, additional_species = ionize_solvated_system(ion_config, u, total_charge=0)

    assert additional_species == []
    assert len(u_out.atoms) == n_atoms_before
    assert len(u_out.select_atoms("resname NA")) == 0
    assert len(u_out.select_atoms("resname CL")) == 0
