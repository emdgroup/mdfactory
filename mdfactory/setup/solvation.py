# ABOUTME: Solvation and ionization routines for MD simulation systems
# ABOUTME: Fills simulation boxes with water, removes clashes, and adds ions
"""Solvation and ionization routines for MD simulation systems."""

from itertools import product
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import distances


def get_water_boxes(wbox: mda.Universe, nx: int, ny: int, nz: int):
    """Generate a list of translated water box atom groups to form a larger box.
    The wbox universe should have its left bottom corner in [0, 0, 0].

    Parameters
    ----------
    wbox : mda.Universe
        The input water box as an MDAnalysis Universe object.
    nx : int
        Number of repetitions along the x-axis.
    ny : int
        Number of repetitions along the y-axis.
    nz : int
        Number of repetitions along the z-axis.

    Returns
    -------
    list of mda.AtomGroup
        List of AtomGroups, each corresponding to a translated copy of the water box.

    Notes
    -----
    - Residue IDs are incremented to ensure uniqueness across copies.

    """
    dx, dy, dz, *_ = wbox.dimensions
    universes = []
    nwater = len(wbox.residues)
    count = 0
    for xx, yy, zz in product(range(nx), range(ny), range(nz)):
        wbox_copy = wbox.copy()
        wbox_copy.residues.resids += count * nwater
        wbox_copy.atoms.translate((xx * dx, yy * dy, zz * dz))
        universes.append(wbox_copy.atoms)
        count += 1
    return universes


def solvate(system: mda.Universe, prune_in_z=False, remove_sphere=None) -> mda.Universe:
    """Fill a rectangular simulation box with water and remove clashing molecules.

    Parameters
    ----------
    system : mda.Universe
        The solute system with box dimensions set. Must have a rectangular box
        (90-degree angles).
    prune_in_z : bool, optional
        If True, remove water molecules whose z-coordinates fall between the
        lipid layer boundaries (offset by 10 A inward). Default is False.
    remove_sphere : tuple of float or None, optional
        If provided, a tuple (cx, cy, cz, r) specifying the center and radius
        of a sphere within which water molecules are removed.

    Returns
    -------
    mda.Universe
        Solvated system with clashing and out-of-bounds waters removed.
        Also written to ``solvated.pdb``.

    Raises
    ------
    ValueError
        If the system box is not rectangular.

    """
    if not np.array_equal(system.dimensions[3:], [90.0, 90.0, 90.0]):
        raise ValueError("System box must be rectangular.")
    wbox = mda.Universe(Path(__file__).parent / "data" / "spc216.gro")
    wbox.add_TopologyAttr("chainID")
    wbox.atoms.chainIDs = "W"
    dx, dy, dz, *_ = wbox.dimensions
    wbox_dimensions = np.array([dx, dy, dz])
    mins = wbox.atoms.positions.min(axis=0)
    wbox.atoms.translate(-1.0 * mins)

    system_box = system.dimensions[:3]

    nbox = np.ceil(system_box / wbox_dimensions).astype(int)

    wb = get_water_boxes(wbox, *nbox)

    solv = mda.Merge(system.atoms, *wb)
    solv.dimensions = [*system_box, 90, 90, 90]

    if prune_in_z:
        lipids = system.select_atoms("not water and not resname NA and not resname CL")
        lipid_z = lipids.positions[:, 2]
        min_z = np.min(lipid_z) + 10
        max_z = np.max(lipid_z) - 10
    else:
        max_z = -1000000
        min_z = -max_z

    # remove waters outside of box
    outside_box = solv.select_atoms(
        f"water and same residue as (prop x >= {system_box[0]} or prop y >= {system_box[1]} or prop z >= {system_box[2]})"  # noqa: E501
    )
    # prune in z
    prune_z = solv.select_atoms(
        f"water and same residue as (prop z <= {max_z} and prop z >= {min_z})"
    )
    # remove clashes
    rad = 2.4
    print("Removing clashes with radius", rad)
    clashes = solv.select_atoms(f"water and same residue as around {rad} (not water)")
    # sphere
    sphere_indices = set()
    if remove_sphere is not None:
        cx, cy, cz, r = remove_sphere
        sphere = solv.select_atoms(f"water and same residue as point {cx} {cy} {cz} {r}")
        sphere_indices = set(sphere.atoms.indices)

    all_atom_indices = set(range(solv.atoms.n_atoms))
    atoms_keep = sorted(
        all_atom_indices
        - set(outside_box.atoms.indices)
        - set(prune_z.atoms.indices)
        - set(clashes.atoms.indices)
        - sphere_indices
    )
    in_box = mda.Merge(solv.atoms[atoms_keep])
    in_box.dimensions = [*system_box, 90, 90, 90]

    in_box = remove_clashes(in_box)
    in_box.dimensions = [*system_box, 90, 90, 90]
    in_box.atoms.write("solvated.pdb", bonds=None)
    return in_box


def remove_clashes(in_box, selection="water"):
    """Remove residues involved in inter-residue atomic clashes.

    Find atom pairs within 0.5 A that belong to different residues in the
    given selection and remove both entire residues for each clashing pair.

    Parameters
    ----------
    in_box : mda.Universe
        The system to check for clashes.
    selection : str, optional
        MDAnalysis selection string identifying atoms to check.
        Default is ``"water"``.

    Returns
    -------
    mda.Universe
        System with clashing residues removed, preserving box dimensions.

    """
    water = in_box.select_atoms(selection)
    water_indices = water.atoms.indices
    pairs = distances.capped_distance(
        water.positions, water.positions, 0.5, box=in_box.dimensions, return_distances=False
    )
    atoms_to_remove = set()
    atoms = water.atoms
    for pair in pairs:
        if atoms[pair[0]].residue != atoms[pair[1]].residue:
            um = water.select_atoms(
                f"same residue as index {water_indices[pair[0]]} or "
                f"same residue as index {water_indices[pair[1]]}"
            )
            assert len(set(um.atoms.indices)) == 6
            atoms_to_remove.update(um.atoms.indices)

    all_atom_indices = set(range(in_box.atoms.n_atoms))
    atoms_keep = sorted(all_atom_indices - atoms_to_remove)
    dims = in_box.dimensions
    in_box = mda.Merge(in_box.atoms[atoms_keep])
    in_box.dimensions = dims
    return in_box


def ionize(u: mda.Universe, num_na: int, num_cl: int, seed: int = None, min_distance=5.0):
    """Replace water molecules in a Universe with Na+ and Cl- ions.

    This function randomly selects water molecules in the given MDAnalysis Universe and replaces
    them with sodium (Na+) and chloride (Cl-) ions, ensuring a
    minimum distance from non-water atoms.
    The replacement is performed by renaming the oxygen atom of the selected water residue and
    removing the hydrogen atoms.
    The resulting Universe contains the original atoms (except the removed hydrogens)
    and the newly created ions.

    Parameters
    ----------
    u : mda.Universe
        The MDAnalysis Universe containing the system to be ionized.
    num_na : int
        Number of sodium (Na+) ions to introduce.
    num_cl : int
        Number of chloride (Cl-) ions to introduce.
    seed : int, optional
        Random seed for reproducibility.
    min_distance : float, optional
        Minimum distance (in Å) from non-water atoms for water molecules to be
        considered for replacement. Default is 5.0 Å.

    Returns
    -------
    newuniverse : mda.Universe
        A new Universe object with the specified number of water
        molecules replaced by Na+ and Cl- ions.

    Raises
    ------
    ValueError
        If there are not enough water molecules to replace with the requested
        number of ions, or if a selected residue does not contain an oxygen atom.

    Notes
    -----
    - The function assumes that the Universe contains standard water residues
        and atom naming conventions.

    """
    if seed is not None:
        np.random.seed(seed)
    u = mda.Merge(u.atoms)
    water = u.select_atoms(
        f"same residue as (water and not around {min_distance} (not water))"
    ).residues
    n_water = len(water)

    n_ions = num_na + num_cl

    if n_water <= n_ions:
        raise ValueError("Not enough water to ionize.")

    selected_residues = np.random.choice(list(water), n_ions, replace=False)
    assert len(selected_residues) == len(set(selected_residues))
    na_residues = selected_residues[:num_na]
    cl_residues = selected_residues[num_na:]

    def replace_water_with_ion(residue, ion_name: str, ion_type: str):
        # Get the oxygen atom (will become the ion)
        oxygen_atom = residue.atoms.select_atoms("name O*")

        if oxygen_atom.n_atoms != 1:
            raise ValueError(f"No oxygen atom found in residue {residue.resid}")

        residue.resname = ion_name

        oxygen_atom[0].name = ion_name
        oxygen_atom[0].type = ion_type
        # oxygen_atom[0].position = residue.atoms.center_of_geometry()

        hydrogenatoms = residue.atoms.select_atoms("name H*")
        assert len(hydrogenatoms) == 2

        # Store indices of atoms to remove
        atomstoremove = []
        for h_atom in hydrogenatoms:
            atomstoremove.append(h_atom.index)

        return atomstoremove, oxygen_atom[0].index

    # Keep track of all atoms to remove
    all_atoms_to_remove = []

    cation_indices = []
    # Replace water molecules with Na+ ions
    for residue in na_residues:
        atoms_to_remove, idx_na = replace_water_with_ion(residue, "NA", "Na+")
        cation_indices.append(idx_na)
        all_atoms_to_remove.extend(atoms_to_remove)

    anion_indices = []
    # Replace water molecules with Cl- ions
    for residue in cl_residues:
        atoms_to_remove, idx_cl = replace_water_with_ion(residue, "CL", "Cl-")
        anion_indices.append(idx_cl)
        all_atoms_to_remove.extend(atoms_to_remove)

    allatomindices = set(range(u.atoms.n_atoms))
    atomstokeepindices = sorted(
        allatomindices - set(all_atoms_to_remove) - set(anion_indices) - set(cation_indices)
    )
    atomstokeep = u.atoms[atomstokeepindices]
    ion_groups = []
    if cation_indices:
        cations = u.atoms[sorted(cation_indices)]
        cations.residues.resids = np.arange(1, cations.n_atoms + 1)
        ion_groups.append(cations)
    if anion_indices:
        anions = u.atoms[sorted(anion_indices)]
        anions.residues.resids = np.arange(1, anions.n_atoms + 1)
        ion_groups.append(anions)
    newuniverse = mda.Merge(atomstokeep, *ion_groups)

    n_atoms = len(newuniverse.atoms)
    n_atoms_orig = len(u.atoms)
    assert n_atoms == n_atoms_orig - 2 * n_ions

    return newuniverse
