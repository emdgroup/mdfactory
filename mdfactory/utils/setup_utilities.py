# ABOUTME: System setup utilities for bilayer and mixedbox construction
# ABOUTME: Handles lipid structure generation, placement, and box packing
"""System setup utilities for bilayer and mixedbox construction."""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

import MDAnalysis as mda
import numpy as np
from loguru import logger
from rdkit import Chem, DistanceGeometry

if TYPE_CHECKING:
    from openff.toolkit import Topology
from rdkit.Chem import AllChem, rdDepictor, rdDistGeom
from scipy.spatial.distance import cdist

from mdfactory.models.composition import BilayerComposition
from mdfactory.models.input import MixedBoxComposition
from mdfactory.models.species import LipidSpecies
from mdfactory.utils.utilities import temporary_working_directory, working_directory

# Avogadro's number, matching the g<->Dalton conventions used in composition.py.
N_AVOGADRO = 6.022e23

# A protein_mixedbox is packed at 0.1x the final density so water and solutes have
# room to be placed around the fixed protein before compression squeezes the box
# down to its requested size (mirrors the mixedbox 0.1x loose-packing idiom).
PROTEIN_MIXEDBOX_DILUTION_FACTOR = 0.1


def generate_lipid_structure_legacy(
    smiles,
    head_indices,
    tail_indices,
    num_conformers=5,
):  # pragma: no cover
    raise DeprecationWarning("This function is deprecated. Use generate_lipid_structure instead.")
    # Generate the molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Compute 2D coordinates for visualization
    rdDepictor.Compute2DCoords(mol)

    # Generate bounds matrix
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)

    # Set the bounds for head and tail distances based on indices
    for head_index in head_indices:
        for tail_index in tail_indices:
            # Assign the maximum distance based on which index is larger
            if head_index > tail_index:
                max_head_tail_distance = bounds[tail_index, head_index]
                bounds[head_index, tail_index] = (
                    max_head_tail_distance - 5
                )  # Max distance between headgroup and tails
                # print(max_head_tail_distance)
            else:
                max_head_tail_distance = bounds[head_index, tail_index]
                bounds[tail_index, head_index] = (
                    max_head_tail_distance - 5
                )  # Max distance between headgroup and tails
                # print(max_head_tail_distance)
    for i in range(len(tail_indices)):
        for j in range(i + 1, len(tail_indices)):
            # Determine the minimum tail-tail distance for the specific pair
            # Assign the minimum distance based on which index is larger
            if tail_indices[i] < tail_indices[j]:
                min_tail_distance = bounds[tail_indices[j], tail_indices[i]]
                bounds[tail_indices[i], tail_indices[j]] = (
                    min_tail_distance + 5
                )  # Min distance between tails
                # print(min_tail_distance)
            else:
                min_tail_distance = bounds[tail_indices[i], tail_indices[j]]
                print(min_tail_distance)
                bounds[tail_indices[j], tail_indices[i]] = (
                    min_tail_distance + 5
                )  # Min distance between tails

    # Smooth the bounds matrix
    DistanceGeometry.DoTriangleSmoothing(bounds)

    # Set up the embedding parameters
    params = rdDistGeom.EmbedParameters()
    params.useExpTorsionAnglePrefs = False
    params.useBasicKnowledge = False
    params.numThreads = 0
    params.randomSeed = 42
    params.SetBoundsMat(bounds)

    # Generate conformers
    rdDistGeom.EmbedMultipleConfs(mol, num_conformers, params)

    # Write only the first optimized conformer to a PDB file
    # with Chem.PDBWriter(output_filename) as writer:
    #     writer.write(mol, confId=cids[0])  # Write only the first conformer
    # print(f"Writing output to: {output_filename}")
    return mol


def generate_lipid_structure(
    smiles,
    head_indices,
    tail_indices,
    store_trajectory=False,
    trajectory_path=None,
):
    """Generate a 3D lipid conformer with extended head-to-tail geometry.

    Embed multiple RDKit conformers, select the one with maximal head-tail
    separation, then minimize with UFF distance constraints to produce a
    stretched lipid structure suitable for bilayer construction.

    Parameters
    ----------
    smiles : str
        SMILES string of the lipid molecule.
    head_indices : list of int
        Atom indices belonging to the head group.
    tail_indices : list of int
        Atom indices belonging to the tail termini.
    store_trajectory : bool, optional
        If True, write intermediate minimization frames to an XYZ file.
        Default is False.
    trajectory_path : str or None, optional
        Path for the trajectory XYZ file. Only used when *store_trajectory*
        is True. Defaults to ``"trajectory.xyz"``.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule with a single optimized conformer.

    Raises
    ------
    ValueError
        If no initial conformers are produced, tail atoms are too far apart,
        head-tail atoms are too close, or any bond length exceeds 2.5 A.

    """
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    Chem.SanitizeMol(rdkit_mol)

    tail_indices = sorted(tail_indices)

    AllChem.EmbedMultipleConfs(
        rdkit_mol, numConfs=50, useRandomCoords=True, randomSeed=42, numThreads=0
    )
    if not rdkit_mol.GetNumConformers():
        raise ValueError("RDKit did not produce any initial conformers.")

    conf_head_tail_distance_sum = {}
    for conf_id in range(rdkit_mol.GetNumConformers()):
        conf = rdkit_mol.GetConformer(conf_id)
        pos = conf.GetPositions()
        dist = 0
        for hi in head_indices:
            for ti in tail_indices:
                d = np.linalg.norm(pos[hi] - pos[ti])
                dist += d
        conf_head_tail_distance_sum[conf_id] = dist
    mini = max(conf_head_tail_distance_sum, key=conf_head_tail_distance_sum.get)

    mol = rdkit_mol
    from rdkit.Chem import rdDistGeom

    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)

    # i1, i2, min, max
    distance_constraints_head_tail = []
    for head_index in head_indices:
        for tail_index in tail_indices:
            if head_index > tail_index:
                max_dist = bounds[tail_index, head_index]
                min_dist = max_dist - 5
                max_dist += 20
                # min_dist = bounds[head_index, tail_index]
            else:
                max_dist = bounds[head_index, tail_index]
                min_dist = max_dist - 5
                # min_dist = bounds[tail_index, head_index]
                max_dist += 20
            assert min_dist < max_dist
            distance_constraints_head_tail.append((head_index, tail_index, min_dist, max_dist))

    distance_constraints_tails = []
    for i in range(len(tail_indices)):
        for j in range(i + 1, len(tail_indices)):
            if tail_indices[i] > tail_indices[j]:
                raise ValueError("Tail indices must be sorted.")
            min_dist = bounds[tail_indices[j], tail_indices[i]]
            max_dist = min_dist + 3
            min_dist = 0
            assert min_dist < max_dist
            distance_constraints_tails.append(
                (tail_indices[i], tail_indices[j], min_dist, max_dist)
            )

    mol2 = Chem.Mol(mol)
    mol2.RemoveAllConformers()
    mol2.AddConformer(mol.GetConformer(id=mini))
    del mol

    ff = AllChem.UFFGetMoleculeForceField(mol2)
    for i1, i2, min_dist, max_dist in distance_constraints_head_tail:
        ff.UFFAddDistanceConstraint(
            i1, i2, relative=False, minLen=min_dist, maxLen=max_dist, forceConstant=10
        )
    for i1, i2, min_dist, max_dist in distance_constraints_tails:
        ff.UFFAddDistanceConstraint(
            i1, i2, relative=False, minLen=min_dist, maxLen=max_dist, forceConstant=10
        )
    if store_trajectory:
        trajectory = []
        max_iterations = 100000 // 50
        print("Running max_iterations", max_iterations)
        for i in range(max_iterations):
            conf = mol2.GetConformer()
            pos = [
                (
                    conf.GetAtomPosition(j).x,
                    conf.GetAtomPosition(j).y,
                    conf.GetAtomPosition(j).z,
                )
                for j in range(mol2.GetNumAtoms())
            ]
            trajectory.append(pos)

            more = ff.Minimize(maxIts=50)
            if more == 0:
                print(f"Converged after {i + 1} iterations")
                break

        if trajectory_path is None:
            trajectory_path = "trajectory.xyz"
        with open(trajectory_path, "w") as f:
            for i, positions in enumerate(trajectory):
                f.write(f"{mol2.GetNumAtoms()}\n")
                f.write(f"Step {i}\n")
                for j, pos in enumerate(positions):
                    atom = mol2.GetAtomWithIdx(j)
                    symbol = atom.GetSymbol()
                    f.write(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    else:
        ff.Minimize(maxIts=100000)

    dmat = Chem.Get3DDistanceMatrix(mol2)
    for i1, i2, min_dist, max_dist in distance_constraints_tails:
        # print(f"Distance {i1}-{i2}: {dmat[i1, i2]:.2f} (target {min_dist:.2f}-{max_dist:.2f})")
        if dmat[i1, i2] > 15.0:
            raise ValueError("Tails are too far away from each other.")
    for i1, i2, min_dist, max_dist in distance_constraints_head_tail:
        # print(f"Distance {i1}-{i2}: {dmat[i1, i2]:.2f} (target {min_dist:.2f}-{max_dist:.2f})")
        if dmat[i1, i2] < 10.0:
            raise ValueError("Head-Tails are too close.")

    for b in mol2.GetBonds():
        i1, i2 = b.GetEndAtomIdx(), b.GetBeginAtomIdx()
        dist = dmat[i1, i2]
        if dist > 2.5:
            raise ValueError(f"Bond length is too long {i1}-{i2}: {dist:.3f}")

    return mol2


# Create the rotation matrix using Rodrigues' rotation formula
def rotation_matrix(axis, theta):
    """Compute a 3D rotation matrix using Rodrigues' rotation formula.

    Parameters
    ----------
    axis : array_like of float
        Rotation axis vector (will be normalized).
    theta : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix.

    """
    axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis
    a = np.cos(theta)
    b = np.sin(theta)
    x, y, z = axis  # Unpack the axis components

    # Create the rotation matrix
    return np.array(
        [
            [a + x * x * (1 - a), x * y * (1 - a) - z * b, x * z * (1 - a) + y * b],
            [y * x * (1 - a) + z * b, a + y * y * (1 - a), y * z * (1 - a) - x * b],
            [z * x * (1 - a) - y * b, z * y * (1 - a) + x * b, a + z * z * (1 - a)],
        ]
    )


def align_lipid_with_z_axis(lipid_universe, tail_atom_ids, head_atom_ids, z_axis):
    """Align a lipid so the tail-to-head vector points along a given axis.

    Parameters
    ----------
    lipid_universe : mda.Universe
        Universe containing the lipid structure.
    tail_atom_ids : list of int
        Atom indices corresponding to the lipid tails.
    head_atom_ids : list of int
        Atom indices corresponding to the lipid head group.
    z_axis : array_like of float
        Target alignment vector.

    Returns
    -------
    mda.Universe
        Copy of the lipid universe with atoms rotated so the tail-to-head
        direction is aligned with *z_axis*.

    Raises
    ------
    ValueError
        If the computed rotation axis has zero norm.

    """
    z_axis = np.array(z_axis)
    # Create a copy of the original lipid universe
    aligned_universe = lipid_universe.copy()
    com = aligned_universe.atoms.center_of_geometry()
    aligned_universe.atoms.translate(-com)
    # Get the positions of the tail and head group atoms
    tail_positions = aligned_universe.atoms[tail_atom_ids].positions
    head_position = aligned_universe.atoms[head_atom_ids].positions.mean(
        axis=0
    )  # Center of mass of head group

    # Calculate the vector from tail to head
    tail_center = tail_positions.mean(axis=0)  # Center of mass of the tails
    vector_to_head = head_position - tail_center

    # Normalize the vector to get the direction
    vector_to_head_normalized = vector_to_head / np.linalg.norm(vector_to_head)

    # Calculate the angle between the two vectors
    cos_angle = np.dot(vector_to_head_normalized, z_axis)
    angle = np.arccos(cos_angle)  # Angle in radians

    if np.isclose(cos_angle, 1.0):
        # parallel
        rotation_axis_normalized = np.array([0, 0, 1])
        angle = 0.0
    elif np.isclose(cos_angle, -1.0):
        # anti-parallel
        axis = np.random.randn(3)
        axis -= axis.dot(z_axis) * z_axis
        other_dot = np.dot(axis, vector_to_head_normalized)
        if not np.isclose(other_dot, 0.0, rtol=0, atol=1e-6):
            raise ValueError(f"Did not yield perpendicular axis! Dot product is {other_dot}.")
        rotation_axis_normalized = axis / np.linalg.norm(axis)
    else:
        # Calculate the rotation axis (cross product)
        rotation_axis = np.cross(vector_to_head_normalized, z_axis)
        rotation_axis_normalized = (
            rotation_axis / np.linalg.norm(rotation_axis)
            if np.linalg.norm(rotation_axis) != 0
            else rotation_axis
        )

    if np.linalg.norm(rotation_axis_normalized) < 1e-12:
        raise ValueError("Rotation axis cannot have zero norm.")
    # Apply the rotation matrix to all atom positions
    rot_matrix = rotation_matrix(rotation_axis_normalized, angle)
    aligned_positions = np.dot(aligned_universe.atoms.positions, rot_matrix.T)

    # Update the positions in the aligned universe
    aligned_universe.atoms.positions = aligned_positions

    return aligned_universe


def create_bilayer_from_model(bilayer_composition: BilayerComposition, check_clashes: bool = True):
    """Build a lipid bilayer (or monolayer) from a composition specification.

    Place lipids on a 2D grid for each leaflet, align them along the z-axis,
    resolve inter-leaflet clashes, and merge into a single Universe with
    contiguous residue numbering.

    Parameters
    ----------
    bilayer_composition : BilayerComposition
        Specification of lipid species, counts, monolayer flag, and z-padding.
    check_clashes : bool, optional
        If True, perform an all-residue pairwise clash check after assembly
        and raise on failure. Default is True.

    Returns
    -------
    mda.Universe
        Assembled bilayer system with box dimensions set.

    Raises
    ------
    ValueError
        If inter-leaflet clashes cannot be resolved or if residue-level
        clashes are detected when *check_clashes* is True.

    """
    max_lipid_width = 0
    max_lipid_height = 0
    lipids = bilayer_composition.species

    # Determine grid size (N x N) based on total lipid count
    total_lipids = sum(lipid.count for lipid in lipids)
    if not bilayer_composition.monolayer:
        total_lipids = total_lipids // 2
        lipids = [
            LipidSpecies(
                smiles=lipid.smiles,
                resname=lipid.resname,
                count=lipid.count // 2,
                head_atoms=lipid.head_atoms,
                tail_atoms=lipid.tail_atoms,
            )
            for lipid in lipids
        ]
    N = int(np.ceil(np.sqrt(total_lipids)))  # N x N grid

    for lipid in lipids:
        u = align_lipid_with_z_axis(lipid.universe, lipid.tail_atoms, lipid.head_atoms, [0, 0, 1])
        atoms = u.atoms
        atoms.write(f"{lipid.resname}.pdb")
        positions = atoms.positions
        # Extract only x-y coordinates
        xy_positions = positions[:, :2]

        # Compute the distance array for the x-y coordinates
        distance_matrix = cdist(xy_positions, xy_positions)

        # Find the maximum distance (excluding the diagonal)
        max_xy_distance = np.max(distance_matrix[np.triu_indices(len(xy_positions), k=1)])

        # NOTE: not so great if tails have different length :)
        # max_z_distance = (
        #     atoms[lipid.head_atoms].center_of_geometry()[2]
        #     - atoms[lipid.tail_atoms].center_of_geometry()[2]
        # )

        # NOTE: better use lowest tail atom z position
        lowest_tail_z = np.atleast_2d(atoms[lipid.tail_atoms].positions).min(axis=0)[2]
        if lowest_tail_z > 0:
            raise ValueError("Lowest tail Z coordinate must be negative.")
        max_z_distance = np.abs(atoms[lipid.head_atoms].center_of_geometry()[2] - lowest_tail_z)

        max_lipid_height = max(max_lipid_height, max_z_distance)
        max_lipid_width = max(max_lipid_width, max_xy_distance)

    del u

    bin_size = round(max_lipid_width + 5, ndigits=3)

    # Create grid points
    grid_points = [(x * bin_size, y * bin_size) for x in range(N) for y in range(N)]

    # Handle empty spots and place lipids for both bilayers
    num_grid_points = len(grid_points)

    if total_lipids < num_grid_points:
        used_grid_points = random.sample(grid_points, total_lipids)
    else:
        used_grid_points = grid_points

    if bilayer_composition.monolayer:
        # lipid middle should be at z=0
        max_lipid_height /= 2

    top_lipid_universes = place_lipids_on_monolayer(
        used_grid_points,
        np.array([0, 0, 1]),
        lipids,
        is_top_layer=True,
        lipid_headgroup_target=max_lipid_height,
    )
    if not bilayer_composition.monolayer:
        bottom_lipid_universes = place_lipids_on_monolayer(
            used_grid_points,
            np.array([0, 0, -1]),
            lipids,
            is_top_layer=False,
            lipid_headgroup_target=max_lipid_height,
        )
    else:
        bottom_lipid_universes = []

    # check z-clashed for layers
    from MDAnalysis.analysis import distances

    def check_clash(u1, u2):
        # no need to consider PBC here, because we're only interested in z-direction clashes
        _, dists = distances.capped_distance(u1.atoms.positions, u2.atoms.positions, max_cutoff=1.1)
        if len(dists) and np.min(dists) <= 1.0:
            return True
        return False

    if not bilayer_composition.monolayer:
        logger.info("Checking clashes in z-direction...")
        assert len(top_lipid_universes) == len(bottom_lipid_universes)
        for u1 in top_lipid_universes:
            for u2 in bottom_lipid_universes:
                assert len(u1.residues) == 1
                assert len(u2.residues) == 1

                if not check_clash(u1, u2):
                    continue
                else:
                    mda.Merge(u1, u2).atoms.write("initial_clash.pdb")

                for _ in range(10):
                    u1.atoms.translate((0, 0, 0.5))
                    u2.atoms.translate((0, 0, -0.5))
                    if not check_clash(u1, u2):
                        mda.Merge(u1, u2).atoms.write("fixed_clash.pdb")
                        break

                if check_clash(u1, u2):
                    mda.Merge(u1, u2).atoms.write("clash.pdb")
                    raise ValueError("Lipid clashes could not be fixed.")
        logger.info("No clashes found or clashes fixed ✅.")

    # Merge all lipid universes together for both layers
    final_bilayer = mda.Merge(*top_lipid_universes, *bottom_lipid_universes)
    # Group by resname
    selections = [final_bilayer.select_atoms(f"resname {lipid.resname}") for lipid in lipids]
    final_bilayer = mda.Merge(*selections)
    # Calculate box dimensions
    positions = final_bilayer.atoms.positions

    z_padding = bilayer_composition.z_padding
    max_x = positions[:, 0].max() + 7
    max_y = positions[:, 1].max() + 7
    min_z = positions[:, 2].min() - z_padding
    max_z = positions[:, 2].max() + z_padding
    z_box = max_z - min_z

    # Set box dimensions and center the bilayer
    final_bilayer.dimensions = [max_x, max_y, z_box, 90, 90, 90]
    final_bilayer.atoms.translate((0, 0, z_box / 2))

    # Number resids contiguously
    for resid, r in enumerate(final_bilayer.residues, start=1):
        r.atoms.residues.resids = resid

    clashes = False
    if check_clashes:
        logger.info("Checking clashes in entire system...")
        for i1, r1 in enumerate(final_bilayer.residues):
            for i2, r2 in enumerate(final_bilayer.residues):
                if i1 >= i2:
                    continue
                _, dists = distances.capped_distance(
                    r1.atoms, r2.atoms, max_cutoff=1.1, box=final_bilayer.dimensions
                )
                if len(dists) and np.min(dists) < 1.0:
                    clashes = True
                    logger.error(f"Clash between {i1}/{r1} and {i2}/{r2}")
                    clash = mda.Merge(r1.atoms, r2.atoms)
                    clash.dimensions = final_bilayer.dimensions
                    clash.atoms.write(f"clash_{i1}_{i2}.pdb")
        if clashes:
            final_bilayer.atoms.write("bilayer_with_clashes.pdb")
            logger.error("Clashes found.")
            raise ValueError("Clashes found.")
        logger.info("System clash check passed.")
    else:
        logger.info("Skipping clash check (check_clashes=False).")

    return final_bilayer


def place_lipids_on_monolayer(
    grid_points, z_axis, lipid_data, is_top_layer=True, lipid_headgroup_target=0.0
):
    """Place lipids onto grid positions for one leaflet of a bilayer.

    Each lipid is aligned along *z_axis*, translated so its head group
    reaches *lipid_headgroup_target*, positioned at a grid point in the
    xy-plane, and randomly rotated around z.

    Parameters
    ----------
    grid_points : list of tuple of float
        (x, y) positions on the grid where lipids will be placed.
    z_axis : numpy.ndarray
        Unit vector defining the leaflet orientation (e.g. [0,0,1] for top).
    lipid_data : list of LipidSpecies
        Lipid species with counts and structural metadata.
    is_top_layer : bool, optional
        If True, head groups point toward +z. Default is True.
    lipid_headgroup_target : float, optional
        Target z-coordinate for the head group center of geometry.
        Default is 0.0.

    Returns
    -------
    list of mda.AtomGroup
        Positioned lipid atom groups, one per grid point used.

    """
    lipid_universes = []

    lipid_dict = {v.resname: v for v in lipid_data}

    # Create a list based on counts
    lipid_queue = []
    for lipid, data in lipid_dict.items():
        lipid_queue.extend([lipid] * data.count)  # Repeat lipid based on count
    # Shuffle the list to randomize order
    random.shuffle(lipid_queue)

    for grid_point, lipid_type in zip(grid_points, lipid_queue):
        lipid_info = lipid_dict[lipid_type]
        # Create a copy of the selected lipid universe for placement
        lipid_copy = lipid_info.universe.copy()
        for residue in lipid_copy.residues:
            residue.resname = lipid_type

        # Align the lipid with the Z-axis
        lipid_copy = align_lipid_with_z_axis(
            lipid_copy, lipid_info.tail_atoms, lipid_info.head_atoms, z_axis
        )

        # Calculate lipid shift based on the head group atom
        target = lipid_headgroup_target
        if not is_top_layer:
            target *= -1.0
        lipid_shift = target - lipid_copy.atoms[lipid_info.head_atoms].center_of_geometry()[2]

        # Adjust the position of the aligned lipid
        lipid_copy.atoms.translate((*grid_point, 0))
        lipid_copy.atoms.translate((0, 0, lipid_shift))

        # Rotate the lipid randomly around the z-axis
        angle = np.random.uniform(0, 360)
        point = lipid_copy.atoms.center_of_geometry()
        lipid_copy.atoms.rotateby(angle=angle, axis=[0, 0, 1], point=point)

        lipid_universes.append(lipid_copy.atoms)

    return lipid_universes


def create_mixed_box_universe(composition: MixedBoxComposition) -> tuple[mda.Universe, Topology]:
    """Pack multiple molecular species into a rectangular simulation box.

    Use packmol to place molecules at the target density, then load the
    result into an MDAnalysis Universe with correct residue names and
    box dimensions.

    Parameters
    ----------
    composition : MixedBoxComposition
        Species list with counts and target density.

    Returns
    -------
    tuple of (mda.Universe, openff.toolkit.Topology)
        The packed system as an MDAnalysis Universe and the corresponding
        OpenFF Topology.

    Raises
    ------
    ValueError
        If the resulting box is not rectangular.

    """
    molecules = [spec.openff_molecule for spec in composition.species]
    number_of_copies = [spec.count for spec in composition.species]

    with temporary_working_directory() as working_dir:
        top = _pack_molecules_into_box(
            molecules=molecules,
            number_of_copies=number_of_copies,
            working_dir=str(working_dir),
            target_density=composition.target_density,
        )
        logger.info(f"Successfully packed {sum(number_of_copies)} molecules into a box.")
        resnames = [m.properties["resname"] for m in top.molecules]
        u = mda.Universe(working_dir / "packmol_output.pdb")
        u.residues.resnames = resnames

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            box_vectors = np.asarray(top.box_vectors.to("angstrom"))
        is_diagonal = np.count_nonzero(box_vectors - np.diag(np.diagonal(box_vectors))) == 0
        if not is_diagonal:
            raise ValueError("Box is not rectangular.")
        u.dimensions = [*np.diagonal(box_vectors), 90, 90, 90]
    return u, top


def _pack_molecules_into_box(molecules, number_of_copies, working_dir, target_density=1.0):
    """Pack molecules into a box using packmol.

    Parameters
    ----------
    molecules : list
        List of molecule objects.
    number_of_copies : list
        Number of copies for each molecule.
    working_dir : str
        Working directory for packmol outputs.
    target_density : float, optional
        Target density in g/cm^3. Default is 1.0.

    Returns
    -------
    Topology
        Packed box topology.

    """
    from openff.interchange.components._packmol import UNIT_CUBE, pack_box
    from openff.units import unit

    return pack_box(
        molecules=molecules,
        number_of_copies=number_of_copies,
        target_density=target_density * unit.gram / unit.centimeter**3,
        box_shape=UNIT_CUBE,
        working_directory=working_dir,
    )


def protein_displaced_volume_a3(
    protein_mass_dalton: float, partial_specific_volume: float
) -> float:
    """Return a protein's displaced volume in Å³ from its mass and partial specific volume.

    ``partial_specific_volume`` is in mL/g. The mass converts to grams via
    Avogadro's number, multiplies by the specific volume to give cm³, and cm³
    converts to Å³ (1 cm³ = 1e24 Å³).
    """
    mass_g = protein_mass_dalton / N_AVOGADRO
    volume_cm3 = mass_g * partial_specific_volume
    return volume_cm3 * 1e24


def cubic_box_edge_for_density(total_mass_dalton: float, target_density: float) -> float:
    """Return box edge (Å) holding ``total_mass_dalton`` at the density."""
    mass_g = total_mass_dalton / N_AVOGADRO
    volume_cm3 = mass_g / target_density
    volume_a3 = volume_cm3 * 1e24
    return float(volume_a3 ** (1.0 / 3.0))


def resolve_solution_counts(
    species,
    box_volume_a3: float,
    protein_mass_dalton: float,
    partial_specific_volume: float,
    basis: str,
) -> list[int]:
    """Resolve each solution species to an integer molecule count.

    A species with an explicit ``count`` keeps it. A species with a molar
    ``concentration`` resolves to ``round(concentration * V_basis * N_A)`` where
    ``V_basis`` is the full box volume (``basis="box"``) or the box minus the
    protein's displaced volume (``basis="protein_excluded"``). A positive
    concentration that rounds to zero molecules is rejected: a larger box is
    required to represent it.

    Parameters
    ----------
    species : list
        Solution species; each has ``count`` or ``concentration`` set.
    box_volume_a3 : float
        Final box volume in Å³.
    protein_mass_dalton : float
        Protein mass in Daltons, for the excluded-volume basis.
    partial_specific_volume : float
        Protein partial specific volume in mL/g.
    basis : str
        ``"box"`` or ``"protein_excluded"``.

    Returns
    -------
    list[int]
        Counts aligned with ``species``.

    """
    if basis == "protein_excluded":
        v_basis_a3 = box_volume_a3 - protein_displaced_volume_a3(
            protein_mass_dalton, partial_specific_volume
        )
    else:
        v_basis_a3 = box_volume_a3
    if v_basis_a3 <= 0:
        raise ValueError(
            f"Concentration reference volume is non-positive ({v_basis_a3:.1f} Å³): the "
            "protein does not fit in the requested box. A larger box is required."
        )
    # 1 Å³ = 1e-27 L, so count = concentration[mol/L] * V[L] * N_A.
    v_basis_liters = v_basis_a3 * 1e-27
    counts = []
    for spec in species:
        if spec.count is not None:
            counts.append(int(spec.count))
            continue
        count = int(round(spec.concentration * v_basis_liters * N_AVOGADRO))
        if spec.concentration > 0 and count == 0:
            raise ValueError(
                f"Species '{spec.resname}' at {spec.concentration} M resolves to zero "
                f"molecules in a {v_basis_a3:.1f} Å³ reference volume. A larger box is required."
            )
        counts.append(count)
    return counts


def create_protein_mixedbox_universe(
    u_protein: mda.Universe,
    molecules,
    number_of_copies,
    resnames,
    box_edge: float,
    working_dir,
    tolerance_a: float = 2.0,
    native_atom_names=None,
) -> mda.Universe:
    """Pack solution molecules around a fixed protein with packmol.

    The protein is centered in a cubic box of edge ``box_edge`` (Å) and handed to
    packmol as a fixed solute (``fixed 0. 0. 0. 0. 0. 0.``), so it appears exactly
    once and its pdb2gmx atom order is preserved. The requested molecules are
    packed around it with a ``tolerance_a`` non-overlap distance. This drives
    packmol's lower-level primitives directly rather than ``pack_box(solute=...)``,
    which would route the protein through an OpenFF Topology and reorder its atoms.

    Parameters
    ----------
    u_protein : mda.Universe
        The processed protein (pdb2gmx atom order); centered in place by this call.
    molecules : list
        OpenFF molecules for each solution species, in packing order.
    number_of_copies : list[int]
        Copies to pack for each molecule, aligned with ``molecules``.
    resnames : list[str]
        Residue name for each molecule, aligned with ``molecules``.
    box_edge : float
        Cubic box edge length in Å (the dilute packing box).
    working_dir : str or Path
        Directory the packmol working files are written into.
    tolerance_a : float, optional
        Packmol non-overlap tolerance in Å. Default 2.0 (matches pack_box).
    native_atom_names : list[list[str] | None] or None, optional
        CHARMM atom names for each species, aligned with ``molecules``. Use
        ``None`` for CGenFF species. These names replace the generic OpenFF PDB
        names for native water and ions.

    Returns
    -------
    mda.Universe
        Protein atoms first (authoritative names/positions), followed by the
        packed solution molecules with residue names set, with cubic box
        dimensions of ``box_edge``.

    """
    import subprocess  # noqa: PLC0415

    from openff.interchange.components._packmol import (  # noqa: PLC0415
        _build_input_file,
        _create_molecule_pdbs,
        _find_packmol,
    )
    from openff.units import Quantity  # noqa: PLC0415

    packmol_path = _find_packmol()
    if packmol_path is None:
        raise OSError("Packmol not found, cannot pack protein_mixedbox.")

    # Center the protein so packmol's fixed (zero-translation) placement keeps it
    # in the middle of the box.
    bbox_min = u_protein.atoms.positions.min(axis=0)
    bbox_max = u_protein.atoms.positions.max(axis=0)
    box_center = np.array([box_edge, box_edge, box_edge]) / 2.0
    u_protein.atoms.translate(box_center - (bbox_min + bbox_max) / 2.0)
    n_protein_atoms = u_protein.atoms.n_atoms

    with working_directory(working_dir, create=True) as wd:
        protein_pdb = wd / "protein_centered.pdb"
        u_protein.atoms.write(str(protein_pdb))

        pdb_file_names = _create_molecule_pdbs(molecules)
        box_size = Quantity(np.array([box_edge, box_edge, box_edge]), "angstrom")
        input_file_path, output_file_path = _build_input_file(
            pdb_file_names,
            number_of_copies,
            str(protein_pdb),
            box_size,
            Quantity(tolerance_a, "angstrom"),
            rectangular=True,
        )

        with open(input_file_path) as file_handle:
            result = subprocess.check_output(
                packmol_path, stdin=file_handle, stderr=subprocess.STDOUT
            )
        if result.decode("utf-8").find("Success!") <= 0:
            raise RuntimeError("Packmol did not report success while packing protein_mixedbox.")

        # The fixed solute is written first, so its atoms/positions are unchanged;
        # take those from the authoritative protein universe and only the packed
        # solution atoms from packmol's output.
        u_packed = mda.Universe(str(wd / output_file_path))

    if native_atom_names is None:
        native_atom_names = [None] * len(molecules)
    if not (len(molecules) == len(number_of_copies) == len(resnames) == len(native_atom_names)):
        raise ValueError("molecules, number_of_copies, resnames, and native_atom_names must align.")

    solution_residue_data = []
    for molecule, resname, count, atom_names in zip(
        molecules, resnames, number_of_copies, native_atom_names
    ):
        solution_residue_data.extend([(resname.upper(), molecule.n_atoms, atom_names)] * count)

    expected_atoms = n_protein_atoms + sum(n_atoms for _, n_atoms, _ in solution_residue_data)
    if u_packed.atoms.n_atoms != expected_atoms:
        raise ValueError(
            f"Packmol output has {u_packed.atoms.n_atoms} atoms, expected "
            f"{expected_atoms}. Cannot reconstruct packed molecules."
        )

    # Packmol preserves structure/copy order, but its repeated residue IDs and
    # chain labels are not reliable molecule boundaries. Rebuild the solution in
    # one allocation, assigning one residue per known molecule. Avoid a Merge per
    # molecule here: realistic systems contain tens of thousands of waters.
    atom_offset = n_protein_atoms
    solution_names: list[str] = []
    solution_resnames: list[str] = []
    atom_resindex: list[int] = []
    for residue_index, (resname, n_atoms, atom_names) in enumerate(solution_residue_data):
        packed_names = list(u_packed.atoms.names[atom_offset : atom_offset + n_atoms])
        if atom_names is not None:
            if len(atom_names) != n_atoms:
                raise ValueError(
                    f"Native residue {resname} has {n_atoms} atoms, but "
                    f"{len(atom_names)} CHARMM atom names were supplied."
                )
            packed_names = atom_names
        solution_names.extend(packed_names)
        solution_resnames.append(resname)
        atom_resindex.extend([residue_index] * n_atoms)
        atom_offset += n_atoms

    n_solution_atoms = expected_atoms - n_protein_atoms
    solution = mda.Universe.empty(
        n_solution_atoms,
        n_residues=len(solution_residue_data),
        atom_resindex=np.asarray(atom_resindex),
        trajectory=True,
    )
    solution.add_TopologyAttr("names", solution_names)
    solution.add_TopologyAttr("types", list(u_packed.atoms.types[n_protein_atoms:expected_atoms]))
    solution.add_TopologyAttr("resnames", solution_resnames)
    solution.add_TopologyAttr("resids", np.arange(1, len(solution_residue_data) + 1))
    solution.add_TopologyAttr("chainIDs", ["S"] * n_solution_atoms)
    solution.atoms.positions = u_packed.atoms.positions[n_protein_atoms:expected_atoms]

    merged = mda.Merge(u_protein.atoms, solution.atoms)
    merged.dimensions = [box_edge, box_edge, box_edge, 90, 90, 90]
    return merged


def charmm_native_atom_names(smiles: str) -> list[str] | None:
    """Return canonical CHARMM36m atom names for supported native species."""
    return {
        "O": ["OW", "HW1", "HW2"],
        "[Na+]": ["NA"],
        "[Cl-]": ["CL"],
    }.get(smiles)
