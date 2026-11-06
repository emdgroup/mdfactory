# ABOUTME: Core build pipeline for constructing MD simulation systems
# ABOUTME: Dispatches to mixedbox, bilayer, LNP, and proteinbox build routines
"""Core build pipeline for constructing MD simulation systems."""

import json
import shutil
from functools import partial
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from pydantic import validate_call

from mdfactory.parametrize import (
    DISPATCH_TOPOLOGY_BUILD,
    NativeForceField,
    resolve_parametrizer,
)

from .check import check_bilayer_buildable
from .models.composition import BilayerComposition, LNPComposition, MixedBoxComposition
from .models.input import BuildInput
from .models.species import SingleMoleculeSpecies
from .run_schedules import write_mdp_files
from .setup.solvation import ionize, solvate
from .utils.setup_utilities import (
    N_AVOGADRO,
    create_bilayer_from_model,
    create_mixed_box_universe,
)
from .utils.utilities import working_directory


def _compute_charge_from_universe(u: mda.Universe, species: list) -> int:
    """Compute total system charge from actual residue counts in the universe.

    This is needed because transformations like sphere cropping or shell projection
    may change residue counts from the input specification.

    Parameters
    ----------
    u : mda.Universe
        The universe with the actual system structure
    species : list
        List of species objects with resname and charge properties

    Returns
    -------
    int
        Total system charge based on actual residue counts

    """
    # Build a map from resname to charge
    charge_by_resname = {}
    for spec in species:
        if hasattr(spec, "charge") and spec.charge is not None:
            charge_by_resname[spec.resname] = spec.charge

    # Count residues and compute total charge
    total_charge = 0
    for resname, charge in charge_by_resname.items():
        count = len(u.select_atoms(f"resname {resname}").residues)
        total_charge += count * charge

    return total_charge


@validate_call
def build_mixedbox(inp: BuildInput):
    """Build a mixed-box system: compress, ionize, write topology and run files."""
    from loguru import logger

    # build the universe with the geometry of the system
    u, parameters, parametrize_function, topology_function = create_mixedbox_by_compression(inp)

    logger.info("Mixedbox system compressed to target density.")

    u, additional_species = ionize_solvated_system(inp.system.ionization, u, inp.system.charge)
    parameters += [parametrize_function(spec) for spec in additional_species]

    topology_function(u, inp.system.species + additional_species, parameters, "mixedbox")
    logger.info("Topology written.")

    # dump system output
    u.atoms.write("system.pdb")

    # TODO: NOT ENGINE-INDEPENDENT AT THE MOMENT...
    # position restraints
    # TODO: maybe rewrite to use universe directly?
    # gromacs_create_position_restraint_file(structure_file="system.pdb", output="posre.itp")

    write_mdp_files(inp)
    logger.info("Run schedule written.")


@validate_call
def create_mixedbox_by_compression(inp: BuildInput, require_target_density: bool = True):
    """Create a mixed-box universe by packing at low density then compressing with OpenMM.

    Parameters
    ----------
    inp : BuildInput
        Mixed-box build input.
    require_target_density : bool, optional
        Raise when compression does not reach ``inp.system.target_density``.
        When False, warn and continue with the box as compressed (LNP cores,
        whose lipid packing often cannot reach the nominal density within the
        compression budget). Default is True.

    Returns
    -------
    tuple
        (universe, parameters, parametrize_function, topology_function)

    """
    from loguru import logger

    loose = inp.system.model_copy()
    loose.target_density = inp.system.target_density * 0.1

    u, _ = create_mixed_box_universe(loose)
    logger.info("Successfully created universe for mixed box system.")

    # parametrize all species in the box for the FF/engine combination
    parametrize_function = resolve_parametrizer(inp.parametrization, inp.engine)
    topology_function = DISPATCH_TOPOLOGY_BUILD.get(inp.engine, None)
    if topology_function is None:
        raise NotImplementedError(f"Cannot build topology for engine {inp.engine}.")

    parameters = [parametrize_function(spec) for spec in inp.system.species]
    logger.info("Parametrization done.")

    # compress box to target density
    with working_directory("compression", create=True, cleanup=True) as wd:
        topology_function(u, inp.system.species, parameters, "mixedbox")

        u.atoms.write("mixedbox.pdb")
        u = mda.Universe("mixedbox.pdb")
        logger.info("Preliminary mixedbox topology prepared.")

        # TODO: currently not independent of GMX-type inputs
        # 3. run openmm-squeeze
        from .simulation.openmm_utils import compress_box

        # 4. get the coordinates of the equilibrated (in-vacuo) mixedbox and set it
        # on the existing universe
        u_equil = compress_box(
            u,
            wd / "mixedbox.pdb",
            wd / "topology.top",
            target_density=inp.system.target_density,
            require_convergence=require_target_density,
        )
        u = u_equil

    return u, parameters, parametrize_function, topology_function


@validate_call
def build_bilayer(inp: BuildInput):
    """Build a bilayer system: compress, solvate, ionize, write topology and run files."""
    from loguru import logger

    u_equil, parameters, parametrize_function, topology_function = create_bilayer_by_compression(
        inp
    )

    logger.info("Bilayer system equilibrated in vacuum.")
    # 5. solvate
    # TODO: solvation settings in model?
    u_solvated = solvate(u_equil, prune_in_z=True)
    u_solvated.dimensions = u_equil.dimensions
    n_water = len(u_solvated.select_atoms("water").residues)

    logger.info(f"Solvation added {n_water} water molecules.")

    if n_water == 0:
        raise ValueError("Solvation did not add any water molecules.")

    # TODO: refactor for more reusability?
    water_spec = SingleMoleculeSpecies(
        count=n_water,
        smiles="O",
        resname="SOL",
    )

    u_ionized, ion_species = ionize_solvated_system(
        inp.system.ionization, u_solvated, inp.system.charge
    )
    additional_species = [water_spec, *ion_species]
    parameters += [parametrize_function(spec) for spec in additional_species]

    u_ionized.atoms.write("system.pdb")

    # 7. write composite topology file for final system with water and ions
    topology_function(u_ionized, inp.system.species + additional_species, parameters, "bilayer")

    # 8. write the run schedule
    write_mdp_files(inp)
    logger.info("Run schedule written.")


def create_bilayer_by_compression(inp):
    """Create a bilayer universe by placing lipids then compressing with OpenMM.

    Returns
    -------
    tuple
        (universe, parameters, parametrize_function, topology_function)

    """
    from loguru import logger

    if not isinstance(inp.system, BilayerComposition):
        raise TypeError("System must specify a bilayer composition.")

    try:
        check_bilayer_buildable(inp.system)
    except Exception as e:
        raise ValueError("Cannot build bilayer system.") from e

    u = create_bilayer_from_model(inp.system)

    # 1. parametrize lipid species
    # parametrize all species in the box for the FF/engine combination
    parametrize_function = resolve_parametrizer(inp.parametrization, inp.engine)

    # preliminary parameters (only lipids)
    parameters = [parametrize_function(spec) for spec in inp.system.species]

    logger.info("Parametrization done.")
    # 2. write temporary toplogy file
    topology_function = DISPATCH_TOPOLOGY_BUILD.get(inp.engine, None)
    if topology_function is None:
        raise NotImplementedError(f"Cannot build topology for engine {inp.engine}.")

    with working_directory("bilayer_squeeze", create=True, cleanup=True) as wd:
        topology_function(u, inp.system.species, parameters, "bilayer")
        u.atoms.write("bilayer.pdb")
        logger.info("Preliminary bilayer topology prepared.")

        # TODO: currently not independent of GMX-type inputs
        # 3. run openmm-squeeze
        from .simulation.openmm_utils import compress_equilibrate_bilayer

        # 4. get the coordinates of the equilibrated (in-vacuo) bilayer and set it
        # on the existing universe
        u_equil = compress_equilibrate_bilayer(u, wd / "bilayer.pdb", wd / "topology.top")

    return u_equil, parameters, parametrize_function, topology_function


@validate_call
def build_lnp(inp: BuildInput):
    """Build an LNP (Lipid Nanoparticle) system.

    The LNP consists of:
    - Core (internal): A compressed sphere of lipid molecules
    - Shell (external): A spherical monolayer wrapping the core

    Workflow:
    1. Create core via create_mixedbox_by_compression → create_sphere
    2. Create shell via create_bilayer_by_compression → shell_from_monolayer
    3. Merge core and shell components
    4. Solvate with spherical exclusion zone
    5. Ionize the solvated system
    6. Write topology and run files
    """
    from loguru import logger

    if not isinstance(inp.system, LNPComposition):
        raise TypeError("System must specify an LNP composition.")

    logger.info(
        f"Building LNP with total radius {inp.system.radius} Å, "
        f"core radius {inp.system.core_radius} Å"
    )

    # Log calculated counts
    for spec in inp.system.core.species:
        logger.info(f"Core species {spec.resname}: {spec.count} molecules")
    for spec in inp.system.shell.species:
        logger.info(f"Shell species {spec.resname}: {spec.count} molecules")

    # Step 1: Create and parametrize core
    u_core, parameters, parametrize_function, topology_function = create_lnp_core(inp)
    logger.info("LNP core created and shaped into sphere.")

    # Step 2: Create shell
    u_shell, shell_params = create_lnp_shell(inp, parametrize_function, topology_function)
    parameters += shell_params
    logger.info("LNP shell created from monolayer projection.")

    # Step 3: Merge core and shell
    u_lnp = merge_lnp_components(
        u_core, u_shell, inp, parameters, parametrize_function, topology_function
    )
    logger.info("LNP components merged.")

    # Step 4: Solvate with spherical exclusion
    # Calculate box size and center for solvation
    mins = u_lnp.atoms.positions.min(axis=0)
    maxs = u_lnp.atoms.positions.max(axis=0)
    center = (mins + maxs) / 2

    # Set box dimensions with padding
    box_size = maxs - mins + 2 * inp.system.padding
    u_lnp.dimensions = [box_size[0], box_size[1], box_size[2], 90, 90, 90]

    # Translate so LNP is centered in the box
    u_lnp.atoms.translate(-mins + inp.system.padding)
    center = u_lnp.atoms.center_of_geometry()

    # Remove water inside the LNP core using remove_sphere
    # Use a slightly smaller radius to avoid water clashing with shell
    exclusion_radius = inp.system.core_radius * 0.9
    u_solvated = solvate(
        u_lnp, prune_in_z=False, remove_sphere=(center[0], center[1], center[2], exclusion_radius)
    )
    u_solvated.dimensions = u_lnp.dimensions
    n_water = len(u_solvated.select_atoms("water").residues)
    logger.info(f"Solvation added {n_water} water molecules.")

    if n_water == 0:
        raise ValueError("Solvation did not add any water molecules.")

    water_spec = SingleMoleculeSpecies(count=n_water, smiles="O", resname="SOL")

    # Step 5: Ionize
    # Compute charge from actual residue counts (not input counts) since transformations
    # like sphere cropping and shell projection change the counts
    all_species = inp.system.core.species + inp.system.shell.species
    actual_charge = _compute_charge_from_universe(u_solvated, all_species)
    logger.info(
        f"Computed charge from universe: {actual_charge} (input config charge: {inp.system.charge})"
    )

    u_ionized, ion_species = ionize_solvated_system(
        inp.system.ionization, u_solvated, actual_charge
    )
    additional_species = [water_spec, *ion_species]

    u_ionized.atoms.write("system.pdb")

    # Step 6: Write topology
    # Get unique LNP species and deduplicate parameters by hash
    unique_lnp_species = inp.system.get_species_with_counts()
    param_by_hash = {param.moleculetype: param for param in parameters}
    unique_lnp_params = [param_by_hash[spec.hash] for spec in unique_lnp_species]
    # Append water/ion species and their parameters in order (like bilayer/mixedbox do)
    all_species = unique_lnp_species + additional_species
    all_params = unique_lnp_params + [parametrize_function(spec) for spec in additional_species]
    topology_function(u_ionized, all_species, all_params, "lnp")
    logger.info("Topology written.")

    # Log composition comparison (input fractions vs actual)
    _log_lnp_composition(u_ionized, inp)

    # Step 7: Write the run schedule (lnp uses the mixedbox protocol)
    write_mdp_files(inp)
    logger.info("Run schedule written.")


def create_lnp_core(inp: BuildInput):
    """Create the LNP core by compressing a mixed box and shaping into a sphere.

    For efficiency, we pack only a unit cell (1/n³ of total molecules) and
    replicate it n×n×n times before cropping to a sphere.

    Returns
    -------
    tuple
        (u_core, parameters, parametrize_function, topology_function)

    """
    from loguru import logger

    from .setup.lnp import replicate_box_3d
    from .simulation.openmm_utils import create_sphere

    # Create a temporary MixedBoxComposition for the core unit cell
    core_comp = MixedBoxComposition(
        species=inp.system.core.species,
        target_density=inp.system.core.target_density,
    )

    # Create a temporary BuildInput for the mixedbox compression
    core_inp = BuildInput(
        simulation_type="mixedbox",
        system=core_comp,
        parametrization=inp.parametrization,
        engine=inp.engine,
    )

    # Use the existing mixedbox compression workflow for the unit cell. The lipid
    # core often cannot reach the nominal density within the compression budget;
    # keep the pre-existing behavior of continuing with a warning instead of failing.
    u_unit, parameters, parametrize_function, topology_function = create_mixedbox_by_compression(
        core_inp, require_target_density=False
    )
    logger.info("Core unit cell compressed.")

    # Replicate the unit cell n×n×n times
    n = inp.system.core.replication_factor
    if n > 1:
        logger.info(f"Replicating unit cell {n}×{n}×{n} = {n**3} times.")
        u_box = replicate_box_3d(u_unit, n)
        logger.info(
            f"Replicated box has {len(u_box.residues)} residues "
            f"(unit cell had {len(u_unit.residues)})."
        )
    else:
        u_box = u_unit

    # Shape into a sphere using create_sphere
    core_radius_nm = inp.system.core_radius / 10.0  # Convert Å to nm

    with working_directory("core_sphere", create=True, cleanup=True) as wd:
        # For topology, use species from the full replicated box
        # We need to update counts to reflect the replicated system
        replicated_species = [s.model_copy() for s in inp.system.core.species]
        for s in replicated_species:
            s.count = s.count * (n**3)

        topology_function(u_box, replicated_species, parameters, "core")
        u_box.atoms.write("core.pdb")

        # Center the box at origin for spherical restraint
        u_box.atoms.translate(-u_box.atoms.center_of_geometry())
        u_box.atoms.write("core_centered.pdb")

        u_core = create_sphere(
            u_box, wd / "core_centered.pdb", wd / "topology.top", radius=core_radius_nm
        )

    # Assign chain ID 'A' to core atoms
    u_core.add_TopologyAttr("chainID")
    u_core.atoms.chainIDs = "A"

    return u_core, parameters, parametrize_function, topology_function


def create_lnp_shell(inp: BuildInput, parametrize_function, topology_function):
    """Create the LNP shell from a monolayer projected onto the core surface.

    Returns
    -------
    tuple
        (u_shell, shell_parameters)

    """
    from loguru import logger

    from .setup.lnp import shell_from_monolayer

    # Create a temporary BilayerComposition for the shell (as monolayer)
    shell_comp = BilayerComposition(
        species=inp.system.shell.species,
        monolayer=True,
        z_padding=20.0,
    )

    # Create a temporary BuildInput for the monolayer
    shell_inp = BuildInput(
        simulation_type="bilayer",
        system=shell_comp,
        parametrization=inp.parametrization,
        engine=inp.engine,
    )

    # Use the existing bilayer compression workflow to create a monolayer
    u_mono, shell_params, _, _ = create_bilayer_by_compression(shell_inp)
    logger.info("Shell monolayer created and equilibrated.")

    # Project monolayer onto spherical surface
    u_shell = shell_from_monolayer(u_mono, R=inp.system.core_radius, z0=inp.system.shell.z0)
    logger.info(f"Shell projected onto sphere with R={inp.system.core_radius} Å.")

    # Assign chain ID 'B' to shell atoms
    u_shell.add_TopologyAttr("chainID")
    u_shell.atoms.chainIDs = "B"

    return u_shell, shell_params


def _log_lnp_composition(u: mda.Universe, inp: BuildInput) -> None:
    """Log a table comparing input fractions vs actual fractions for core and shell."""
    from loguru import logger

    # Count residues by resname in core (chainID A) and shell (chainID B)
    core_atoms = u.select_atoms("chainID A")
    shell_atoms = u.select_atoms("chainID B")

    def count_by_resname(atoms):
        counts = {}
        for res in atoms.residues:
            counts[res.resname] = counts.get(res.resname, 0) + 1
        return counts

    core_counts = count_by_resname(core_atoms)
    shell_counts = count_by_resname(shell_atoms)

    # Build table for core
    lines = ["\n=== LNP Composition Summary ===\n"]
    lines.append("Core:")
    lines.append(f"  {'Resname':<10} {'Input %':>10} {'Actual %':>10} {'Count':>8}")
    lines.append(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
    core_total = sum(core_counts.values())
    for spec in inp.system.core.species:
        # Use stored input_fractions to get original YAML values
        input_frac = inp.system.core.input_fractions.get(spec.resname, 0) * 100
        actual_count = core_counts.get(spec.resname, 0)
        actual_frac = (actual_count / core_total * 100) if core_total > 0 else 0
        lines.append(
            f"  {spec.resname:<10} {input_frac:>10.2f} {actual_frac:>10.2f} {actual_count:>8}"
        )

    # Build table for shell
    lines.append("\nShell:")
    lines.append(f"  {'Resname':<10} {'Input %':>10} {'Actual %':>10} {'Count':>8}")
    lines.append(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
    shell_total = sum(shell_counts.values())
    for spec in inp.system.shell.species:
        # Use stored input_fractions to get original YAML values
        input_frac = inp.system.shell.input_fractions.get(spec.resname, 0) * 100
        actual_count = shell_counts.get(spec.resname, 0)
        actual_frac = (actual_count / shell_total * 100) if shell_total > 0 else 0
        lines.append(
            f"  {spec.resname:<10} {input_frac:>10.2f} {actual_frac:>10.2f} {actual_count:>8}"
        )

    # Calculate actual LNP radius from core + shell atoms only (exclude water/ions)
    lnp_atoms = u.select_atoms("chainID A or chainID B")
    center = lnp_atoms.center_of_geometry()
    distances = np.linalg.norm(lnp_atoms.positions - center, axis=1)
    actual_radius = distances.max()

    lines.append("\nLNP Radius:")
    lines.append(f"  Target: {inp.system.radius:>8.1f} Å")
    lines.append(f"  Actual: {actual_radius:>8.1f} Å")

    logger.info("\n".join(lines))


def merge_lnp_components(
    u_core, u_shell, inp: BuildInput, parameters, parametrize_function, topology_function
):
    """Merge core and shell into a single LNP universe.

    The shell is positioned around the core and the combined system
    undergoes a brief equilibration with spherical restraints.

    Returns
    -------
    mda.Universe
        Merged LNP universe.

    """
    from loguru import logger

    from .simulation.openmm_utils import create_sphere

    # Center the core at origin
    core_center = u_core.atoms.center_of_geometry()
    u_core.atoms.translate(-core_center)

    # Center the shell at origin (it should already be roughly centered)
    shell_center = u_shell.atoms.center_of_geometry()
    u_shell.atoms.translate(-shell_center)

    # Merge the two universes
    u_lnp = mda.Merge(u_core.atoms, u_shell.atoms)

    # Set box dimensions large enough to contain the LNP
    max_radius = inp.system.radius * 1.2  # Add 20% buffer
    box_dim = 2 * max_radius
    u_lnp.dimensions = [box_dim, box_dim, box_dim, 90, 90, 90]

    # Center at box center
    u_lnp.atoms.translate([box_dim / 2, box_dim / 2, box_dim / 2])

    logger.info("Core and shell merged. Running equilibration...")

    # Brief equilibration to relax the combined structure
    lnp_radius_nm = inp.system.radius / 10.0  # Convert Å to nm

    with working_directory("lnp_equilibrate", create=True, cleanup=True) as wd:
        # Use unique species and match parameters by hash
        unique_species = inp.system.get_unique_species_for_parametrization()
        # Build a map from moleculetype (hash) to parameter
        param_by_hash = {param.moleculetype: param for param in parameters}
        # Get parameters in the same order as unique_species
        unique_params = [param_by_hash[spec.hash] for spec in unique_species]
        topology_function(u_lnp, unique_species, unique_params, "lnp")
        u_lnp.atoms.write("lnp.pdb")

        # Center at origin for spherical restraint
        u_lnp.atoms.translate(-u_lnp.atoms.center_of_geometry())
        u_lnp.atoms.write("lnp_centered.pdb")

        u_equil = create_sphere(
            u_lnp, wd / "lnp_centered.pdb", wd / "topology.top", radius=lnp_radius_nm, steps=50_000
        )

    logger.info("LNP equilibration complete.")
    return u_equil


def ionize_solvated_system(ion_config, u_solvated, total_charge, solvent_volume_a3=None):
    """Add ions for neutralization and salt concentration.

    Parameters
    ----------
    ion_config : IonizationConfig or None
        Ionization settings (concentration, neutralize, etc.)
    u_solvated : mda.Universe
        Solvated system universe
    total_charge : int
        Total charge of the system before ionization
    solvent_volume_a3 : float or None, optional
        Volume in Å³ to resolve the salt concentration against. When given, the
        salt pair count is ``round(concentration · volume · N_A)`` on that volume
        (the protein_mixedbox concentration-volume basis). When None, the count
        is estimated from the water-molecule count as ``ceil(c · n_water / 55.55)``.

    Notes
    -----
    The resolved salt is added *on top of* neutralization and any ions already
    packed as explicit solution species -- it is not a target total ionic
    strength. So the final Na+/Cl- counts are ``salt_pairs + neutralizing_ions``
    plus whatever ion species were packed. Set ``ion_config.concentration = 0``
    to add no salt and control ionic strength purely through packed ion species
    (neutralization still applies unless ``neutralize`` is also disabled).

    Returns
    -------
    tuple[mda.Universe, list[SingleMoleculeSpecies]]
        Ionized universe and list of ion species added

    """
    from loguru import logger

    if ion_config is None:
        return u_solvated, []

    add_na = 0
    add_cl = 0
    if ion_config.neutralize:
        # total_charge = system.charge
        if total_charge > 0:
            add_cl = total_charge
        elif total_charge < 0:
            add_na = -total_charge
        logger.info(
            f"System has total charge {total_charge}, adding {add_na} Na+ and "
            f"{add_cl} Cl- ions for neutralization."
        )

    c_ions = ion_config.concentration  # mol/l
    if c_ions == 0 and add_na == 0 and add_cl == 0:
        logger.info("No ions to add; skipping ionization.")
        return u_solvated, []

    n_water = len(u_solvated.select_atoms("water").residues)
    if n_water == 0:
        raise ValueError("Cannot ionize system without water.")
    if solvent_volume_a3 is not None:
        # 1 Å³ = 1e-27 L, so n_salt = concentration[mol/L] * V[L] * N_A.
        v_liters = solvent_volume_a3 * 1e-27
        n_ions = int(round(c_ions * v_liters * N_AVOGADRO))
    else:
        M_water = 55.55  # mol/l
        n_ions = np.ceil(c_ions * n_water / M_water).astype(int)

    num_na = n_ions + add_na
    num_cl = n_ions + add_cl
    logger.info(
        f"Adding {num_na} Na+ and {num_cl} Cl- ions to the system for neutralization "
        f"and {c_ions} M salt concentration."
    )
    if num_na == 0 and num_cl == 0:
        logger.info("No ions to add; skipping ionization.")
        return u_solvated, []

    u_ionized = ionize(
        u_solvated,
        num_na=num_na,
        num_cl=num_cl,
        min_distance=ion_config.min_distance,
        seed=ion_config.seed,
    )
    u_ionized.dimensions = u_solvated.dimensions
    logger.info(f"Added {n_ions} Na+/Cl- ions.")

    additional_species = []
    if num_na > 0:
        additional_species.append(SingleMoleculeSpecies(count=num_na, smiles="[Na+]", resname="NA"))
    if num_cl > 0:
        additional_species.append(SingleMoleculeSpecies(count=num_cl, smiles="[Cl-]", resname="CL"))
    return u_ionized, additional_species


def _ion_counts(ion_species) -> tuple[int, int]:
    """Return ``(num_na, num_cl)`` from the species ``ionize_solvated_system`` added.

    ``ionize_solvated_system`` returns only the ion species it actually added, so
    one-sided neutralization yields a single species. Counts are therefore looked
    up by residue name; indexing by position would report a lone Cl- as Na+.
    """
    counts = {spec.resname: spec.count for spec in ion_species}
    return counts.get("NA", 0), counts.get("CL", 0)


def _center_in_cubic_box(u: mda.Universe, padding: float) -> float:
    """Center atoms in a cubic box leaving at least ``padding`` on every face.

    The box is centered on the atoms' bounding-box center (not the center of
    geometry), so an asymmetric molecule keeps the requested clearance to each
    face. Returns the cubic box edge length in Angstroms.
    """
    bbox_min = u.atoms.positions.min(axis=0)
    bbox_max = u.atoms.positions.max(axis=0)
    extent = bbox_max - bbox_min
    box_size = float(extent.max() + 2 * padding)
    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]
    box_center = np.array([box_size / 2, box_size / 2, box_size / 2])
    bbox_center = (bbox_min + bbox_max) / 2
    u.atoms.translate(box_center - bbox_center)
    return box_size


def _prepare_protein(protein, spec):
    """Check GROMACS, clean the protein PDB, and run pdb2gmx in the current directory.

    The protein force field and water model come from the resolved family
    ``spec``; the preparation options (``ignore_hydrogens``, ``merge_all``) and
    the chain/disulfide/protonation settings from the ``protein`` species. A
    force field that needs a minimum GROMACS version (amber14sb) is checked
    against the binary first, so an old GROMACS fails before any protein
    preparation.
    """
    from loguru import logger

    from .setup.protein import (
        check_gmx_available,
        clean_pdb,
        require_gmx_version,
        run_pdb2gmx,
    )

    gmx_bin = check_gmx_available()
    logger.info(f"Using GROMACS: {gmx_bin}")
    require_gmx_version(gmx_bin, spec.protein)
    cleaned_pdb = Path("cleaned.pdb")
    clean_pdb(protein.pdb_path.resolve(), cleaned_pdb)
    params = run_pdb2gmx(
        pdb_path=cleaned_pdb,
        forcefield=spec.protein,
        water_model=spec.water_model,
        ignore_hydrogens=protein.ignore_hydrogens,
        merge_all=protein.merge_all,
        disulfide_bonds=protein.disulfide_bonds,
        protonation_states=protein.protonation_states,
        output_dir=Path("pdb2gmx_output"),
        chains=protein.chains,
    )
    logger.info(f"pdb2gmx output: {params.structure_file}")
    return params


def _relax_protein_system(u, topology, itps, ff_dir_name, protein_indices, **relax_kwargs):
    """Relax ``u`` with OpenMM in a scratch ``relaxation/`` dir, protein restrained.

    OpenMM's ``GromacsTopFile`` resolves ``#include`` lines relative to the
    topology, so the itps and force-field directory are copied next to it.
    """
    from .simulation.openmm_utils import relax_with_protein_restraints

    pre_relax_structure = Path("system_pre_relax.pdb").resolve()
    u.atoms.write(str(pre_relax_structure))
    topology = topology.resolve()
    itps = [f.resolve() for f in itps]

    with working_directory("relaxation", create=True, cleanup=True) as wd:
        shutil.copy(pre_relax_structure, wd / "system.pdb")
        shutil.copy(topology, wd / "topology.top")
        for itp in itps:
            shutil.copy(itp, wd / itp.name)
        if ff_dir_name is not None:
            shutil.copytree(topology.parent / ff_dir_name, wd / ff_dir_name, dirs_exist_ok=True)
        return relax_with_protein_restraints(
            u,
            wd / "system.pdb",
            wd / "topology.top",
            protein_indices=protein_indices,
            **relax_kwargs,
        )


def _finalize_protein_build(inp, topology, extra_intermediates=()):
    """Write the run schedule, validate it with grompp, and drop build intermediates.

    The nonbonded block follows the family's nonbonded scheme, so Amber builds
    get the Sage nonbonded settings. Cleanup only runs on success so a failed
    build keeps its intermediates. ``solvated.pdb`` and ``relaxation/`` are left
    in place, as in the bilayer build.
    """
    from loguru import logger

    from .setup.protein import validate_with_grompp

    mdp_files = write_mdp_files(inp)
    logger.info("Run schedule written.")

    validate_with_grompp(
        topology,
        Path("system.pdb"),
        mdp_files["minimization"],
        Path.cwd(),
        allow_net_charge=not inp.system.ionization.neutralize,
    )

    for name in ("cleaned.pdb", "system_pre_relax.pdb", *extra_intermediates):
        Path(name).unlink(missing_ok=True)
    shutil.rmtree("pdb2gmx_output", ignore_errors=True)


@validate_call
def build_proteinbox(inp: BuildInput):
    """Build a protein-in-waterbox system: clean, parametrize, solvate, ionize, relax."""
    from loguru import logger

    from .models.composition import ProteinBoxComposition
    from .setup.protein import bundle_forcefield_into_topology, update_topology_molecules

    if not isinstance(inp.system, ProteinBoxComposition):
        raise TypeError("System must specify a ProteinBoxComposition.")

    params = _prepare_protein(inp.system.protein, inp.parametrization)

    # Center protein in a cubic box
    u = mda.Universe(str(params.structure_file))
    box_size = _center_in_cubic_box(u, inp.system.padding)
    logger.info(f"Protein centered in cubic box: {box_size:.1f} A")

    # Solvate (reuse existing solvation)
    u_solvated = solvate(u, prune_in_z=False)
    u_solvated.dimensions = u.dimensions
    n_water = len(u_solvated.select_atoms("water").residues)
    logger.info(f"Solvation added {n_water} water molecules.")

    if n_water == 0:
        raise ValueError("Solvation did not add any water molecules.")

    # Ionize (reuse existing ionization)
    u_ionized, ion_species = ionize_solvated_system(
        inp.system.ionization, u_solvated, params.total_charge
    )
    num_na, num_cl = _ion_counts(ion_species)
    n_water_final = len(u_ionized.select_atoms("water").residues)

    # Update topology [ molecules ] section. topol.top references its per-chain and
    # posre .itp files by bare name, so they must sit alongside it.
    topology_dest = Path("topology.top")
    shutil.copy(params.topology_file, topology_dest)
    for include_file in params.topology_include_files:
        shutil.copy(include_file, Path(include_file.name))
    ff_dir_name = bundle_forcefield_into_topology(topology_dest)
    update_topology_molecules(topology_dest, n_water_final, num_na, num_cl)

    # OpenMM relaxation with protein position restraints
    u_relaxed = _relax_protein_system(
        u_ionized,
        topology_dest,
        params.topology_include_files,
        ff_dir_name,
        u_ionized.select_atoms("protein and not name H*").indices.tolist(),
        steps=inp.system.relax_steps,
    )
    logger.info("OpenMM relaxation complete.")

    u_relaxed.atoms.write("system.pdb")
    logger.info("Final system written to system.pdb")

    _finalize_protein_build(inp, topology_dest.resolve())
    logger.info("Proteinbox build complete.")


# Post-compression protein verification thresholds.
PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A = 0.5
PROTEIN_MIXEDBOX_CLASH_CUTOFF_A = 0.8


def _reimage_protein_centered(u: mda.Universe, n_protein_atoms: int, protein_reference) -> None:
    """Make the protein whole across PBC, center it, and wrap the solution by residue.

    Rebuilds the protein from ``protein_reference`` (its pre-compression
    coordinates, ordered like the first ``n_protein_atoms`` atoms of ``u``) by
    correcting each atom's displacement from atom 0 with whole box vectors. This
    assumes a rigid translation plus PBC image jumps; a large rotation is caught
    by the RMSD and clash checks in :func:`_reimage_and_verify`. Mutates ``u``.
    """
    box = u.dimensions[:3].copy()
    pos = u.atoms.positions.copy()
    protein = pos[:n_protein_atoms]

    reference_displacement = protein_reference - protein_reference[0]
    current_displacement = protein - protein[0]
    delta = current_displacement - reference_displacement
    protein_whole = protein - box * np.round(delta / box)

    bbox_center = (protein_whole.min(axis=0) + protein_whole.max(axis=0)) / 2.0
    shift = box / 2.0 - bbox_center
    pos[:n_protein_atoms] = protein_whole
    pos += shift
    u.atoms.positions = pos

    if n_protein_atoms < u.atoms.n_atoms:
        solution = u.atoms[n_protein_atoms:]
        solution.positions = solution.wrap(compound="residues")


def _check_no_protein_clashes(
    u: mda.Universe, n_protein_atoms: int, cutoff: float = PROTEIN_MIXEDBOX_CLASH_CUTOFF_A
) -> None:
    """Raise if any protein atom is closer than ``cutoff`` Å to a solution atom (periodic)."""
    if n_protein_atoms >= u.atoms.n_atoms:
        return
    from MDAnalysis.lib.distances import capped_distance

    protein_positions = u.atoms.positions[:n_protein_atoms]
    solution_positions = u.atoms.positions[n_protein_atoms:]
    pairs, _ = capped_distance(
        protein_positions, solution_positions, max_cutoff=cutoff, box=u.dimensions
    )
    if len(pairs) > 0:
        raise ValueError(
            f"{len(pairs)} protein-solution atom pairs are closer than {cutoff} Å after "
            "re-imaging; the protein was likely mis-imaged across the PBC boundary."
        )


def _check_no_solution_clashes(
    u: mda.Universe, n_protein_atoms: int, cutoff: float = PROTEIN_MIXEDBOX_CLASH_CUTOFF_A
) -> None:
    """Raise when atoms from different solution residues overlap under PBC."""
    if n_protein_atoms >= u.atoms.n_atoms:
        return
    from MDAnalysis.lib.distances import capped_distance

    solution = u.atoms[n_protein_atoms:]
    pairs, _ = capped_distance(
        solution.positions, solution.positions, max_cutoff=cutoff, box=u.dimensions
    )
    if len(pairs) == 0:
        return
    different_residues = solution.resindices[pairs[:, 0]] != solution.resindices[pairs[:, 1]]
    unique_pairs = pairs[:, 0] < pairs[:, 1]
    n_clashes = int(np.count_nonzero(different_residues & unique_pairs))
    if n_clashes:
        raise ValueError(
            f"{n_clashes} solution-solution atom pairs from different residues are "
            f"closer than {cutoff} Å."
        )


def _reimage_and_verify(u: mda.Universe, n_protein_atoms: int, protein_reference, stage: str):
    """Re-image the protein, then check its internal RMSD and all contacts.

    Returns the protein RMSD (Å) after best-fit superposition onto the reference;
    raises ``ValueError`` on an RMSD or clash violation. ``stage`` names the build
    step in the error message.
    """
    from MDAnalysis.analysis.rms import rmsd

    _reimage_protein_centered(u, n_protein_atoms, protein_reference)
    protein_rmsd = float(
        rmsd(
            u.atoms.positions[:n_protein_atoms], protein_reference, center=True, superposition=True
        )
    )
    if protein_rmsd > PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A:
        raise ValueError(
            f"Protein RMSD {protein_rmsd:.2f} Å after {stage} exceeds the "
            f"{PROTEIN_MIXEDBOX_RMSD_TOLERANCE_A} Å tolerance; the restraints did not hold."
        )
    _check_no_protein_clashes(u, n_protein_atoms)
    _check_no_solution_clashes(u, n_protein_atoms)
    return protein_rmsd


def _stage_protein_topology(params, protein_ff_dir, dest_dir, species, parameters, counts):
    """Write the composite protein + solution topology and its inputs into ``dest_dir``.

    Copies the pdb2gmx topology (force-field includes already relative), its
    itps, and the bundled force field next to the output, then returns the path
    of the generated ``topology.top``.
    """
    from .parametrize import generate_gromacs_topology_with_protein

    top_src = dest_dir / "topol_protein.top"
    ff_dir = dest_dir / protein_ff_dir.name
    shutil.copy(params.topology_file, top_src)
    shutil.copytree(protein_ff_dir, ff_dir, dirs_exist_ok=True)
    for include_file in params.topology_include_files:
        shutil.copy(include_file, dest_dir / include_file.name)
    topology = dest_dir / "topology.top"
    generate_gromacs_topology_with_protein(
        top_src, ff_dir, species, parameters, counts, "protein_mixedbox", out_path=topology
    )
    return topology


@validate_call
def build_protein_mixedbox(inp: BuildInput):
    """Build a protein packed with water, ions, and SMILES solutes.

    Under ``charmm`` the protein uses CHARMM36m and the solutes CGenFF; under
    ``amber`` the protein uses amber14sb and the solutes OpenFF Sage. Native
    water and ions come from the bundled protein force field in both cases.

    Prepares the protein with pdb2gmx, packs the solution molecules around it in a
    dilute cubic box, compresses to the target box or density with the protein's
    internal structure restrained, re-images the protein whole and centered,
    ionizes on the chosen concentration-volume basis, and relaxes with the protein
    restrained before writing the final composite topology and structure.
    """
    from loguru import logger

    from .models.composition import (
        FixedBoxSizing,
        ProteinMixedBoxComposition,
    )
    from .setup.protein import bundle_forcefield_into_topology, extract_mass_from_topology
    from .simulation.openmm_utils import compress_box
    from .utils.setup_utilities import (
        PROTEIN_MIXEDBOX_DILUTION_FACTOR,
        concentration_reference_volume_a3,
        create_protein_mixedbox_universe,
        cubic_box_edge_for_density,
        resolve_solution_counts,
        topology_atom_names,
    )

    if not isinstance(inp.system, ProteinMixedBoxComposition):
        raise TypeError("System must specify a ProteinMixedBoxComposition.")

    system = inp.system
    fixed_box = isinstance(system.sizing, FixedBoxSizing)

    params = _prepare_protein(system.protein, inp.parametrization)

    # Bundle the selected force field immediately. Native water/ion parameters
    # and small-molecule compatibility checks must use this exact directory
    # rather than an unrelated force field (e.g. one shipped with CGenFF).
    protein_ff_name = bundle_forcefield_into_topology(params.topology_file)
    if protein_ff_name is None:
        raise RuntimeError("pdb2gmx topology does not include a force field.")
    protein_ff_dir = params.topology_file.parent / protein_ff_name

    # Protein mass and charge from the generated topology
    protein_mass = extract_mass_from_topology(params.topology_file)
    protein_charge = params.total_charge
    logger.info(f"Protein mass {protein_mass:.1f} Da, charge {protein_charge}.")

    # Final cubic box edge
    if fixed_box:
        box_edge = float(system.sizing.box_size)
    else:
        species_mass = sum(spec.mass * spec.count for spec in system.species)
        box_edge = cubic_box_edge_for_density(
            protein_mass + species_mass, system.sizing.target_density
        )
    box_volume_a3 = box_edge**3
    logger.info(f"Final cubic box edge {box_edge:.1f} Å ({box_volume_a3:.3e} Å³).")

    # Reject a box that cannot hold the protein with the requested padding.
    u_protein = mda.Universe(str(params.structure_file))
    extent = float(
        (u_protein.atoms.positions.max(axis=0) - u_protein.atoms.positions.min(axis=0)).max()
    )
    if box_edge < extent + 2 * system.padding:
        raise ValueError(
            f"Protein extent {extent:.1f} Å needs a box of at least "
            f"{extent + 2 * system.padding:.1f} Å (extent + 2 × padding {system.padding} Å), "
            f"but the final box is {box_edge:.1f} Å. Use a larger box or less padding."
        )

    # Resolve each solution species to a molecule count on the chosen basis.
    counts = resolve_solution_counts(
        system.species,
        box_volume_a3,
        protein_mass,
        system.partial_specific_volume,
        system.concentration_volume_basis,
    )

    # Parametrize the solution species with the family's small-molecule force field
    # (CGenFF or Sage); native water/ions short-circuit to the bundled force field.
    native_forcefield = NativeForceField(
        name=inp.parametrization.protein,
        directory=protein_ff_dir,
        water_model=inp.parametrization.water_model,
    )
    parametrize_solute = resolve_parametrizer(inp.parametrization, inp.engine, native_forcefield)
    parameters = [parametrize_solute(spec) for spec in system.species]
    logger.info("Solution-species parametrization complete.")

    # Drop species that resolve to zero copies from packing and topology alike.
    packed = [
        (spec, par, count)
        for spec, par, count in zip(system.species, parameters, counts)
        if count > 0
    ]
    packed_species = [spec for spec, _, _ in packed]
    packed_parameters = [par for _, par, _ in packed]
    packed_counts = [count for _, _, count in packed]

    n_protein_atoms = u_protein.atoms.n_atoms
    protein_indices = list(range(n_protein_atoms))

    # Pack the solution around the centered protein in a dilute cubic box (0.1×
    # density idiom), then compress it down to the target box.
    dilute_edge = box_edge / PROTEIN_MIXEDBOX_DILUTION_FACTOR ** (1.0 / 3.0)
    size_kw = (
        {"target_volume": box_volume_a3}
        if fixed_box
        else {"target_density": system.sizing.target_density}
    )

    with working_directory("compression", create=True, cleanup=True) as wd:
        u_packed = create_protein_mixedbox_universe(
            u_protein,
            [spec.openff_molecule for spec in packed_species],
            packed_counts,
            [spec.resname for spec in packed_species],
            dilute_edge,
            wd / "packmol",
            native_atom_names=[
                topology_atom_names(spec, par)
                for spec, par in zip(packed_species, packed_parameters)
            ],
        )
        logger.info(
            f"Packed {sum(packed_counts)} solution molecules around the protein "
            f"in a {dilute_edge:.1f} Å dilute box."
        )

        # The centered protein positions are the compression restraint reference,
        # used to make the protein whole after compression.
        protein_reference = u_packed.atoms.positions[:n_protein_atoms].copy()

        comp_topology = _stage_protein_topology(
            params, protein_ff_dir, wd, packed_species, packed_parameters, packed_counts
        )
        packed_pdb = wd / "packed.pdb"
        u_packed.atoms.write(str(packed_pdb))
        u_compressed = compress_box(
            u_packed, packed_pdb, comp_topology, protein_indices=protein_indices, **size_kw
        )
    logger.info("Compression complete.")

    # Re-image: make the protein whole across PBC, center it, wrap the solution by
    # residue, and verify the protein geometry survived compression.
    protein_rmsd = _reimage_and_verify(
        u_compressed, n_protein_atoms, protein_reference, "compression"
    )
    logger.info(f"Protein re-imaged whole and centered (RMSD {protein_rmsd:.2f} Å).")

    # Charge from the protein topology plus the actually packed species.
    species_charge = sum(
        count * (spec.charge or 0) for spec, count in zip(packed_species, packed_counts)
    )
    total_charge = protein_charge + species_charge

    # Ionize by replacing water on the chosen concentration-volume basis.
    reference_volume = partial(
        concentration_reference_volume_a3,
        protein_mass_dalton=protein_mass,
        partial_specific_volume=system.partial_specific_volume,
        basis=system.concentration_volume_basis,
    )
    actual_box_volume_a3 = float(u_compressed.trajectory.ts.volume)
    u_ionized, ion_species = ionize_solvated_system(
        system.ionization,
        u_compressed,
        total_charge,
        solvent_volume_a3=reference_volume(actual_box_volume_a3),
    )
    n_replaced = sum(spec.count for spec in ion_species)

    # Final [ molecules ] counts, positionally aligned with the ionize output order
    # (protein → packed solution with water reduced by the replaced count → Na → Cl).
    # Track by position, not resname, so an explicit ion species that shares a
    # resname with the ionization ions is not double-counted.
    final_counts = list(packed_counts)
    for i, spec in enumerate(packed_species):
        if spec.is_water:
            final_counts[i] -= n_replaced
            break
    # Final count of each requested species for the build report; ionization
    # species are reported separately.
    final_iter = iter(final_counts)
    final_input_counts = [next(final_iter) if count > 0 else 0 for count in counts]
    final_species = [*packed_species, *ion_species]
    final_parameters = [*packed_parameters, *(parametrize_solute(s) for s in ion_species)]
    final_counts += [spec.count for spec in ion_species]

    # Write the final composite topology in the build directory.
    topology_dest = _stage_protein_topology(
        params, protein_ff_dir, Path(), final_species, final_parameters, final_counts
    ).resolve()

    # Restrained relaxation of the solution around the fixed protein. The
    # composite topology includes the per-chain/posre itps, the merged
    # small-molecule parameters, and each small-molecule itp; all sit beside
    # topology.top.
    u_relaxed = _relax_protein_system(
        u_ionized,
        topology_dest,
        Path.cwd().glob("*.itp"),
        protein_ff_name,
        protein_indices,
        steps=system.relax_steps,
        pressure=None if fixed_box else 1.0,
    )
    logger.info("OpenMM relaxation complete.")

    # Make the protein whole and centered one final time and validate every
    # intermolecular contact before writing output.
    _reimage_and_verify(u_relaxed, n_protein_atoms, protein_reference, "relaxation")

    u_relaxed.atoms.write("system.pdb")
    logger.info("Final system written to system.pdb")

    final_box_volume_a3 = float(u_relaxed.trajectory.ts.volume)
    final_concentration_volume_a3 = reference_volume(final_box_volume_a3)
    final_concentration_volume_l = final_concentration_volume_a3 * 1e-27
    build_report = {
        "simulation_type": "protein_mixedbox",
        "requested_box_volume_a3": box_volume_a3,
        "compressed_box_volume_a3": actual_box_volume_a3,
        "final_box_volume_a3": final_box_volume_a3,
        "concentration_volume_basis": system.concentration_volume_basis,
        "final_concentration_volume_a3": final_concentration_volume_a3,
        "species": [
            {
                "resname": spec.resname,
                "smiles": spec.smiles,
                "requested_count": spec.count,
                "requested_concentration_molar": spec.concentration,
                "resolved_count": resolved_count,
                "final_count": final_count,
                "achieved_concentration_molar": final_count
                / (final_concentration_volume_l * N_AVOGADRO),
            }
            for spec, resolved_count, final_count in zip(system.species, counts, final_input_counts)
        ],
        "ionization_species": [
            {"resname": spec.resname, "smiles": spec.smiles, "count": spec.count}
            for spec in ion_species
        ],
    }
    Path("build_metadata.json").write_text(json.dumps(build_report, indent=2) + "\n")
    logger.info("Resolved counts and achieved concentrations written to build_metadata.json.")

    _finalize_protein_build(inp, topology_dest, extra_intermediates=("topol_protein.top",))
    logger.info("Protein_mixedbox build complete.")
