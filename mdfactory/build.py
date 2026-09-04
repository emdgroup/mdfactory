# ABOUTME: Core build pipeline for constructing MD simulation systems
# ABOUTME: Dispatches to mixedbox, bilayer, and LNP build routines
"""Core build pipeline for constructing MD simulation systems."""

import os
from functools import partial

import MDAnalysis as mda
import numpy as np
from pydantic import validate_call

from mdfactory.parametrize import (
    DISPATCH_ENGINE_PARAMETRIZE,
    DISPATCH_TOPOLOGY_BUILD,
    retrieve_or_deposit_parameters,
)

from .check import check_bilayer_buildable
from .models.composition import BilayerComposition, LNPComposition, MixedBoxComposition
from .models.input import BuildInput
from .models.parametrization import SmirnoffConfig
from .models.species import SingleMoleculeSpecies
from .run_schedules import RunScheduleManager
from .setup.solvation import ionize, solvate
from .utils.setup_utilities import create_bilayer_from_model, create_mixed_box_universe
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


def _get_parametrize_function(inp: BuildInput):
    """Get the parametrization function, optionally wrapping with config for SMIRNOFF."""
    parametrize_function = DISPATCH_ENGINE_PARAMETRIZE.get(inp.engine, {}).get(
        inp.parametrization, None
    )
    if parametrize_function is None:
        raise NotImplementedError(
            f"Cannot parametrize with {inp.parametrization} for engine {inp.engine}"
        )

    # Wrap to include config for smirnoff
    if inp.parametrization == "smirnoff" and isinstance(inp.parametrization_config, SmirnoffConfig):
        parametrize_function = partial(
            parametrize_function, smirnoff_config=inp.parametrization_config
        )

    return parametrize_function


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

    # copy run files from template directory (run_schedules.yaml)
    manager = RunScheduleManager()
    manager.copy_run_files_with_check(
        engine=inp.engine, system_type="mixedbox", target_folder=os.getcwd(), force_copy=True
    )
    logger.info("Run schedule files copied.")


@validate_call
def create_mixedbox_by_compression(inp: BuildInput):
    """Create a mixed-box universe by packing at low density then compressing with OpenMM.

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
    parametrize_function = _get_parametrize_function(inp)
    topology_function = DISPATCH_TOPOLOGY_BUILD.get(inp.engine, None)
    if topology_function is None:
        raise NotImplementedError(f"Cannot build topology for engine {inp.engine}.")

    dec = retrieve_or_deposit_parameters(engine=inp.engine, parametrization=inp.parametrization)
    parametrize_with_db = dec(parametrize_function)
    parameters = [parametrize_with_db(spec) for spec in inp.system.species]
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
            u, wd / "mixedbox.pdb", wd / "topology.top", target_density=inp.system.target_density
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

    # 8. copy run schedule files
    manager = RunScheduleManager()
    manager.copy_run_files_with_check(
        engine=inp.engine, system_type="bilayer", target_folder=os.getcwd(), force_copy=True
    )
    logger.info("Run schedule files copied.")


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
    parametrize_function = _get_parametrize_function(inp)

    # preliminary parameters (only lipids)
    dec = retrieve_or_deposit_parameters(engine=inp.engine, parametrization=inp.parametrization)
    parametrize_with_db = dec(parametrize_function)
    parameters = [parametrize_with_db(spec) for spec in inp.system.species]

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

    # Step 7: Copy run schedule files
    manager = RunScheduleManager()
    manager.copy_run_files_with_check(
        engine=inp.engine, system_type="lnp", target_folder=os.getcwd(), force_copy=True
    )
    logger.info("Run schedule files copied.")


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
        parametrization_config=inp.parametrization_config,
        engine=inp.engine,
    )

    # Use the existing mixedbox compression workflow for the unit cell
    u_unit, parameters, parametrize_function, topology_function = create_mixedbox_by_compression(
        core_inp
    )
    logger.info("Core unit cell compressed to target density.")

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
        parametrization_config=inp.parametrization_config,
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


def ionize_solvated_system(ion_config, u_solvated, total_charge):
    """Add ions for neutralization and salt concentration.

    Parameters
    ----------
    ion_config : IonizationConfig or None
        Ionization settings (concentration, neutralize, etc.)
    u_solvated : mda.Universe
        Solvated system universe
    total_charge : int
        Total charge of the system before ionization

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

    n_water = len(u_solvated.select_atoms("water").residues)
    if n_water == 0:
        raise ValueError("Cannot ionize system without water.")
    c_ions = ion_config.concentration  # mol/l
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

    na_spec = SingleMoleculeSpecies(count=num_na, smiles="[Na+]", resname="NA")
    cl_spec = SingleMoleculeSpecies(count=num_cl, smiles="[Cl-]", resname="CL")
    return u_ionized, [na_spec, cl_spec]
