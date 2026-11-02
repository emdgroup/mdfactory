# ABOUTME: OpenMM simulation utilities for box compression, equilibration, and relaxation
# ABOUTME: Provides convergence checks, density-targeted compression, and protein restraints
"""OpenMM simulation utilities for box compression and equilibration."""

import sys

import cycler
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import openmm as mm
from openmm import app, unit


def _add_protein_rmsd_restraint(system, positions, protein_indices, restraint_k):
    """Restrain protein shape without opposing barostat translations.

    ``RMSDForce`` removes the best-fit rigid translation and rotation before
    measuring displacement, so the barostat can move molecule centers freely.
    """
    rmsd = mm.RMSDForce(positions, list(protein_indices))
    restraint = mm.CustomCVForce("k*rmsd^2")
    # rmsd^2 is a mean; scaling by the atom count gives k * sum(displacement**2).
    effective_k = restraint_k * len(protein_indices)
    restraint.addGlobalParameter("k", effective_k * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addCollectiveVariable("rmsd", rmsd)
    system.addForce(restraint)
    return restraint


def _state_data_reporter(target):
    """Return a StateDataReporter logging energy, temperature, density, and volume."""
    return app.StateDataReporter(
        target,
        1000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True,
        volume=True,
    )


def _universe_from_simulation(simulation, u, stem):
    """Write ``<stem>.pdb``/``<stem>.mmcif`` and return a copy of *u* with the final state.

    The returned universe keeps the topology of *u* and takes positions and
    box dimensions from the simulation's current state.
    """
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with open(f"{stem}.pdb", "w") as f:
        app.PDBFile.writeFile(simulation.topology, positions, f, keepIds=True)
    with open(f"{stem}.mmcif", "w") as f:
        app.PDBxFile.writeFile(simulation.topology, positions, f, keepIds=True)
    u_tmp = mda.Universe(app.PDBxFile(f"{stem}.mmcif"))
    # Use Merge instead of copy() to avoid file reference issues
    ret = mda.Merge(u.atoms)
    ret.atoms.positions = u_tmp.atoms.positions
    ret.dimensions = u_tmp.dimensions
    return ret


def rescale_to_volume(context, target_volume_nm3, rigid_indices=None):
    """Isotropically expand a rectangular box and its molecule centers to a target volume.

    Each molecule moves rigidly with its center of mass, so internal geometry is
    unchanged and intermolecular distances only grow. Atoms in *rigid_indices*
    (e.g. a multi-chain protein) are treated as a single rigid unit even when
    they span several molecules, so the unit is not pulled apart.

    Parameters
    ----------
    context : openmm.Context
        Context whose positions and box vectors are updated in place.
    target_volume_nm3 : float
        Target box volume in nm^3. Must not be smaller than the current volume.
    rigid_indices : iterable of int or None, optional
        Particle indices to move together as one unit. Default is None.

    Returns
    -------
    float
        Linear scale factor applied to the box edges.

    Raises
    ------
    ValueError
        If the target volume is smaller than the current volume (shrinking by
        scaling would create overlaps).

    """
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    a, b, c = (v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors())
    volume = a[0] * b[1] * c[2]
    if target_volume_nm3 < volume:
        raise ValueError(
            f"Target volume {target_volume_nm3:.3f} nm³ is smaller than the current "
            f"volume {volume:.3f} nm³; rescaling can only expand the box."
        )
    scale = (target_volume_nm3 / volume) ** (1.0 / 3.0)

    system = context.getSystem()
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(len(positions))]
    )

    rigid = set(rigid_indices) if rigid_indices is not None else set()
    groups = []
    rigid_group = []
    for molecule in context.getMolecules():
        if rigid and rigid.intersection(molecule):
            rigid_group.extend(molecule)
        else:
            groups.append(list(molecule))
    if rigid_group:
        groups.append(rigid_group)

    new_positions = positions.copy()
    for group in groups:
        group_masses = masses[group]
        weights = group_masses if group_masses.sum() > 0 else np.ones(len(group))
        center = (positions[group] * weights[:, None]).sum(axis=0) / weights.sum()
        new_positions[group] += (scale - 1.0) * center

    context.setPeriodicBoxVectors(
        mm.Vec3(*(np.asarray(a) * scale)) * unit.nanometer,
        mm.Vec3(*(np.asarray(b) * scale)) * unit.nanometer,
        mm.Vec3(*(np.asarray(c) * scale)) * unit.nanometer,
    )
    context.setPositions(new_positions * unit.nanometer)
    return scale


def finish_compression(
    context,
    target_volume_nm3,
    *,
    converged,
    require_convergence,
    not_reached_message,
    overshoot_tolerance=0.15,
    rigid_indices=None,
):
    """Settle a compressed box exactly on its target volume.

    Parameters
    ----------
    context : openmm.Context
        Context after the compression run; updated in place when rescaled.
    target_volume_nm3 : float
        Target box volume in nm^3 (for a density target, total mass / density).
    converged : bool
        Whether the compression loop reached the target.
    require_convergence : bool
        Raise when not converged; otherwise warn and keep the box as is.
    not_reached_message : str
        Message used when the target was not reached.
    overshoot_tolerance : float, optional
        Largest relative density overshoot (``target_volume / volume - 1``)
        that is corrected by rescaling. Default is 0.15.
    rigid_indices : iterable of int or None, optional
        Particles moved as one rigid unit during rescaling.

    Returns
    -------
    float or None
        Relative overshoot that was corrected, or None when the target was not
        reached.

    Raises
    ------
    RuntimeError
        If the target was not reached and *require_convergence* is True, or the
        overshoot exceeds *overshoot_tolerance*.

    """
    if not converged:
        if require_convergence:
            raise RuntimeError(not_reached_message)
        print(f"WARNING: {not_reached_message} Continuing with the box as compressed.")
        return None

    box = context.getState().getPeriodicBoxVectors()
    volume_nm3 = (box[0][0] * box[1][1] * box[2][2]).value_in_unit(unit.nanometer**3)
    overshoot = target_volume_nm3 / volume_nm3 - 1.0
    if overshoot > overshoot_tolerance:
        raise RuntimeError(
            f"Box compression overshot the target density by {overshoot:.1%}, exceeding "
            f"the {overshoot_tolerance:.1%} that can be corrected by rescaling."
        )
    if volume_nm3 < target_volume_nm3:
        scale = rescale_to_volume(context, target_volume_nm3, rigid_indices=rigid_indices)
        print(
            f"Compression overshot the target density by {overshoot:.2%}; expanded the box "
            f"from {volume_nm3:.3f} to {target_volume_nm3:.3f} nm³ (edge x{scale:.4f})."
        )
    return max(overshoot, 0.0)


def monitor_volume_convergence(volumes, window_size=10, tolerance=0.02):
    """Check if box volume has converged by comparing running averages.

    Parameters
    ----------
    volumes : list of float
        Volume values recorded at each reporting interval.
    window_size : int, optional
        Number of steps for each running-average window. Default is 10.
    tolerance : float, optional
        Relative tolerance for convergence (e.g. 0.02 = 2%). Default is 0.02.

    Returns
    -------
    converged : bool
        Whether the relative change between consecutive windows is below
        *tolerance*.
    current_avg : float or None
        Mean volume over the most recent window, or None if insufficient data.
    previous_avg : float or None
        Mean volume over the preceding window, or None if insufficient data.

    """
    if len(volumes) < 2 * window_size:
        return False, None, None

    # Calculate running averages for current and previous windows
    current_avg = np.mean(volumes[-window_size:])
    previous_avg = np.mean(volumes[-2 * window_size : -window_size])

    # Check convergence
    relative_change = abs(current_avg - previous_avg) / previous_avg
    converged = relative_change < tolerance

    return converged, current_avg, previous_avg


def simulate_compression_until_constant_volume(
    simulation: app.Simulation, steps: int = 100000, tolerance: float = 0.05
) -> tuple[list[float], int, bool]:
    """Run an NPT simulation until the box volume converges.

    Step the simulation forward in 1000-step increments, recording the volume
    after each increment, and stop early when ``monitor_volume_convergence``
    signals convergence.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        An initialized OpenMM simulation with a barostat.
    steps : int, optional
        Maximum number of MD steps. Default is 100000.
    tolerance : float, optional
        Relative volume-change tolerance passed to
        ``monitor_volume_convergence``. Default is 0.05.

    Returns
    -------
    volumes : list of float
        Recorded box volumes in nm^3.
    step : int
        Total number of steps executed.
    converged : bool
        Whether volume convergence was reached before *steps* was exhausted.

    """
    volumes = []
    report_interval = 1000
    convergence_check_interval = 1000
    window_size = 10

    converged = False
    step = 0

    for _ in range(1, steps // report_interval + 1):
        simulation.step(report_interval)
        step += report_interval

        # Get current state
        state = simulation.context.getState()
        box_vectors = state.getPeriodicBoxVectors()

        # Calculate volume (in nm³)
        volume = (box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]).value_in_unit(
            unit.nanometer**3
        )
        # energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        volumes.append(volume)
        # energies_phase1.append(energy)

        # Check convergence every 1000 steps
        if step % convergence_check_interval == 0:
            converged, current_avg, previous_avg = monitor_volume_convergence(
                volumes, window_size=window_size, tolerance=tolerance
            )

            print(f"Step {step:6d}: Volume = {volume:.2f} nm³")
            if converged:
                print(
                    f"Volume converged! Current avg: {current_avg:.2f}, "
                    f"Previous avg: {previous_avg:.2f}"
                )
                break
            elif len(volumes) >= 2 * window_size:  # Only show convergence info after enough data
                rel_change = (
                    abs(current_avg - previous_avg) / previous_avg * 100
                    if current_avg and previous_avg
                    else np.nan
                )
                print(f"  Convergence check: {rel_change:.2f}% change")
    return volumes, step, converged


def simulate_compression_until_density_reached(
    simulation: app.Simulation, mass, steps: int = 100000, target_density: float = 1.0
) -> tuple[list[float], int, bool]:
    """Run an NPT simulation until the system density reaches a target value.

    Step the simulation in 25-step increments (one default barostat interval),
    computing the instantaneous density from the total mass and box volume.
    Stop early once the density meets or exceeds *target_density*.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        An initialized OpenMM simulation with a barostat.
    mass : openmm.unit.Quantity
        Total mass of the system (with OpenMM units attached).
    steps : int, optional
        Maximum number of MD steps. Default is 100000.
    target_density : float, optional
        Target density in g/mL. Default is 1.0.

    Returns
    -------
    densities : list of float
        Recorded densities in g/mL at each reporting interval.
    step : int
        Total number of steps executed.
    converged : bool
        Whether the target density was reached before *steps* was exhausted.

    """
    densities = []
    # Check more frequently than the reporting cadence so high-pressure volume
    # moves cannot run far past the target between observations.
    report_interval = 25
    convergence_check_interval = 1000

    converged = False
    step = 0

    for _ in range(1, steps // report_interval + 1):
        simulation.step(report_interval)
        step += report_interval

        # Get current state
        state = simulation.context.getState()
        box_vectors = state.getPeriodicBoxVectors()

        # Calculate volume (in nm³)
        volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
        density = (mass / volume).value_in_unit(unit.gram / unit.item / unit.milliliter)

        densities.append(density)

        if density >= target_density:
            converged = True
            print(f"Density reached target: {density:.4f} g/mL at step {step}")
            break
        if step % convergence_check_interval == 0:
            print(f"Step {step:6d}: Density = {density:.4f} g/mL")
    return densities, step, converged


def create_sphere(u, pdb, top, radius, steps: int = 100_000):
    """Compress a system into a sphere using a harmonic radial restraint.

    Apply a custom external force that penalizes atoms beyond *radius* from
    the origin, minimize, and run Langevin dynamics to pack the system into
    a spherical shape.

    Parameters
    ----------
    u : mda.Universe
        Reference MDAnalysis Universe whose topology is preserved in the
        returned object.
    pdb : str or Path
        Path to the input PDB file for OpenMM.
    top : str or Path
        Path to the GROMACS topology file.
    radius : float
        Sphere radius in nanometers.
    steps : int, optional
        Number of MD steps to run after minimization. Default is 100000.

    Returns
    -------
    mda.Universe
        Universe with updated positions packed into the sphere. Box
        dimensions are set to None (non-periodic).

    """
    # TODO: use proper logging
    print("Loading GROMACS files...")
    gro = app.PDBFile(str(pdb))
    top = app.GromacsTopFile(str(top))

    # Create system
    print("Creating OpenMM system...")
    system = top.createSystem(
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    print("Radius in nm lol", radius)
    force = mm.CustomExternalForce(f"10*max(0, r-{radius})^2; r=sqrt(x*x+y*y+z*z)")
    system.addForce(force)
    for i in range(system.getNumParticles()):
        force.addParticle(i, [])

    # Create integrator and simulation
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    simulation = app.Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    simulation.reporters.append(app.PDBReporter("sphere_creation.pdb", reportInterval=5000))

    simulation.reporters.append(_state_data_reporter("data.txt"))
    simulation.reporters.append(_state_data_reporter(sys.stdout))
    print("Minimizing energy...")
    simulation.minimizeEnergy()

    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    with open("min.pdb", "w") as f_min:
        app.PDBFile.writeFile(simulation.topology, minimized_positions, f_min)

    simulation.step(steps)

    positions = simulation.context.getState(getPositions=True).getPositions()
    with open("sphere.pdb", "w") as f_sphere_pdb:
        app.PDBFile.writeFile(simulation.topology, positions, f_sphere_pdb, keepIds=True)
    with open("sphere.mmcif", "w") as f_sphere_mmcif:
        app.PDBxFile.writeFile(simulation.topology, positions, f_sphere_mmcif, keepIds=True)
    u_tmp = mda.Universe(app.PDBxFile("sphere.mmcif"))
    # Use Merge instead of copy() to avoid file reference issues
    ret = mda.Merge(u.atoms)
    ret.dimensions = None
    ret.atoms.positions = u_tmp.atoms.positions
    return ret


def compress_box(
    u,
    pdb,
    top,
    target_density=1.0,
    pressure=1000.0,
    steps_compression=200_000,
    target_volume=None,
    protein_indices=None,
    restraint_k=1000.0,
    overshoot_tolerance=0.15,
    require_convergence=True,
):
    """Compress a periodic box to a target density using high-pressure NPT.

    Apply a Monte Carlo barostat at elevated pressure, minimize, and run
    dynamics until the system density reaches *target_density* (or, when
    *target_volume* is given, the equivalent density for that volume). When
    *protein_indices* is given, their best-fit RMSD from the starting structure
    is restrained through minimization and compression.

    At high pressure the box can sweep past the target between two checks.
    Any such overshoot is undone by expanding the box and molecule centers to
    the exact target volume (see :func:`rescale_to_volume`).

    Parameters
    ----------
    u : mda.Universe
        Reference MDAnalysis Universe whose topology is preserved in the
        returned object.
    pdb : str or Path
        Path to the input PDB file for OpenMM.
    top : str or Path
        Path to the GROMACS topology file.
    target_density : float, optional
        Target density in g/mL. Default is 1.0. Ignored when *target_volume*
        is given.
    pressure : float, optional
        Barostat pressure in bar. Default is 1000.0.
    steps_compression : int, optional
        Maximum number of compression MD steps. Default is 200000.
    target_volume : float or None, optional
        Target box volume in Å^3. When given, it replaces *target_density*
        (the ``fixed_box`` sizing mode). Default is None.
    protein_indices : list of int or None, optional
        0-based atom indices to hold with harmonic position restraints. Default
        is None (no restraints).
    restraint_k : float, optional
        Restraint force constant in kJ/mol/nm^2. Default is 1000.0.
    overshoot_tolerance : float, optional
        Largest relative density overshoot corrected by rescaling; larger
        overshoots raise. Default is 0.15 (15%).
    require_convergence : bool, optional
        Raise when the target is not reached within *steps_compression*. When
        False, only warn and return the box as compressed so far (the behavior
        LNP core compression relies on). Default is True.

    Returns
    -------
    mda.Universe
        Universe with compressed positions and updated box dimensions.
        Positions are wrapped by residue unless *protein_indices* is given, in
        which case the caller is responsible for re-imaging the protein.

    Raises
    ------
    RuntimeError
        If the requested density or volume is not reached (and
        *require_convergence* is True), or compression overshoots its target
        beyond *overshoot_tolerance*.

    """
    # TODO: use proper logging
    print("Loading GROMACS files...")
    gro = app.PDBFile(str(pdb))
    top = app.GromacsTopFile(str(top), periodicBoxVectors=gro.topology.getPeriodicBoxVectors())

    # Create system
    print("Creating OpenMM system...")
    system = top.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds
    )

    if protein_indices is not None:
        _add_protein_rmsd_restraint(system, gro.positions, protein_indices, restraint_k)

    barostat = mm.MonteCarloBarostat(
        pressure * unit.bar,
        300 * unit.kelvin,
    )
    system.addForce(barostat)

    # Create integrator and simulation
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    simulation = app.Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    simulation.reporters.append(app.PDBReporter("compress.pdb", reportInterval=5000))

    simulation.reporters.append(_state_data_reporter("data.txt"))

    state = simulation.context.getState()
    box = state.getPeriodicBoxVectors()
    totalMass = 0 * unit.dalton
    for i in range(system.getNumParticles()):
        totalMass += system.getParticleMass(i)
    volume = box[0][0] * box[1][1] * box[2][2]
    print(system.getNumParticles())
    density = (totalMass / volume).value_in_unit(unit.gram / unit.item / unit.milliliter)
    print(density, totalMass, volume)
    # return

    # Energy minimization
    print("Minimizing energy...")
    simulation.minimizeEnergy()

    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(simulation.topology, minimized_positions, open("min.pdb", "w"))

    if target_volume is not None:
        # target_volume is in Å^3; OpenMM box volumes are in nm^3.
        target_volume_nm3 = target_volume / 1000.0
        target_density = (totalMass / target_volume_nm3 / unit.nanometer**3).value_in_unit(
            unit.gram / unit.item / unit.milliliter
        )
    else:
        target_volume_nm3 = (
            totalMass / (target_density * unit.gram / unit.item / unit.milliliter)
        ).value_in_unit(unit.nanometer**3)

    densities, completed_steps, converged = simulate_compression_until_density_reached(
        simulation, mass=totalMass, steps=steps_compression, target_density=target_density
    )
    final_density = f"{densities[-1]:.3f} g/mL" if densities else "nan"

    finish_compression(
        simulation.context,
        target_volume_nm3=target_volume_nm3,
        converged=converged,
        require_convergence=require_convergence,
        overshoot_tolerance=overshoot_tolerance,
        rigid_indices=protein_indices,
        not_reached_message=(
            f"Box compression did not reach {target_volume_nm3:.3f} nm³ "
            f"({target_density:.3f} g/mL) in {completed_steps} steps; "
            f"final density was {final_density}."
        ),
    )

    ret = _universe_from_simulation(simulation, u, "compressed_system")
    # A restrained protein can cross the PBC boundary; the caller re-images it.
    if protein_indices is None:
        ret.atoms.positions = ret.atoms.wrap(compound="residues")
    return ret


def compress_equilibrate_bilayer(
    u,
    pdb,
    top,
    pressure=1000.0,
    steps_compression=200_000,
    # steps_equil=500_000,
    # steps_equil=200_000,
    steps_equil=0,
):
    """Compress and equilibrate a lipid bilayer in two phases.

    Phase 1 applies z-coordinate restraints with a membrane barostat at
    elevated pressure (XY-isotropic, Z-fixed) until volume converges.
    Phase 2 removes the restraints and continues equilibration at 1 bar.

    Parameters
    ----------
    u : mda.Universe
        Reference MDAnalysis Universe whose topology is preserved in the
        returned object.
    pdb : str or Path
        Path to the input PDB file for OpenMM.
    top : str or Path
        Path to the GROMACS topology file.
    pressure : float, optional
        Barostat pressure in bar for Phase 1. Default is 1000.0.
    steps_compression : int, optional
        Maximum MD steps for Phase 1 compression. Default is 200000.
    steps_equil : int, optional
        Maximum MD steps for Phase 2 equilibration. Default is 0.

    Returns
    -------
    mda.Universe
        Equilibrated bilayer Universe with updated positions and box
        dimensions. An ``equilibration_analysis.png`` plot is also saved.

    """
    # TODO: use proper logging
    print("Loading GROMACS files...")
    gro = app.PDBFile(str(pdb))
    top = app.GromacsTopFile(str(top), periodicBoxVectors=gro.topology.getPeriodicBoxVectors())

    # Create system
    print("Creating OpenMM system...")
    system = top.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds
    )

    # PHASE 1: Equilibration WITH z-coordinate restraints
    print("\n=== PHASE 1: Equilibration with Z-restraints ===")

    # Add z-coordinate restraints
    restraint_force = mm.CustomExternalForce("k*((z-z0)^2)")
    restraint_force.addGlobalParameter("k", 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint_force.addPerParticleParameter("z0")

    for i in range(system.getNumParticles()):
        z_coord = gro.positions[i][2]
        restraint_force.addParticle(i, [z_coord])

    restraint_force_index = system.addForce(restraint_force)

    # Add anisotropic pressure control
    # barostat = mm.MonteCarloAnisotropicBarostat(
    #     (pressure * unit.bar, pressure * unit.bar, 1.0 * unit.bar),
    #     300 * unit.kelvin,
    #     True,
    #     True,
    #     False,
    # )
    barostat = mm.MonteCarloMembraneBarostat(
        pressure * unit.bar,
        200 * unit.bar * unit.nanometer,
        300 * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFixed,
    )
    barostat_index = system.addForce(barostat)

    # Create integrator and simulation
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    simulation = app.Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    ####
    nonbonded = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    charges = []
    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        charges.append(charge)
    print("total charge", np.sum(charges))
    ####

    simulation.reporters.append(app.PDBReporter("compress.pdb", reportInterval=5000))
    # simulation.reporters.append(app.PDBReporter("compress.pdb", reportInterval=100))

    # Energy minimization
    print("Minimizing energy...")
    simulation.minimizeEnergy()

    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(simulation.topology, minimized_positions, open("min.pdb", "w"))

    # Set initial velocities
    # simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Phase 1 equilibration with volume monitoring
    print("Starting Phase 1 equilibration with z-restraints...")

    volumes_phase1, step, converged = simulate_compression_until_constant_volume(
        simulation, steps=steps_compression, tolerance=0.02
    )
    print(f"Phase 1 completed after {step} steps. Converged: {converged}")

    # PHASE 2: Equilibration WITHOUT z-coordinate restraints
    print("\n=== PHASE 2: Equilibration without Z-restraints ===")

    # Remove z-coordinate restraints
    print("Removing z-coordinate restraints...")
    system.removeForce(barostat_index)
    system.removeForce(restraint_force_index)

    # barostat_unrestricted = mm.MonteCarloAnisotropicBarostat(
    #     (1.0 * unit.bar, 1.0 * unit.bar, 1.0 * unit.bar),
    #     300 * unit.kelvin,
    #     True,
    #     True,
    #     False,
    # )
    barostat_unrestricted = mm.MonteCarloMembraneBarostat(
        1.0 * unit.bar,
        200 * unit.bar * unit.nanometer,
        300 * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFixed,
    )
    system.addForce(barostat_unrestricted)

    # Reinitialize the simulation with updated system
    integrator2 = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    simulation2 = app.Simulation(top.topology, system, integrator2)
    simulation2.context.setState(simulation.context.getState(getPositions=True, getVelocities=True))
    simulation2.reporters.append(app.PDBReporter("equil.pdb", reportInterval=50000))

    # Phase 2 equilibration
    print("Starting Phase 2 equilibration without z-restraints...")
    volumes_phase2, step, converged = simulate_compression_until_constant_volume(
        simulation2, steps=steps_equil, tolerance=1e-14
    )
    print("Phase 2 completed!")

    # Save final state
    print("Saving final equilibrated state...")
    ret = _universe_from_simulation(simulation2, u, "equilibrated_system")

    # Analysis and plotting
    print("\n=== ANALYSIS ===")

    # Calculate convergence statistics
    final_volume_avg = (
        np.mean(volumes_phase1[-1000:]) if len(volumes_phase1) >= 1000 else np.mean(volumes_phase1)
    )
    volume_std = (
        np.std(volumes_phase1[-1000:]) if len(volumes_phase1) >= 1000 else np.std(volumes_phase1)
    )

    print(f"Phase 1 - Final average volume: {final_volume_avg:.2f} ± {volume_std:.2f} nm³")
    print(f"Phase 1 - Volume range: {min(volumes_phase1):.2f} - {max(volumes_phase1):.2f} nm³")

    if len(volumes_phase2) > 0:
        phase2_avg = np.mean(volumes_phase2)
        phase2_std = np.std(volumes_phase2)
        print(f"Phase 2 - Average volume: {phase2_avg:.2f} ± {phase2_std:.2f} nm³")
        print(f"Phase 2 - Volume range: {min(volumes_phase2):.2f} - {max(volumes_phase2):.2f} nm³")

    # Create plots
    # Enable grid and update its appearance
    plt.rcParams.update({"axes.grid": True})
    plt.rcParams.update({"grid.color": "silver"})
    plt.rcParams.update({"grid.linestyle": "--"})

    # Set figure resolution
    plt.rcParams.update({"figure.dpi": 150})

    # Hide the top and right spines
    plt.rcParams.update({"axes.spines.top": False})
    plt.rcParams.update({"axes.spines.right": False})

    # Increase font sizes
    plt.rcParams.update({"font.size": 12})  # General font size
    plt.rcParams.update({"axes.titlesize": 14})  # Title font size
    plt.rcParams.update({"axes.labelsize": 12})  # Axis label font size

    plt.rcParams.update({"axes.prop_cycle": cycler.cycler("color", ["#0F69AF"])})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Volume plots
    steps1 = np.arange(len(volumes_phase1))
    ax1.plot(steps1, volumes_phase1, "b-", alpha=0.7, label="Phase 1 (with Z-restraints)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Volume (nm³)")
    ax1.set_title("Box Volume - Phase 1")
    ax1.grid(True, alpha=0.3)

    if len(volumes_phase2) > 0:
        steps2 = np.arange(len(volumes_phase2))
        ax2.plot(steps2, volumes_phase2, "r-", alpha=0.7, label="Phase 2 (no Z-restraints)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Volume (nm³)")
        ax2.set_title("Box Volume - Phase 2")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("equilibration_analysis.png", dpi=300, bbox_inches="tight")

    return ret


def relax_with_protein_restraints(
    u,
    pdb,
    top,
    protein_indices,
    restraint_k=1000.0,
    steps=10000,
    pressure=1.0,
):
    """Minimize and briefly equilibrate with the protein structure restrained.

    Runs a short simulation with a best-fit RMSD restraint on the specified
    protein atoms, allowing water and ions to relax around the fixed protein
    structure without opposing barostat translations. By default it uses NPT at
    1 bar; pass ``pressure=None`` for NVT and an invariant box.

    Parameters
    ----------
    u : mda.Universe
        Reference MDAnalysis Universe whose topology is preserved.
    pdb : str or Path
        Path to the input PDB file for OpenMM.
    top : str or Path
        Path to the GROMACS topology file.
    protein_indices : list of int
        0-based atom indices to restrain (typically protein heavy atoms).
    restraint_k : float, optional
        Restraint force constant in kJ/mol/nm^2. Default is 1000.0.
    steps : int, optional
        Number of MD steps after minimization. Default is 10000.
    pressure : float or None, optional
        Isotropic pressure in bar. If None, do not add a barostat. Default is
        1.0 bar.

    Returns
    -------
    mda.Universe
        Universe with relaxed solvent positions. Protein positions
        remain at their restrained coordinates.

    """
    print("Loading GROMACS files for protein relaxation...")
    gro = app.PDBFile(str(pdb))
    top_file = app.GromacsTopFile(str(top), periodicBoxVectors=gro.topology.getPeriodicBoxVectors())

    print("Creating OpenMM system...")
    system = top_file.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    _add_protein_rmsd_restraint(system, gro.positions, protein_indices, restraint_k)

    if pressure is not None:
        barostat = mm.MonteCarloBarostat(pressure * unit.bar, 300 * unit.kelvin)
        system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    simulation = app.Simulation(top_file.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    simulation.reporters.append(_state_data_reporter(sys.stdout))

    print("Minimizing energy (protein restrained)...")
    simulation.minimizeEnergy()

    ensemble = "NPT" if pressure is not None else "NVT"
    print(f"Running {steps} steps of {ensemble} relaxation (protein restrained)...")
    simulation.step(steps)

    ret = _universe_from_simulation(simulation, u, "relaxed_system")

    print("Protein relaxation complete.")
    return ret
