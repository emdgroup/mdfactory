# ABOUTME: OpenMM simulation utilities for box compression, equilibration, and relaxation
# ABOUTME: Provides volume convergence monitoring, density-targeted compression, and protein restraints
"""OpenMM simulation utilities for box compression and equilibration."""

import sys

import cycler
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import openmm as mm
from openmm import app, unit


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

    Step the simulation in 1000-step increments and compute the instantaneous
    density from the total mass and current box volume. Stop early once the
    density meets or exceeds *target_density*.

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
    report_interval = 1000
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

        # Check convergence every 1000 steps
        if step % convergence_check_interval == 0:
            print(f"Step {step:6d}: Density = {density:.4f} g/mL")
            if density >= target_density:
                converged = True
                print(f"Density reached target: {density:.4f} g/mL")
                break
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

    simulation.reporters.append(
        app.StateDataReporter(
            "data.txt",
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            volume=True,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            volume=True,
        )
    )
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


def compress_box(u, pdb, top, target_density=1.0, pressure=1000.0, steps_compression=200_000):
    """Compress a periodic box to a target density using high-pressure NPT.

    Apply a Monte Carlo barostat at elevated pressure, minimize, and run
    dynamics until the system density reaches *target_density*.

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
        Target density in g/mL. Default is 1.0.
    pressure : float, optional
        Barostat pressure in bar. Default is 1000.0.
    steps_compression : int, optional
        Maximum number of compression MD steps. Default is 200000.

    Returns
    -------
    mda.Universe
        Universe with compressed positions and updated box dimensions.
        Positions are wrapped by residue.

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

    simulation.reporters.append(
        app.StateDataReporter(
            "data.txt",
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            volume=True,
        )
    )

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

    simulate_compression_until_density_reached(
        simulation, mass=totalMass, steps=steps_compression, target_density=target_density
    )

    state = simulation.context.getState()
    positions = simulation.context.getState(getPositions=True).getPositions()
    box_vectors = state.getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)
    app.PDBFile.writeFile(
        simulation.topology, positions, open("compressed_system.pdb", "w"), keepIds=True
    )
    app.PDBxFile.writeFile(
        simulation.topology, positions, open("compressed_system.mmcif", "w"), keepIds=True
    )
    u_tmp = mda.Universe(app.PDBxFile("compressed_system.mmcif"))
    # Use Merge instead of copy() to avoid file reference issues
    ret = mda.Merge(u.atoms)
    ret.atoms.positions = u_tmp.atoms.positions
    ret.dimensions = u_tmp.dimensions
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
    positions = simulation2.context.getState(getPositions=True).getPositions()

    state = simulation2.context.getState()
    box_vectors = state.getPeriodicBoxVectors()
    simulation2.topology.setPeriodicBoxVectors(box_vectors)
    app.PDBFile.writeFile(
        simulation2.topology, positions, open("equilibrated_system.pdb", "w"), keepIds=True
    )
    app.PDBxFile.writeFile(
        simulation2.topology, positions, open("equilibrated_system.mmcif", "w"), keepIds=True
    )
    u_tmp = mda.Universe(app.PDBxFile("equilibrated_system.mmcif"))
    # Use Merge instead of copy() to avoid file reference issues
    ret = mda.Merge(u.atoms)
    ret.atoms.positions = u_tmp.atoms.positions
    ret.dimensions = u_tmp.dimensions

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
):
    """Minimize and briefly equilibrate with protein atoms position-restrained.

    Runs a short NPT simulation at 1 bar with harmonic restraints on
    the specified protein atom indices, allowing water and ions to relax
    around the fixed protein structure.

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

    Returns
    -------
    mda.Universe
        Universe with relaxed solvent positions. Protein positions
        remain at their restrained coordinates.

    """
    print("Loading GROMACS files for protein relaxation...")
    gro = app.PDBFile(str(pdb))
    top_file = app.GromacsTopFile(
        str(top), periodicBoxVectors=gro.topology.getPeriodicBoxVectors()
    )

    print("Creating OpenMM system...")
    system = top_file.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    # Position restraints on protein atoms
    restraint_force = mm.CustomExternalForce(
        "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint_force.addGlobalParameter(
        "k", restraint_k * unit.kilojoules_per_mole / unit.nanometer**2
    )
    restraint_force.addPerParticleParameter("x0")
    restraint_force.addPerParticleParameter("y0")
    restraint_force.addPerParticleParameter("z0")

    for idx in protein_indices:
        pos = gro.positions[idx]
        restraint_force.addParticle(
            idx,
            [
                pos[0].value_in_unit(unit.nanometer),
                pos[1].value_in_unit(unit.nanometer),
                pos[2].value_in_unit(unit.nanometer),
            ],
        )

    system.addForce(restraint_force)

    # Isotropic barostat at 1 bar
    barostat = mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin)
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    simulation = app.Simulation(top_file.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            volume=True,
        )
    )

    print("Minimizing energy (protein restrained)...")
    simulation.minimizeEnergy()

    print(f"Running {steps} steps of NPT relaxation (protein restrained)...")
    simulation.step(steps)

    positions = simulation.context.getState(getPositions=True).getPositions()
    state = simulation.context.getState()
    box_vectors = state.getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)

    with open("relaxed_system.pdb", "w") as f:
        app.PDBFile.writeFile(simulation.topology, positions, f, keepIds=True)
    with open("relaxed_system.mmcif", "w") as f:
        app.PDBxFile.writeFile(simulation.topology, positions, f, keepIds=True)

    u_tmp = mda.Universe(app.PDBxFile("relaxed_system.mmcif"))
    ret = mda.Merge(u.atoms)
    ret.atoms.positions = u_tmp.atoms.positions
    ret.dimensions = u_tmp.dimensions
    ret.atoms.positions = ret.atoms.wrap(compound="residues")

    print("Protein relaxation complete.")
    return ret
