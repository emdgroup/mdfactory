# ABOUTME: Tests for staged OpenMM compression and its NaN recovery
# ABOUTME: Drive run_staged_compression with fake simulations; no OpenMM context is created
"""Tests for staged box compression and NaN retry in ``openmm_utils``."""

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from mdfactory.simulation import openmm_utils
from mdfactory.simulation.openmm_utils import (
    COMPRESSION_STAGE_FRACTIONS,
    _is_nan_failure,
    run_staged_compression,
)

NAN_MESSAGES = [
    "Particle coordinate is NaN.  For more information, see https://example.org",
    "NaN encountered during RMSD force calculation",
]
MASS = 1000.0 * unit.dalton
TO_G_PER_ML = unit.gram / unit.item / unit.milliliter
INITIAL_VOLUME = 1000.0  # nm^3
TARGET_VOLUME = 10.0  # nm^3
TARGET_DENSITY = (MASS / (TARGET_VOLUME * unit.nanometer**3)).value_in_unit(TO_G_PER_ML)


class World:
    """Shared record of everything the fake simulations do."""

    def __init__(self, fail_on_step_calls=(), fail_message=NAN_MESSAGES[0], fail_on_minimize=()):
        self.fail_on_step_calls = set(fail_on_step_calls)
        self.fail_message = fail_message
        self.fail_on_minimize = set(fail_on_minimize)
        self.step_calls = 0
        self.minimize_calls = 0
        self.minimized_edges = []
        self.made_dts = []
        self.restored_edges = []


class FakeSimulation:
    """Shrinks the cubic box a little on every ``step`` call."""

    def __init__(self, world, edge=INITIAL_VOLUME ** (1 / 3), integrator=None, topology=None):
        self.world = world
        self.edge = edge
        self.context = self
        self.integrator = integrator
        self.topology = topology
        self.reporters = []

    def getState(self, getPositions=False):
        return self

    def getPositions(self):
        return [self.edge]

    def getPeriodicBoxVectors(self):
        return (
            mm.Vec3(self.edge, 0, 0) * unit.nanometer,
            mm.Vec3(0, self.edge, 0) * unit.nanometer,
            mm.Vec3(0, 0, self.edge) * unit.nanometer,
        )

    def setPeriodicBoxVectors(self, a, b, c):
        self.edge = a[0].value_in_unit(unit.nanometer)
        self.world.restored_edges.append(self.edge)

    def setPositions(self, positions):
        pass

    def step(self, _steps):
        self.world.step_calls += 1
        if self.world.step_calls in self.world.fail_on_step_calls:
            raise mm.OpenMMException(self.world.fail_message)
        self.edge *= 0.98

    def minimizeEnergy(self, reporter=None):
        self.world.minimize_calls += 1
        if self.world.minimize_calls in self.world.fail_on_minimize:
            raise mm.OpenMMException(NAN_MESSAGES[0])
        self.world.minimized_edges.append(self.edge)


def _run(world, simulation=None, steps=100_000, max_nan_retries=3):
    def make_simulation(dt):
        world.made_dts.append(dt)
        return FakeSimulation(world, edge=0.0)

    return run_staged_compression(
        simulation or FakeSimulation(world),
        make_simulation,
        mass=MASS,
        target_density=TARGET_DENSITY,
        steps=steps,
        dt=0.002,
        max_nan_retries=max_nan_retries,
    )


@pytest.mark.parametrize("message", NAN_MESSAGES)
def test_nan_messages_are_recognised(message):
    assert _is_nan_failure(mm.OpenMMException(message))


def test_other_failures_are_not_nan():
    assert not _is_nan_failure(mm.OpenMMException("Unknown force"))
    assert not _is_nan_failure(ValueError("NaN"))


def test_stages_minimize_between_but_not_after_the_last():
    world = World()
    _, densities, steps, converged = _run(world)
    assert converged
    assert world.minimize_calls == len(COMPRESSION_STAGE_FRACTIONS) - 1
    assert steps > 0
    assert densities[-1] >= TARGET_DENSITY
    assert world.made_dts == []


def test_stages_already_passed_are_skipped():
    world = World()
    # Start at the 0.7 target-density mark: the 0.3 and 0.6 stages are already done.
    start_edge = (TARGET_VOLUME / 0.7) ** (1 / 3)
    _, _, _, converged = _run(world, simulation=FakeSimulation(world, edge=start_edge))
    assert converged
    assert world.minimize_calls == 0


def _step_calls_to_reach(fraction):
    """Step calls the unperturbed fake needs to reach *fraction* of the target density."""
    edge = INITIAL_VOLUME ** (1 / 3)
    threshold = (TARGET_VOLUME / fraction) ** (1 / 3)
    calls = 0
    while edge > threshold:
        edge *= 0.98
        calls += 1
    return calls


@pytest.mark.parametrize("message", NAN_MESSAGES)
def test_nan_retries_from_checkpoint_with_halved_dt(message):
    # Fail on the first step of the second stage: the checkpoint is the box left by
    # the first stage's minimization, not the starting box.
    world = World(
        fail_on_step_calls={_step_calls_to_reach(COMPRESSION_STAGE_FRACTIONS[0]) + 1},
        fail_message=message,
    )
    _, _, _, converged = _run(world)
    assert converged
    assert world.made_dts == [0.001]
    assert world.restored_edges == [world.minimized_edges[0]]


def test_nan_in_minimization_restarts_the_stage():
    world = World(fail_on_minimize={1})
    _, _, _, converged = _run(world)
    assert converged
    assert world.made_dts == [0.001]
    # The failed minimization came before any checkpoint beyond the start.
    assert world.restored_edges == [INITIAL_VOLUME ** (1 / 3)]


def test_retries_are_bounded_and_the_error_is_actionable():
    world = World(fail_on_step_calls=range(1, 1000))
    with pytest.raises(RuntimeError, match="NaN.*lower target density") as excinfo:
        _run(world, max_nan_retries=2)
    assert world.made_dts == [0.001, 0.0005]
    assert isinstance(excinfo.value.__cause__, mm.OpenMMException)


def test_non_nan_openmm_errors_are_not_retried():
    world = World(fail_on_step_calls={1}, fail_message="Unknown force")
    with pytest.raises(mm.OpenMMException, match="Unknown force"):
        _run(world)
    assert world.made_dts == []


def test_unconverged_run_stops_in_the_first_unfinished_stage():
    world = World()
    _, densities, steps, converged = _run(world, steps=100)
    assert not converged
    assert steps <= 100
    assert densities[-1] < TARGET_DENSITY
    # It stopped in the first stage instead of starting the next.
    assert world.minimize_calls == 0


def test_step_budget_is_shared_across_stages_and_the_last_stage_can_fall_short():
    # The budget finishes the first two stages but runs out two checks before the
    # target. Were each stage given the full budget, the last one would converge.
    calls = _step_calls_to_reach(COMPRESSION_STAGE_FRACTIONS[-1]) - 2
    world = World()
    _, densities, steps, converged = _run(world, steps=25 * calls)
    assert not converged
    assert steps == 25 * calls
    assert world.step_calls == calls
    assert densities[-1] < TARGET_DENSITY
    # Both inter-stage minimizations ran; none after the unfinished last stage.
    assert world.minimize_calls == len(COMPRESSION_STAGE_FRACTIONS) - 1


class FakeTopology:
    def getPeriodicBoxVectors(self):
        return None


@pytest.mark.parametrize("message", NAN_MESSAGES)
def test_compress_box_retries_with_one_restraint_and_hands_the_new_simulation_on(
    message, monkeypatch, tmp_path
):
    """Drive compress_box through a NaN retry with OpenMM's file and simulation I/O faked."""
    monkeypatch.chdir(tmp_path)
    world = World(fail_on_step_calls={1}, fail_message=message)
    gro = type("Gro", (), {"topology": FakeTopology(), "positions": [mm.Vec3(0, 0, 0)] * 2})()
    system = mm.System()
    for _ in range(2):
        system.addParticle(MASS.value_in_unit(unit.dalton) / 2)
    simulations = []
    restraint_calls = []
    finished = []
    exported = []

    class FakePDBFile:
        def __new__(cls, path):
            return gro

        @staticmethod
        def writeFile(*args, **kwargs):
            pass

    class FakeTopFile:
        def __init__(self, path, periodicBoxVectors=None):
            self.topology = gro.topology

        def createSystem(self, **kwargs):
            return system

    def fake_simulation(topology, system_, integrator):
        sim = FakeSimulation(world, integrator=integrator, topology=topology)
        simulations.append(sim)
        return sim

    monkeypatch.setattr(openmm_utils.app, "PDBFile", FakePDBFile)
    monkeypatch.setattr(openmm_utils.app, "GromacsTopFile", FakeTopFile)
    monkeypatch.setattr(openmm_utils.app, "Simulation", fake_simulation)
    monkeypatch.setattr(openmm_utils.app, "PDBReporter", lambda *a, **k: object())
    monkeypatch.setattr(openmm_utils, "_state_data_reporter", lambda target: object())
    monkeypatch.setattr(
        openmm_utils,
        "_add_protein_rmsd_restraint",
        lambda *args: restraint_calls.append(args),
    )
    monkeypatch.setattr(
        openmm_utils, "finish_compression", lambda context, *a, **k: finished.append(context)
    )
    monkeypatch.setattr(
        openmm_utils,
        "_universe_from_simulation",
        lambda simulation, u, stem: exported.append(simulation) or "universe",
    )

    result = openmm_utils.compress_box(
        None,
        "in.pdb",
        "in.top",
        target_volume=TARGET_VOLUME * 1000.0,  # Å^3
        protein_indices=[0, 1],
        dt=0.004,
        max_nan_retries=1,
    )

    assert result == "universe"
    # The restraint reference is captured once from the input positions, not per attempt.
    assert len(restraint_calls) == 1
    assert restraint_calls[0][1] is gro.positions
    # A retry builds a second simulation at half the requested time step.
    assert len(simulations) == 2
    step_sizes = [
        sim.integrator.getStepSize().value_in_unit(unit.picoseconds) for sim in simulations
    ]
    assert np.allclose(step_sizes, [0.004, 0.002])
    # The result comes from the replacement simulation, not the one that hit NaN.
    assert finished == [simulations[1].context]
    assert exported == [simulations[1]]


def test_minimization_progress_is_logged():
    """A real (tiny) minimization calls the reporter, which logs energy through loguru."""
    messages = []
    sink_id = openmm_utils.logger.add(lambda message: messages.append(str(message)))
    try:
        system = mm.System()
        force = mm.NonbondedForce()
        rng = np.random.default_rng(0)
        for _ in range(30):
            system.addParticle(40.0)
            force.addParticle(0.0, 0.3, 1.0)
        system.addForce(force)
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(rng.uniform(0.0, 0.5, size=(30, 3)))

        simulation = type("Sim", (), {})()
        simulation.minimizeEnergy = lambda reporter: mm.LocalEnergyMinimizer.minimize(
            context, reporter=reporter
        )
        openmm_utils._minimize(simulation, log_interval=0.0)
    finally:
        openmm_utils.logger.remove(sink_id)

    assert any("Minimization iteration" in m and "potential energy" in m for m in messages)
    assert any("Minimization finished" in m for m in messages)
