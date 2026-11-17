# ABOUTME: Tests for staged OpenMM compression and its NaN recovery
# ABOUTME: Drive run_staged_compression with fake simulations; no OpenMM context is created
"""Tests for staged box compression and NaN retry in ``openmm_utils``."""

import openmm as mm
import pytest
from openmm import unit

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

    def __init__(self, world, edge=INITIAL_VOLUME ** (1 / 3)):
        self.world = world
        self.edge = edge
        self.context = self

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

    def minimizeEnergy(self):
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


def _step_calls_to_finish_first_stage():
    """Step calls the unperturbed fake needs to reach the first stage's density."""
    edge = INITIAL_VOLUME ** (1 / 3)
    threshold = (TARGET_VOLUME / COMPRESSION_STAGE_FRACTIONS[0]) ** (1 / 3)
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
        fail_on_step_calls={_step_calls_to_finish_first_stage() + 1}, fail_message=message
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


def test_step_budget_is_shared_and_unconverged_runs_stop_early():
    world = World()
    _, densities, steps, converged = _run(world, steps=100)
    assert not converged
    assert steps <= 100
    assert densities[-1] < TARGET_DENSITY
    # It stopped in the first unfinished stage instead of starting the next.
    assert world.minimize_calls <= 1
