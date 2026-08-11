# ABOUTME: Tests for GROMACS error classification and Parsl exception unwrapping
# ABOUTME: Verifies physics vs infrastructure failure detection
"""Tests for mdfactory.orchestration.errors."""

from mdfactory.orchestration.errors import (
    FailureType,
    _matches_physics_patterns,
    _unwrap_parsl_exception,
    classify_failure,
)


class TestMatchesPhysicsPatterns:
    def test_lincs_warning(self):
        assert _matches_physics_patterns("LINCS WARNING: relative constraint deviation")

    def test_coordinate_constraints(self):
        assert _matches_physics_patterns(
            "step 1234: the coordinate constraints could not be satisfied"
        )

    def test_particle_moved(self):
        assert _matches_physics_patterns("Particle 42 in box 3 moved more than 5.0 nm")

    def test_domain_decomposition(self):
        assert _matches_physics_patterns("The domain decomposition cell size is too small")

    def test_blowing_up(self):
        assert _matches_physics_patterns("Fatal error: blowing up")

    def test_force_not_finite(self):
        assert _matches_physics_patterns("force 1.#INF is not finite")

    def test_water_molecule_starting_at(self):
        assert _matches_physics_patterns(
            "step 1000: Water molecule starting at (1.2, 3.4, 5.6) cannot be settled"
        )

    def test_too_many_lincs_warnings(self):
        assert _matches_physics_patterns("Too many LINCS warnings")

    def test_can_not_continue(self):
        assert _matches_physics_patterns("can not continue")

    def test_abnormal_termination(self):
        assert _matches_physics_patterns("Abnormal termination of GROMACS")

    def test_perturbed_nonbonded_pairs(self):
        assert _matches_physics_patterns(
            "There are 42 perturbed non-bonded pair interactions beyond the pair-list cutoff"
        )

    def test_atoms_too_many_constraints(self):
        assert _matches_physics_patterns(
            "atoms 123 and 456 are involved in more than 6 constraints"
        )

    def test_normal_output_no_match(self):
        assert not _matches_physics_patterns("Step 1000, time 2.0 ps")

    def test_empty_string_no_match(self):
        assert not _matches_physics_patterns("")


class TestClassifyFailure:
    def test_physics_from_exception_message(self):
        exc = RuntimeError("LINCS WARNING: bonds to H are constrained")
        assert classify_failure(exc) == FailureType.PHYSICS

    def test_physics_from_coordinate_error(self):
        exc = RuntimeError("step 500: the coordinate constraints could not be satisfied")
        assert classify_failure(exc) == FailureType.PHYSICS

    def test_infrastructure_oom(self):
        exc = RuntimeError("slurmstepd: error: Detected 1 oom-kill event")
        assert classify_failure(exc) == FailureType.INFRASTRUCTURE

    def test_infrastructure_timeout(self):
        exc = RuntimeError("JOB 12345 CANCELLED AT DUE TO TIME LIMIT")
        assert classify_failure(exc) == FailureType.INFRASTRUCTURE

    def test_infrastructure_preempted(self):
        exc = RuntimeError("Job was preempted by higher priority job")
        assert classify_failure(exc) == FailureType.INFRASTRUCTURE

    def test_unknown_for_generic_error(self):
        exc = RuntimeError("Something went wrong")
        assert classify_failure(exc) == FailureType.UNKNOWN

    def test_unwraps_parsl_legacy_exception(self):
        class LegacyWrapper(Exception):
            pass

        exc = LegacyWrapper("wrapper")
        exc.e_value = RuntimeError("LINCS WARNING in bonds")
        assert classify_failure(exc) == FailureType.PHYSICS

    def test_physics_from_log_file(self, tmp_path, monkeypatch):
        from mdfactory.orchestration.stages import StageSpec

        mock_spec = StageSpec(
            name="EM",
            deffnm="min",
            mdp_file="em.mdp",
            gro_in="system.pdb",
            gro_out="min.gro",
            tpr_file="min.tpr",
            cpt_file="min.cpt",
            prereq_cpt=None,
            traj_files=(),
            ref_file=None,
            maxwarn=1,
            supports_pme_gpu=False,
        )
        monkeypatch.setattr("mdfactory.orchestration.stages.STAGE_BY_NAME", {"EM": mock_spec})
        log_file = tmp_path / "min.log"
        log_file.write_text("Step 100\nLINCS WARNING\nStep 101\n")

        exc = RuntimeError("Process returned non-zero exit code 1")
        result = classify_failure(exc, sim_dir=tmp_path, stage="EM")
        assert result == FailureType.PHYSICS

    def test_unwraps_dependency_error_with_cause(self):
        """DependencyError (via __cause__) is unwrapped to classify the root cause."""
        root = RuntimeError("LINCS WARNING: bonds to H are constrained")
        wrapper = RuntimeError("Dependency failure for task 5")
        wrapper.__cause__ = root
        assert classify_failure(wrapper) == FailureType.PHYSICS

    def test_dependency_error_infra_from_root_cause(self):
        """DependencyError wrapping an infrastructure failure is classified correctly."""
        root = RuntimeError("slurmstepd: error: Detected 1 oom-kill event")
        wrapper = RuntimeError("Dependency failure for task 3")
        wrapper.__cause__ = root
        assert classify_failure(wrapper) == FailureType.INFRASTRUCTURE

    def test_dependency_error_unknown_root_cause(self):
        """DependencyError wrapping an unrecognised error yields UNKNOWN."""
        root = RuntimeError("Neither gmx nor gmx_mpi found in PATH")
        wrapper = RuntimeError("Dependency failure for task 1")
        wrapper.__cause__ = root
        assert classify_failure(wrapper) == FailureType.UNKNOWN


class TestUnwrapParslException:
    """Tests for _unwrap_parsl_exception helper."""

    def test_plain_exception_returned_as_is(self):
        exc = RuntimeError("simple error")
        assert _unwrap_parsl_exception(exc) is exc

    def test_unwraps_cause(self):
        root = ValueError("the real error")
        wrapper = RuntimeError("Dependency failure for task 5")
        wrapper.__cause__ = root
        assert _unwrap_parsl_exception(wrapper) is root

    def test_unwraps_legacy_e_value(self):
        class LegacyWrapper(Exception):
            pass

        root = RuntimeError("LINCS WARNING in bonds")
        wrapper = LegacyWrapper("wrapper message")
        wrapper.e_value = root
        assert _unwrap_parsl_exception(wrapper) is root

    def test_cause_takes_precedence_over_e_value(self):
        """If both __cause__ and .e_value exist, __cause__ wins."""
        cause_exc = ValueError("from __cause__")
        e_val_exc = RuntimeError("from e_value")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cause_exc
        wrapper.e_value = e_val_exc
        assert _unwrap_parsl_exception(wrapper) is cause_exc

    def test_chained_cause(self):
        """Deeply nested __cause__ chain — returns the immediate __cause__."""
        deepest = ValueError("deepest root")
        middle = RuntimeError("middle")
        middle.__cause__ = deepest
        outer = RuntimeError("Dependency failure")
        outer.__cause__ = middle
        # Returns the immediate __cause__ (middle), not the deepest.
        # Parsl's _find_any_root_cause already resolves __cause__ to
        # the deepest, so this is the expected behaviour.
        assert _unwrap_parsl_exception(outer) is middle
