# ABOUTME: Tests for rescue retry loop
# ABOUTME: Verifies rescue activation, tier progression, and failure handling
"""Tests for mdfactory.orchestration.rescue."""

from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.errors import FailureType
from mdfactory.orchestration.rescue import (
    _clean_partial_outputs,
    execute_stage_with_rescue,
)


@pytest.fixture
def sim_dir(tmp_path):
    """Create a simulation directory with MDP files."""
    (tmp_path / "em.mdp").write_text("integrator = steep\nemstep = 0.01\nnsteps = 50000\n")
    (tmp_path / "nvt.mdp").write_text("integrator = md\ndt = 0.002\nnsteps = 50000\n")
    (tmp_path / "system.pdb").touch()
    (tmp_path / "topology.top").touch()
    return tmp_path


class TestCleanPartialOutputs:
    def test_removes_stage_files(self, tmp_path):
        (tmp_path / "min.tpr").touch()
        (tmp_path / "min.cpt").touch()
        (tmp_path / "min.gro").touch()
        (tmp_path / "min.log").touch()
        (tmp_path / "min.edr").touch()

        _clean_partial_outputs(tmp_path, "EM")

        assert not (tmp_path / "min.tpr").exists()
        assert not (tmp_path / "min.cpt").exists()
        assert not (tmp_path / "min.gro").exists()
        assert not (tmp_path / "min.log").exists()
        assert not (tmp_path / "min.edr").exists()

    def test_handles_missing_files_gracefully(self, tmp_path):
        _clean_partial_outputs(tmp_path, "EM")

    def test_production_cleans_trajectory(self, tmp_path):
        (tmp_path / "prod.xtc").touch()
        (tmp_path / "prod.trr").touch()
        (tmp_path / "prod.tpr").touch()
        _clean_partial_outputs(tmp_path, "Production")
        assert not (tmp_path / "prod.xtc").exists()
        assert not (tmp_path / "prod.trr").exists()


class TestExecuteStageWithRescue:
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_success_on_first_try(self, mock_run_stage, sim_dir):
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_stage.return_value = mock_future

        result = execute_stage_with_rescue(
            sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3
        )
        assert result is mock_future
        assert mock_run_stage.call_count == 1

    @patch("mdfactory.orchestration.rescue.classify_failure")
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_rescue_on_physics_failure(self, mock_run_stage, mock_classify, sim_dir):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("LINCS WARNING")
        success_future = MagicMock()
        success_future.result.return_value = None
        mock_run_stage.side_effect = [fail_future, success_future]
        mock_classify.return_value = FailureType.PHYSICS

        result = execute_stage_with_rescue(
            sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3
        )
        assert result is success_future
        assert mock_run_stage.call_count == 2
        assert (sim_dir / "em_rescue_t1.mdp").exists()

    @patch("mdfactory.orchestration.rescue.classify_failure")
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_rescue_exhausted_raises(self, mock_run_stage, mock_classify, sim_dir):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("LINCS WARNING")
        mock_run_stage.return_value = fail_future
        mock_classify.return_value = FailureType.PHYSICS

        with pytest.raises(RuntimeError, match="LINCS WARNING"):
            execute_stage_with_rescue(sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=2)
        # tier 0 + tier 1 + tier 2 = 3 attempts
        assert mock_run_stage.call_count == 3

    @patch("mdfactory.orchestration.rescue.classify_failure")
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_infra_failure_not_rescued(self, mock_run_stage, mock_classify, sim_dir):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("Out of memory")
        mock_run_stage.return_value = fail_future
        mock_classify.return_value = FailureType.INFRASTRUCTURE

        with pytest.raises(RuntimeError, match="Out of memory"):
            execute_stage_with_rescue(sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3)
        assert mock_run_stage.call_count == 1

    @patch("mdfactory.orchestration.rescue.classify_failure")
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_mdp_override_passed_to_run_stage(self, mock_run_stage, mock_classify, sim_dir):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("LINCS")
        success_future = MagicMock()
        success_future.result.return_value = None
        mock_run_stage.side_effect = [fail_future, success_future]
        mock_classify.return_value = FailureType.PHYSICS

        execute_stage_with_rescue(sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3)

        # First call: no mdp_override (tier 0)
        first_call = mock_run_stage.call_args_list[0]
        assert first_call.kwargs.get("mdp_override") is None

        # Second call: has mdp_override (tier 1)
        second_call = mock_run_stage.call_args_list[1]
        assert second_call.kwargs.get("mdp_override") == "em_rescue_t1.mdp"

    @patch("mdfactory.orchestration.rescue.classify_failure")
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_unknown_failure_not_rescued(self, mock_run_stage, mock_classify, sim_dir):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("mysterious error")
        mock_run_stage.return_value = fail_future
        mock_classify.return_value = FailureType.UNKNOWN

        with pytest.raises(RuntimeError, match="mysterious error"):
            execute_stage_with_rescue(sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3)
        assert mock_run_stage.call_count == 1

    @patch("mdfactory.orchestration.rescue.classify_failure")
    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_rescue_logs_warning_with_tier_and_stage(self, mock_run_stage, mock_classify, sim_dir):
        """Rescue activation emits WARNING log with tier number and stage name."""
        from loguru import logger

        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("LINCS WARNING")
        success_future = MagicMock()
        success_future.result.return_value = None
        mock_run_stage.side_effect = [fail_future, success_future]
        mock_classify.return_value = FailureType.PHYSICS

        warnings = []
        handler_id = logger.add(
            lambda msg: warnings.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            execute_stage_with_rescue(sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3)
        finally:
            logger.remove(handler_id)

        # Should have at least one warning mentioning rescue tier and stage
        rescue_warnings = [w for w in warnings if "RESCUE" in w and "EM" in w]
        assert len(rescue_warnings) >= 1
        # Check tier info is present
        assert any("tier 1" in w for w in rescue_warnings)

    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_dependency_error_surfaces_root_cause(self, mock_run_stage, sim_dir):
        """A grompp DependencyError is unwrapped to classify the real root cause.

        When grompp fails, mdrun receives a DependencyError with __cause__
        pointing to the actual grompp exception. The rescue loop must
        unwrap this to classify correctly and show an actionable message.
        """
        # Simulate Parsl DependencyError wrapping a grompp failure
        root_cause = RuntimeError("Neither gmx nor gmx_mpi found in PATH")
        dependency_error = RuntimeError(
            "Dependency failure for task 5. The representative cause is via task 0"
        )
        dependency_error.__cause__ = root_cause

        fail_future = MagicMock()
        fail_future.result.side_effect = dependency_error
        mock_run_stage.return_value = fail_future

        # Should raise the original DependencyError (not retry it),
        # since "gmx not found" is UNKNOWN, not PHYSICS.
        with pytest.raises(RuntimeError, match="Dependency failure"):
            execute_stage_with_rescue(sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3)
        # Should NOT retry — gmx-not-found is not a physics failure
        assert mock_run_stage.call_count == 1

    @patch("mdfactory.orchestration.rescue.run_stage")
    def test_dependency_error_with_physics_root_is_rescued(self, mock_run_stage, sim_dir):
        """A DependencyError wrapping a physics failure IS rescued."""
        root_cause = RuntimeError("LINCS WARNING: bonds to H are constrained")
        dependency_error = RuntimeError("Dependency failure for task 5")
        dependency_error.__cause__ = root_cause

        fail_future = MagicMock()
        fail_future.result.side_effect = dependency_error
        success_future = MagicMock()
        success_future.result.return_value = None
        mock_run_stage.side_effect = [fail_future, success_future]

        result = execute_stage_with_rescue(
            sim_dir, "EM", None, MagicMock(), MagicMock(), max_rescue=3
        )
        assert result is success_future
        assert mock_run_stage.call_count == 2
