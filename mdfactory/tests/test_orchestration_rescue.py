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
