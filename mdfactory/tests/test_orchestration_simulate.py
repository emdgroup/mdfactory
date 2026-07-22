# ABOUTME: Unit tests for Parsl-based GROMACS simulation orchestration
# ABOUTME: Tests checkpoint detection, validation, and dry-run mode
"""Unit tests for simulation orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.config import ExecutorConfig
from mdfactory.orchestration.simulate import (
    _detect_needed_stages,
    _log_dry_run_plan,
    _validate_simulation_dir,
    run_simulations,
)


@pytest.fixture
def mock_sim_dir(tmp_path):
    """Create mock simulation directory with required files."""
    sim_dir = tmp_path / "abc123"
    sim_dir.mkdir()

    # Required files
    (sim_dir / "system.pdb").write_text("FAKE PDB")
    (sim_dir / "topology.top").write_text("FAKE TOP")
    (sim_dir / "em.mdp").write_text("FAKE MDP")
    (sim_dir / "nvt.mdp").write_text("FAKE MDP")
    (sim_dir / "npt.mdp").write_text("FAKE MDP")
    (sim_dir / "md.mdp").write_text("FAKE MDP")

    return sim_dir


def test_validate_simulation_dir_success(mock_sim_dir):
    """Validation passes when all files present."""
    _validate_simulation_dir(mock_sim_dir, ["EM", "NVT", "NPT", "Production"])


def test_validate_simulation_dir_missing_files(tmp_path):
    """Validation raises when files missing."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing files"):
        _validate_simulation_dir(sim_dir, ["EM"])


def test_validate_simulation_dir_missing_mdp(mock_sim_dir):
    """Validation raises when MDP file missing."""
    (mock_sim_dir / "em.mdp").unlink()

    with pytest.raises(FileNotFoundError, match="Missing files.*em.mdp"):
        _validate_simulation_dir(mock_sim_dir, ["EM"])


def test_checkpoint_auto_skips_completed(mock_sim_dir):
    """Checkpoint mode auto skips stages with outputs."""
    (mock_sim_dir / "min.gro").write_text("FAKE OUTPUT")

    needed = _detect_needed_stages(mock_sim_dir, ["EM", "NVT"], "auto")
    assert "EM" not in needed
    assert "NVT" in needed


def test_checkpoint_auto_skips_empty_file(mock_sim_dir):
    """Checkpoint mode auto does not skip empty files."""
    (mock_sim_dir / "min.gro").write_text("")  # Empty file

    needed = _detect_needed_stages(mock_sim_dir, ["EM"], "auto")
    assert "EM" in needed


def test_checkpoint_skip_always_skips(mock_sim_dir):
    """Checkpoint mode skip always skips if file exists."""
    (mock_sim_dir / "min.gro").write_text("")  # Empty file

    needed = _detect_needed_stages(mock_sim_dir, ["EM"], "skip")
    assert "EM" not in needed


def test_checkpoint_force_never_skips(mock_sim_dir):
    """Checkpoint mode force reruns everything."""
    (mock_sim_dir / "min.gro").write_text("FAKE OUTPUT")

    needed = _detect_needed_stages(mock_sim_dir, ["EM"], "force")
    assert "EM" in needed


def test_checkpoint_all_stages_complete(mock_sim_dir):
    """All stages skipped when outputs exist."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    (mock_sim_dir / "prod.xtc").write_text("FAKE")

    needed = _detect_needed_stages(mock_sim_dir, ["EM", "NVT", "NPT", "Production"], "auto")
    assert len(needed) == 0


def test_dry_run_no_parsl_load(mock_sim_dir):
    """Dry-run logs plan without loading Parsl."""
    results = run_simulations([mock_sim_dir], ExecutorConfig(), dry_run=True)

    assert len(results) == 1
    assert results[0]["hash"] == "abc123"
    assert results[0]["stages"] == ["EM", "NVT", "NPT", "Production"]


def test_dry_run_with_checkpoint(mock_sim_dir):
    """Dry-run respects checkpoint detection."""
    (mock_sim_dir / "min.gro").write_text("FAKE OUTPUT")
    (mock_sim_dir / "nvt.gro").write_text("FAKE OUTPUT")

    results = run_simulations(
        [mock_sim_dir], ExecutorConfig(), checkpoint_mode="auto", dry_run=True
    )

    # EM and NVT should be skipped
    assert results[0]["stages"] == ["NPT", "Production"]


def test_log_dry_run_plan_format(mock_sim_dir):
    """Dry-run plan includes expected fields."""
    work_plan = [
        {
            "sim_dir": mock_sim_dir,
            "hash": "abc123",
            "stages": ["EM", "NVT"],
        }
    ]

    config = ExecutorConfig()
    result = _log_dry_run_plan(work_plan, config)

    assert result == work_plan


@patch("mdfactory.orchestration.simulate.parsl_session")
@patch("mdfactory.orchestration.simulate.run_full_pipeline")
def test_parallel_submission(mock_pipeline, mock_session, tmp_path):
    """Multiple simulations submit in parallel."""
    # Create 10 mock simulations
    sim_dirs = []
    for i in range(10):
        sim_dir = tmp_path / f"sim{i:03d}"
        sim_dir.mkdir()
        (sim_dir / "system.pdb").write_text("FAKE")
        (sim_dir / "topology.top").write_text("FAKE")
        for mdp in ["em", "nvt", "npt", "md"]:
            (sim_dir / f"{mdp}.mdp").write_text("FAKE")
        sim_dirs.append(sim_dir)

    # Mock pipeline returns future
    mock_future = MagicMock()
    mock_future.result.return_value = {"status": "success"}
    mock_future.done.return_value = True
    mock_pipeline.return_value = mock_future

    # Mock session context
    mock_session.return_value.__enter__.return_value = MagicMock()

    # Run
    results = run_simulations(sim_dirs, ExecutorConfig(), stages=["EM", "NVT", "NPT", "Production"])

    # Verify 10 pipelines submitted
    assert mock_pipeline.call_count == 10


def test_stage_filtering(mock_sim_dir):
    """Only requested stages are included in work plan."""
    results = run_simulations(
        [mock_sim_dir],
        ExecutorConfig(),
        stages=["EM", "NVT"],
        dry_run=True,
    )

    # Plan should only include EM and NVT
    assert results[0]["stages"] == ["EM", "NVT"]


def test_validation_before_submission(tmp_path):
    """Validation catches missing files before Parsl submission."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").write_text("FAKE")
    # Missing topology.top

    with pytest.raises(FileNotFoundError, match="Missing files.*topology.top"):
        run_simulations([sim_dir], ExecutorConfig(), dry_run=True)


def test_multiple_simulations_mixed_states(tmp_path):
    """Handle multiple simulations in different completion states."""
    # Sim 1: nothing done
    sim1 = tmp_path / "sim1"
    sim1.mkdir()
    for f in ["system.pdb", "topology.top", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
        (sim1 / f).write_text("FAKE")

    # Sim 2: EM complete
    sim2 = tmp_path / "sim2"
    sim2.mkdir()
    for f in ["system.pdb", "topology.top", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
        (sim2 / f).write_text("FAKE")
    (sim2 / "min.gro").write_text("FAKE")

    # Sim 3: all complete
    sim3 = tmp_path / "sim3"
    sim3.mkdir()
    for f in ["system.pdb", "topology.top", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
        (sim3 / f).write_text("FAKE")
    (sim3 / "min.gro").write_text("FAKE")
    (sim3 / "nvt.gro").write_text("FAKE")
    (sim3 / "npt.gro").write_text("FAKE")
    (sim3 / "prod.xtc").write_text("FAKE")

    results = run_simulations([sim1, sim2, sim3], ExecutorConfig(), dry_run=True)

    assert len(results) == 3
    assert results[0]["stages"] == ["EM", "NVT", "NPT", "Production"]
    assert results[1]["stages"] == ["NVT", "NPT", "Production"]
    assert results[2]["stages"] == []


def test_checkpoint_production_output(mock_sim_dir):
    """Production stage uses prod.xtc as checkpoint file."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    (mock_sim_dir / "prod.xtc").write_text("FAKE")

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")
    assert "Production" not in needed


def test_stage_order_preservation(mock_sim_dir):
    """Requested stages maintain order."""
    needed = _detect_needed_stages(mock_sim_dir, ["Production", "NPT", "EM", "NVT"], "force")
    assert needed == ["Production", "NPT", "EM", "NVT"]


@patch("mdfactory.orchestration.simulate.parsl_session")
@patch("mdfactory.orchestration.simulate.run_full_pipeline")
def test_wait_false_returns_futures(mock_pipeline, mock_session, mock_sim_dir):
    """When wait=False, returns raw futures."""
    mock_future = MagicMock()
    mock_pipeline.return_value = mock_future

    mock_session_obj = MagicMock()
    mock_session.return_value.__enter__.return_value = mock_session_obj

    results = run_simulations([mock_sim_dir], ExecutorConfig(), wait=False)

    # Should return futures, not results
    assert results == [mock_future]
    # Session should be detached
    mock_session_obj.detach.assert_called_once()


def test_empty_sim_list():
    """Handle empty simulation list gracefully."""
    results = run_simulations([], ExecutorConfig(), dry_run=True)
    assert results == []
