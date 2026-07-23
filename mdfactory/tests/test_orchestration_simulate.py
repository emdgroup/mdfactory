# ABOUTME: Unit tests for Parsl-based GROMACS simulation orchestration
# ABOUTME: Tests checkpoint detection, validation, and dry-run mode
"""Unit tests for simulation orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.config import ExecutorConfig
from mdfactory.orchestration.apps import _build_mdrun_script
from mdfactory.orchestration.simulate import (
    _detect_needed_stages,
    _execute_stage_list,
    _log_dry_run_plan,
    _validate_simulation_dir,
    _validate_stage_prerequisites,
    _validate_trajectory_complete,
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
    """All stages skipped when outputs AND prerequisite checkpoints exist."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.cpt").write_text("FAKE CPT")  # NPT needs this
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")  # Production needs this
    # Create large trajectory file (> 10 MB) to pass size fallback validation
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 11_000_000)

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
@patch("mdfactory.orchestration.simulate._execute_stage_list")
def test_parallel_submission(mock_execute, mock_session, tmp_path):
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

    # Mock execute_stage_list returns future
    mock_future = MagicMock()
    mock_future.result.return_value = {"status": "success"}
    mock_future.done.return_value = True
    mock_execute.return_value = mock_future

    # Mock session context
    mock_session.return_value.__enter__.return_value = MagicMock()

    # Run
    results = run_simulations(sim_dirs, ExecutorConfig(), stages=["EM", "NVT", "NPT", "Production"])

    # Verify 10 stage lists executed
    assert mock_execute.call_count == 10


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

    # Sim 3: all complete (with prerequisite checkpoints)
    sim3 = tmp_path / "sim3"
    sim3.mkdir()
    for f in ["system.pdb", "topology.top", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
        (sim3 / f).write_text("FAKE")
    (sim3 / "min.gro").write_text("FAKE")
    (sim3 / "nvt.gro").write_text("FAKE")
    (sim3 / "nvt.cpt").write_text("FAKE CPT")  # NPT prerequisite
    (sim3 / "npt.gro").write_text("FAKE")
    (sim3 / "npt.cpt").write_text("FAKE CPT")  # Production prerequisite
    # Create large trajectory file (> 10 MB) to pass size fallback validation
    (sim3 / "prod.xtc").write_bytes(b"X" * 11_000_000)

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
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")  # Production prerequisite
    # Create large trajectory file (> 10 MB) to pass size fallback validation
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 11_000_000)

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")
    assert "Production" not in needed


def test_stage_order_preservation(mock_sim_dir):
    """Requested stages maintain order."""
    needed = _detect_needed_stages(mock_sim_dir, ["Production", "NPT", "EM", "NVT"], "force")
    assert needed == ["Production", "NPT", "EM", "NVT"]


@patch("mdfactory.orchestration.simulate.parsl_session")
@patch("mdfactory.orchestration.simulate._execute_stage_list")
def test_wait_false_returns_futures(mock_execute, mock_session, mock_sim_dir):
    """When wait=False, returns raw futures."""
    mock_future = MagicMock()
    mock_execute.return_value = mock_future

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


# === Decision 2: Strict Checkpoint Validation Tests ===


def test_checkpoint_auto_requires_cpt_files(mock_sim_dir):
    """NPT requires both npt.gro AND nvt.cpt for auto mode."""
    # Create npt.gro but NOT nvt.cpt
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    # nvt.cpt is missing!

    needed = _detect_needed_stages(mock_sim_dir, ["EM", "NVT", "NPT"], "auto")

    # NPT should still be needed because nvt.cpt is missing
    assert "EM" not in needed  # min.gro exists
    assert "NVT" not in needed  # nvt.gro exists
    assert "NPT" in needed  # nvt.cpt missing


def test_checkpoint_auto_skips_with_all_files(mock_sim_dir):
    """NPT skipped when both npt.gro AND nvt.cpt exist."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.cpt").write_text("FAKE CPT")
    (mock_sim_dir / "npt.gro").write_text("FAKE")

    needed = _detect_needed_stages(mock_sim_dir, ["NPT"], "auto")

    assert "NPT" not in needed  # Both npt.gro and nvt.cpt exist


def test_checkpoint_production_requires_npt_cpt(mock_sim_dir):
    """Production requires prod.xtc AND npt.cpt."""
    (mock_sim_dir / "prod.xtc").write_text("FAKE TRAJ")
    # npt.cpt is missing!

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")

    # Production should be needed because npt.cpt is missing
    assert "Production" in needed


def test_checkpoint_production_skips_with_all_files(mock_sim_dir):
    """Production skipped when both prod.xtc AND npt.cpt exist."""
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")
    # Create large trajectory file (> 10 MB) to pass size fallback validation
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 11_000_000)

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")

    assert "Production" not in needed


def test_validate_prerequisites_em_no_requirements():
    """EM has no prerequisites."""
    # EM doesn't need any input files (uses system.pdb which is validated elsewhere)
    from pathlib import Path

    sim_dir = Path("/tmp/fake")
    _validate_stage_prerequisites(sim_dir, "EM")  # Should not raise


def test_validate_prerequisites_nvt_requires_min_gro(mock_sim_dir):
    """NVT requires min.gro from EM."""
    # min.gro is missing
    with pytest.raises(FileNotFoundError, match="Cannot start NVT.*min.gro"):
        _validate_stage_prerequisites(mock_sim_dir, "NVT")


def test_validate_prerequisites_nvt_succeeds_with_min_gro(mock_sim_dir):
    """NVT validation passes when min.gro exists."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    _validate_stage_prerequisites(mock_sim_dir, "NVT")  # Should not raise


def test_validate_prerequisites_npt_requires_nvt_files(mock_sim_dir):
    """NPT requires both nvt.gro AND nvt.cpt."""
    # Only create nvt.gro, not nvt.cpt
    (mock_sim_dir / "nvt.gro").write_text("FAKE")

    with pytest.raises(FileNotFoundError, match="Cannot start NPT.*nvt.cpt"):
        _validate_stage_prerequisites(mock_sim_dir, "NPT")


def test_validate_prerequisites_npt_succeeds_with_all_files(mock_sim_dir):
    """NPT validation passes when nvt.gro AND nvt.cpt exist."""
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.cpt").write_text("FAKE CPT")
    _validate_stage_prerequisites(mock_sim_dir, "NPT")  # Should not raise


def test_validate_prerequisites_production_requires_npt_files(mock_sim_dir):
    """Production requires both npt.gro AND npt.cpt."""
    # Only create npt.gro, not npt.cpt
    (mock_sim_dir / "npt.gro").write_text("FAKE")

    with pytest.raises(FileNotFoundError, match="Cannot start Production.*npt.cpt"):
        _validate_stage_prerequisites(mock_sim_dir, "Production")


def test_validate_prerequisites_production_succeeds_with_all_files(mock_sim_dir):
    """Production validation passes when npt.gro AND npt.cpt exist."""
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")
    _validate_stage_prerequisites(mock_sim_dir, "Production")  # Should not raise


def test_validate_prerequisites_error_message_includes_fix(mock_sim_dir):
    """Error message includes actionable fix commands."""
    with pytest.raises(FileNotFoundError) as exc_info:
        _validate_stage_prerequisites(mock_sim_dir, "NPT")

    error_msg = str(exc_info.value)
    assert "mdfactory simulate" in error_msg  # Fix command included
    assert "--stages EM NVT" in error_msg  # Suggests running earlier stages
    assert "--checkpoint force" in error_msg  # Alternative fix


@patch("mdfactory.orchestration.simulate.parsl_session")
def test_prerequisite_validation_before_parsl_session(mock_session, mock_sim_dir):
    """Prerequisites validated before Parsl session starts."""
    # Create incomplete state: NVT complete but missing nvt.cpt (needed for NPT)
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    # nvt.cpt is missing!

    # Try to run NPT (which requires nvt.cpt)
    with pytest.raises(FileNotFoundError, match="Cannot start NPT"):
        run_simulations(
            [mock_sim_dir],
            ExecutorConfig(),
            stages=["NPT", "Production"],
            checkpoint_mode="force",  # Force to ensure NPT is in work plan
        )

    # Parsl session should NOT have been started
    mock_session.assert_not_called()


# === Decision 3: Resource Auto-Detection Tests ===


def test_mdrun_app_cpu_fallback():
    """Mdrun bash app uses default thread count when env vars unset."""
    bash_script = _build_mdrun_script(deffnm="test", work_dir="/tmp/test")

    # Should use default of 12 threads
    assert "NTHR=${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-12}}" in bash_script
    # Should use CPU mode (no GPU flags) with dynamic binary detection
    assert "$GMX_BIN mdrun -deffnm test -nt $NTHR" in bash_script


def test_mdrun_app_gpu_detection():
    """Mdrun bash app detects GPU from CUDA_VISIBLE_DEVICES."""
    bash_script = _build_mdrun_script(deffnm="test", work_dir="/tmp/test")

    # Should check for CUDA_VISIBLE_DEVICES
    assert "CUDA_VISIBLE_DEVICES" in bash_script
    # GPU mode should use -ntmpi 1 -ntomp -gpu_id -nb gpu -pme gpu
    assert "-ntmpi 1" in bash_script
    assert "-ntomp $NTHR" in bash_script
    assert "-gpu_id $GPU_ID" in bash_script
    assert "-nb gpu" in bash_script
    assert "-pme gpu" in bash_script


def test_mdrun_app_logging():
    """Mdrun bash app logs resource configuration to stderr."""
    bash_script = _build_mdrun_script(deffnm="test", work_dir="/tmp/test")

    # Should log what resources are being used
    assert "Running on GPU" in bash_script
    assert "Running on CPU" in bash_script
    assert ">&2" in bash_script  # Logs to stderr


def test_mdrun_app_slurm_priority():
    """SLURM_CPUS_PER_TASK has priority over OMP_NUM_THREADS."""
    bash_script = _build_mdrun_script(deffnm="test", work_dir="/tmp/test")

    # Check priority order in bash variable expansion
    assert "${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-12}}" in bash_script


def test_stage_functions_no_hardcoded_nt():
    """Stage functions don't pass hardcoded nt parameter to mdrun."""
    from mdfactory.orchestration import stages
    import inspect

    # Check all stage functions
    for name in ["run_em_stage", "run_nvt_stage", "run_npt_stage", "run_production_stage"]:
        func = getattr(stages, name)
        source = inspect.getsource(func)
        # Should NOT have nt=12 anywhere
        assert "nt=12" not in source, f"{name} still has hardcoded nt=12"
        assert "nt=24" not in source, f"{name} has hardcoded nt"


# === Decision 1: Explicit Stage Execution Tests ===


def test_execute_stage_list_empty():
    """Execute stage list returns None when no stages provided."""
    result = _execute_stage_list(Path("/tmp/test"), [], None, None)
    assert result is None


def test_execute_stage_list_validates_order():
    """Execute stage list validates stages are in dependency order."""
    with pytest.raises(ValueError, match="must be in dependency order"):
        # NPT before NVT is invalid
        _execute_stage_list(Path("/tmp/test"), ["NPT", "NVT"], None, None)


def test_execute_stage_list_validates_unknown_stage():
    """Execute stage list rejects unknown stage names."""
    with pytest.raises(ValueError, match="Unknown stage.*Equilibration"):
        _execute_stage_list(Path("/tmp/test"), ["Equilibration"], None, None)


@patch("mdfactory.orchestration.simulate.run_em_stage")
def test_execute_stage_list_em_only(mock_em):
    """Execute stage list runs EM without dependencies."""
    mock_future = MagicMock()
    mock_em.return_value = mock_future

    result = _execute_stage_list(
        Path("/tmp/test"), ["EM"], MagicMock(), MagicMock()
    )

    # EM should be called once
    mock_em.assert_called_once()
    assert result == mock_future


@patch("mdfactory.orchestration.simulate.run_em_stage")
@patch("mdfactory.orchestration.simulate.run_nvt_stage")
def test_execute_stage_list_em_nvt_chain(mock_nvt, mock_em):
    """Execute stage list chains EM → NVT dependencies."""
    em_future = MagicMock()
    nvt_future = MagicMock()
    mock_em.return_value = em_future
    mock_nvt.return_value = nvt_future

    grompp_app = MagicMock()
    mdrun_app = MagicMock()

    result = _execute_stage_list(
        Path("/tmp/test"), ["EM", "NVT"], grompp_app, mdrun_app
    )

    # EM called first
    mock_em.assert_called_once_with(Path("/tmp/test"), grompp_app, mdrun_app)
    # NVT called with EM future as dependency
    mock_nvt.assert_called_once_with(
        Path("/tmp/test"), em_future, grompp_app, mdrun_app
    )
    # Returns final NVT future
    assert result == nvt_future


@patch("mdfactory.orchestration.simulate.run_em_stage")
@patch("mdfactory.orchestration.simulate.run_nvt_stage")
@patch("mdfactory.orchestration.simulate.run_npt_stage")
@patch("mdfactory.orchestration.simulate.run_production_stage")
def test_execute_stage_list_full_pipeline(mock_prod, mock_npt, mock_nvt, mock_em):
    """Execute stage list chains all 4 stages correctly."""
    em_fut = MagicMock()
    nvt_fut = MagicMock()
    npt_fut = MagicMock()
    prod_fut = MagicMock()

    mock_em.return_value = em_fut
    mock_nvt.return_value = nvt_fut
    mock_npt.return_value = npt_fut
    mock_prod.return_value = prod_fut

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    result = _execute_stage_list(
        sim_dir, ["EM", "NVT", "NPT", "Production"], grompp_app, mdrun_app
    )

    # Verify call chain
    mock_em.assert_called_once_with(sim_dir, grompp_app, mdrun_app)
    mock_nvt.assert_called_once_with(sim_dir, em_fut, grompp_app, mdrun_app)
    mock_npt.assert_called_once_with(sim_dir, nvt_fut, grompp_app, mdrun_app)
    mock_prod.assert_called_once_with(sim_dir, npt_fut, grompp_app, mdrun_app)

    # Returns final production future
    assert result == prod_fut


@patch("mdfactory.orchestration.simulate.run_nvt_stage")
@patch("mdfactory.orchestration.simulate.run_npt_stage")
def test_execute_stage_list_partial_pipeline_from_checkpoint(mock_npt, mock_nvt):
    """Execute stage list handles checkpoint resume (first stage not EM)."""
    # Resume from NVT (EM already complete)
    nvt_fut = MagicMock()
    npt_fut = MagicMock()
    mock_nvt.return_value = nvt_fut
    mock_npt.return_value = npt_fut

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    # Execute NVT → NPT (EM was already complete)
    result = _execute_stage_list(sim_dir, ["NVT", "NPT"], grompp_app, mdrun_app)

    # NVT called with prev_future=None (checkpoint resume)
    mock_nvt.assert_called_once_with(sim_dir, None, grompp_app, mdrun_app)
    # NPT called with NVT's future
    mock_npt.assert_called_once_with(sim_dir, nvt_fut, grompp_app, mdrun_app)
    # Returns final NPT future
    assert result == npt_fut


@patch("mdfactory.orchestration.simulate.parsl_session")
@patch("mdfactory.orchestration.simulate._execute_stage_list")
def test_run_simulations_uses_execute_stage_list(mock_execute, mock_session, mock_sim_dir):
    """run_simulations dispatcher now uses _execute_stage_list."""
    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.result.return_value = {"status": "success"}
    mock_execute.return_value = mock_future

    mock_session.return_value.__enter__.return_value = MagicMock()

    # Create partial checkpoint state (EM complete)
    (mock_sim_dir / "min.gro").write_text("FAKE")

    results = run_simulations(
        [mock_sim_dir], ExecutorConfig(), checkpoint_mode="auto"
    )

    # Should call _execute_stage_list with only needed stages
    mock_execute.assert_called_once()
    call_args = mock_execute.call_args
    # Second argument is the stages list
    stages_arg = call_args[0][1]
    # EM should be skipped, others needed
    assert "EM" not in stages_arg
    assert "NVT" in stages_arg


# === Decision 4: GROMACS Error Handling Tests ===


def test_grompp_app_has_error_handling():
    """Grompp bash app includes robust error handling."""
    from mdfactory.orchestration.apps import _build_grompp_script

    bash_script = _build_grompp_script(
        mdp_file="em.mdp",
        gro_file="system.pdb",
        top_file="topology.top",
        tpr_file="min.tpr",
        work_dir="/tmp/test",
    )

    # Should have bash strict mode
    assert "set -euo pipefail" in bash_script
    # Should check grompp exit code (with dynamic binary)
    assert "if ! $GMX_BIN grompp" in bash_script
    # Should verify TPR was created
    assert "if [ ! -f min.tpr ]" in bash_script
    # Should log errors to stderr
    assert ">&2" in bash_script


def test_grompp_app_logs_descriptive_errors():
    """Grompp error messages include actionable guidance."""
    from mdfactory.orchestration.apps import _build_grompp_script

    bash_script = _build_grompp_script(
        mdp_file="em.mdp",
        gro_file="system.pdb",
        top_file="topology.top",
        tpr_file="min.tpr",
        work_dir="/tmp/test",
    )

    # Error message should mention the failed file
    assert "ERROR: grompp failed for min.tpr" in bash_script
    # Should suggest checking inputs
    assert "Check that input files" in bash_script


def test_mdrun_app_has_error_handling():
    """Mdrun bash app includes robust error handling."""
    # Pass gro_out so output-verification block is included
    bash_script = _build_mdrun_script(deffnm="min", work_dir="/tmp/test", gro_out="min.gro")

    # Should have bash strict mode
    assert "set -euo pipefail" in bash_script
    # Should check mdrun exit code (both GPU and CPU paths, with dynamic binary)
    assert "if ! $GMX_BIN mdrun" in bash_script
    assert bash_script.count("if ! $GMX_BIN mdrun") == 4  # GPU and CPU × {MPI, thread-MPI}
    # Should verify output was created
    assert "if [ ! -f" in bash_script
    # Should log errors to stderr
    assert "ERROR: mdrun failed" in bash_script


def test_mdrun_app_verifies_stage_specific_outputs():
    """Mdrun verifies correct output file for each stage (spec-driven, no deffnm comparisons)."""
    # EM stage: spec passes gro_out="min.gro"
    em_script = _build_mdrun_script(deffnm="min", work_dir="/tmp/test", gro_out="min.gro")
    assert "min.gro" in em_script
    assert 'deffnm" == "min"' not in em_script  # no hardcoded deffnm comparison

    # Production: spec passes traj_files
    prod_script = _build_mdrun_script(
        deffnm="prod", work_dir="/tmp/test", traj_files=("prod.xtc", "prod.trr")
    )
    assert "prod.xtc" in prod_script
    assert "prod.trr" in prod_script
    assert 'deffnm" == "prod"' not in prod_script  # no hardcoded deffnm comparison

    # NVT: spec passes gro_out="nvt.gro"
    nvt_script = _build_mdrun_script(deffnm="nvt", work_dir="/tmp/test", gro_out="nvt.gro")
    assert "nvt.gro" in nvt_script


def test_mdrun_app_error_message_includes_log_hint():
    """Mdrun error messages direct users to md.log."""
    bash_script = _build_mdrun_script(deffnm="prod", work_dir="/tmp/test")

    # Should suggest checking md.log
    assert "Check md.log for details" in bash_script


def test_bash_apps_use_strict_mode():
    """All bash apps use set -euo pipefail."""
    from mdfactory.orchestration.apps import _build_grompp_script

    grompp_script = _build_grompp_script(
        mdp_file="em.mdp",
        gro_file="system.pdb",
        top_file="topology.top",
        tpr_file="min.tpr",
        work_dir="/tmp/test",
    )
    mdrun_script = _build_mdrun_script(deffnm="min", work_dir="/tmp/test")

    # Both should use bash strict mode
    assert "set -euo pipefail" in grompp_script
    assert "set -euo pipefail" in mdrun_script


# === Decision 8: Trajectory Frame Count Validation Tests ===


def test_validate_trajectory_returns_false_for_missing_file(mock_sim_dir):
    """Trajectory validation returns False if file doesn't exist."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    assert result is False


def test_validate_trajectory_returns_false_for_empty_file(mock_sim_dir):
    """Trajectory validation returns False for empty files."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    (mock_sim_dir / "prod.xtc").write_text("")  # Empty file
    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    assert result is False


def test_validate_trajectory_uses_size_fallback_without_structure(mock_sim_dir):
    """Trajectory validation uses size heuristic when no structure file found."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    # Create trajectory but no structure files
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 15_000_000)  # 15 MB

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    # Should pass size heuristic (> 10 MB)
    assert result is True


def test_validate_trajectory_size_fallback_rejects_small_files(mock_sim_dir):
    """Trajectory validation rejects small files in fallback mode."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    # Create small trajectory (< 10 MB threshold)
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 1000)

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    # Should fail size heuristic
    assert result is False


@patch("mdfactory.orchestration.simulate.mda")
def test_validate_trajectory_with_mdanalysis_complete(mock_mda, mock_sim_dir):
    """Trajectory validation uses MDAnalysis to count frames (complete)."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    # Setup mocks
    (mock_sim_dir / "prod.xtc").write_bytes(b"FAKE XTC")
    (mock_sim_dir / "system.pdb").write_text("FAKE PDB")

    mock_traj = MagicMock()
    mock_traj.__len__.return_value = 100  # 100 frames
    mock_universe = MagicMock()
    mock_universe.trajectory = mock_traj
    mock_mda.Universe.return_value = mock_universe

    result = _validate_trajectory_complete(
        mock_sim_dir, "prod.xtc", expected_frames=100
    )

    # Should pass (100 >= 100)
    assert result is True


@patch("mdfactory.orchestration.simulate.mda")
def test_validate_trajectory_with_mdanalysis_incomplete(mock_mda, mock_sim_dir):
    """Trajectory validation detects incomplete trajectories."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    # Setup mocks
    (mock_sim_dir / "prod.xtc").write_bytes(b"FAKE XTC")
    (mock_sim_dir / "system.pdb").write_text("FAKE PDB")

    mock_traj = MagicMock()
    mock_traj.__len__.return_value = 50  # Only 50 frames
    mock_universe = MagicMock()
    mock_universe.trajectory = mock_traj
    mock_mda.Universe.return_value = mock_universe

    result = _validate_trajectory_complete(
        mock_sim_dir, "prod.xtc", expected_frames=100
    )

    # Should fail (50 < 100)
    assert result is False


@patch("mdfactory.orchestration.simulate.mda")
def test_validate_trajectory_without_expected_frames(mock_mda, mock_sim_dir):
    """Trajectory validation without expected_frames checks readability only."""
    from mdfactory.orchestration.simulate import _validate_trajectory_complete

    # Setup mocks
    (mock_sim_dir / "prod.xtc").write_bytes(b"FAKE XTC")
    (mock_sim_dir / "system.pdb").write_text("FAKE PDB")

    mock_traj = MagicMock()
    mock_traj.__len__.return_value = 1  # Just 1 frame
    mock_universe = MagicMock()
    mock_universe.trajectory = mock_traj
    mock_mda.Universe.return_value = mock_universe

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")

    # Should pass (> 0 frames, no expectation)
    assert result is True


def test_find_structure_file_priority_order(mock_sim_dir):
    """Find structure file uses correct priority order."""
    from mdfactory.orchestration.simulate import find_structure_file

    # Create files in reverse priority order
    (mock_sim_dir / "system.pdb").write_text("FAKE")
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "npt.gro").write_text("FAKE")

    # Should prioritize npt.gro over min.gro over system.pdb
    result = find_structure_file(mock_sim_dir)
    assert result == mock_sim_dir / "npt.gro"


def test_find_structure_file_returns_none_if_missing(tmp_path):
    """Find structure file returns None if no candidates exist."""
    from mdfactory.orchestration.simulate import find_structure_file

    # Create empty directory with no structure files
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = find_structure_file(empty_dir)
    assert result is None


def test_extract_expected_frames_from_mdp(mock_sim_dir):
    """Extract expected frames from MDP file."""
    from mdfactory.orchestration.simulate import _extract_expected_frames_from_mdp

    # Create MDP with nsteps and nstxout-compressed
    mdp_content = """
    ; Production MDP
    nsteps = 500000
    nstxout-compressed = 5000  ; Save every 5000 steps
    dt = 0.002
    """
    (mock_sim_dir / "md.mdp").write_text(mdp_content)

    result = _extract_expected_frames_from_mdp(mock_sim_dir, "Production")

    # 500000 / 5000 = 100 frames
    assert result == 100


def test_extract_expected_frames_handles_comments(mock_sim_dir):
    """MDP parser ignores comments correctly."""
    from mdfactory.orchestration.simulate import _extract_expected_frames_from_mdp

    mdp_content = """
    ; nsteps = 999999  ; This is a comment
    nsteps = 100000  ; Real value
    ; nstxout-compressed = 123
    nstxout-compressed = 1000  ; Real value
    """
    (mock_sim_dir / "md.mdp").write_text(mdp_content)

    result = _extract_expected_frames_from_mdp(mock_sim_dir, "Production")

    # 100000 / 1000 = 100 frames
    assert result == 100


def test_extract_expected_frames_returns_none_for_missing_file(mock_sim_dir):
    """MDP parser returns None if file doesn't exist."""
    from mdfactory.orchestration.simulate import _extract_expected_frames_from_mdp

    result = _extract_expected_frames_from_mdp(mock_sim_dir, "Production")
    assert result is None


def test_extract_expected_frames_returns_none_for_incomplete_mdp(mock_sim_dir):
    """MDP parser returns None if required parameters missing."""
    from mdfactory.orchestration.simulate import _extract_expected_frames_from_mdp

    # Only nsteps, no nstxout-compressed
    (mock_sim_dir / "md.mdp").write_text("nsteps = 100000\n")

    result = _extract_expected_frames_from_mdp(mock_sim_dir, "Production")
    assert result is None


# Tests for GROMACS binary detection


def test_get_gromacs_detect_script_returns_string():
    """GROMACS detect function returns valid bash script."""
    from mdfactory.orchestration.apps import _get_gromacs_detect_script

    script = _get_gromacs_detect_script()

    assert isinstance(script, str)
    assert len(script) > 0


def test_get_gromacs_detect_script_contains_required_logic():
    """Script contains binary detection logic."""
    from mdfactory.orchestration.apps import _get_gromacs_detect_script

    script = _get_gromacs_detect_script()

    # Check for gmx detection
    assert "command -v gmx" in script
    assert "command -v gmx_mpi" in script

    # Check for GMX_BIN assignment
    assert 'GMX_BIN="gmx"' in script
    assert 'GMX_BIN="gmx_mpi"' in script

    # Check for error handling
    assert "ERROR" in script
    assert "exit 1" in script


def test_get_gromacs_detect_script_prefers_gmx_over_gmx_mpi():
    """Script checks for gmx first (thread-MPI preferred for local execution)."""
    from mdfactory.orchestration.apps import _get_gromacs_detect_script

    script = _get_gromacs_detect_script()

    # gmx should be checked before gmx_mpi
    gmx_pos = script.find("command -v gmx ")
    gmx_mpi_pos = script.find("command -v gmx_mpi")

    assert gmx_pos < gmx_mpi_pos, "gmx should be checked before gmx_mpi"


# === Decision 7: Checkpoint restart flag tests ===


# === Decision 7: Restart wiring through stage functions and execute_stage_list ===


@patch("mdfactory.orchestration.simulate.run_nvt_stage")
def test_execute_stage_list_passes_restart_to_stage_fn(mock_nvt):
    """_execute_stage_list forwards stage_restarts to the stage function."""
    mock_nvt.return_value = MagicMock()
    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    _execute_stage_list(
        sim_dir,
        ["NVT"],
        grompp_app,
        mdrun_app,
        stage_restarts={"NVT": "/sim/nvt.cpt"},
    )

    # NVT should receive restart_from_cpt kwarg
    mock_nvt.assert_called_once_with(
        sim_dir, None, grompp_app, mdrun_app, restart_from_cpt="/sim/nvt.cpt"
    )


@patch("mdfactory.orchestration.simulate.run_npt_stage")
@patch("mdfactory.orchestration.simulate.run_nvt_stage")
def test_execute_stage_list_restart_only_for_matching_stage(mock_nvt, mock_npt):
    """Only the stage with a cpt entry gets restart_from_cpt; others run normally."""
    mock_nvt.return_value = MagicMock()
    mock_npt.return_value = MagicMock()
    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    _execute_stage_list(
        sim_dir,
        ["NVT", "NPT"],
        grompp_app,
        mdrun_app,
        stage_restarts={"NPT": "/sim/npt.cpt"},
    )

    # NVT runs normally (no restart kwarg)
    mock_nvt.assert_called_once_with(sim_dir, None, grompp_app, mdrun_app)
    # NPT gets the restart kwarg
    mock_npt.assert_called_once_with(
        sim_dir, mock_nvt.return_value, grompp_app, mdrun_app,
        restart_from_cpt="/sim/npt.cpt",
    )


def test_nvt_stage_skips_grompp_on_restart():
    """run_nvt_stage skips grompp and passes restart_from_cpt to mdrun_app."""
    from mdfactory.orchestration.stages import run_nvt_stage

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    mdrun_app.return_value = MagicMock()

    run_nvt_stage(
        Path("/sim"),
        em_future=None,
        grompp_app=grompp_app,
        mdrun_app=mdrun_app,
        restart_from_cpt="/sim/nvt.cpt",
    )

    # grompp must NOT be called
    grompp_app.assert_not_called()
    # mdrun called with restart kwarg
    mdrun_app.assert_called_once()
    call_kwargs = mdrun_app.call_args[1]
    assert call_kwargs.get("restart_from_cpt") == "/sim/nvt.cpt"


def test_npt_stage_skips_grompp_on_restart():
    """run_npt_stage skips grompp and passes restart_from_cpt to mdrun_app."""
    from mdfactory.orchestration.stages import run_npt_stage

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    mdrun_app.return_value = MagicMock()

    run_npt_stage(
        Path("/sim"),
        nvt_future=None,
        grompp_app=grompp_app,
        mdrun_app=mdrun_app,
        restart_from_cpt="/sim/npt.cpt",
    )

    grompp_app.assert_not_called()
    mdrun_app.assert_called_once()
    call_kwargs = mdrun_app.call_args[1]
    assert call_kwargs.get("restart_from_cpt") == "/sim/npt.cpt"


def test_production_stage_skips_grompp_on_restart():
    """run_production_stage skips grompp and passes restart_from_cpt to mdrun_app."""
    from mdfactory.orchestration.stages import run_production_stage

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    mdrun_app.return_value = MagicMock()

    run_production_stage(
        Path("/sim"),
        npt_future=None,
        grompp_app=grompp_app,
        mdrun_app=mdrun_app,
        restart_from_cpt="/sim/prod.cpt",
    )

    grompp_app.assert_not_called()
    mdrun_app.assert_called_once()
    call_kwargs = mdrun_app.call_args[1]
    assert call_kwargs.get("restart_from_cpt") == "/sim/prod.cpt"


# === Decision 6: Per-stage resource config wiring tests ===


@patch("mdfactory.orchestration.simulate.run_em_stage")
def test_execute_stage_list_passes_stage_config_when_slurm(mock_em):
    """_execute_stage_list calls get_stage_config and forwards to EM stage."""
    from mdfactory.orchestration.config import SlurmExecutorConfig

    cfg = SlurmExecutorConfig(
        account="acct",
        cpus_per_node=12,
        stage_overrides={"EM": {"cpus_per_node": 4, "gres": None}},
    )
    mock_em.return_value = MagicMock()
    grompp_app = MagicMock()
    mdrun_app = MagicMock()

    _execute_stage_list(Path("/tmp/test"), ["EM"], grompp_app, mdrun_app, config=cfg)

    # stage_config kwarg should be the EM-overridden config
    mock_em.assert_called_once()
    _, call_kwargs = mock_em.call_args
    em_cfg = call_kwargs.get("stage_config")
    assert em_cfg is not None
    assert em_cfg.cpus_per_node == 4


@patch("mdfactory.orchestration.simulate.run_em_stage")
def test_execute_stage_list_no_stage_config_for_local(mock_em):
    """_execute_stage_list does not inject stage_config for LocalProvider (no overrides)."""
    from mdfactory.orchestration.config import ExecutorConfig

    cfg = ExecutorConfig()  # no get_stage_config method
    mock_em.return_value = MagicMock()

    _execute_stage_list(Path("/tmp/test"), ["EM"], MagicMock(), MagicMock(), config=cfg)

    # No stage_config kwarg should be injected
    _, call_kwargs = mock_em.call_args
    assert "stage_config" not in call_kwargs


def test_mdrun_app_ntasks_override_used_when_set():
    """Mdrun bash app uses explicit NTHR when ntasks > 0."""
    bash_script = _build_mdrun_script(deffnm="nvt", work_dir="/tmp/test", ntasks=8)

    assert "NTHR=8" in bash_script
    # Should NOT fall back to SLURM auto-detection
    assert "${SLURM_CPUS_PER_TASK" not in bash_script


def test_mdrun_app_cpu_mode_forced_when_disable_gpu():
    """Mdrun bash app uses CPU-only path when disable_gpu=True."""
    bash_script = _build_mdrun_script(deffnm="min", work_dir="/tmp/test", disable_gpu=True)

    # GPU condition should be replaced with literal "false"
    assert "if false;" in bash_script


def test_em_stage_passes_resource_hints_to_mdrun():
    """run_em_stage extracts resource hints from stage_config and passes to mdrun_app."""
    from mdfactory.orchestration.config import SlurmExecutorConfig
    from mdfactory.orchestration.stages import run_em_stage

    cfg = SlurmExecutorConfig(account="acct", cpus_per_node=4, gres=None)

    grompp_app = MagicMock()
    grompp_app.return_value = MagicMock()
    mdrun_app = MagicMock()
    mdrun_app.return_value = MagicMock()

    run_em_stage(Path("/sim"), grompp_app, mdrun_app, stage_config=cfg)

    mdrun_app.assert_called_once()
    _, call_kwargs = mdrun_app.call_args
    assert call_kwargs.get("ntasks") == 4
    assert call_kwargs.get("disable_gpu") is True  # gres=None → no GPU


def test_mdrun_app_no_cpt_flags_by_default():
    """Mdrun bash app omits -cpi/-append when no restart_from_cpt given."""
    bash_script = _build_mdrun_script(deffnm="nvt", work_dir="/tmp/test")

    assert "-cpi" not in bash_script
    assert "-append" not in bash_script


def test_mdrun_app_cpt_flags_added_when_restart_set():
    """Mdrun bash app adds -cpi <file> -append when restart_from_cpt provided."""
    bash_script = _build_mdrun_script(
        deffnm="npt",
        work_dir="/tmp/test",
        restart_from_cpt="/sim/npt.cpt",
    )

    assert "-cpi /sim/npt.cpt -append" in bash_script


def test_mdrun_app_cpt_flags_in_both_cpu_and_gpu_branches():
    """Checkpoint restart flags appear in both CPU and GPU command lines."""
    bash_script = _build_mdrun_script(
        deffnm="prod",
        work_dir="/tmp/test",
        restart_from_cpt="/sim/prod.cpt",
    )

    # Both GPU and CPU branches should carry the -cpi flag
    cpi_count = bash_script.count("-cpi /sim/prod.cpt -append")
    assert cpi_count >= 2, f"Expected -cpi flag in both GPU and CPU branches, found {cpi_count}"


def test_mdrun_app_cpt_log_message_mentions_file():
    """Mdrun bash app logs the checkpoint file it is resuming from."""
    bash_script = _build_mdrun_script(
        deffnm="prod",
        work_dir="/tmp/test",
        restart_from_cpt="/sim/prod.cpt",
    )

    assert "resuming from /sim/prod.cpt" in bash_script
