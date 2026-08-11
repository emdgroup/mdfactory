# ABOUTME: Unit tests for Parsl-based GROMACS simulation orchestration
# ABOUTME: Tests checkpoint detection, validation, and dry-run mode
"""Unit tests for simulation orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.apps import (
    _assemble_mdrun_command,
    _build_mdrun_script,
    _resolve_binary_token,
    _resolve_gpu_flags,
    _resolve_is_mpi,
    _resolve_restart_flags,
    _resolve_thread_count_expr,
    _resolve_thread_flags,
)
from mdfactory.orchestration.config import ExecutorConfig
from mdfactory.orchestration.simulate import (
    _detect_needed_stages,
    _execute_stage_list,
    _log_dry_run_plan,
    _missing_build_files,
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


# ---------------------------------------------------------------------------
# _missing_build_files
# ---------------------------------------------------------------------------


def test_missing_build_files_complete(mock_sim_dir):
    """Returns empty list when all required files are present."""
    assert _missing_build_files(mock_sim_dir, ["EM", "NVT", "NPT", "Production"]) == []


def test_missing_build_files_empty_dir(tmp_path):
    """Returns all required files when directory is empty."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    missing = _missing_build_files(sim_dir, ["EM"])
    assert "system.pdb" in missing
    assert "topology.top" in missing
    assert "em.mdp" in missing


def test_missing_build_files_partial(mock_sim_dir):
    """Returns only the missing file when one is removed."""
    (mock_sim_dir / "npt.mdp").unlink()
    missing = _missing_build_files(mock_sim_dir, ["EM", "NVT", "NPT", "Production"])
    assert missing == ["npt.mdp"]


# ---------------------------------------------------------------------------
# run_simulations — incomplete-build filtering
# ---------------------------------------------------------------------------


def test_run_simulations_skips_incomplete_build(tmp_path):
    """run_simulations warns and skips dirs with missing build outputs.

    Skipped dirs appear in the results as status='skipped' entries; only the
    ready dir appears in the dry-run work-plan portion of the results.
    """
    ready = tmp_path / "ready"
    ready.mkdir()
    for f in ["system.pdb", "topology.top", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
        (ready / f).write_text("FAKE")

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    # No files written — build never completed

    config = ExecutorConfig()
    results = run_simulations([ready, incomplete], config, dry_run=True)

    # The incomplete dir is reported as skipped, not silently dropped.
    skipped = [r for r in results if r.get("status") == "skipped"]
    skipped_hashes = [r["hash"] for r in skipped]
    assert "incomplete" in skipped_hashes

    # The ready dir appears in the work-plan portion (has a sim_dir key).
    plan_items = [r for r in results if "sim_dir" in r]
    plan_dirs = [item["sim_dir"] for item in plan_items]
    assert ready in plan_dirs
    assert incomplete not in plan_dirs


def test_run_simulations_all_incomplete_returns_skipped_entries(tmp_path):
    """run_simulations returns skipped entries (not []) when no directories are ready."""
    empty = tmp_path / "empty"
    empty.mkdir()

    config = ExecutorConfig()
    results = run_simulations([empty], config, dry_run=True)
    assert len(results) == 1
    assert results[0]["status"] == "skipped"
    assert results[0]["hash"] == "empty"


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


@patch("mdfactory.orchestration.simulate._validate_trajectory_complete", return_value=True)
def test_checkpoint_all_stages_complete(mock_validate, mock_sim_dir):
    """All stages skipped when outputs AND prerequisite checkpoints exist."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.cpt").write_text("FAKE CPT")  # NPT needs this
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")  # Production needs this
    (mock_sim_dir / "prod.xtc").write_text("FAKE")  # Trajectory present

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
    run_simulations(sim_dirs, ExecutorConfig(), stages=["EM", "NVT", "NPT", "Production"])

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
    """Incomplete builds are skipped before Parsl submission with a skipped entry."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").write_text("FAKE")
    # Missing topology.top and all MDP files — build never completed

    # Should not raise; returns a skipped entry (not []) so callers can account
    # for the skipped directory.
    results = run_simulations([sim_dir], ExecutorConfig(), dry_run=True)
    assert len(results) == 1
    assert results[0]["status"] == "skipped"


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
    (sim3 / "prod.xtc").write_text("FAKE")  # Trajectory present

    # Mock trajectory validation so fake XTC counts as complete
    with patch("mdfactory.orchestration.simulate._validate_trajectory_complete", return_value=True):
        results = run_simulations([sim1, sim2, sim3], ExecutorConfig(), dry_run=True)

    assert len(results) == 3
    assert results[0]["stages"] == ["EM", "NVT", "NPT", "Production"]
    assert results[1]["stages"] == ["NVT", "NPT", "Production"]
    assert results[2]["stages"] == []


@patch("mdfactory.orchestration.simulate._validate_trajectory_complete", return_value=True)
def test_checkpoint_production_output(mock_validate, mock_sim_dir):
    """Production stage skipped when prod.xtc is validated as complete."""
    (mock_sim_dir / "min.gro").write_text("FAKE")
    (mock_sim_dir / "nvt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.gro").write_text("FAKE")
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")  # Production prerequisite
    (mock_sim_dir / "prod.xtc").write_text("FAKE")  # Trajectory present

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


@patch("mdfactory.orchestration.simulate.parsl_session")
@patch("mdfactory.orchestration.simulate._execute_stage_list")
def test_wait_false_emits_warning(mock_execute, mock_session, mock_sim_dir):
    """wait=False logs a warning that raw futures resolve to None, not dicts."""
    mock_execute.return_value = MagicMock()
    mock_session.return_value.__enter__.return_value = MagicMock()

    with patch("mdfactory.orchestration.simulate.logger") as mock_logger:
        run_simulations([mock_sim_dir], ExecutorConfig(), wait=False)

    # Warning must be emitted and mention the None / parsl.clear() contract
    mock_logger.warning.assert_called()
    warning_text = " ".join(str(a) for call in mock_logger.warning.call_args_list for a in call[0])
    assert "None" in warning_text or "raw" in warning_text
    assert "parsl.clear()" in warning_text


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


@patch("mdfactory.orchestration.simulate._validate_trajectory_complete", return_value=True)
def test_checkpoint_production_skips_with_all_files(mock_validate, mock_sim_dir):
    """Production skipped when both prod.xtc AND npt.cpt exist (trajectory validated)."""
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")
    (mock_sim_dir / "prod.xtc").write_text("FAKE")  # Trajectory present

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")

    assert "Production" not in needed


@patch("mdfactory.orchestration.simulate._validate_trajectory_complete", return_value=True)
def test_checkpoint_production_complete_with_trr_only(mock_validate, mock_sim_dir):
    """Production marked complete when only prod.trr exists (TRR-only MDP config)."""
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")
    # XTC absent; TRR present and mocked as validated
    (mock_sim_dir / "prod.trr").write_text("FAKE")

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")

    assert "Production" not in needed


def test_checkpoint_production_partial_with_neither_traj(mock_sim_dir):
    """Production is partial (not complete) when neither prod.xtc nor prod.trr exists."""
    (mock_sim_dir / "npt.cpt").write_text("FAKE CPT")
    (mock_sim_dir / "prod.cpt").write_text("FAKE CPT")
    (mock_sim_dir / "prod.tpr").write_text("FAKE TPR")

    needed = _detect_needed_stages(mock_sim_dir, ["Production"], "auto")

    assert "Production" in needed


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
# Tests refactored to target resolver functions directly (pure Python, no bash parsing).


def test_resolve_thread_count_expr_auto():
    """Auto thread count uses three-level SLURM > OMP > nproc fallback."""
    expr = _resolve_thread_count_expr(0)
    assert expr == "${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-$(nproc)}}"


def test_resolve_thread_count_expr_explicit():
    """Explicit ntasks produces a literal decimal string."""
    assert _resolve_thread_count_expr(8) == "8"
    assert _resolve_thread_count_expr(12) == "12"


def test_resolve_thread_count_expr_nproc_fallback():
    """Three-level fallback uses nproc (not a fixed number) as the last resort.

    Guards against accidental regression to the old two-level form
    ``${SLURM_CPUS_PER_TASK:-$(nproc)}``, which ignores OMP_NUM_THREADS
    and would cause a thread-count mismatch in Parsl worker environments.
    """
    expr = _resolve_thread_count_expr(0)
    # All three levels must be present in the correct order
    assert "SLURM_CPUS_PER_TASK" in expr
    assert "OMP_NUM_THREADS" in expr
    assert "$(nproc)" in expr
    assert expr.index("SLURM_CPUS_PER_TASK") < expr.index("OMP_NUM_THREADS") < expr.index("nproc")


def test_resolve_gpu_flags_gpu_and_pme_gpu():
    """GPU mode with PME-on-GPU emits correct GPU flags."""
    flags = _resolve_gpu_flags(has_gpu=True, pme_gpu=True)
    assert flags == "-nb gpu -pme gpu -gpu_id $GPU_ID"


def test_resolve_gpu_flags_gpu_no_pme():
    """GPU mode with PME-on-CPU (e.g. EM) emits -pme cpu."""
    flags = _resolve_gpu_flags(has_gpu=True, pme_gpu=False)
    assert flags == "-nb gpu -pme cpu -gpu_id $GPU_ID"


def test_resolve_gpu_flags_no_gpu():
    """CPU-only mode emits an empty string."""
    assert _resolve_gpu_flags(has_gpu=False, pme_gpu=True) == ""
    assert _resolve_gpu_flags(has_gpu=False, pme_gpu=False) == ""


def test_resolve_is_mpi():
    """Binary selector resolves to expected is_mpi values."""
    assert _resolve_is_mpi("gmx_mpi") is True
    assert _resolve_is_mpi("gmx") is False
    assert _resolve_is_mpi("auto") is None


def test_resolve_thread_flags_explicit_mpi():
    """MPI build always uses -ntomp regardless of GPU mode."""
    assert _resolve_thread_flags(is_mpi=True, has_gpu=True) == "-ntomp $NTHR"
    assert _resolve_thread_flags(is_mpi=True, has_gpu=False) == "-ntomp $NTHR"


def test_resolve_thread_flags_tmpi_gpu():
    """Thread-MPI build with GPU uses -ntmpi 1 -ntomp."""
    assert _resolve_thread_flags(is_mpi=False, has_gpu=True) == "-ntmpi 1 -ntomp $NTHR"


def test_resolve_thread_flags_tmpi_cpu():
    """Thread-MPI build without GPU uses -nt (all-thread count)."""
    assert _resolve_thread_flags(is_mpi=False, has_gpu=False) == "-nt $NTHR"


def test_resolve_thread_flags_auto():
    """Auto mode defers to shell variable $MDRUN_THREAD_FLAGS."""
    assert _resolve_thread_flags(is_mpi=None, has_gpu=True) == "$MDRUN_THREAD_FLAGS"
    assert _resolve_thread_flags(is_mpi=None, has_gpu=False) == "$MDRUN_THREAD_FLAGS"


def test_resolve_binary_token():
    """Binary token resolver returns expected strings."""
    assert _resolve_binary_token("gmx") == "gmx"
    assert _resolve_binary_token("gmx_mpi") == "gmx_mpi"
    assert _resolve_binary_token("auto") == "$GMX_BIN"


def test_assemble_mdrun_command_gpu_tmpi():
    """Assembler produces correct GPU + thread-MPI command."""
    cmd = _assemble_mdrun_command(
        "gmx",
        "prod",
        "-cpi prod.cpt -append",
        "-ntmpi 1 -ntomp $NTHR",
        "-nb gpu -pme gpu -gpu_id $GPU_ID",
    )
    assert cmd == (
        "gmx mdrun -deffnm prod -cpi prod.cpt -append "
        "-ntmpi 1 -ntomp $NTHR -nb gpu -pme gpu -gpu_id $GPU_ID"
    )


def test_assemble_mdrun_command_cpu_mpi_no_restart():
    """Assembler skips empty parts — no duplicate spaces."""
    cmd = _assemble_mdrun_command("gmx_mpi", "min", "", "-ntomp $NTHR", "")
    assert cmd == "gmx_mpi mdrun -deffnm min -ntomp $NTHR"
    assert "  " not in cmd  # no double spaces


def test_mdrun_app_logging():
    """Mdrun bash app logs starting message to stderr."""
    bash_script = _build_mdrun_script(deffnm="test", work_dir="/tmp/test")

    # Starting message is always present
    assert 'echo "GROMACS mdrun: Starting test with $NTHR threads' in bash_script
    assert ">&2" in bash_script  # Logs to stderr


def test_mdrun_app_slurm_priority():
    """SLURM_CPUS_PER_TASK has priority over OMP_NUM_THREADS in the fallback chain."""
    expr = _resolve_thread_count_expr(0)
    # Three-level form; SLURM_CPUS_PER_TASK is checked before OMP_NUM_THREADS
    assert "SLURM_CPUS_PER_TASK" in expr
    assert "OMP_NUM_THREADS" in expr
    assert expr.index("SLURM_CPUS_PER_TASK") < expr.index("OMP_NUM_THREADS")


def test_stage_functions_no_hardcoded_nt():
    """Stage functions don't pass hardcoded nt parameter to mdrun."""
    import inspect

    from mdfactory.orchestration import stages

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
    result = _execute_stage_list(Path("/tmp/test"), [], None, None, max_rescue=0)
    assert result is None


def test_execute_stage_list_validates_order():
    """Execute stage list validates stages are in dependency order."""
    with pytest.raises(ValueError, match="must be in dependency order"):
        # NPT before NVT is invalid
        _execute_stage_list(Path("/tmp/test"), ["NPT", "NVT"], None, None, max_rescue=0)


def test_execute_stage_list_validates_unknown_stage():
    """Execute stage list rejects unknown stage names."""
    with pytest.raises(ValueError, match="Unknown stage.*Equilibration"):
        _execute_stage_list(Path("/tmp/test"), ["Equilibration"], None, None, max_rescue=0)


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_em_only(mock_run_stage):
    """Execute stage list dispatches EM via run_stage with correct spec and no prev_future."""
    mock_future = MagicMock()
    mock_run_stage.return_value = mock_future

    result = _execute_stage_list(Path("/tmp/test"), ["EM"], MagicMock(), MagicMock(), max_rescue=0)

    mock_run_stage.assert_called_once()
    call_args = mock_run_stage.call_args[0]
    assert call_args[0].name == "EM"  # spec
    assert call_args[2] is None  # prev_future = None for first stage
    assert result == mock_future


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_em_nvt_chain(mock_run_stage):
    """Execute stage list chains EM → NVT: NVT receives EM's future as prev_future."""
    em_future = MagicMock()
    nvt_future = MagicMock()
    mock_run_stage.side_effect = [em_future, nvt_future]

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    result = _execute_stage_list(sim_dir, ["EM", "NVT"], grompp_app, mdrun_app, max_rescue=0)

    assert mock_run_stage.call_count == 2
    calls = mock_run_stage.call_args_list
    # First call: EM, prev_future=None
    assert calls[0][0][0].name == "EM"
    assert calls[0][0][2] is None
    # Second call: NVT, prev_future=em_future (chained)
    assert calls[1][0][0].name == "NVT"
    assert calls[1][0][2] is em_future
    assert result == nvt_future


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_full_pipeline(mock_run_stage):
    """Execute stage list chains all 4 stages via run_stage with correct futures."""
    stage_futures = [MagicMock() for _ in range(4)]
    mock_run_stage.side_effect = stage_futures

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    result = _execute_stage_list(
        sim_dir, ["EM", "NVT", "NPT", "Production"], grompp_app, mdrun_app, max_rescue=0
    )

    assert mock_run_stage.call_count == 4
    calls = mock_run_stage.call_args_list
    # Verify spec names and future chaining
    assert calls[0][0][0].name == "EM"
    assert calls[0][0][2] is None  # no prev for first stage
    assert calls[1][0][0].name == "NVT"
    assert calls[1][0][2] is stage_futures[0]
    assert calls[2][0][0].name == "NPT"
    assert calls[2][0][2] is stage_futures[1]
    assert calls[3][0][0].name == "Production"
    assert calls[3][0][2] is stage_futures[2]
    assert result is stage_futures[3]


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_partial_pipeline_from_checkpoint(mock_run_stage):
    """Checkpoint resume: first stage is not EM; prev_future starts as None."""
    nvt_fut = MagicMock()
    npt_fut = MagicMock()
    mock_run_stage.side_effect = [nvt_fut, npt_fut]

    grompp_app = MagicMock()
    mdrun_app = MagicMock()
    sim_dir = Path("/tmp/test")

    result = _execute_stage_list(sim_dir, ["NVT", "NPT"], grompp_app, mdrun_app, max_rescue=0)

    calls = mock_run_stage.call_args_list
    assert calls[0][0][0].name == "NVT"
    assert calls[0][0][2] is None  # no preceding future
    assert calls[1][0][0].name == "NPT"
    assert calls[1][0][2] is nvt_fut  # chained
    assert result is npt_fut


@patch("mdfactory.orchestration.rescue.execute_stage_with_rescue")
@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_routes_to_rescue_when_enabled(mock_run_stage, mock_rescue):
    """Rescue-eligible stages dispatch to execute_stage_with_rescue when max_rescue > 0."""
    rescue_future = MagicMock()
    mock_rescue.return_value = rescue_future
    sim_dir = Path("/tmp/test")

    result = _execute_stage_list(sim_dir, ["EM"], MagicMock(), MagicMock(), max_rescue=3)

    mock_rescue.assert_called_once()
    mock_run_stage.assert_not_called()
    assert result is rescue_future


@patch("mdfactory.orchestration.rescue.execute_stage_with_rescue")
@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_production_bypasses_rescue(mock_run_stage, mock_rescue):
    """Production stage uses run_stage even when max_rescue > 0."""
    prod_future = MagicMock()
    mock_run_stage.return_value = prod_future
    sim_dir = Path("/tmp/test")

    result = _execute_stage_list(sim_dir, ["Production"], MagicMock(), MagicMock(), max_rescue=3)

    mock_run_stage.assert_called_once()
    mock_rescue.assert_not_called()
    assert result is prod_future


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

    run_simulations([mock_sim_dir], ExecutorConfig(), checkpoint_mode="auto")

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
    """Mdrun bash app uses strict mode and spec-driven output verification.

    The new implementation relies on ``set -euo pipefail`` (not inline
    ``if ! mdrun ...``) to propagate mdrun errors, and uses a single
    unconditional mdrun command with no surrounding if/then/fi wrapper.
    """
    # Pass gro_out so output-verification block is included
    bash_script = _build_mdrun_script(deffnm="min", work_dir="/tmp/test", gro_out="min.gro")

    # bash strict mode is present
    assert "set -euo pipefail" in bash_script
    # Output check is present (spec-driven, not inline command check)
    assert 'if [ ! -f "min.gro" ]' in bash_script
    # No 4-branch if/then/fi command wrappers — mdrun appears as a single line
    assert bash_script.count("if ! $GMX_BIN mdrun") == 0
    # The mdrun command is on exactly one line (no surrounding if/fi)
    mdrun_lines = [ln for ln in bash_script.splitlines() if "mdrun -deffnm" in ln]
    assert len(mdrun_lines) == 1, f"Expected 1 mdrun command line, found: {mdrun_lines}"


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
    """Mdrun command appears as a single unconditional line (no if/fi wrapper).

    The new implementation emits ``set -euo pipefail`` for error propagation
    instead of inline ``if ! mdrun ...`` guards, producing a readable script
    with one unconditional mdrun command.
    """
    bash_script = _build_mdrun_script(deffnm="prod", work_dir="/tmp/test")

    # mdrun command is a single line — no surrounding if/then/fi
    mdrun_lines = [ln.strip() for ln in bash_script.splitlines() if "mdrun -deffnm" in ln]
    assert len(mdrun_lines) == 1, f"Expected 1 mdrun line, got: {mdrun_lines}"
    cmd_line = mdrun_lines[0]
    assert not cmd_line.startswith("if"), "mdrun command must not be wrapped in an if-check"
    assert "mdrun -deffnm prod" in cmd_line


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

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    assert result is False


def test_validate_trajectory_returns_false_for_empty_file(mock_sim_dir):
    """Trajectory validation returns False for empty files."""

    (mock_sim_dir / "prod.xtc").write_text("")  # Empty file
    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    assert result is False


def test_validate_trajectory_incomplete_without_structure(mock_sim_dir):
    """Trajectory validation returns False when no structure file is available.

    Without a topology we cannot parse the trajectory, so the result is
    conservatively False — the caller should trigger a partial restart rather
    than silently skipping a stage with an unverifiable trajectory.
    """
    # Create trajectory but no structure files
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 15_000_000)  # 15 MB

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    # No structure → cannot validate → treat as incomplete
    assert result is False


def test_validate_trajectory_returns_false_for_non_parseable_file(mock_sim_dir):
    """Trajectory validation returns False for non-parseable trajectory data."""
    # Small file that can't be parsed as an XTC/TRR
    (mock_sim_dir / "prod.xtc").write_bytes(b"X" * 1000)

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc")
    assert result is False


@patch("mdfactory.orchestration.simulate.mda")
def test_validate_trajectory_with_mdanalysis_complete(mock_mda, mock_sim_dir):
    """Trajectory validation uses MDAnalysis to count frames (complete)."""

    # Setup mocks
    (mock_sim_dir / "prod.xtc").write_bytes(b"FAKE XTC")
    (mock_sim_dir / "system.pdb").write_text("FAKE PDB")

    mock_traj = MagicMock()
    mock_traj.__len__.return_value = 100  # 100 frames
    mock_universe = MagicMock()
    mock_universe.trajectory = mock_traj
    mock_mda.Universe.return_value = mock_universe

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc", expected_frames=100)

    # Should pass (100 >= 100)
    assert result is True


@patch("mdfactory.orchestration.simulate.mda")
def test_validate_trajectory_with_mdanalysis_incomplete(mock_mda, mock_sim_dir):
    """Trajectory validation detects incomplete trajectories."""

    # Setup mocks
    (mock_sim_dir / "prod.xtc").write_bytes(b"FAKE XTC")
    (mock_sim_dir / "system.pdb").write_text("FAKE PDB")

    mock_traj = MagicMock()
    mock_traj.__len__.return_value = 50  # Only 50 frames
    mock_universe = MagicMock()
    mock_universe.trajectory = mock_traj
    mock_mda.Universe.return_value = mock_universe

    result = _validate_trajectory_complete(mock_sim_dir, "prod.xtc", expected_frames=100)

    # Should fail (50 < 100)
    assert result is False


@patch("mdfactory.orchestration.simulate.mda")
def test_validate_trajectory_without_expected_frames(mock_mda, mock_sim_dir):
    """Trajectory validation without expected_frames checks readability only."""

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


def test_find_structure_file_candidates_derived_from_registry():
    """_STRUCTURE_CANDIDATES matches the current STAGE_REGISTRY gro_out fields."""
    from mdfactory.orchestration.simulate import _STRUCTURE_CANDIDATES
    from mdfactory.orchestration.stages import STAGE_REGISTRY

    expected = [spec.gro_out for spec in reversed(STAGE_REGISTRY) if spec.gro_out] + ["system.pdb"]
    assert _STRUCTURE_CANDIDATES == expected


def test_find_structure_file_picks_up_new_stage_gro(tmp_path, monkeypatch):
    """find_structure_file returns a new stage's .gro when _STRUCTURE_CANDIDATES is extended."""
    import mdfactory.orchestration.simulate as sim_module
    from mdfactory.orchestration.simulate import find_structure_file

    # Simulate a hypothetical future 'Heating' stage by patching the candidate list.
    extended = ["heat.gro"] + sim_module._STRUCTURE_CANDIDATES
    monkeypatch.setattr(sim_module, "_STRUCTURE_CANDIDATES", extended)

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "heat.gro").write_text("FAKE HEATED COORDS")

    result = find_structure_file(sim_dir)
    assert result == sim_dir / "heat.gro"


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


def test_build_gmx_detect_block_no_extras():
    """Core detection block sets GMX_BIN only when no extras supplied."""
    from mdfactory.orchestration.apps import _build_gmx_detect_block

    block = _build_gmx_detect_block()
    assert "GMX_BIN=gmx\n" in block
    assert "GMX_BIN=gmx_mpi" in block
    assert "MDRUN_THREAD_FLAGS" not in block
    assert block.startswith("if command -v gmx")
    assert block.endswith("fi")


def test_build_gmx_detect_block_with_extras():
    """Extra assignments are appended to each branch via semicolons."""
    from mdfactory.orchestration.apps import _build_gmx_detect_block

    block = _build_gmx_detect_block(
        gmx_extra='MDRUN_THREAD_FLAGS="-nt $NTHR"',
        gmx_mpi_extra='MDRUN_THREAD_FLAGS="-ntomp $NTHR"',
    )
    assert 'GMX_BIN=gmx; MDRUN_THREAD_FLAGS="-nt $NTHR"' in block
    assert 'GMX_BIN=gmx_mpi; MDRUN_THREAD_FLAGS="-ntomp $NTHR"' in block


def test_build_gmx_detect_block_error_handling():
    """Block contains the canonical error message and exit 1."""
    from mdfactory.orchestration.apps import _build_gmx_detect_block

    block = _build_gmx_detect_block()
    assert "ERROR: Neither gmx nor gmx_mpi found in PATH" in block
    assert "exit 1" in block


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

    # Check for GMX_BIN assignment (unquoted, consistent with _build_gmx_detect_block)
    assert "GMX_BIN=gmx\n" in script  # gmx branch (not gmx_mpi)
    assert "GMX_BIN=gmx_mpi" in script

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


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_passes_restart_to_stage_fn(mock_run_stage):
    """_execute_stage_list forwards stage_restarts to run_stage as restart_from_cpt."""
    mock_run_stage.return_value = MagicMock()
    sim_dir = Path("/tmp/test")

    _execute_stage_list(
        sim_dir,
        ["NVT"],
        MagicMock(),
        MagicMock(),
        stage_restarts={"NVT": "/sim/nvt.cpt"},
    )

    mock_run_stage.assert_called_once()
    call_kwargs = mock_run_stage.call_args[1]
    assert call_kwargs.get("restart_from_cpt") == "/sim/nvt.cpt"


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_restart_only_for_matching_stage(mock_run_stage):
    """Only the stage with a cpt entry gets a non-empty restart_from_cpt."""
    nvt_fut = MagicMock()
    mock_run_stage.side_effect = [nvt_fut, MagicMock()]
    sim_dir = Path("/tmp/test")

    _execute_stage_list(
        sim_dir,
        ["NVT", "NPT"],
        MagicMock(),
        MagicMock(),
        stage_restarts={"NPT": "/sim/npt.cpt"},
        max_rescue=0,
    )

    calls = mock_run_stage.call_args_list
    # NVT: no restart (empty string)
    assert calls[0][1].get("restart_from_cpt") == ""
    # NPT: gets restart path
    assert calls[1][1].get("restart_from_cpt") == "/sim/npt.cpt"


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


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_passes_stage_config_when_slurm(mock_run_stage):
    """_execute_stage_list resolves per-stage config and forwards as stage_config to run_stage."""
    from mdfactory.orchestration.config import SlurmExecutorConfig

    cfg = SlurmExecutorConfig(
        account="acct",
        cpus_per_node=12,
        stage_overrides={"EM": {"cpus_per_node": 4, "gres": None}},
    )
    mock_run_stage.return_value = MagicMock()

    _execute_stage_list(
        Path("/tmp/test"), ["EM"], MagicMock(), MagicMock(), config=cfg, max_rescue=0
    )

    mock_run_stage.assert_called_once()
    _, call_kwargs = mock_run_stage.call_args
    em_cfg = call_kwargs.get("stage_config")
    assert em_cfg is not None
    assert em_cfg.cpus_per_node == 4


@patch("mdfactory.orchestration.simulate.run_stage")
def test_execute_stage_list_no_stage_config_for_local(mock_run_stage):
    """_execute_stage_list does not inject stage_config for LocalProvider (no overrides)."""
    from mdfactory.orchestration.config import ExecutorConfig

    mock_run_stage.return_value = MagicMock()

    _execute_stage_list(
        Path("/tmp/test"),
        ["EM"],
        MagicMock(),
        MagicMock(),
        config=ExecutorConfig(),
        max_rescue=0,
    )

    _, call_kwargs = mock_run_stage.call_args
    assert "stage_config" not in call_kwargs


def test_mdrun_app_ntasks_override_used_when_set():
    """Explicit ntasks embeds a literal NTHR= in the script."""
    # Check via resolver (pure Python)
    assert _resolve_thread_count_expr(8) == "8"

    # Confirm it flows through _build_mdrun_script to the preamble
    bash_script = _build_mdrun_script(deffnm="nvt", work_dir="/tmp/test", ntasks=8)
    assert "NTHR=8" in bash_script
    # Should NOT fall back to SLURM auto-detection
    assert "${SLURM_CPUS_PER_TASK" not in bash_script


def test_mdrun_app_cpu_mode_forced_when_disable_gpu():
    """disable_gpu=True produces a script with no GPU flags anywhere."""
    # Check via resolver (pure Python)
    assert _resolve_gpu_flags(has_gpu=False, pme_gpu=True) == ""

    # Confirm the generated script has no GPU-related content
    bash_script = _build_mdrun_script(deffnm="min", work_dir="/tmp/test", disable_gpu=True)
    assert "-nb gpu" not in bash_script
    assert "-gpu_id" not in bash_script
    assert "GPU_ID" not in bash_script


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
    """No restart_from_cpt produces empty restart flags."""
    # Check via resolver (pure Python)
    assert _resolve_restart_flags("") == ""

    # Confirm the generated script has no checkpoint flags
    bash_script = _build_mdrun_script(deffnm="nvt", work_dir="/tmp/test")
    assert "-cpi" not in bash_script
    assert "-append" not in bash_script


def test_mdrun_app_cpt_flags_added_when_restart_set():
    """restart_from_cpt adds -cpi <file> -append to the mdrun command."""
    # Check via resolver (pure Python)
    assert _resolve_restart_flags("x.cpt") == "-cpi x.cpt -append"

    # Confirm it flows into the generated script
    bash_script = _build_mdrun_script(
        deffnm="npt",
        work_dir="/tmp/test",
        restart_from_cpt="/sim/npt.cpt",
    )
    assert "-cpi /sim/npt.cpt -append" in bash_script


def test_mdrun_app_cpt_flags_in_single_command():
    """Checkpoint restart flags appear exactly once in the single mdrun command.

    The new implementation emits one unconditional mdrun command, so -cpi
    appears exactly once (not duplicated across GPU/CPU branches).
    """
    bash_script = _build_mdrun_script(
        deffnm="prod",
        work_dir="/tmp/test",
        restart_from_cpt="/sim/prod.cpt",
    )

    cpi_count = bash_script.count("-cpi /sim/prod.cpt -append")
    assert cpi_count == 1, (
        f"Expected -cpi flag exactly once in the single mdrun command, found {cpi_count}"
    )


def test_mdrun_app_cpt_log_message_mentions_file():
    """Mdrun bash app logs the checkpoint file it is resuming from."""
    bash_script = _build_mdrun_script(
        deffnm="prod",
        work_dir="/tmp/test",
        restart_from_cpt="/sim/prod.cpt",
    )

    assert "resuming from /sim/prod.cpt" in bash_script


# ---------------------------------------------------------------------------
# T1: Checkpoint restart detection from disk state (Finding 5)
# ---------------------------------------------------------------------------


def test_detect_stage_state_partial_restart_nvt(tmp_path):
    """Partial NVT: .cpt + .tpr present, .gro absent → status=partial, restart=True."""
    from mdfactory.orchestration.simulate import _detect_stage_state

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "nvt.cpt").write_bytes(b"x")
    (sim_dir / "nvt.tpr").write_bytes(b"x")
    # nvt.gro intentionally absent

    state = _detect_stage_state(sim_dir, "NVT", "auto")

    assert state["status"] == "partial"
    assert state["restart"] is True
    assert state["cpt_file"] == sim_dir / "nvt.cpt"


def test_detect_stage_state_partial_restart_em(tmp_path):
    """Partial EM: min.cpt + min.tpr present, min.gro absent → status=partial."""
    from mdfactory.orchestration.simulate import _detect_stage_state

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "min.cpt").write_bytes(b"x")
    (sim_dir / "min.tpr").write_bytes(b"x")

    state = _detect_stage_state(sim_dir, "EM", "auto")

    assert state["status"] == "partial"
    assert state["restart"] is True
    assert state["cpt_file"] == sim_dir / "min.cpt"


def test_detect_stage_state_not_started_when_no_files(tmp_path):
    """Empty directory → status=not_started, restart=False."""
    from mdfactory.orchestration.simulate import _detect_stage_state

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()

    state = _detect_stage_state(sim_dir, "NVT", "auto")

    assert state["status"] == "not_started"
    assert state["restart"] is False
    assert state["cpt_file"] is None


def test_detect_needed_stages_with_restart_info_partial(tmp_path):
    """Restart info propagated into the work-plan when partial progress exists."""
    from mdfactory.orchestration.simulate import _detect_needed_stages_with_restart_info

    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "nvt.cpt").write_bytes(b"x")
    (sim_dir / "nvt.tpr").write_bytes(b"x")

    items = _detect_needed_stages_with_restart_info(sim_dir, ["NVT"], "auto")

    assert len(items) == 1
    assert items[0]["stage"] == "NVT"
    assert items[0]["restart"] is True
    assert items[0]["cpt_file"] == sim_dir / "nvt.cpt"


@patch("mdfactory.orchestration.simulate.parsl_session")
@patch("mdfactory.orchestration.simulate._execute_stage_list")
def test_run_simulations_propagates_stage_restarts(mock_execute, mock_session, tmp_path):
    """stage_restarts dict assembled from checkpoint detection is passed to _execute_stage_list."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    for f in ["system.pdb", "topology.top", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
        (sim_dir / f).write_text("FAKE")
    # NVT partial: .cpt + .tpr present
    (sim_dir / "nvt.cpt").write_bytes(b"x")
    (sim_dir / "nvt.tpr").write_bytes(b"x")
    # EM already complete
    (sim_dir / "min.gro").write_text("FAKE")

    mock_execute.return_value = MagicMock()
    mock_session.return_value.__enter__.return_value = MagicMock()
    mock_session.return_value.__exit__.return_value = False

    from mdfactory.orchestration.config import ExecutorConfig

    run_simulations([sim_dir], ExecutorConfig(), wait=False)

    # _execute_stage_list must be called; assert stage_restarts contains NVT entry
    assert mock_execute.called
    call_kwargs = mock_execute.call_args[1]
    restarts = call_kwargs.get("stage_restarts", {})
    assert "NVT" in restarts
    assert str(sim_dir / "nvt.cpt") == restarts["NVT"]


# ---------------------------------------------------------------------------
# T3: run_stage grompp-kwarg wiring (Finding 7)
# ---------------------------------------------------------------------------


def test_run_stage_nvt_passes_ref_file_to_grompp(tmp_path):
    """NVT: run_stage calls grompp_app with ref_file=min.gro (position restraints)."""
    from mdfactory.orchestration.stages import STAGE_BY_NAME, run_stage

    grompp_app = MagicMock(return_value=MagicMock())
    mdrun_app = MagicMock(return_value=MagicMock())

    run_stage(STAGE_BY_NAME["NVT"], tmp_path, None, grompp_app, mdrun_app)

    call_kwargs = grompp_app.call_args[1]
    assert call_kwargs["ref_file"] == "min.gro"
    # NVT has no prereq_cpt; cpt_file kwarg should be absent
    assert "cpt_file" not in call_kwargs


def test_run_stage_npt_passes_ref_and_cpt_to_grompp(tmp_path):
    """NPT: run_stage calls grompp_app with both ref_file=nvt.gro and cpt_file=nvt.cpt."""
    from mdfactory.orchestration.stages import STAGE_BY_NAME, run_stage

    grompp_app = MagicMock(return_value=MagicMock())
    mdrun_app = MagicMock(return_value=MagicMock())

    run_stage(STAGE_BY_NAME["NPT"], tmp_path, None, grompp_app, mdrun_app)

    call_kwargs = grompp_app.call_args[1]
    assert call_kwargs["ref_file"] == "nvt.gro"
    assert call_kwargs["cpt_file"] == "nvt.cpt"
    assert call_kwargs["maxwarn"] == 2


def test_run_stage_em_has_no_ref_or_cpt_flags(tmp_path):
    """EM: run_stage calls grompp_app without ref_file or cpt_file (none specified in spec)."""
    from mdfactory.orchestration.stages import STAGE_BY_NAME, run_stage

    grompp_app = MagicMock(return_value=MagicMock())
    mdrun_app = MagicMock(return_value=MagicMock())

    run_stage(STAGE_BY_NAME["EM"], tmp_path, None, grompp_app, mdrun_app)

    call_kwargs = grompp_app.call_args[1]
    assert "ref_file" not in call_kwargs
    assert "cpt_file" not in call_kwargs


def test_run_stage_production_has_cpt_but_no_ref(tmp_path):
    """Production: run_stage calls grompp_app with cpt_file=npt.cpt but no ref_file."""
    from mdfactory.orchestration.stages import STAGE_BY_NAME, run_stage

    grompp_app = MagicMock(return_value=MagicMock())
    mdrun_app = MagicMock(return_value=MagicMock())

    run_stage(STAGE_BY_NAME["Production"], tmp_path, None, grompp_app, mdrun_app)

    call_kwargs = grompp_app.call_args[1]
    assert "ref_file" not in call_kwargs
    assert call_kwargs["cpt_file"] == "npt.cpt"


# ---------------------------------------------------------------------------
# T5: _bash_result_to_dict (Finding 15)
# ---------------------------------------------------------------------------


def test_bash_result_to_dict_none_becomes_success():
    """None (bash_app exit) → success dict."""
    from mdfactory.orchestration.simulate import _bash_result_to_dict

    result = _bash_result_to_dict(None, "abc123")
    assert result == {"hash": "abc123", "status": "success"}


def test_bash_result_to_dict_int_becomes_success():
    """Integer exit code → success dict."""
    from mdfactory.orchestration.simulate import _bash_result_to_dict

    result = _bash_result_to_dict(0, "abc123")
    assert result == {"hash": "abc123", "status": "success"}


def test_bash_result_to_dict_dict_passthrough():
    """Dict values (python_app callers) pass through unchanged."""
    from mdfactory.orchestration.simulate import _bash_result_to_dict

    d = {"hash": "abc123", "status": "failed", "error": "mdrun crashed"}
    assert _bash_result_to_dict(d, "abc123") is d


# ---------------------------------------------------------------------------
# T2: grompp -r/-t flag generation (Finding 6)
# ---------------------------------------------------------------------------


def test_build_grompp_script_generates_ref_flag():
    """_build_grompp_script emits -r flag when ref_file is provided."""
    from mdfactory.orchestration.apps import _build_grompp_script

    script = _build_grompp_script(
        "nvt.mdp", "min.gro", "topology.top", "nvt.tpr", "/sim", ref_file="min.gro"
    )
    assert "-r min.gro" in script


def test_build_grompp_script_generates_cpt_flag():
    """_build_grompp_script emits -t flag when cpt_file is provided."""
    from mdfactory.orchestration.apps import _build_grompp_script

    script = _build_grompp_script(
        "npt.mdp", "nvt.gro", "topology.top", "npt.tpr", "/sim", cpt_file="nvt.cpt"
    )
    assert "-t nvt.cpt" in script


def test_build_grompp_script_omits_flags_by_default():
    """_build_grompp_script omits -r and -t when no ref_file / cpt_file given."""
    from mdfactory.orchestration.apps import _build_grompp_script

    script = _build_grompp_script("em.mdp", "system.pdb", "topology.top", "em.tpr", "/sim")
    assert "-r " not in script
    assert "-t " not in script


def test_build_grompp_script_can_emit_both_flags():
    """_build_grompp_script emits both -r and -t when both are provided."""
    from mdfactory.orchestration.apps import _build_grompp_script

    script = _build_grompp_script(
        "npt.mdp",
        "nvt.gro",
        "topology.top",
        "npt.tpr",
        "/sim",
        ref_file="nvt.gro",
        cpt_file="nvt.cpt",
    )
    assert "-r nvt.gro" in script
    assert "-t nvt.cpt" in script


# ---------------------------------------------------------------------------
# T6: gmx_binary selection in _build_mdrun_script (Finding 16)
# ---------------------------------------------------------------------------


def test_build_mdrun_script_auto_includes_binary_detection_preamble():
    """gmx_binary='auto' emits the runtime binary detection if/elif block."""
    script = _build_mdrun_script("prod", "/sim", gmx_binary="auto")
    # The detection preamble checks for 'gmx' and 'gmx_mpi' at runtime
    assert "command -v gmx" in script


def test_build_mdrun_script_explicit_gmx_omits_detection_preamble():
    """gmx_binary='gmx' uses literal 'gmx' binary, no runtime detection."""
    script = _build_mdrun_script("prod", "/sim", gmx_binary="gmx")
    assert "command -v gmx" not in script
    # Literal binary name appears in the mdrun command
    assert "gmx mdrun" in script


def test_build_mdrun_script_explicit_gmx_mpi():
    """gmx_binary='gmx_mpi' uses literal 'gmx_mpi' binary, no detection preamble."""
    script = _build_mdrun_script("prod", "/sim", gmx_binary="gmx_mpi")
    assert "command -v gmx" not in script
    assert "gmx_mpi mdrun" in script


# ---------------------------------------------------------------------------
# T7: _extract_resource_hints GPU / gmx_binary branches (Finding 18)
# ---------------------------------------------------------------------------


def test_extract_resource_hints_gpu_gres_disables_disable_gpu():
    """GPU gres string → disable_gpu=False (GPU mode active)."""
    from mdfactory.orchestration.stages import _extract_resource_hints

    cfg = MagicMock(
        cpus_per_node=12, gres="gpu:l40s:1", gmx_binary="gmx_mpi", max_workers_per_node=1
    )
    hints = _extract_resource_hints(cfg)

    assert hints.disable_gpu is False
    assert hints.gmx_binary == "gmx_mpi"
    assert hints.ntasks == 12


def test_extract_resource_hints_non_gpu_gres_sets_disable_gpu():
    """Non-GPU gres string → disable_gpu=True (no GPU)."""
    from mdfactory.orchestration.stages import _extract_resource_hints

    cfg = MagicMock(cpus_per_node=4, gres="ssd:1", gmx_binary="auto", max_workers_per_node=1)
    hints = _extract_resource_hints(cfg)

    assert hints.disable_gpu is True


def test_extract_resource_hints_none_gres_sets_disable_gpu():
    """gres=None → disable_gpu=True (no GPU)."""
    from mdfactory.orchestration.stages import _extract_resource_hints

    cfg = MagicMock(cpus_per_node=8, gres=None, gmx_binary="auto", max_workers_per_node=1)
    hints = _extract_resource_hints(cfg)

    assert hints.disable_gpu is True


def test_extract_resource_hints_none_stage_config_is_cpu_safe():
    """stage_config=None → disable_gpu=True (safe default for local runs)."""
    from mdfactory.orchestration.stages import _extract_resource_hints

    hints = _extract_resource_hints(None)

    assert hints.disable_gpu is True
    assert hints.ntasks == 0
    assert hints.gmx_binary == "auto"


def test_extract_resource_hints_divides_by_max_workers():
    """ntasks is divided by max_workers_per_node when >1 to avoid oversubscription."""
    from mdfactory.orchestration.stages import _extract_resource_hints

    cfg = MagicMock(cpus_per_node=12, gres=None, gmx_binary="auto", max_workers_per_node=2)
    hints = _extract_resource_hints(cfg)

    assert hints.ntasks == 6  # 12 // 2
