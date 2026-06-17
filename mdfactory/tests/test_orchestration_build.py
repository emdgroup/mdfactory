# ABOUTME: Tests for orchestration build dispatch and dry-run functions
# ABOUTME: Validates Parsl submission, dry-run output, and error handling
"""Tests for orchestration build dispatch."""

from unittest.mock import MagicMock, patch

import pytest

parsl = pytest.importorskip("parsl", reason="parsl not installed")

from mdfactory.orchestration.config import ExecutorConfig  # noqa: E402


class FakeBuildInput:
    """Fake BuildInput for testing isinstance checks."""

    def __init__(
        self,
        hash="TEST",
        simulation_type="mixedbox",
        parametrization="cgenff",
        engine="gromacs",
    ):
        self.hash = hash
        self.simulation_type = simulation_type
        self.parametrization = parametrization
        self.engine = engine

    def model_dump(self):
        """Return dict representation."""
        return {
            "simulation_type": self.simulation_type,
            "parametrization": self.parametrization,
            "engine": self.engine,
        }


def test_build_systems_dry_run(monkeypatch, tmp_path):
    """build_systems with dry_run=True describes planned builds without loading Parsl."""
    import mdfactory.orchestration.build as build_mod
    from mdfactory.orchestration.build import build_systems

    mock_model = FakeBuildInput(
        hash="ABC123", simulation_type="bilayer", parametrization="smirnoff"
    )
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)

    results = build_systems(
        [mock_model],
        ExecutorConfig(),
        output_dir=tmp_path,
        dry_run=True,
    )

    assert len(results) == 1
    assert results[0]["hash"] == "ABC123"
    assert results[0]["simulation_type"] == "bilayer"
    assert str(tmp_path / "ABC123") in results[0]["output_directory"]


def test_build_systems_dry_run_multiple(monkeypatch, tmp_path):
    """build_systems with dry_run=True handles multiple inputs."""
    import mdfactory.orchestration.build as build_mod
    from mdfactory.orchestration.build import build_systems

    models = [FakeBuildInput(hash=f"HASH{i}") for i in range(3)]
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)

    results = build_systems(models, ExecutorConfig(), output_dir=tmp_path, dry_run=True)
    assert len(results) == 3
    assert [r["hash"] for r in results] == ["HASH0", "HASH1", "HASH2"]


def test_build_systems_submits_correct_count(monkeypatch, tmp_path):
    """build_systems submits one Parsl task per input."""
    import parsl

    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = {"hash": "H1", "status": "success", "directory": "/tmp/H1"}
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(parsl, "load", MagicMock())
    monkeypatch.setattr(parsl, "clear", MagicMock())
    monkeypatch.setattr(parsl, "dfk", MagicMock(side_effect=RuntimeError("No DFK")))

    mock_model = FakeBuildInput(hash="H1")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)

    cfg = ExecutorConfig()
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    results = build_mod.build_systems([mock_model], cfg, output_dir=tmp_path)

    assert mock_app_fn.call_count == 1
    assert len(results) == 1
    assert results[0]["status"] == "success"


def test_build_systems_handles_failed_future(monkeypatch, tmp_path):
    """build_systems captures exceptions from failed futures."""
    import parsl

    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_future.result.side_effect = RuntimeError("CUDA OOM")
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(parsl, "load", MagicMock())
    monkeypatch.setattr(parsl, "clear", MagicMock())
    monkeypatch.setattr(parsl, "dfk", MagicMock(side_effect=RuntimeError("No DFK")))

    mock_model = FakeBuildInput(hash="FAIL1", simulation_type="bilayer")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    cfg = ExecutorConfig()
    results = build_mod.build_systems([mock_model], cfg, output_dir=tmp_path)

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "CUDA OOM" in results[0]["error"]
    # Finding 10: failure metadata is surfaced for future retry logic.
    assert results[0]["failure_type"] == "RuntimeError"
    assert results[0]["error_detail"] == "CUDA OOM"


def test_build_systems_no_wait(monkeypatch, tmp_path):
    """build_systems with wait=False returns futures directly."""
    import parsl

    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(parsl, "load", MagicMock())

    mock_model = FakeBuildInput(hash="NW1")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    cfg = ExecutorConfig()
    futures = build_mod.build_systems([mock_model], cfg, output_dir=tmp_path, wait=False)

    assert futures == [mock_future]


def test_build_systems_dry_run_invalid_input_type(tmp_path):
    """build_systems with dry_run=True raises TypeError for invalid input."""
    from mdfactory.orchestration.build import build_systems

    with pytest.raises(TypeError, match="Expected BuildInput or dict"):
        build_systems([42], ExecutorConfig(), output_dir=tmp_path, dry_run=True)


# --- Finding 5: Tests for _build_system_impl ---


def test_build_system_impl_strips_internal_keys_and_chdirs(monkeypatch, tmp_path):
    """_build_system_impl strips _-prefixed keys and chdirs to build_dir."""
    from mdfactory.orchestration.apps import _build_system_impl

    build_dir = tmp_path / "OUT"
    captured_model = {}

    def mock_run_build(model):
        import os

        captured_model["hash"] = model.hash
        captured_model["cwd"] = os.getcwd()

    monkeypatch.setattr("mdfactory.workflows.run_build_from_dict", mock_run_build)

    # Provide minimal valid BuildInput fields plus internal keys
    input_dict = {
        "simulation_type": "mixedbox",
        "parametrization": "cgenff",
        "engine": "gromacs",
        "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
        "_build_dir": str(build_dir),
        "_internal_flag": "should_be_stripped",
    }
    result = _build_system_impl(input_dict)

    assert result["status"] == "success"
    assert result["directory"] == str(build_dir.resolve())
    assert build_dir.exists()
    # Verify we chdir'd to build_dir during execution
    assert captured_model["cwd"] == str(build_dir)


def test_build_system_impl_restores_cwd_on_failure(monkeypatch, tmp_path):
    """_build_system_impl restores original cwd even if build fails."""
    import os

    from mdfactory.orchestration.apps import _build_system_impl

    build_dir = tmp_path / "FAIL"
    original_cwd = os.getcwd()

    def mock_run_build_fail(model):
        raise RuntimeError("Build exploded")

    monkeypatch.setattr("mdfactory.workflows.run_build_from_dict", mock_run_build_fail)

    input_dict = {
        "simulation_type": "mixedbox",
        "parametrization": "cgenff",
        "engine": "gromacs",
        "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
        "_build_dir": str(build_dir),
    }
    with pytest.raises(RuntimeError, match="Build exploded"):
        _build_system_impl(input_dict)

    # cwd should be restored
    assert os.getcwd() == original_cwd


# --- Finding 9: Tests for SLURM cleanup functions ---


def test_shutdown_parsl_calls_clear_and_scancel(monkeypatch):
    """_shutdown_parsl calls parsl.clear() and scancels SLURM jobs."""
    import parsl as parsl_mod

    from mdfactory.orchestration.build import _shutdown_parsl

    mock_dfk = MagicMock()
    mock_executor = MagicMock()
    mock_executor.provider.resources = {"block-1": {"remote_job_id": "12345"}}
    mock_dfk.executors = {"htex": mock_executor}

    monkeypatch.setattr(parsl_mod, "dfk", MagicMock(return_value=mock_dfk))
    monkeypatch.setattr(parsl_mod, "clear", MagicMock())

    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0)
        _shutdown_parsl()

    parsl_mod.clear.assert_called_once()
    mock_subprocess.assert_called_once()
    assert "12345" in mock_subprocess.call_args[0][0]


def test_shutdown_parsl_attempts_scancel_even_if_clear_fails(monkeypatch):
    """_shutdown_parsl runs scancel even when parsl.clear() raises."""
    import parsl as parsl_mod

    from mdfactory.orchestration.build import _shutdown_parsl

    mock_dfk = MagicMock()
    mock_executor = MagicMock()
    mock_executor.provider.resources = {"block-1": {"remote_job_id": "99999"}}
    mock_dfk.executors = {"htex": mock_executor}

    monkeypatch.setattr(parsl_mod, "dfk", MagicMock(return_value=mock_dfk))
    monkeypatch.setattr(parsl_mod, "clear", MagicMock(side_effect=RuntimeError("timeout")))

    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0)
        _shutdown_parsl()

    # scancel should still be called despite clear() failing
    mock_subprocess.assert_called_once()
    assert "99999" in mock_subprocess.call_args[0][0]


def test_get_slurm_job_ids_handles_missing_resources(monkeypatch):
    """_get_slurm_job_ids handles executors without resources attribute."""
    from mdfactory.orchestration.build import _get_slurm_job_ids

    mock_dfk = MagicMock()
    mock_executor = MagicMock(spec=[])  # no attributes
    mock_dfk.executors = {"htex": mock_executor}

    result = _get_slurm_job_ids(mock_dfk)
    assert result == []


def test_keyboard_interrupt_triggers_shutdown(monkeypatch, tmp_path):
    """KeyboardInterrupt during _wait_with_progress triggers _shutdown_parsl."""
    import parsl as parsl_mod

    import mdfactory.orchestration.build as build_mod

    mock_app_fn = MagicMock()
    # Future that raises KeyboardInterrupt when polled
    mock_future = MagicMock()
    mock_future.done.side_effect = KeyboardInterrupt()
    mock_app_fn.return_value = mock_future

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(parsl_mod, "load", MagicMock())
    monkeypatch.setattr(parsl_mod, "clear", MagicMock())
    monkeypatch.setattr(parsl_mod, "dfk", MagicMock(side_effect=RuntimeError("No DFK")))

    mock_model = FakeBuildInput(hash="INT1")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    cfg = ExecutorConfig()
    with pytest.raises(KeyboardInterrupt):
        build_mod.build_systems([mock_model], cfg, output_dir=tmp_path)

    # parsl.clear should have been called via _shutdown_parsl in the finally block
    assert parsl_mod.clear.called


# --- Finding 10: Tests for dict input path ---


def test_build_systems_dry_run_with_dict_input(monkeypatch, tmp_path):
    """build_systems with dry_run=True handles dict input correctly."""
    from mdfactory.orchestration.build import build_systems

    input_dict = {
        "simulation_type": "mixedbox",
        "parametrization": "cgenff",
        "engine": "gromacs",
        "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
    }

    results = build_systems([input_dict], ExecutorConfig(), output_dir=tmp_path, dry_run=True)

    assert len(results) == 1
    assert results[0]["simulation_type"] == "mixedbox"
    assert results[0]["parametrization"] == "cgenff"
    assert results[0]["hash"]  # hash should be computed


def test_build_systems_with_dict_input(monkeypatch, tmp_path):
    """build_systems handles dict input and injects _build_dir."""
    import parsl as parsl_mod

    import mdfactory.orchestration.build as build_mod

    captured_args = []
    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = {"hash": "X", "status": "success", "directory": "/tmp/X"}

    def capture_app_call(input_dict):
        captured_args.append(input_dict)
        return mock_future

    mock_app_fn.side_effect = capture_app_call

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(parsl_mod, "load", MagicMock())
    monkeypatch.setattr(parsl_mod, "clear", MagicMock())
    monkeypatch.setattr(parsl_mod, "dfk", MagicMock(side_effect=RuntimeError("No DFK")))
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    input_dict = {
        "simulation_type": "mixedbox",
        "parametrization": "cgenff",
        "engine": "gromacs",
        "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
    }

    cfg = ExecutorConfig()
    results = build_mod.build_systems([input_dict], cfg, output_dir=tmp_path)

    assert len(captured_args) == 1
    assert "_build_dir" in captured_args[0]
    assert results[0]["status"] == "success"


# --- Finding 9: Multi-iteration polling test ---


def test_wait_with_progress_multi_iteration(monkeypatch):
    """_wait_with_progress exercises multi-iteration polling when future is not immediately done."""
    from mdfactory.orchestration.build import _wait_with_progress

    mock_future = MagicMock()
    mock_future.done.side_effect = [False, False, True]
    mock_future.result.return_value = {"hash": "POLL1", "status": "success", "directory": "/tmp"}
    mock_future.task_status.return_value = "running"

    results = _wait_with_progress([mock_future], hashes=["POLL1"], poll_interval=0.01)

    assert len(results) == 1
    assert results[0]["status"] == "success"
    # done() should have been called at least twice (False then True)
    assert mock_future.done.call_count >= 2


# --- Finding 10: Assert non-cleanup in no_wait ---


def test_build_systems_no_wait_does_not_clear(monkeypatch, tmp_path):
    """build_systems with wait=False does NOT call parsl.clear()."""
    import parsl

    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(parsl, "load", MagicMock())
    mock_clear = MagicMock()
    monkeypatch.setattr(parsl, "clear", mock_clear)

    mock_model = FakeBuildInput(hash="NW2")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    cfg = ExecutorConfig()
    futures = build_mod.build_systems([mock_model], cfg, output_dir=tmp_path, wait=False)

    assert futures == [mock_future]
    mock_clear.assert_not_called()


# --- Finding 10: failure description helper ---


def test_describe_failure_plain_exception():
    """_describe_failure returns the exception type name and message."""
    from mdfactory.orchestration.build import _describe_failure

    failure_type, detail = _describe_failure(RuntimeError("boom"))
    assert failure_type == "RuntimeError"
    assert detail == "boom"


def test_describe_failure_unwraps_legacy_e_value():
    """_describe_failure unwraps a legacy Parsl wrapper exposing .e_value."""
    from mdfactory.orchestration.build import _describe_failure

    class FakeAppFailure(Exception):
        """Mimic an older Parsl wrapper carrying the underlying error."""

        def __init__(self, e_value):
            super().__init__("wrapped")
            self.e_value = e_value

    underlying = ValueError("real GROMACS crash")
    failure_type, detail = _describe_failure(FakeAppFailure(underlying))
    assert failure_type == "ValueError"
    assert detail == "real GROMACS crash"


# --- Finding 12: completeness guard ---


def test_collect_results_returns_complete_list():
    """_collect_results returns all results when every slot is captured."""
    from mdfactory.orchestration.build import _collect_results

    results = [{"hash": "A"}, {"hash": "B"}]
    assert _collect_results(results, ["A", "B"]) == results


def test_collect_results_raises_on_uncaptured_slot():
    """_collect_results raises rather than silently dropping a None slot."""
    from mdfactory.orchestration.build import _collect_results

    results = [{"hash": "A"}, None]
    with pytest.raises(RuntimeError, match="never captured"):
        _collect_results(results, ["AAAA", "BBBB"])
