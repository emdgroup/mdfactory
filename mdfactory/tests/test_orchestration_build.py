# ABOUTME: Tests for orchestration build dispatch and dry-run functions
# ABOUTME: Validates Parsl submission, dry-run output, and error handling
"""Tests for orchestration build dispatch."""

from unittest.mock import MagicMock

import pytest

from mdfactory.orchestration.config import ExecutorConfig


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
    """build_systems_dry_run describes planned builds without loading Parsl."""
    import mdfactory.orchestration.build as build_mod
    from mdfactory.orchestration.build import build_systems_dry_run

    mock_model = FakeBuildInput(
        hash="ABC123", simulation_type="bilayer", parametrization="smirnoff"
    )
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)

    results = build_systems_dry_run(
        [mock_model],
        ExecutorConfig(),
        output_dir=tmp_path,
    )

    assert len(results) == 1
    assert results[0]["hash"] == "ABC123"
    assert results[0]["simulation_type"] == "bilayer"
    assert str(tmp_path / "ABC123") in results[0]["output_directory"]


def test_build_systems_dry_run_multiple(monkeypatch, tmp_path):
    """build_systems_dry_run handles multiple inputs."""
    import mdfactory.orchestration.build as build_mod
    from mdfactory.orchestration.build import build_systems_dry_run

    models = [FakeBuildInput(hash=f"HASH{i}") for i in range(3)]
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)

    results = build_systems_dry_run(models, ExecutorConfig(), output_dir=tmp_path)
    assert len(results) == 3
    assert [r["hash"] for r in results] == ["HASH0", "HASH1", "HASH2"]


def test_build_systems_submits_correct_count(monkeypatch, tmp_path):
    """build_systems submits one Parsl task per input."""
    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = {"hash": "H1", "status": "success", "directory": "/tmp/H1"}
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(build_mod.parsl, "load", MagicMock())
    monkeypatch.setattr(build_mod.parsl, "clear", MagicMock())

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
    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_future.result.side_effect = RuntimeError("CUDA OOM")
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(build_mod.parsl, "load", MagicMock())
    monkeypatch.setattr(build_mod.parsl, "clear", MagicMock())

    mock_model = FakeBuildInput(hash="FAIL1", simulation_type="bilayer")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    cfg = ExecutorConfig()
    results = build_mod.build_systems([mock_model], cfg, output_dir=tmp_path)

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "CUDA OOM" in results[0]["error"]


def test_build_systems_no_wait(monkeypatch, tmp_path):
    """build_systems with wait=False returns futures directly."""
    mock_app_fn = MagicMock()
    mock_future = MagicMock()
    mock_app_fn.return_value = mock_future

    import mdfactory.orchestration.build as build_mod

    monkeypatch.setattr(build_mod, "get_build_app", lambda: mock_app_fn)
    monkeypatch.setattr(build_mod.parsl, "load", MagicMock())

    mock_model = FakeBuildInput(hash="NW1")
    monkeypatch.setattr(build_mod, "BuildInput", FakeBuildInput)
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())

    cfg = ExecutorConfig()
    futures = build_mod.build_systems([mock_model], cfg, output_dir=tmp_path, wait=False)

    assert futures == [mock_future]


def test_build_systems_dry_run_invalid_input_type(tmp_path):
    """build_systems_dry_run raises TypeError for invalid input."""
    from mdfactory.orchestration.build import build_systems_dry_run

    with pytest.raises(TypeError, match="Expected BuildInput or dict"):
        build_systems_dry_run([42], ExecutorConfig(), output_dir=tmp_path)
