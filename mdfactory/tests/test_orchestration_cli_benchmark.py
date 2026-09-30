# ABOUTME: Unit tests for the `mdfactory benchmark` CLI command
# ABOUTME: Covers result reporting and argument wiring into the sweep
"""Tests for the benchmark CLI command."""

from unittest.mock import MagicMock, patch

from mdfactory.cli import _report_benchmark_result, benchmark
from mdfactory.performance.benchmark import BenchmarkResult, TrialResult


def _make_result(optimum: TrialResult | None) -> BenchmarkResult:
    """Build a BenchmarkResult with one success, one failure, and *optimum*."""
    trials = [
        TrialResult(cpu_count=1, gpu_replicas=0, ns_per_day=10.0, wall_seconds=1.0),
        TrialResult(cpu_count=2, gpu_replicas=0, ns_per_day=None, wall_seconds=1.0, error="boom"),
    ]
    return BenchmarkResult(
        system_path="/tmp/sim",
        trials=trials,
        optimum=optimum,
        selection_criterion="ns_per_day",
    )


def test_report_benchmark_result_logs_optimum():
    """The selected optimum is reported via the logger."""
    optimum = TrialResult(cpu_count=1, gpu_replicas=0, ns_per_day=10.0, wall_seconds=1.0)
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_benchmark_result(_make_result(optimum))
    messages = " ".join(str(call) for call in mock_logger.info.call_args_list)
    assert "Optimum" in messages


def test_report_benchmark_result_without_optimum_warns():
    """No successful trial produces a warning rather than an optimum line."""
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_benchmark_result(_make_result(None))
    mock_logger.warning.assert_called_once()


def test_benchmark_command_wires_arguments_into_sweep(tmp_path):
    """CLI flags are forwarded into BenchmarkConfig for each resolved sim dir."""
    (tmp_path / "system.pdb").touch()
    config_path = tmp_path / "slurm.yaml"
    config_path.write_text("provider: slurm\n")

    with (
        patch("mdfactory.cli._load_executor_config", return_value=MagicMock()),
        patch("mdfactory.performance.benchmark.run_benchmark_sweep") as mock_sweep,
        patch("mdfactory.cli._report_benchmark_result") as mock_report,
    ):
        mock_sweep.return_value = MagicMock()
        benchmark(
            source=tmp_path,
            slurm=str(config_path),
            cpus=[1, 2],
            gpus=None,
            duration_ps=5.0,
            selection="efficiency",
            dry_run=False,
        )

    assert mock_sweep.call_count == 1
    _, _, benchmark_config = mock_sweep.call_args.args
    assert benchmark_config.cpu_counts == [1, 2]
    assert benchmark_config.selection == "efficiency"
    assert benchmark_config.duration_ps == 5.0
    # dry_run=False must be forwarded explicitly (finding 11): a dropped
    # flag would make a preview run submit real work.
    assert mock_sweep.call_args.kwargs["dry_run"] is False
    mock_report.assert_called_once()


def test_benchmark_command_forwards_dry_run(tmp_path):
    """--dry-run reaches run_benchmark_sweep so no work is submitted."""
    (tmp_path / "system.pdb").touch()
    config_path = tmp_path / "slurm.yaml"
    config_path.write_text("provider: slurm\n")

    with (
        patch("mdfactory.cli._load_executor_config", return_value=MagicMock()),
        patch("mdfactory.performance.benchmark.run_benchmark_sweep") as mock_sweep,
        patch("mdfactory.cli._report_benchmark_result"),
    ):
        mock_sweep.return_value = MagicMock()
        benchmark(source=tmp_path, slurm=str(config_path), dry_run=True)

    assert mock_sweep.call_count == 1
    assert mock_sweep.call_args.kwargs["dry_run"] is True


def test_benchmark_command_gpu_ties_cores_to_gpus(tmp_path):
    """--gpus sets cores 1:1 to GPUs; an explicit --cpus is ignored."""
    (tmp_path / "system.pdb").touch()
    config_path = tmp_path / "slurm.yaml"
    config_path.write_text("provider: slurm\n")

    with (
        patch("mdfactory.cli._load_executor_config") as mock_load,
        patch("mdfactory.performance.benchmark.run_benchmark_sweep") as mock_sweep,
        patch("mdfactory.cli._report_benchmark_result"),
    ):
        mock_load.return_value = MagicMock(cpus_per_node=8)
        mock_sweep.return_value = MagicMock()
        benchmark(source=tmp_path, slurm=str(config_path), cpus=[8], gpus=[1, 2])

    _, _, benchmark_config = mock_sweep.call_args.args
    assert benchmark_config.cpu_counts == [1, 2]  # cores = GPUs, --cpus ignored
    assert benchmark_config.gpu_replicas == [1, 2]
