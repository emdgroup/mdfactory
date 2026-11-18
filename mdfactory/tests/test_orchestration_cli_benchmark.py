# ABOUTME: Unit tests for the `mdfactory benchmark` CLI command
# ABOUTME: Covers result reporting and argument wiring into the sweep
"""Tests for the benchmark CLI command."""

from unittest.mock import MagicMock, patch

import pytest

from mdfactory.cli import _report_benchmark_result, benchmark
from mdfactory.performance.benchmark import BenchmarkConfig, BenchmarkResult, TrialResult


def _result(throughputs, threshold=0.75):
    """Build a BenchmarkResult from ``(cpu_count, ns_per_day)`` pairs."""
    return BenchmarkResult(
        system_path="/tmp/sim",
        trials=[
            TrialResult(cpu_count=cpus, ns_per_day=ns, wall_seconds=1.0) for cpus, ns in throughputs
        ],
        efficiency_threshold=threshold,
    )


def test_report_benchmark_result_names_the_highest_justified_count():
    """The answer reported is the highest count that meets the threshold."""
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_benchmark_result(_result([(1, 4.0), (2, 7.0), (4, 11.0)]))

    messages = " ".join(str(call) for call in mock_logger.info.call_args_list)
    assert "cpus=1: 4.000 ns/day, 100% efficiency" in messages
    assert "cpus=4: 11.000 ns/day, 69% efficiency" in messages
    assert "Highest justified CPU count: 2" in messages


def test_report_benchmark_result_warns_when_nothing_is_justified():
    """A fully failed sweep reports a warning instead of a CPU count."""
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_benchmark_result(_result([(1, None), (2, None)]))

    mock_logger.warning.assert_called_once()


@pytest.fixture
def slurm_config(tmp_path):
    """Provide a prepared sim dir plus a SLURM config path."""
    (tmp_path / "system.pdb").touch()
    config_path = tmp_path / "slurm.yaml"
    config_path.write_text("provider: slurm\n")
    return tmp_path, str(config_path)


@pytest.mark.parametrize(
    ("kwargs", "expected_counts"),
    [
        ({}, [1, 2, 4, 8, 16, 32, 64]),
        ({"cpus": [2, 8, 32]}, [2, 8, 32]),
    ],
)
def test_benchmark_command_wires_flags_into_the_sweep(slurm_config, kwargs, expected_counts):
    """Flags and the swept core counts reach BenchmarkConfig unchanged."""
    source, slurm = slurm_config

    with (
        patch("mdfactory.cli._load_executor_config", return_value=MagicMock()),
        patch("mdfactory.performance.benchmark.run_benchmark_sweep") as mock_sweep,
        patch("mdfactory.cli._report_benchmark_result") as mock_report,
    ):
        mock_sweep.return_value = MagicMock()
        benchmark(source=source, slurm=slurm, duration_ps=5.0, efficiency=0.8, **kwargs)

    assert mock_sweep.call_count == 1
    _, _, config = mock_sweep.call_args.args
    assert config == BenchmarkConfig(
        cpu_counts=expected_counts, duration_ps=5.0, efficiency_threshold=0.8
    )
    # A dropped flag would make a preview submit real work.
    assert mock_sweep.call_args.kwargs["dry_run"] is False
    mock_report.assert_called_once()


def test_benchmark_command_forwards_dry_run(slurm_config):
    """--dry-run reaches run_benchmark_sweep so no work is submitted."""
    source, slurm = slurm_config

    with (
        patch("mdfactory.cli._load_executor_config", return_value=MagicMock()),
        patch("mdfactory.performance.benchmark.run_benchmark_sweep") as mock_sweep,
        patch("mdfactory.cli._report_benchmark_result"),
    ):
        mock_sweep.return_value = MagicMock()
        benchmark(source=source, slurm=slurm, dry_run=True)

    assert mock_sweep.call_args.kwargs["dry_run"] is True
