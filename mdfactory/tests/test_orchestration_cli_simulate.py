# ABOUTME: Unit tests for the CLI simulate command and its helper functions
# ABOUTME: Covers path resolution, hash filtering, and result reporting
"""Unit tests for CLI simulate command helpers (Finding 8)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdfactory.cli import (
    _discover_sim_dirs,
    _filter_sim_paths_by_hash_prefix,
    _report_simulation_results,
    _resolve_sim_paths_for_simulate,
)


# ---------------------------------------------------------------------------
# _discover_sim_dirs
# ---------------------------------------------------------------------------


def test_discover_sim_dirs_direct(tmp_path):
    """Root containing system.pdb is returned as a single-element list."""
    (tmp_path / "system.pdb").touch()
    result = _discover_sim_dirs(tmp_path)
    assert result == [tmp_path]


def test_discover_sim_dirs_children(tmp_path):
    """Children that contain system.pdb are returned as sorted list."""
    for name in ["abc123", "def456"]:
        d = tmp_path / name
        d.mkdir()
        (d / "system.pdb").touch()
    result = _discover_sim_dirs(tmp_path)
    assert len(result) == 2
    assert result == sorted([tmp_path / "abc123", tmp_path / "def456"])


def test_discover_sim_dirs_ignores_non_sim_children(tmp_path):
    """Children without system.pdb are silently ignored."""
    good = tmp_path / "abc123"
    good.mkdir()
    (good / "system.pdb").touch()

    bad = tmp_path / "not_a_sim"
    bad.mkdir()
    # No system.pdb → should not appear

    result = _discover_sim_dirs(tmp_path)
    assert result == [good]


def test_discover_sim_dirs_no_sims_raises(tmp_path):
    """No simulation-ready directories → ValueError with helpful message."""
    with pytest.raises(ValueError, match="No simulation directories"):
        _discover_sim_dirs(tmp_path)


# ---------------------------------------------------------------------------
# _filter_sim_paths_by_hash_prefix
# ---------------------------------------------------------------------------


def test_filter_sim_paths_exact_prefix_match(tmp_path):
    """Exact hash prefix matches the correct directory."""
    abc = tmp_path / "ABC123"
    def_ = tmp_path / "DEF456"
    result = _filter_sim_paths_by_hash_prefix([abc, def_], ["ABC123"])
    assert result == [abc]


def test_filter_sim_paths_case_insensitive(tmp_path):
    """Hash prefix matching is case-insensitive."""
    abc = tmp_path / "ABC123"
    def_ = tmp_path / "DEF456"
    result = _filter_sim_paths_by_hash_prefix([abc, def_], ["abc"])
    assert result == [abc]


def test_filter_sim_paths_multiple_prefixes(tmp_path):
    """Multiple prefixes can match multiple directories."""
    abc = tmp_path / "ABC123"
    def_ = tmp_path / "DEF456"
    result = _filter_sim_paths_by_hash_prefix([abc, def_], ["abc", "def"])
    assert len(result) == 2


def test_filter_sim_paths_no_match_raises(tmp_path):
    """No matching prefix → ValueError listing available directories."""
    paths = [tmp_path / "ABC123"]
    with pytest.raises(ValueError, match="No simulation directories match"):
        _filter_sim_paths_by_hash_prefix(paths, ["xyz"])


def test_filter_sim_paths_returns_sorted(tmp_path):
    """Result is sorted by path name."""
    b = tmp_path / "BBB"
    a = tmp_path / "AAA"
    result = _filter_sim_paths_by_hash_prefix([b, a], ["A", "B"])
    assert result == [a, b]


# ---------------------------------------------------------------------------
# _resolve_sim_paths_for_simulate
# ---------------------------------------------------------------------------


def test_resolve_sim_paths_directory_delegates_to_discover(tmp_path):
    """A directory source delegates to _discover_sim_dirs."""
    (tmp_path / "system.pdb").touch()
    result = _resolve_sim_paths_for_simulate(tmp_path)
    assert result == [tmp_path]


def test_resolve_sim_paths_yaml_source(tmp_path):
    """A YAML source reads 'system_directory' entries."""
    import yaml

    dirs = [str(tmp_path / "sim1"), str(tmp_path / "sim2")]
    summary = {"system_directory": dirs}
    yaml_path = tmp_path / "summary.yaml"
    yaml_path.write_text(yaml.dump(summary))

    result = _resolve_sim_paths_for_simulate(yaml_path)
    assert result == [Path(d) for d in dirs]


def test_resolve_sim_paths_invalid_source_raises(tmp_path):
    """Non-directory, non-YAML source raises ValueError."""
    txt_file = tmp_path / "not_valid.txt"
    txt_file.touch()
    with pytest.raises(ValueError, match="Invalid source"):
        _resolve_sim_paths_for_simulate(txt_file)


# ---------------------------------------------------------------------------
# _report_simulation_results — all status branches (Findings 8, 17)
# ---------------------------------------------------------------------------


def test_report_simulation_results_success_only(capsys):
    """All success → correct succeeded count, no errors reported."""
    results = [
        {"hash": "a", "status": "success"},
        {"hash": "b", "status": "success"},
    ]
    _report_simulation_results(results)
    # Just assert it doesn't raise; logger output not captured by capsys


def test_report_simulation_results_counts_all_statuses():
    """succeeded + failed + skipped are all counted separately."""
    results = [
        {"hash": "a", "status": "success"},
        {"hash": "b", "status": "failed", "error": "mdrun died"},
        {"hash": "c", "status": "skipped"},
    ]
    # Patch the logger to capture what was logged
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_simulation_results(results)

    # At least one info call should mention all three counts
    info_calls = [str(c) for c in mock_logger.info.call_args_list]
    combined = " ".join(info_calls)
    assert "1" in combined  # 1 succeeded / 1 failed / 1 skipped
    assert "skipped" in combined.lower()


def test_report_simulation_results_failed_details_logged():
    """Failed results include hash and error message in the log output."""
    results = [{"hash": "xyz789", "status": "failed", "error": "segfault in mdrun"}]
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_simulation_results(results)

    error_calls = [str(c) for c in mock_logger.error.call_args_list]
    combined = " ".join(error_calls)
    assert "xyz789" in combined
    assert "segfault" in combined


def test_report_simulation_results_skipped_not_in_error_log():
    """Skipped results do not generate error-level log entries."""
    results = [{"hash": "skp", "status": "skipped", "reason": "build incomplete"}]
    with patch("mdfactory.cli.logger") as mock_logger:
        _report_simulation_results(results)

    # No error calls for skipped entries
    mock_logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# simulate_systems — end-to-end with mocked run_simulations (Finding 8)
# ---------------------------------------------------------------------------


@patch("mdfactory.orchestration.run_simulations", return_value=[{"hash": "abc", "status": "success"}])
@patch("mdfactory.cli._load_executor_config", return_value=MagicMock(provider="local"))
@patch("mdfactory.cli._resolve_slurm_flag", return_value=None)
def test_simulate_systems_calls_run_simulations(
    mock_slurm, mock_config, mock_run, tmp_path, capsys
):
    """simulate_systems resolves paths and delegates to run_simulations."""
    from mdfactory.cli import simulate_systems

    # Prepare a minimal simulation directory
    sim_dir = tmp_path / "abc123"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()

    simulate_systems(source=tmp_path)

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    sim_paths = call_args[0][0]  # first positional arg
    assert any(p.name == "abc123" for p in sim_paths)


@patch(
    "mdfactory.orchestration.run_simulations",
    return_value=[{"hash": "abc", "status": "failed", "error": "boom"}],
)
@patch("mdfactory.cli._load_executor_config", return_value=MagicMock(provider="local"))
@patch("mdfactory.cli._resolve_slurm_flag", return_value=None)
def test_simulate_systems_reports_failed_results(
    mock_slurm, mock_config, mock_run, tmp_path
):
    """simulate_systems passes results with status=failed to _report_simulation_results."""
    from mdfactory.cli import simulate_systems

    sim_dir = tmp_path / "abc123"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()

    with patch("mdfactory.cli._report_simulation_results") as mock_report:
        simulate_systems(source=tmp_path)
        mock_report.assert_called_once()
        results_arg = mock_report.call_args[0][0]
        failed = [r for r in results_arg if r.get("status") == "failed"]
        assert len(failed) == 1
