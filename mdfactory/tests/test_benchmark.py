# ABOUTME: Tests for the scalability benchmark sweep module
# ABOUTME: Covers md.log parser, config validation, MDP generation, sweep logic, JSON persistence
"""Tests for mdfactory.performance.benchmark."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.apps import MDRUN_PERF_MARKER
from mdfactory.performance.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    TrialResult,
    _build_sweep_configs,
    _generate_benchmark_mdp,
    _select_optimum,
    _sweep_lock,
    parse_mdlog_performance,
    parse_performance_marker,
    read_performance_file,
    run_benchmark_sweep,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic GROMACS log content
# ---------------------------------------------------------------------------

#: Typical GROMACS md.log performance block (end of file)
NORMAL_LOG = """\
               Core t (s)   Wall t (s)        (%)
       Time:      120.000       15.000      800.0
                         0:15
                 (ns/day)    (hour/ns)
   Performance:        5.234        4.586
"""

#: Log with two performance blocks (restart with -append)
RESTART_LOG = """\
               Core t (s)   Wall t (s)        (%)
       Time:       60.000        7.500      800.0
                         0:07
                 (ns/day)    (hour/ns)
   Performance:        3.100        7.742

   <restart marker>

               Core t (s)   Wall t (s)        (%)
       Time:      120.000       15.000      800.0
                         0:15
                 (ns/day)    (hour/ns)
   Performance:        5.234        4.586
"""

#: Truncated log without performance block
TRUNCATED_LOG = """\
Step    Time         Lambda
   0    0.00000        0
  10    0.02000        0

Writing checkpoint, step 10 at Mon Sep  8 12:00:00 2025
"""

#: Minimal production MDP content for testing
SAMPLE_MDP = """\
; Production MDP
integrator  = md
dt          = 0.002   ; ps
nsteps      = 500000  ; 1 ns
nstxout     = 5000
nstvout     = 5000
nstfout     = 0
nstenergy   = 1000
nstlog      = 1000
nstxout-compressed = 5000
"""


# ---------------------------------------------------------------------------
# T1: md.log performance parser
# ---------------------------------------------------------------------------


class TestParseMdlogPerformance:
    """Tests for parse_mdlog_performance."""

    def test_normal_log(self, tmp_path):
        """Normal log file returns ns/day."""
        log = tmp_path / "prod.log"
        log.write_text(NORMAL_LOG)
        result = parse_mdlog_performance(log)
        assert result == pytest.approx(5.234)

    def test_restart_log_uses_last_block(self, tmp_path):
        """Restart log with multiple performance blocks uses the last one."""
        log = tmp_path / "prod.log"
        log.write_text(RESTART_LOG)
        result = parse_mdlog_performance(log)
        assert result == pytest.approx(5.234)

    def test_truncated_log_returns_none(self, tmp_path):
        """Truncated log without performance block returns None."""
        log = tmp_path / "prod.log"
        log.write_text(TRUNCATED_LOG)
        assert parse_mdlog_performance(log) is None

    def test_missing_file_returns_none(self, tmp_path):
        """Non-existent file returns None."""
        assert parse_mdlog_performance(tmp_path / "nonexistent.log") is None

    def test_empty_file_returns_none(self, tmp_path):
        """Empty file returns None."""
        log = tmp_path / "empty.log"
        log.write_text("")
        assert parse_mdlog_performance(log) is None


class TestParsePerformanceMarker:
    """Tests for parse_performance_marker."""

    def test_extracts_value(self):
        """Marker line yields the ns/day value."""
        stdout = f"some mdrun output\n{MDRUN_PERF_MARKER}5.234\n"
        assert parse_performance_marker(stdout) == pytest.approx(5.234)

    def test_uses_last_marker(self):
        """When several markers are present, the last one wins."""
        stdout = f"{MDRUN_PERF_MARKER}1.0\n{MDRUN_PERF_MARKER}9.75\n"
        assert parse_performance_marker(stdout) == pytest.approx(9.75)

    def test_missing_marker_returns_none(self):
        """stdout without the marker returns None."""
        assert parse_performance_marker("no marker here") is None

    def test_empty_marker_returns_none(self):
        """Marker with an empty value returns None (regex requires a digit)."""
        assert parse_performance_marker(f"{MDRUN_PERF_MARKER}\n") is None

    def test_none_stdout_returns_none(self):
        """None or empty stdout returns None."""
        assert parse_performance_marker(None) is None
        assert parse_performance_marker("") is None


class TestReadPerformanceFile:
    """Tests for read_performance_file."""

    def test_reads_marker_from_file(self, tmp_path):
        """Sidecar with a marker line yields the value."""
        perf = tmp_path / "bench.deadbeef.perf"
        perf.write_text(f"{MDRUN_PERF_MARKER}5.234\n")
        assert read_performance_file(perf) == pytest.approx(5.234)

    def test_missing_file_returns_none(self, tmp_path):
        """A sidecar that was never written returns None."""
        assert read_performance_file(tmp_path / "nope.perf") is None

    def test_empty_value_returns_none(self, tmp_path):
        """A sidecar with an empty marker value returns None."""
        perf = tmp_path / "bench.perf"
        perf.write_text(f"{MDRUN_PERF_MARKER}\n")
        assert read_performance_file(perf) is None


class TestSweepLock:
    """Tests for the concurrent-sweep guard."""

    def test_second_sweep_rejected(self, tmp_path):
        """A second concurrent sweep raises instead of sharing trial dirs."""
        with _sweep_lock(tmp_path):
            with pytest.raises(RuntimeError, match="already running"):
                with _sweep_lock(tmp_path):
                    pass

    def test_lock_released_after_exit(self, tmp_path):
        """The lock is released when the holding sweep exits."""
        with _sweep_lock(tmp_path):
            pass
        with _sweep_lock(tmp_path):
            pass  # must not raise

    def test_lock_held_before_mdp_generation(self, tmp_path):
        """A concurrent sweep fails before touching the shared benchmark.mdp
        (finding 2 regression: MDP generation used to run before the lock,
        letting a losing sweep mutate a running sweep's trials)."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)

        with _sweep_lock(sim_dir):
            with pytest.raises(RuntimeError, match="already running"):
                run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))
            # The losing sweep must not have regenerated the shared MDP
            assert not (sim_dir / "benchmark.mdp").exists()


# ---------------------------------------------------------------------------
# T2: BenchmarkConfig validation
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig validation."""

    def test_defaults(self):
        """Default config has sensible values."""
        cfg = BenchmarkConfig()
        assert cfg.cpu_counts == [1, 2, 4, 8]
        assert cfg.gpu_replicas == []
        assert cfg.duration_ps == 100.0
        assert cfg.selection == "ns_per_day"

    def test_empty_cpu_counts_rejected(self):
        """Empty cpu_counts raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            BenchmarkConfig(cpu_counts=[])

    def test_negative_duration_rejected(self):
        """Non-positive duration_ps raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            BenchmarkConfig(duration_ps=0)

    def test_custom_values(self):
        """Custom config values are accepted."""
        cfg = BenchmarkConfig(
            cpu_counts=[16, 32],
            gpu_replicas=[1, 2],
            duration_ps=50.0,
            selection="efficiency",
            mdp_overrides={"nstxout-compressed": "0"},
        )
        assert cfg.cpu_counts == [16, 32]
        assert cfg.gpu_replicas == [1, 2]
        assert cfg.selection == "efficiency"


# ---------------------------------------------------------------------------
# T3: TrialResult and optimum selection
# ---------------------------------------------------------------------------


class TestTrialResult:
    """Tests for TrialResult and _select_optimum."""

    def test_efficiency_property(self):
        """Efficiency is ns_per_day / cpu_count."""
        t = TrialResult(cpu_count=8, gpu_replicas=0, ns_per_day=16.0, wall_seconds=10.0)
        assert t.efficiency == pytest.approx(2.0)

    def test_efficiency_none_when_no_performance(self):
        """Efficiency is None when ns_per_day is None."""
        t = TrialResult(cpu_count=8, gpu_replicas=0, ns_per_day=None, wall_seconds=None)
        assert t.efficiency is None

    def test_efficiency_none_when_zero_cpus(self):
        """Efficiency is None when cpu_count is 0 (guard against division by zero)."""
        t = TrialResult(cpu_count=0, gpu_replicas=0, ns_per_day=5.0, wall_seconds=10.0)
        assert t.efficiency is None

    def test_select_optimum_ns_per_day(self):
        """Select by ns_per_day picks highest throughput."""
        trials = [
            TrialResult(cpu_count=4, gpu_replicas=0, ns_per_day=5.0, wall_seconds=10.0),
            TrialResult(cpu_count=8, gpu_replicas=0, ns_per_day=8.0, wall_seconds=10.0),
            TrialResult(cpu_count=16, gpu_replicas=0, ns_per_day=7.5, wall_seconds=10.0),
        ]
        opt = _select_optimum(trials, "ns_per_day")
        assert opt is not None
        assert opt.cpu_count == 8

    def test_select_optimum_efficiency(self):
        """Select by efficiency picks best ns/day per core."""
        trials = [
            TrialResult(cpu_count=4, gpu_replicas=0, ns_per_day=5.0, wall_seconds=10.0),
            TrialResult(cpu_count=8, gpu_replicas=0, ns_per_day=8.0, wall_seconds=10.0),
            TrialResult(cpu_count=16, gpu_replicas=0, ns_per_day=7.5, wall_seconds=10.0),
        ]
        opt = _select_optimum(trials, "efficiency")
        assert opt is not None
        # 5/4=1.25, 8/8=1.0, 7.5/16=0.47 → 4 cores wins
        assert opt.cpu_count == 4

    def test_select_optimum_all_failed(self):
        """All-failed trials return None."""
        trials = [
            TrialResult(cpu_count=4, gpu_replicas=0, ns_per_day=None, wall_seconds=5.0),
        ]
        assert _select_optimum(trials, "ns_per_day") is None


# ---------------------------------------------------------------------------
# T4: Benchmark MDP generation
# ---------------------------------------------------------------------------


class TestGenerateBenchmarkMdp:
    """Tests for _generate_benchmark_mdp."""

    def test_nsteps_override(self, tmp_path):
        """nsteps is set to achieve the target duration."""
        src = tmp_path / "md.mdp"
        src.write_text(SAMPLE_MDP)
        out = tmp_path / "benchmark.mdp"
        _generate_benchmark_mdp(src, out, duration_ps=100.0)
        content = out.read_text()
        # dt=0.002, duration=100 → nsteps=50000
        assert "50000" in content

    def test_io_frequency_reduced(self, tmp_path):
        """Output frequencies are set to nsteps to minimize I/O."""
        src = tmp_path / "md.mdp"
        src.write_text(SAMPLE_MDP)
        out = tmp_path / "benchmark.mdp"
        _generate_benchmark_mdp(src, out, duration_ps=100.0)
        content = out.read_text()
        # nstxout, nstvout, nstenergy, nstlog should be 50000
        assert content.count("50000") >= 4

    def test_extra_overrides_applied(self, tmp_path):
        """Extra MDP overrides are applied."""
        src = tmp_path / "md.mdp"
        src.write_text(SAMPLE_MDP)
        out = tmp_path / "benchmark.mdp"
        _generate_benchmark_mdp(
            src, out, duration_ps=100.0, extra_overrides={"nstxout_compressed": "99999"}
        )
        content = out.read_text()
        assert "99999" in content

    def test_disabled_outputs_preserved(self, tmp_path):
        """Outputs set to 0 (disabled) are not overridden."""
        src = tmp_path / "md.mdp"
        src.write_text(SAMPLE_MDP)
        out = tmp_path / "benchmark.mdp"
        _generate_benchmark_mdp(src, out, duration_ps=100.0)

        from mdfactory.orchestration.mdp import get_mdp_value, parse_mdp

        parsed = parse_mdp(out)
        # nstfout was 0 in SAMPLE_MDP — should remain 0
        assert get_mdp_value(parsed, "nstfout") == "0"
        # nstxout was 5000 — should be overridden to nsteps (50000)
        assert get_mdp_value(parsed, "nstxout") == "50000"

    def test_explicit_dt_override(self, tmp_path):
        """Explicit dt parameter overrides the source MDP value."""
        src = tmp_path / "md.mdp"
        src.write_text(SAMPLE_MDP)
        out = tmp_path / "benchmark.mdp"
        _generate_benchmark_mdp(src, out, duration_ps=100.0, dt=0.001)
        content = out.read_text()
        # dt=0.001, duration=100 → nsteps=100000
        assert "100000" in content


# ---------------------------------------------------------------------------
# T5: Sweep config generation
# ---------------------------------------------------------------------------


class TestBuildSweepConfigs:
    """Tests for _build_sweep_configs."""

    def test_cpu_only_sweep(self):
        """CPU-only sweep generates one point per cpu_count."""
        base = MagicMock()
        cfg = BenchmarkConfig(cpu_counts=[2, 4, 8])
        points = _build_sweep_configs(base, cfg)
        assert len(points) == 3
        assert [p["cpu_count"] for p in points] == [2, 4, 8]
        assert all(p["gpu_replicas"] == 0 for p in points)

    def test_cpu_only_points_normalize_max_workers(self):
        """CPU-only points force max_workers_per_node=1 so ntasks equals the
        labeled cpu_count regardless of the base config (finding 1)."""
        base = MagicMock()
        cfg = BenchmarkConfig(cpu_counts=[1, 2, 4, 8])
        points = _build_sweep_configs(base, cfg)
        assert all(p["config_overrides"]["max_workers_per_node"] == 1 for p in points)
        assert [p["config_overrides"]["cpus_per_node"] for p in points] == [1, 2, 4, 8]

    def test_cpu_gpu_sweep(self):
        """CPU × GPU sweep generates the cross-product."""
        base = MagicMock()
        cfg = BenchmarkConfig(cpu_counts=[4, 8], gpu_replicas=[1, 2])
        points = _build_sweep_configs(base, cfg)
        assert len(points) == 4  # 2 cpus × 2 gpu
        combos = [(p["cpu_count"], p["gpu_replicas"]) for p in points]
        assert (4, 1) in combos
        assert (4, 2) in combos
        assert (8, 1) in combos
        assert (8, 2) in combos


# ---------------------------------------------------------------------------
# T6: Dry-run mode
# ---------------------------------------------------------------------------


class TestRunBenchmarkSweepDryRun:
    """Tests for run_benchmark_sweep dry-run mode."""

    def test_dry_run_no_parsl(self, tmp_path):
        """Dry-run does not import or load Parsl."""
        from mdfactory.orchestration.config import ExecutorConfig

        cfg = ExecutorConfig()
        bench_cfg = BenchmarkConfig(cpu_counts=[2, 4])

        result = run_benchmark_sweep(tmp_path, cfg, bench_cfg, dry_run=True)

        assert len(result.trials) == 2
        assert all(t.ns_per_day is None for t in result.trials)
        assert result.optimum is None
        assert result.selection_criterion == "ns_per_day"

    def test_dry_run_with_gpu(self, tmp_path):
        """Dry-run with GPU replicas generates cross-product."""
        from mdfactory.orchestration.config import ExecutorConfig

        cfg = ExecutorConfig()
        bench_cfg = BenchmarkConfig(cpu_counts=[4], gpu_replicas=[1, 2])

        result = run_benchmark_sweep(tmp_path, cfg, bench_cfg, dry_run=True)

        assert len(result.trials) == 2
        assert result.trials[0].gpu_replicas == 1
        assert result.trials[1].gpu_replicas == 2


# ---------------------------------------------------------------------------
# T7: JSON persistence round-trip
# ---------------------------------------------------------------------------


class TestBenchmarkResultPersistence:
    """Tests for BenchmarkResult save/load."""

    def test_json_round_trip(self, tmp_path):
        """Save and load produces equivalent result."""
        trials = [
            TrialResult(cpu_count=4, gpu_replicas=0, ns_per_day=5.0, wall_seconds=10.0),
            TrialResult(cpu_count=8, gpu_replicas=0, ns_per_day=8.0, wall_seconds=12.0),
        ]
        optimum = trials[1]
        result = BenchmarkResult(
            system_path="/tmp/sim",
            trials=trials,
            optimum=optimum,
            selection_criterion="ns_per_day",
        )

        path = tmp_path / "benchmark_result.json"
        result.save(path)

        loaded = BenchmarkResult.load(path)
        assert loaded.system_path == result.system_path
        assert len(loaded.trials) == 2
        assert loaded.optimum is not None
        assert loaded.optimum.ns_per_day == pytest.approx(8.0)
        assert loaded.selection_criterion == "ns_per_day"

    def test_json_contains_all_fields(self, tmp_path):
        """Saved JSON contains expected top-level keys."""
        import json

        result = BenchmarkResult(
            system_path="/tmp/sim",
            trials=[
                TrialResult(cpu_count=4, gpu_replicas=0, ns_per_day=5.0, wall_seconds=10.0),
            ],
        )
        path = tmp_path / "result.json"
        result.save(path)

        data = json.loads(path.read_text())
        assert "system_path" in data
        assert "trials" in data
        assert "optimum" in data
        assert "selection_criterion" in data


# ---------------------------------------------------------------------------
# T8: Mocked execution path (dry_run=False)
# ---------------------------------------------------------------------------


def _setup_sim_dir(tmp_path):
    """Create a minimal simulation directory for benchmark tests."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").write_text("FAKE")
    (sim_dir / "topology.top").write_text("FAKE")
    # Companion topology include — must be staged into each trial directory
    (sim_dir / "extra_params.itp").write_text("FAKE")
    # Production MDP
    (sim_dir / "md.mdp").write_text(SAMPLE_MDP)
    return sim_dir


def _mdrun_writes_perf(perf_text: str):
    """Build a mock mdrun side_effect that writes a performance sidecar.

    The real mdrun script writes ns/day to ``performance_file`` inside
    ``work_dir``; this mimics that so ``read_performance_file`` finds it.
    """

    def _fake(**kwargs):
        Path(kwargs["work_dir"], kwargs["performance_file"]).write_text(perf_text)
        fut = MagicMock()
        fut.result.return_value = 0
        return fut

    return _fake


class TestRunBenchmarkSweepExecution:
    """Tests for the live execution path (dry_run=False)."""

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_mocked_sweep_runs_all_points(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """Mocked sweep runs every point inside a single allocation."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_grompp = MagicMock()
        mock_grompp.return_value.result.return_value = "ok"
        mock_grompp_app.return_value = mock_grompp

        mock_mdrun = MagicMock()
        # mdrun writes ns/day to the (uniquely named) performance sidecar
        mock_mdrun.side_effect = _mdrun_writes_perf(f"{MDRUN_PERF_MARKER}5.234\n")
        mock_mdrun_app.return_value = mock_mdrun

        cfg = ExecutorConfig()
        bench_cfg = BenchmarkConfig(cpu_counts=[2, 4])

        result = run_benchmark_sweep(sim_dir, cfg, bench_cfg)

        # One allocation for the whole sweep, two trials inside it
        assert mock_session.call_count == 1
        assert mock_grompp.call_count == 2
        assert mock_mdrun.call_count == 2
        assert len(result.trials) == 2
        assert all(t.ns_per_day == pytest.approx(5.234) for t in result.trials)
        assert all(t.error is None for t in result.trials)
        assert result.optimum is not None
        # Result saved as JSON sidecar
        assert (sim_dir / "benchmark_result.json").exists()

        # The allocation is sized to the largest requested CPU count...
        allocation_config = mock_session.call_args.args[0]
        assert allocation_config.cpus_per_node == 4
        # ...with a single worker (CPU-only sweep runs one mdrun at a time).
        assert allocation_config.max_workers_per_node == 1
        # ...while mdrun thread counts follow the individual sweep points.
        assert [call.kwargs["ntasks"] for call in mock_mdrun.call_args_list] == [2, 4]

        # Regression: ns/day must be extracted on the worker into a uniquely
        # named sidecar, not by reading the log from the driver.
        perf_names = [call.kwargs["performance_file"] for call in mock_mdrun.call_args_list]
        assert all(name for name in perf_names)
        assert len(set(perf_names)) == len(perf_names)  # unique per trial

        # Regression: apps must receive an explicit input dependency list.
        # Omitting it forwards ``inputs=None``, which Parsl cannot iterate.
        assert mock_grompp.call_args.kwargs["inputs"] == []
        assert mock_mdrun.call_args.kwargs["inputs"] == [mock_grompp.return_value]

        # Regression: topology include files must be staged into the trial dir
        # (GROMACS resolves ``#include`` relative to the topology file).
        trial_dir = sim_dir / ".benchmark" / "cpus2_gpu0"
        assert (trial_dir / "extra_params.itp").is_symlink()

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_cpu_threads_match_labels_with_base_max_workers(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """The labeled cpu_count equals the actual mdrun thread count even
        when the base config declares max_workers_per_node > 1 (finding 1)."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_grompp = MagicMock()
        mock_grompp.return_value.result.return_value = "ok"
        mock_grompp_app.return_value = mock_grompp

        mock_mdrun = MagicMock()
        mock_mdrun.side_effect = _mdrun_writes_perf(f"{MDRUN_PERF_MARKER}5.234\n")
        mock_mdrun_app.return_value = mock_mdrun

        # Base config claims 4 workers per node — previously this leaked into
        # the per-point hints and produced ntasks 0, 0, 1, 2.
        cfg = ExecutorConfig(max_workers_per_node=4)
        bench_cfg = BenchmarkConfig(cpu_counts=[1, 2, 4, 8])

        run_benchmark_sweep(sim_dir, cfg, bench_cfg)

        assert [c.kwargs["ntasks"] for c in mock_mdrun.call_args_list] == [1, 2, 4, 8]
        # The allocation itself stays single-worker; thread counts come from
        # the per-point overrides.
        assert mock_session.call_args.args[0].max_workers_per_node == 1

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_sweep_allocation_is_exclusive(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """The sweep allocation requests exclusive use of the node so neighbour
        jobs cannot skew the measurement (a shared node swung the 8-core point
        ~37% between identical runs)."""
        from mdfactory.orchestration.config import SlurmExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_grompp = MagicMock()
        mock_grompp.return_value.result.return_value = "ok"
        mock_grompp_app.return_value = mock_grompp

        mock_mdrun = MagicMock()
        mock_mdrun.side_effect = _mdrun_writes_perf(f"{MDRUN_PERF_MARKER}5.234\n")
        mock_mdrun_app.return_value = mock_mdrun

        # Base config opts OUT of exclusive — the sweep must still opt IN.
        cfg = SlurmExecutorConfig(account="acct", partition="cpu", exclusive=False)

        run_benchmark_sweep(sim_dir, cfg, BenchmarkConfig(cpu_counts=[2, 4]))

        allocation_config = mock_session.call_args.args[0]
        assert allocation_config.exclusive is True
        # ...and the base config object is not mutated.
        assert cfg.exclusive is False

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_trial_md_mdp_resolves_to_benchmark_mdp(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """Each trial's md.mdp is the short benchmark MDP, not the production
        MDP (finding 4 regression)."""
        from mdfactory.orchestration.config import ExecutorConfig
        from mdfactory.orchestration.mdp import get_mdp_value, parse_mdp

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_grompp = MagicMock()
        mock_grompp.return_value.result.return_value = "ok"
        mock_grompp_app.return_value = mock_grompp

        mock_mdrun = MagicMock()
        mock_mdrun.side_effect = _mdrun_writes_perf(f"{MDRUN_PERF_MARKER}5.234\n")
        mock_mdrun_app.return_value = mock_mdrun

        run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))

        trial_dir = sim_dir / ".benchmark" / "cpus2_gpu0"
        md_mdp = trial_dir / "md.mdp"
        # The trial's md.mdp must resolve to the shared benchmark MDP...
        assert md_mdp.resolve() == (sim_dir / "benchmark.mdp").resolve()
        # ...which is the short trial (100 ps / 0.002 ps = 50000 steps), not
        # the production MDP (500000 steps).  A regression here would run the
        # full-length production MDP for every trial.
        assert get_mdp_value(parse_mdp(md_mdp), "nsteps") == "50000"
        # grompp consumed the trial's md.mdp
        assert mock_grompp.call_args.kwargs["mdp_file"] == "md.mdp"

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_gpu_replicas_run_concurrently_with_aggregate_throughput(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """A gpu_replicas=2 point launches two concurrent mdruns sharing the
        GPU, each with half the threads, and records the aggregate ns/day
        (findings 3 and 10)."""
        from mdfactory.orchestration.config import SlurmExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_grompp = MagicMock()
        mock_grompp.return_value.result.return_value = "ok"
        mock_grompp_app.return_value = mock_grompp

        events = []

        def mdrun_side_effect(**kwargs):
            events.append(f"submit:{kwargs['deffnm']}")
            Path(kwargs["work_dir"], kwargs["performance_file"]).write_text(
                f"{MDRUN_PERF_MARKER}5.234\n"
            )
            fut = MagicMock()

            def _join(*args, **_kwargs):
                events.append(f"join:{kwargs['deffnm']}")
                return 0

            fut.result.side_effect = _join
            return fut

        mock_app = MagicMock(side_effect=mdrun_side_effect)
        mock_mdrun_app.return_value = mock_app

        cfg = SlurmExecutorConfig(account="acct", partition="gpu", gres="gpu:l40s:1")
        bench_cfg = BenchmarkConfig(cpu_counts=[4], gpu_replicas=[2])

        result = run_benchmark_sweep(sim_dir, cfg, bench_cfg)

        # One grompp (shared TPR), two concurrent mdruns
        assert mock_grompp.call_count == 1
        assert mock_app.call_count == 2
        # Both replicas are submitted before either is joined => concurrent
        assert events == ["submit:bench_r0", "submit:bench_r1", "join:bench_r0", "join:bench_r1"]

        mdrun_calls = mock_app.call_args_list
        # Distinct deffnms and unique perf sidecars per replica
        assert [c.kwargs["deffnm"] for c in mdrun_calls] == ["bench_r0", "bench_r1"]
        perf_names = [c.kwargs["performance_file"] for c in mdrun_calls]
        assert len(set(perf_names)) == 2
        # Thread division: 4 cpus across 2 replicas => 2 threads each
        assert [c.kwargs["ntasks"] for c in mdrun_calls] == [2, 2]
        # GPU stays enabled via the base config's gres
        assert all(c.kwargs["disable_gpu"] is False for c in mdrun_calls)

        # Allocation sized for the largest point with one worker per replica
        allocation_config = mock_session.call_args.args[0]
        assert allocation_config.cpus_per_node == 4
        assert allocation_config.max_workers_per_node == 2

        # Aggregate throughput across the two replicas (5.234 + 5.234)
        assert len(result.trials) == 1
        trial = result.trials[0]
        assert trial.cpu_count == 4
        assert trial.gpu_replicas == 2
        assert trial.ns_per_day == pytest.approx(10.468)
        assert trial.error is None

        # Replicas share the single grompp TPR through per-replica symlinks
        trial_dir = sim_dir / ".benchmark" / "cpus4_gpu2"
        assert (trial_dir / "bench_r0.tpr").is_symlink()
        assert (trial_dir / "bench_r1.tpr").is_symlink()
        assert (trial_dir / "bench_r0.tpr").readlink() == Path("bench.tpr")

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_missing_performance_data_sets_error(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """When mdrun succeeds but emits no marker, error is set."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_grompp = MagicMock()
        mock_grompp.return_value.result.return_value = "ok"
        mock_grompp_app.return_value = mock_grompp

        mock_mdrun = MagicMock()
        # No performance sidecar is written -> no performance data
        mock_mdrun.return_value.result.return_value = 0
        mock_mdrun_app.return_value = mock_mdrun

        cfg = ExecutorConfig()
        bench_cfg = BenchmarkConfig(cpu_counts=[4])

        result = run_benchmark_sweep(sim_dir, cfg, bench_cfg)

        assert len(result.trials) == 1
        assert result.trials[0].ns_per_day is None
        assert result.trials[0].error is not None
        assert "no performance data" in result.trials[0].error
        assert result.optimum is None

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    @patch("mdfactory.orchestration.session.parsl_session")
    @patch("mdfactory.orchestration.apps.get_grompp_app")
    @patch("mdfactory.orchestration.apps.get_mdrun_app")
    def test_failed_trial_continues_sweep(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """A failed trial records the error and continues to the next point."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # First trial fails, second succeeds
        mock_grompp = MagicMock()
        mock_grompp_app.return_value = mock_grompp

        call_count = {"n": 0}

        def grompp_side_effect(*args, **kwargs):
            call_count["n"] += 1
            future = MagicMock()
            if call_count["n"] == 1:
                future.result.side_effect = RuntimeError("GROMACS grompp failed")
            else:
                future.result.return_value = "ok"
            return future

        mock_grompp.side_effect = grompp_side_effect

        mock_mdrun = MagicMock()
        mock_mdrun.side_effect = _mdrun_writes_perf(f"{MDRUN_PERF_MARKER}8.0\n")
        mock_mdrun_app.return_value = mock_mdrun

        cfg = ExecutorConfig()
        bench_cfg = BenchmarkConfig(cpu_counts=[2, 4])

        result = run_benchmark_sweep(sim_dir, cfg, bench_cfg)

        assert len(result.trials) == 2
        # First trial failed
        assert result.trials[0].ns_per_day is None
        assert result.trials[0].error is not None
        assert "grompp failed" in result.trials[0].error
        # Second trial succeeded
        assert result.trials[1].ns_per_day == pytest.approx(8.0)
        # Optimum selected from successful trials only
        assert result.optimum is not None
        assert result.optimum.cpu_count == 4


# ---------------------------------------------------------------------------
# T9: Error paths
# ---------------------------------------------------------------------------


class TestRunBenchmarkSweepErrors:
    """Tests for error handling in run_benchmark_sweep."""

    def test_missing_production_mdp(self, tmp_path):
        """Missing production MDP raises FileNotFoundError."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        # No md.mdp file

        with pytest.raises(FileNotFoundError, match="Production MDP not found"):
            run_benchmark_sweep(sim_dir, ExecutorConfig())

    @patch("mdfactory.orchestration.trajectory.find_structure_file")
    def test_missing_structure_file(self, mock_find, tmp_path):
        """Missing structure file raises FileNotFoundError."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)
        mock_find.return_value = None

        with pytest.raises(FileNotFoundError, match="No structure file found"):
            run_benchmark_sweep(sim_dir, ExecutorConfig())

    def test_multi_node_allocation_rejected(self, tmp_path):
        """A multi-node allocation is rejected because trials are single-rank."""
        from mdfactory.orchestration.config import SlurmExecutorConfig

        cfg = SlurmExecutorConfig(account="acct", partition="cpu", nodes=2)
        with pytest.raises(ValueError, match="single-node"):
            run_benchmark_sweep(tmp_path, cfg, BenchmarkConfig(cpu_counts=[2]))
