# ABOUTME: Tests for the scalability benchmark sweep module
# ABOUTME: Covers md.log parser, config validation, MDP generation, sweep logic, JSON persistence
"""Tests for mdfactory.performance.benchmark."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mdfactory.performance.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    TrialResult,
    _build_sweep_configs,
    _generate_benchmark_mdp,
    _select_optimum,
    parse_mdlog_performance,
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
            src, out, duration_ps=100.0, extra_overrides={"nstxout_compressed": "0"}
        )
        content = out.read_text()
        # nstxout-compressed should be overridden to 0
        assert "0" in content

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
