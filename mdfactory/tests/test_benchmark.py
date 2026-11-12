# ABOUTME: Tests for the CPU scalability benchmark sweep module
# ABOUTME: Covers sidecar parser, config validation, MDP generation, sweep logic, persistence
"""Tests for mdfactory.performance.benchmark."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.apps import _PERF_EXTRACT, MDRUN_PERF_MARKER
from mdfactory.performance.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    TrialResult,
    _generate_benchmark_mdp,
    _sweep_lock,
    read_performance_file,
    run_benchmark_sweep,
)

#: Typical GROMACS md.log performance block (end of file)
NORMAL_LOG = """\
                 (ns/day)    (hour/ns)
   Performance:        5.234        4.586
"""

#: Log with two performance blocks (restart with -append)
RESTART_LOG = """\
   Performance:        3.100        7.742

   <restart marker>

   Performance:        5.234        4.586
"""

#: Truncated log without performance block
TRUNCATED_LOG = """\
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


def _run_extract(tmp_path, log_text: str) -> str:
    """Run ``{_PERF_EXTRACT} <log> | tail -1`` exactly as the mdrun script does."""
    log = tmp_path / "bench.log"
    log.write_text(log_text)
    proc = subprocess.run(
        ["bash", "-c", f'{_PERF_EXTRACT} "{log}" | tail -1'],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


class TestReadPerformanceFile:
    """The sidecar parser is the sole source of live ns/day values."""

    def test_reads_marker_from_file(self, tmp_path):
        perf = tmp_path / "bench.deadbeef.perf"
        perf.write_text(f"{MDRUN_PERF_MARKER}5.234\n")
        assert read_performance_file(perf) == pytest.approx(5.234)

    def test_missing_file_returns_none(self, tmp_path):
        """A sidecar that was never written (failed trial) reads as None."""
        assert read_performance_file(tmp_path / "nope.perf") is None

    def test_empty_value_returns_none(self, tmp_path):
        """A sidecar written from a truncated log holds no value."""
        perf = tmp_path / "bench.perf"
        perf.write_text(f"{MDRUN_PERF_MARKER}\n")
        assert read_performance_file(perf) is None

    def test_last_marker_wins(self, tmp_path):
        perf = tmp_path / "bench.perf"
        perf.write_text(f"{MDRUN_PERF_MARKER}1.0\n{MDRUN_PERF_MARKER}9.75\n")
        assert read_performance_file(perf) == pytest.approx(9.75)


class TestPerfExtractProgram:
    """Execute the worker-side sed program against realistic logs."""

    def test_extracts_ns_per_day_from_log(self, tmp_path):
        assert _run_extract(tmp_path, NORMAL_LOG) == "5.234"

    def test_restart_log_yields_last_block(self, tmp_path):
        """``| tail -1`` collapses restart logs to the final block."""
        assert _run_extract(tmp_path, RESTART_LOG) == "5.234"

    def test_truncated_log_yields_empty(self, tmp_path):
        """A log with no Performance line yields an empty marker value."""
        assert _run_extract(tmp_path, TRUNCATED_LOG) == ""

    def test_extracted_value_round_trips_sidecar(self, tmp_path):
        """The sed output round-trips through the file the driver reads."""
        value = _run_extract(tmp_path, NORMAL_LOG)
        perf = tmp_path / "bench.perf"
        perf.write_text(f"{MDRUN_PERF_MARKER}{value}\n")
        assert read_performance_file(perf) == pytest.approx(5.234)


class TestSweepLock:
    """Tests for the concurrent-sweep guard."""

    def test_second_sweep_rejected(self, tmp_path):
        """A second concurrent sweep raises instead of sharing trial dirs."""
        with _sweep_lock(tmp_path):
            with pytest.raises(RuntimeError, match="already running"):
                with _sweep_lock(tmp_path):
                    pass

    def test_lock_released_after_exit(self, tmp_path):
        with _sweep_lock(tmp_path):
            pass
        with _sweep_lock(tmp_path):
            pass  # must not raise

    def test_lock_held_before_mdp_generation(self, tmp_path):
        """A concurrent sweep fails before touching the shared benchmark.mdp."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir = _setup_sim_dir(tmp_path)

        with _sweep_lock(sim_dir):
            with pytest.raises(RuntimeError, match="already running"):
                run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))
            assert not (sim_dir / "benchmark.mdp").exists()


class TestBenchmarkConfig:
    """Sweep parameter validation."""

    def test_defaults(self):
        config = BenchmarkConfig()
        assert config.cpu_counts == [1, 2, 4, 8, 16, 32, 64]
        assert config.efficiency_threshold == pytest.approx(0.75)
        assert config.duration_ps == pytest.approx(100.0)

    def test_empty_cpu_counts_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            BenchmarkConfig(cpu_counts=[])

    def test_zero_cpu_count_rejected(self):
        """A 0 would fall back to node-wide auto-detection under a false label."""
        with pytest.raises(ValueError, match=">= 1"):
            BenchmarkConfig(cpu_counts=[0, 4])

    def test_non_positive_duration_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            BenchmarkConfig(duration_ps=0)

    @pytest.mark.parametrize("threshold", [0, -0.5, 1.5])
    def test_threshold_must_be_a_fraction(self, threshold):
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            BenchmarkConfig(efficiency_threshold=threshold)


class TestBenchmarkResult:
    """Efficiency accounting and the justified-core-count answer."""

    def _result(self, throughputs, threshold=0.75):
        trials = [
            TrialResult(cpu_count=cpus, ns_per_day=ns, wall_seconds=1.0) for cpus, ns in throughputs
        ]
        return BenchmarkResult(
            system_path="/tmp/sim", trials=trials, efficiency_threshold=threshold
        )

    def test_baseline_is_smallest_successful_trial(self):
        result = self._result([(8, 40.0), (2, 10.0), (4, None)])
        assert result.baseline.cpu_count == 2

    def test_linear_scaling_is_full_efficiency(self):
        result = self._result([(1, 5.0), (4, 20.0)])
        assert result.efficiency(result.trials[1]) == pytest.approx(1.0)

    def test_efficiency_is_throughput_per_core_ratio(self):
        """4 cores at 15 ns/day against a 5 ns/day baseline = 75%."""
        result = self._result([(1, 5.0), (4, 15.0)])
        assert result.efficiency(result.trials[1]) == pytest.approx(0.75)

    def test_max_justified_stops_at_the_threshold(self):
        """Efficiency decays as 1/cpu_count -- only 2 cores stay justified."""
        result = self._result([(1, 4.0), (2, 7.0), (4, 11.0), (8, 16.0), (16, 16.0)])
        assert [result.efficiency(t) for t in result.trials] == pytest.approx(
            [1.0, 0.875, 0.6875, 0.5, 0.25]
        )
        assert result.max_justified.cpu_count == 2

    def test_max_justified_skips_failed_trials(self):
        result = self._result([(1, 4.0), (2, None), (4, 16.0)])
        assert result.max_justified.cpu_count == 4

    def test_max_justified_none_without_successful_trial(self):
        result = self._result([(1, None), (2, None)])
        assert result.baseline is None
        assert result.max_justified is None

    def test_zero_throughput_baseline_yields_no_efficiency(self):
        """A degenerate 0 ns/day baseline cannot be divided by."""
        result = self._result([(1, 0.0), (2, 5.0)])
        assert result.efficiency(result.trials[1]) is None

    def test_json_round_trip(self, tmp_path):
        result = self._result([(1, 5.0), (4, 15.0)])
        path = tmp_path / "benchmark_result.json"
        result.save(path)

        loaded = BenchmarkResult.load(path)
        assert loaded == result
        assert loaded.max_justified.cpu_count == 4
        assert set(json.loads(path.read_text())) == {
            "system_path",
            "trials",
            "efficiency_threshold",
        }


class TestGenerateBenchmarkMdp:
    """Short-trial MDP derivation."""

    def _values(self, tmp_path, **kwargs):
        from mdfactory.orchestration.mdp import parse_mdp

        source = tmp_path / "md.mdp"
        source.write_text(SAMPLE_MDP)
        output = tmp_path / "benchmark.mdp"
        _generate_benchmark_mdp(source, output, **kwargs)
        return {key: value for _, key, value in parse_mdp(output)}

    def test_nsteps_override(self, tmp_path):
        """100 ps at dt = 0.002 is 50000 steps, not the production 500000."""
        assert self._values(tmp_path, duration_ps=100.0)["nsteps"] == "50000"

    def test_output_frequency_reduced(self, tmp_path):
        values = self._values(tmp_path, duration_ps=100.0)
        assert values["nstxout"] == "50000"
        assert values["nstenergy"] == "50000"

    def test_disabled_outputs_preserved(self, tmp_path):
        """Keys already 0 stay 0 instead of being switched back on."""
        assert self._values(tmp_path, duration_ps=100.0)["nstfout"] == "0"

    def test_override_replaces_present_key(self, tmp_path):
        values = self._values(tmp_path, duration_ps=100.0, extra_overrides={"nstxout": "0"})
        assert values["nstxout"] == "0"

    def test_override_appended_when_key_absent(self, tmp_path):
        """An override naming an absent key is appended, not silently dropped."""
        values = self._values(
            tmp_path, duration_ps=100.0, extra_overrides={"verlet-buffer-tolerance": "0.005"}
        )
        assert values["verlet_buffer_tolerance"] == "0.005"

    def test_explicit_dt(self, tmp_path):
        assert self._values(tmp_path, duration_ps=10.0, dt=0.002)["nsteps"] == "5000"


class TestRunBenchmarkSweepDryRun:
    """Dry-run mode runs without cluster or filesystem side effects."""

    def test_dry_run_lists_the_plan_only(self, tmp_path):
        from mdfactory.orchestration.config import ExecutorConfig

        result = run_benchmark_sweep(
            tmp_path, ExecutorConfig(), BenchmarkConfig(cpu_counts=[4, 2]), dry_run=True
        )

        assert [t.cpu_count for t in result.trials] == [2, 4]
        assert all(t.ns_per_day is None for t in result.trials)
        assert result.max_justified is None
        assert not (tmp_path / ".benchmark").exists()
        assert not (tmp_path / "benchmark_result.json").exists()


def _setup_sim_dir(tmp_path):
    """Create a minimal prepared simulation directory."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").write_text("FAKE")
    (sim_dir / "topology.top").write_text("FAKE")
    # Companion topology include — must be staged into each trial directory
    (sim_dir / "extra_params.itp").write_text("FAKE")
    (sim_dir / "md.mdp").write_text(SAMPLE_MDP)
    return sim_dir


def _mock_sweep(tmp_path, ns_per_day="5.234"):
    """Wire grompp/mdrun mocks whose mdrun writes the performance sidecar."""
    sim_dir = _setup_sim_dir(tmp_path)

    def fake_mdrun(**kwargs):
        Path(kwargs["work_dir"], kwargs["performance_file"]).write_text(
            f"{MDRUN_PERF_MARKER}{ns_per_day}\n"
        )
        future = MagicMock()
        future.result.return_value = 0
        return future

    grompp = MagicMock()
    grompp.return_value.result.return_value = "ok"
    mdrun = MagicMock(side_effect=fake_mdrun)
    return sim_dir, grompp, mdrun


@patch("mdfactory.orchestration.trajectory.find_structure_file")
@patch("mdfactory.orchestration.session.parsl_session")
@patch("mdfactory.orchestration.apps.get_grompp_app")
@patch("mdfactory.orchestration.apps.get_mdrun_app")
class TestRunBenchmarkSweepExecution:
    """The live execution path, with Parsl and the apps mocked."""

    def _run(self, tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, **kwargs):
        sim_dir, grompp, mdrun = _mock_sweep(tmp_path, kwargs.pop("ns_per_day", "5.234"))
        mock_find.return_value = sim_dir / "system.pdb"
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_grompp_app.return_value = grompp
        mock_mdrun_app.return_value = mdrun
        return sim_dir, grompp, mdrun

    def test_runs_every_point_in_one_cpu_only_allocation(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir, grompp, mdrun = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )

        result = run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2, 4]))

        assert mock_session.call_count == 1
        assert grompp.call_count == 2
        assert mdrun.call_count == 2
        assert all(t.ns_per_day == pytest.approx(5.234) for t in result.trials)
        assert all(t.error is None for t in result.trials)
        assert (sim_dir / "benchmark_result.json").exists()

        allocation = mock_session.call_args.args[0]
        assert allocation.cpus_per_node == 4  # sized to the largest point
        assert allocation.max_workers_per_node == 1
        assert allocation.max_blocks == 1

    def test_mdrun_threads_match_labels(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """Thread count is the code under test's, never the base config's."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir, _, mdrun = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )

        # Base config claims 4 workers per node; the sweep must ignore that.
        run_benchmark_sweep(
            sim_dir, ExecutorConfig(max_workers_per_node=4), BenchmarkConfig(cpu_counts=[1, 2, 4])
        )

        assert [c.kwargs["ntasks"] for c in mdrun.call_args_list] == [1, 2, 4]
        # CPU-only: a GPU-enabled config must not leak into the trials.
        assert all(c.kwargs["disable_gpu"] is True for c in mdrun.call_args_list)

    def test_sweep_allocation_forces_exclusive_single_block(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """Exclusive so neighbours cannot skew it; base config left untouched."""
        from mdfactory.orchestration.config import SlurmExecutorConfig

        sim_dir, _, _ = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )
        cfg = SlurmExecutorConfig(account="acct", partition="cpu", exclusive=False, max_blocks=4)

        run_benchmark_sweep(sim_dir, cfg, BenchmarkConfig(cpu_counts=[2, 4]))

        allocation = mock_session.call_args.args[0]
        assert allocation.exclusive is True
        assert allocation.max_blocks == 1
        assert cfg.exclusive is False
        assert cfg.max_blocks == 4

    def test_trial_dir_is_staged_with_the_short_mdp(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """Each trial gets a linked copy of the inputs and the short MDP."""
        from mdfactory.orchestration.config import ExecutorConfig
        from mdfactory.orchestration.mdp import get_mdp_value, parse_mdp

        sim_dir, grompp, _ = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )

        run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))

        trial_dir = sim_dir / ".benchmark" / "cpus2"
        # GROMACS resolves #include relative to the topology file.
        assert (trial_dir / "extra_params.itp").is_symlink()
        md_mdp = trial_dir / "md.mdp"
        assert md_mdp.resolve() == (sim_dir / "benchmark.mdp").resolve()
        # 100 ps, not the 1 ns production MDP.
        assert get_mdp_value(parse_mdp(md_mdp), "nsteps") == "50000"
        assert grompp.call_args.kwargs["mdp_file"] == "md.mdp"
        assert grompp.call_args.kwargs["inputs"] == []

    def test_performance_sidecar_is_unique_per_trial(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """A never-reused sidecar name cannot be served from a stale cache entry."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir, grompp, mdrun = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )

        run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[1, 2, 4]))

        names = [c.kwargs["performance_file"] for c in mdrun.call_args_list]
        assert all(names)
        assert len(set(names)) == len(names)
        # mdrun must depend on grompp's TPR, or Parsl cannot order the two.
        assert mdrun.call_args.kwargs["inputs"] == [grompp.return_value]

    def test_gmx_binary_is_forwarded(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """A pinned gmx_mpi config reaches mdrun instead of falling back to auto."""
        from mdfactory.orchestration.config import SlurmExecutorConfig

        sim_dir, _, mdrun = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )
        cfg = SlurmExecutorConfig(account="acct", partition="cpu", gmx_binary="gmx_mpi")

        run_benchmark_sweep(sim_dir, cfg, BenchmarkConfig(cpu_counts=[2]))

        assert mdrun.call_args.kwargs["gmx_binary"] == "gmx_mpi"

    def test_short_mdp_reaches_the_written_file(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """--duration-ps and MDP overrides are honoured by the written MDP."""
        from mdfactory.orchestration.config import ExecutorConfig
        from mdfactory.orchestration.mdp import get_mdp_value, parse_mdp

        sim_dir, _, _ = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )

        run_benchmark_sweep(
            sim_dir,
            ExecutorConfig(),
            BenchmarkConfig(cpu_counts=[2], duration_ps=10.0, mdp_overrides={"nstxout": "0"}),
        )

        parsed = parse_mdp(sim_dir / ".benchmark" / "cpus2" / "md.mdp")
        assert get_mdp_value(parsed, "nsteps") == "5000"
        assert get_mdp_value(parsed, "nstxout") == "0"

    def test_efficiency_threshold_selects_the_answer(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """The persisted result answers the rule of thumb, not raw throughput."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir, _, _ = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )

        config = BenchmarkConfig(cpu_counts=[1, 2, 4], efficiency_threshold=0.9)

        result = run_benchmark_sweep(sim_dir, ExecutorConfig(), config)

        # Identical ns/day at every count: efficiency falls as 1/cpu_count.
        assert result.max_justified.cpu_count == 1
        assert BenchmarkResult.load(sim_dir / "benchmark_result.json").max_justified.cpu_count == 1

    def test_missing_performance_data_sets_error(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """mdrun succeeding without a marker is an error, not a zero result."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir, _, mdrun = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )
        mdrun.side_effect = None
        mdrun.return_value.result.return_value = 0

        result = run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[4]))

        assert result.trials[0].ns_per_day is None
        assert "no performance data" in result.trials[0].error
        assert result.max_justified is None

    def test_failed_trial_records_error_and_continues(
        self, mock_mdrun_app, mock_grompp_app, mock_session, mock_find, tmp_path
    ):
        """One failing trial must not abort the rest of the sweep."""
        from mdfactory.orchestration.config import ExecutorConfig

        sim_dir, grompp, _ = self._run(
            tmp_path, mock_mdrun_app, mock_grompp_app, mock_session, mock_find
        )
        first = MagicMock()
        first.result.side_effect = RuntimeError("GROMACS mdrun failed")
        grompp.return_value = first

        result = run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))

        assert result.trials[0].ns_per_day is None
        assert "mdrun failed" in result.trials[0].error


def test_missing_production_mdp_raises(tmp_path):
    """A sweep needs a prepared directory, so an absent MDP fails fast."""
    from mdfactory.orchestration.config import ExecutorConfig

    sim_dir = _setup_sim_dir(tmp_path)
    (sim_dir / "md.mdp").unlink()

    with pytest.raises(FileNotFoundError, match="Production MDP not found"):
        run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))


def test_missing_structure_file_raises(tmp_path):
    from mdfactory.orchestration.config import ExecutorConfig

    sim_dir = _setup_sim_dir(tmp_path)

    with (
        patch("mdfactory.orchestration.trajectory.find_structure_file", return_value=None),
        pytest.raises(FileNotFoundError, match="No structure file found"),
    ):
        run_benchmark_sweep(sim_dir, ExecutorConfig(), BenchmarkConfig(cpu_counts=[2]))


def test_multi_node_allocation_rejected(tmp_path):
    """Trials are one mdrun on one node, so a multi-node allocation is refused."""
    from mdfactory.orchestration.config import SlurmExecutorConfig

    cfg = SlurmExecutorConfig(account="acct", partition="cpu", nodes=2)
    with pytest.raises(ValueError, match="single-node"):
        run_benchmark_sweep(tmp_path, cfg, BenchmarkConfig(cpu_counts=[2]))
