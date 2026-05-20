# ABOUTME: Smoke tests for analysis submit helpers, verifying job submission
# ABOUTME: configuration and script generation for analysis pipelines.
"""Smoke tests for analysis submit helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mdfactory.analysis import submit as submit_mod


def test_resolve_simulation_paths_from_yaml(tmp_path):
    build_yaml = tmp_path / "systems.yaml"
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    build_yaml.write_text(f"system_directory:\n  - {sim_dir}\n")

    paths = submit_mod.resolve_simulation_paths_from_yaml(build_yaml)

    assert paths == [sim_dir.resolve()]


def test_normalize_slurm_time():
    assert submit_mod.normalize_slurm_time("2h") == "02:00:00"
    assert submit_mod.normalize_slurm_time("30m") == "00:30:00"
    assert submit_mod.normalize_slurm_time("90") == "01:30:00"
    assert submit_mod.normalize_slurm_time("1d") == "1-00:00:00"
    assert submit_mod.normalize_slurm_time("01:00:00") == "01:00:00"


def test_run_simulation_analyses_skips_existing(monkeypatch, tmp_path):
    class DummyBuildInput:
        simulation_type = "bilayer"
        hash = "HASH"

    class DummySimulation:
        def __init__(self, path, structure_file=None, trajectory_file=None):
            self.path = Path(path)
            self.build_input = DummyBuildInput()

        def list_analyses(self):
            return ["analysis_a"]

        def run_analysis(self, name, **kwargs):
            raise AssertionError("run_analysis should not be called when skipping")

    monkeypatch.setattr(
        submit_mod,
        "ANALYSIS_REGISTRY",
        {"bilayer": {"analysis_a": object()}},
    )
    monkeypatch.setattr(submit_mod, "Simulation", DummySimulation)

    result = submit_mod.run_simulation_analyses(
        tmp_path,
        ["analysis_a"],
        structure_file="system.pdb",
        trajectory_file="prod.xtc",
        skip_existing=True,
    )

    assert result["status"] == "skipped"
    assert result["analyses"] == []
    assert result["results"] == []


def test_run_simulation_analyses_success(monkeypatch, tmp_path):
    class DummyBuildInput:
        simulation_type = "bilayer"
        hash = "HASH"

    class DummySimulation:
        def __init__(self, path, structure_file=None, trajectory_file=None):
            self.path = Path(path)
            self.build_input = DummyBuildInput()

        def list_analyses(self):
            return []

        def run_analysis(self, name, **kwargs):
            return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(
        submit_mod,
        "ANALYSIS_REGISTRY",
        {"bilayer": {"analysis_a": object()}},
    )
    monkeypatch.setattr(submit_mod, "Simulation", DummySimulation)

    result = submit_mod.run_simulation_analyses(
        tmp_path,
        ["analysis_a"],
        structure_file="system.pdb",
        trajectory_file="prod.xtc",
        skip_existing=True,
    )

    assert result["status"] == "success"
    assert result["analyses"] == ["analysis_a"]
    assert result["results"] == [{"analysis": "analysis_a", "rows": 1}]


def test_run_simulation_analyses_forwards_analysis_kwargs(monkeypatch, tmp_path):
    class DummyBuildInput:
        simulation_type = "bilayer"
        hash = "HASH"

    captured_kwargs: dict[str, object] = {}

    class DummySimulation:
        def __init__(self, path, structure_file=None, trajectory_file=None):
            self.path = Path(path)
            self.build_input = DummyBuildInput()

        def list_analyses(self):
            return []

        def run_analysis(self, name, **kwargs):
            captured_kwargs.update(kwargs)
            return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(
        submit_mod,
        "ANALYSIS_REGISTRY",
        {"bilayer": {"analysis_a": object()}},
    )
    monkeypatch.setattr(submit_mod, "Simulation", DummySimulation)

    submit_mod.run_simulation_analyses(
        tmp_path,
        ["analysis_a"],
        structure_file="system.pdb",
        trajectory_file="prod.xtc",
        backend="serial",
        n_workers=1,
        analysis_kwargs={"start_ns": 50.0, "stride": 2},
    )

    assert captured_kwargs["backend"] == "serial"
    assert captured_kwargs["n_workers"] == 1
    assert captured_kwargs["start_ns"] == 50.0
    assert captured_kwargs["stride"] == 2


def test_run_analyses_local_smoke(monkeypatch, tmp_path):
    sim1 = tmp_path / "sim1"
    sim2 = tmp_path / "sim2"
    sim1.mkdir()
    sim2.mkdir()

    def fake_run(sim_path, analysis_names, **kwargs):
        return {
            "path": str(sim_path),
            "status": "success",
            "analyses": analysis_names or [],
            "results": [],
            "duration_seconds": 0.0,
            "hash": "HASH",
        }

    monkeypatch.setattr(submit_mod, "run_simulation_analyses", fake_run)

    df = submit_mod.run_analyses_local(
        [sim1, sim2],
        ["analysis_a"],
        structure_file="system.pdb",
        trajectory_file="prod.xtc",
    )

    assert len(df) == 2
    assert set(df["path"]) == {str(sim1), str(sim2)}


# --- Hash filtering tests ---


class TestParseHashFilter:
    def test_single_hash(self):
        assert submit_mod.parse_hash_filter(["abc123"]) == {"abc123"}

    def test_comma_separated(self):
        assert submit_mod.parse_hash_filter(["abc123,def456"]) == {"abc123", "def456"}

    def test_repeated_entries(self):
        assert submit_mod.parse_hash_filter(["abc123", "def456"]) == {"abc123", "def456"}

    def test_mixed_commas_and_repeated(self):
        result = submit_mod.parse_hash_filter(["abc,def", "ghi"])
        assert result == {"abc", "def", "ghi"}

    def test_strips_whitespace(self):
        assert submit_mod.parse_hash_filter([" abc , def "]) == {"abc", "def"}

    def test_deduplicates(self):
        assert submit_mod.parse_hash_filter(["abc,abc", "abc"]) == {"abc"}

    def test_empty_tokens_ignored(self):
        assert submit_mod.parse_hash_filter(["abc,,def", ""]) == {"abc", "def"}

    def test_empty_input(self):
        assert submit_mod.parse_hash_filter([]) == set()


class TestResolveHashPrefix:
    def test_exact_match(self):
        available = {"abc123", "def456", "ghi789"}
        assert submit_mod.resolve_hash_prefix("abc123", available) == "abc123"

    def test_unique_prefix(self):
        available = {"abc123", "def456", "ghi789"}
        assert submit_mod.resolve_hash_prefix("abc", available) == "abc123"

    def test_no_match(self):
        available = {"abc123", "def456"}
        assert submit_mod.resolve_hash_prefix("zzz", available) is None

    def test_ambiguous_prefix(self):
        available = {"abc123", "abc456", "def789"}
        import pytest

        with pytest.raises(ValueError, match="Ambiguous hash prefix"):
            submit_mod.resolve_hash_prefix("abc", available)


class TestFilterPathsByHash:
    def test_filters_to_matching_hashes(self, monkeypatch, tmp_path):
        sim1 = tmp_path / "sim1"
        sim2 = tmp_path / "sim2"
        sim1.mkdir()
        sim2.mkdir()

        class FakeBuildInput:
            def __init__(self, h):
                self.hash = h
                self.simulation_type = "bilayer"

        class FakeSimulation:
            def __init__(self, path, h):
                self.path = path
                self.build_input = FakeBuildInput(h)

        fake_df = pd.DataFrame(
            {
                "hash": ["hash_aaa", "hash_bbb"],
                "path": [str(sim1), str(sim2)],
                "simulation": [
                    FakeSimulation(sim1, "hash_aaa"),
                    FakeSimulation(sim2, "hash_bbb"),
                ],
            }
        )

        class FakeStore:
            def __init__(self, roots, trajectory_file=None, structure_file=None):
                pass

            def discover(self):
                return fake_df

        monkeypatch.setattr(
            "mdfactory.analysis.store.SimulationStore",
            FakeStore,
        )

        result = submit_mod.filter_paths_by_hash(
            [sim1, sim2],
            ["hash_aaa"],
        )
        assert result == [sim1]

    def test_prefix_matching(self, monkeypatch, tmp_path):
        sim1 = tmp_path / "sim1"
        sim2 = tmp_path / "sim2"
        sim1.mkdir()
        sim2.mkdir()

        class FakeBuildInput:
            def __init__(self, h):
                self.hash = h
                self.simulation_type = "bilayer"

        class FakeSimulation:
            def __init__(self, path, h):
                self.path = path
                self.build_input = FakeBuildInput(h)

        fake_df = pd.DataFrame(
            {
                "hash": ["hash_aaa111", "hash_bbb222"],
                "path": [str(sim1), str(sim2)],
                "simulation": [
                    FakeSimulation(sim1, "hash_aaa111"),
                    FakeSimulation(sim2, "hash_bbb222"),
                ],
            }
        )

        class FakeStore:
            def __init__(self, roots, trajectory_file=None, structure_file=None):
                pass

            def discover(self):
                return fake_df

        monkeypatch.setattr(
            "mdfactory.analysis.store.SimulationStore",
            FakeStore,
        )

        result = submit_mod.filter_paths_by_hash(
            [sim1, sim2],
            ["hash_aaa"],
        )
        assert result == [sim1]

    def test_no_match_raises(self, monkeypatch, tmp_path):
        sim1 = tmp_path / "sim1"
        sim1.mkdir()

        fake_df = pd.DataFrame(
            {
                "hash": ["hash_aaa"],
                "path": [str(sim1)],
                "simulation": [None],
            }
        )

        class FakeStore:
            def __init__(self, roots, trajectory_file=None, structure_file=None):
                pass

            def discover(self):
                return fake_df

        monkeypatch.setattr(
            "mdfactory.analysis.store.SimulationStore",
            FakeStore,
        )

        import pytest

        with pytest.raises(ValueError, match="None of the requested hashes"):
            submit_mod.filter_paths_by_hash([sim1], ["zzz"])


def test_submit_analyses_slurm_forwards_analysis_kwargs(monkeypatch, tmp_path):
    submitted_kwargs: list[dict[str, object]] = []

    class DummyJob:
        def __init__(self, job_id: str):
            self.job_id = job_id

        def result(self):
            return {"status": "success"}

    class DummyExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs):
            return None

        def submit(self, func, sim_path, analysis_names, **kwargs):
            submitted_kwargs.append(kwargs)
            return DummyJob(job_id=f"job-{len(submitted_kwargs)}")

    monkeypatch.setitem(sys.modules, "submitit", SimpleNamespace(AutoExecutor=DummyExecutor))

    sim_path = tmp_path / "sim1"
    sim_path.mkdir()
    slurm_cfg = submit_mod.SlurmConfig(account="acct")

    submit_mod.submit_analyses_slurm(
        [sim_path],
        ["analysis_a"],
        structure_file="system.pdb",
        trajectory_file="prod.xtc",
        slurm=slurm_cfg,
        log_dir=tmp_path / "logs",
        wait=False,
        analysis_kwargs={"start_ns": 50.0, "stride": 2},
    )

    assert len(submitted_kwargs) == 1
    assert submitted_kwargs[0]["analysis_kwargs"] == {"start_ns": 50.0, "stride": 2}


# --- SlurmConfig.from_cluster tests ---


class TestSlurmConfigFromCluster:
    """Tests for SlurmConfig.from_cluster() autodiscovery method."""

    def test_from_cluster_success(self, monkeypatch):
        """Test successful autodiscovery creates correct config."""
        from mdfactory.performance import cluster as cluster_mod

        # Create mock cluster info
        mock_partition = cluster_mod.Partition(
            name="compute",
            state="up",
            max_time="3-00:00:00",
            default_time="1:00:00",
            node_types=[
                cluster_mod.NodeType(cpus=32, memory_mb=128 * 1024, gpus=0),
            ],
            total_nodes=10,
            is_default=True,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[mock_partition],
            accounts=["myaccount", "otheraccount"],
            qos_policies=["normal", "high"],
            default_account="myaccount",
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        config = submit_mod.SlurmConfig.from_cluster(
            time="4h",
            cpus_per_task=8,
            mem_gb=16,
        )

        assert config.account == "myaccount"
        assert config.partition == "compute"
        assert config.time == "4h"
        assert config.cpus_per_task == 8
        assert config.mem_gb == 16

    def test_from_cluster_no_slurm_raises(self, monkeypatch):
        """Test that missing SLURM raises RuntimeError."""
        from mdfactory.performance import cluster as cluster_mod

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: None)

        import pytest

        with pytest.raises(RuntimeError, match="not running on a SLURM cluster"):
            submit_mod.SlurmConfig.from_cluster()

    def test_from_cluster_no_account_raises(self, monkeypatch):
        """Test that missing default account raises RuntimeError."""
        from mdfactory.performance import cluster as cluster_mod

        mock_partition = cluster_mod.Partition(
            name="compute",
            state="up",
            max_time="1-00:00:00",
            default_time="1:00:00",
            node_types=[cluster_mod.NodeType(cpus=16, memory_mb=64 * 1024)],
            total_nodes=5,
            is_default=True,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[mock_partition],
            accounts=[],
            qos_policies=[],
            default_account=None,
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        import pytest

        with pytest.raises(RuntimeError, match="no default account found"):
            submit_mod.SlurmConfig.from_cluster()

    def test_from_cluster_no_suitable_partition_raises(self, monkeypatch):
        """Test that no matching partition raises RuntimeError."""
        from mdfactory.performance import cluster as cluster_mod

        # Partition with only 4 CPUs - won't match request for 32 CPUs
        mock_partition = cluster_mod.Partition(
            name="tiny",
            state="up",
            max_time="1:00:00",
            default_time="0:30:00",
            node_types=[cluster_mod.NodeType(cpus=4, memory_mb=8 * 1024)],
            total_nodes=2,
            is_default=True,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[mock_partition],
            accounts=["myaccount"],
            qos_policies=[],
            default_account="myaccount",
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        import pytest

        with pytest.raises(RuntimeError, match="No suitable partition found"):
            submit_mod.SlurmConfig.from_cluster(min_cpus=32)

    def test_from_cluster_selects_gpu_partition(self, monkeypatch):
        """Test that needs_gpu=True selects GPU partition."""
        from mdfactory.performance import cluster as cluster_mod

        cpu_partition = cluster_mod.Partition(
            name="cpu",
            state="up",
            max_time="7-00:00:00",
            default_time="1:00:00",
            node_types=[cluster_mod.NodeType(cpus=64, memory_mb=256 * 1024, gpus=0)],
            total_nodes=100,
            is_default=True,
        )
        gpu_partition = cluster_mod.Partition(
            name="gpu",
            state="up",
            max_time="2-00:00:00",
            default_time="1:00:00",
            node_types=[
                cluster_mod.NodeType(
                    cpus=32, memory_mb=128 * 1024, gpus=4, gpu_type="a100"
                )
            ],
            total_nodes=20,
            is_default=False,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[cpu_partition, gpu_partition],
            accounts=["myaccount"],
            qos_policies=[],
            default_account="myaccount",
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        config = submit_mod.SlurmConfig.from_cluster(needs_gpu=True)

        assert config.partition == "gpu"
        assert config.account == "myaccount"
