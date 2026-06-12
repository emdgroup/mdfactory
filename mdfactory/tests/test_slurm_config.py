# ABOUTME: Unit tests for BaseSlurmConfig, SlurmConfig, and normalize_slurm_time.
# ABOUTME: Uses mocked discover_cluster() so tests run on non-SLURM machines.
"""Unit tests for mdfactory.performance.slurm_config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mdfactory.performance import cluster as cluster_mod
from mdfactory.performance.slurm_config import (
    BaseSlurmConfig,
    SlurmConfig,
    normalize_slurm_time,
)

# ---------------------------------------------------------------------------
# normalize_slurm_time
# ---------------------------------------------------------------------------


class TestNormalizeSlurmTime:
    """Edge-case coverage for the time-string normaliser."""

    def test_hours_shorthand(self):
        assert normalize_slurm_time("2h") == "02:00:00"

    def test_hours_single_digit(self):
        assert normalize_slurm_time("1h") == "01:00:00"

    def test_hours_large(self):
        assert normalize_slurm_time("24h") == "24:00:00"

    def test_minutes_shorthand(self):
        assert normalize_slurm_time("30m") == "00:30:00"

    def test_minutes_overflow(self):
        assert normalize_slurm_time("90m") == "01:30:00"

    def test_minutes_as_integer(self):
        assert normalize_slurm_time("90") == "01:30:00"
        assert normalize_slurm_time("120") == "02:00:00"

    def test_days_shorthand(self):
        assert normalize_slurm_time("1d") == "1-00:00:00"
        assert normalize_slurm_time("3d") == "3-00:00:00"

    def test_hms_passthrough(self):
        assert normalize_slurm_time("01:00:00") == "01:00:00"
        assert normalize_slurm_time("3-00:00:00") == "3-00:00:00"

    def test_strips_whitespace(self):
        assert normalize_slurm_time("  2h  ") == "02:00:00"


# ---------------------------------------------------------------------------
# SlurmConfig — Pydantic model behaviour
# ---------------------------------------------------------------------------


class TestSlurmConfigModel:
    """Tests for the SlurmConfig Pydantic model (no SLURM required)."""

    def test_minimal_construction(self):
        cfg = SlurmConfig(account="mygroup")
        assert cfg.account == "mygroup"
        assert cfg.partition == "cpu"
        assert cfg.time == "02:00:00"  # "2h" normalised on construction
        assert cfg.cpus_per_task == 4
        assert cfg.mem_gb == 8
        assert cfg.job_name_prefix == "mdfactory-analysis"
        assert cfg.qos is None
        assert cfg.constraint is None

    def test_time_normalised_on_construction(self):
        cfg = SlurmConfig(account="grp", time="4h")
        assert cfg.time == "04:00:00"

    def test_time_hms_passthrough(self):
        cfg = SlurmConfig(account="grp", time="02:00:00")
        assert cfg.time == "02:00:00"

    def test_time_minutes_shorthand(self):
        cfg = SlurmConfig(account="grp", time="30m")
        assert cfg.time == "00:30:00"

    def test_time_integer_minutes(self):
        cfg = SlurmConfig(account="grp", time="120")
        assert cfg.time == "02:00:00"

    def test_frozen(self):
        cfg = SlurmConfig(account="grp")
        with pytest.raises(Exception):  # ValidationError or AttributeError
            cfg.account = "other"  # type: ignore[misc]

    def test_inherits_base_slurm_config(self):
        assert issubclass(SlurmConfig, BaseSlurmConfig)

    def test_optional_fields(self):
        cfg = SlurmConfig(
            account="grp",
            partition="gpu",
            qos="high",
            constraint="a100",
            time="1d",
            cpus_per_task=16,
            mem_gb=64,
            job_name_prefix="my-job",
        )
        assert cfg.partition == "gpu"
        assert cfg.qos == "high"
        assert cfg.constraint == "a100"
        assert cfg.time == "1-00:00:00"
        assert cfg.cpus_per_task == 16
        assert cfg.mem_gb == 64
        assert cfg.job_name_prefix == "my-job"


# ---------------------------------------------------------------------------
# SlurmConfig.from_yaml
# ---------------------------------------------------------------------------


class TestSlurmConfigFromYaml:
    def test_round_trip(self, tmp_path: Path):
        data = {
            "account": "mygroup",
            "partition": "compute",
            "time": "4h",
            "cpus_per_task": 8,
            "mem_gb": 32,
        }
        yaml_file = tmp_path / "slurm.yaml"
        yaml_file.write_text(yaml.dump(data))

        cfg = SlurmConfig.from_yaml(yaml_file)

        assert cfg.account == "mygroup"
        assert cfg.partition == "compute"
        assert cfg.time == "04:00:00"  # normalised
        assert cfg.cpus_per_task == 8
        assert cfg.mem_gb == 32

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SlurmConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_defaults_applied(self, tmp_path: Path):
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text("account: grp\n")

        cfg = SlurmConfig.from_yaml(yaml_file)
        assert cfg.time == "02:00:00"
        assert cfg.cpus_per_task == 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_slurm_settings(monkeypatch):
    """Return None for all [slurm] settings — simulates an unconfigured machine.

    Without this fixture, tests that expect autodiscovery to be the sole source
    of truth would behave differently on machines that have ACCOUNT / PARTITION_CPU
    set in their config.ini.
    """
    from mdfactory.settings import Settings

    monkeypatch.setattr(Settings, "slurm_account", property(lambda self: None))
    monkeypatch.setattr(Settings, "slurm_partition_cpu", property(lambda self: None))
    monkeypatch.setattr(Settings, "slurm_partition_gpu", property(lambda self: None))
    monkeypatch.setattr(Settings, "slurm_qos", property(lambda self: None))


# ---------------------------------------------------------------------------
# Helpers to build mock cluster objects
# ---------------------------------------------------------------------------


def _make_cpu_partition(name: str = "compute", cpus: int = 32) -> cluster_mod.Partition:
    return cluster_mod.Partition(
        name=name,
        state="up",
        max_time="3-00:00:00",
        default_time="1:00:00",
        node_types=[cluster_mod.NodeType(cpus=cpus, memory_mb=128 * 1024, gpu_specs=(), count=10)],
        total_nodes=10,
        is_default=True,
    )


def _make_gpu_partition(name: str = "gpu") -> cluster_mod.Partition:
    return cluster_mod.Partition(
        name=name,
        state="up",
        max_time="2-00:00:00",
        default_time="1:00:00",
        node_types=[
            cluster_mod.NodeType(
                cpus=32,
                memory_mb=128 * 1024,
                gpu_specs=((4, "a100"),),
                count=20,
            )
        ],
        total_nodes=20,
        is_default=False,
    )


def _make_cluster(
    partitions: list[cluster_mod.Partition] | None = None,
    default_account: str | None = "myaccount",
) -> cluster_mod.ClusterInfo:
    if partitions is None:
        partitions = [_make_cpu_partition()]
    return cluster_mod.ClusterInfo(
        partitions=partitions,
        accounts=[default_account] if default_account else [],
        qos_policies=["normal"],
        default_account=default_account,
    )


# ---------------------------------------------------------------------------
# BaseSlurmConfig.from_cluster — 3-tier precedence
# ---------------------------------------------------------------------------


class TestBaseSlurmConfigFromCluster:
    """Tests run without a real SLURM cluster (discover_cluster mocked)."""

    def test_autodiscovery_populates_account_and_partition(self, monkeypatch, no_slurm_settings):
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        cfg = SlurmConfig.from_cluster()
        assert cfg.account == "myaccount"
        assert cfg.partition == "compute"

    def test_no_slurm_raises(self, monkeypatch, no_slurm_settings):
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: None)
        with pytest.raises(RuntimeError, match="SLURM autodiscovery failed and no account"):
            SlurmConfig.from_cluster()

    def test_no_default_account_raises(self, monkeypatch, no_slurm_settings):
        cluster = _make_cluster(default_account=None)
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: cluster)
        with pytest.raises(RuntimeError, match="No SLURM account available"):
            SlurmConfig.from_cluster()

    def test_no_suitable_partition_raises(self, monkeypatch, no_slurm_settings):
        # Tiny partition: 4 CPUs — won't satisfy min_cpus=32
        cluster = _make_cluster(partitions=[_make_cpu_partition(cpus=4)])
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: cluster)
        with pytest.raises(RuntimeError, match="No suitable partition found"):
            SlurmConfig.from_cluster(min_cpus=32)

    def test_autodiscovery_fails_no_partition_in_config_raises(
        self, monkeypatch, no_slurm_settings
    ):
        """cluster=None with account supplied but no partition in config raises RuntimeError."""
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: None)
        # Provide account so resolution reaches the partition step
        monkeypatch.setattr(Settings, "slurm_account", property(lambda self: "myaccount"))
        with pytest.raises(RuntimeError, match="no partition configured"):
            SlurmConfig.from_cluster()

    def test_needs_gpu_selects_gpu_partition(self, monkeypatch, no_slurm_settings):
        cluster = _make_cluster(partitions=[_make_cpu_partition(), _make_gpu_partition()])
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: cluster)
        cfg = SlurmConfig.from_cluster(needs_gpu=True)
        assert cfg.partition == "gpu"

    # --- tier 2: config.ini overrides autodiscovery ---

    def test_config_account_overrides_autodiscovery(self, monkeypatch):
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        monkeypatch.setattr(Settings, "slurm_account", property(lambda self: "config-account"))
        cfg = SlurmConfig.from_cluster()
        assert cfg.account == "config-account"

    def test_config_cpu_partition_overrides_autodiscovery(self, monkeypatch):
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        monkeypatch.setattr(Settings, "slurm_partition_cpu", property(lambda self: "config-cpu"))
        cfg = SlurmConfig.from_cluster(needs_gpu=False)
        assert cfg.partition == "config-cpu"

    def test_config_gpu_partition_overrides_autodiscovery(self, monkeypatch):
        from mdfactory.settings import Settings

        cluster = _make_cluster(partitions=[_make_cpu_partition(), _make_gpu_partition()])
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: cluster)
        monkeypatch.setattr(Settings, "slurm_partition_gpu", property(lambda self: "config-gpu"))
        cfg = SlurmConfig.from_cluster(needs_gpu=True)
        assert cfg.partition == "config-gpu"

    def test_config_qos_propagated(self, monkeypatch):
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        monkeypatch.setattr(Settings, "slurm_qos", property(lambda self: "high"))
        cfg = SlurmConfig.from_cluster()
        assert cfg.qos == "high"

    # --- tier 1: explicit kwargs override everything ---

    def test_explicit_account_overrides_config_and_autodiscovery(self, monkeypatch):
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        monkeypatch.setattr(Settings, "slurm_account", property(lambda self: "config-account"))
        cfg = SlurmConfig.from_cluster(account="explicit-account")
        assert cfg.account == "explicit-account"

    def test_explicit_partition_overrides_config_and_autodiscovery(self, monkeypatch):
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        monkeypatch.setattr(
            Settings, "slurm_partition_cpu", property(lambda self: "config-partition")
        )
        cfg = SlurmConfig.from_cluster(partition="explicit-partition")
        assert cfg.partition == "explicit-partition"

    def test_explicit_qos_overrides_config(self, monkeypatch):
        from mdfactory.settings import Settings

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        monkeypatch.setattr(Settings, "slurm_qos", property(lambda self: "high"))
        cfg = SlurmConfig.from_cluster(qos="debug")
        assert cfg.qos == "debug"

    def test_explicit_constraint_forwarded(self, monkeypatch):
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        cfg = SlurmConfig.from_cluster(constraint="epyc")
        assert cfg.constraint == "epyc"

    # --- submitit-specific extra fields forwarded ---

    def test_submitit_fields_forwarded(self, monkeypatch):
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        cfg = SlurmConfig.from_cluster(
            time="4h",
            cpus_per_task=8,
            mem_gb=32,
            job_name_prefix="my-prefix",
        )
        assert cfg.time == "04:00:00"
        assert cfg.cpus_per_task == 8
        assert cfg.mem_gb == 32
        assert cfg.job_name_prefix == "my-prefix"

    def test_returns_slurm_config_type(self, monkeypatch):
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: _make_cluster())
        result = SlurmConfig.from_cluster()
        assert isinstance(result, SlurmConfig)


# ---------------------------------------------------------------------------
# Backward-compat import from mdfactory.analysis.submit
# ---------------------------------------------------------------------------


def test_backward_compat_import_slurm_config():
    """from mdfactory.analysis.submit import SlurmConfig must still work."""
    from mdfactory.analysis.submit import SlurmConfig as SubmitSlurmConfig

    cfg = SubmitSlurmConfig(account="grp")
    assert cfg.account == "grp"


def test_backward_compat_import_normalize_slurm_time():
    """from mdfactory.analysis.submit import normalize_slurm_time must still work."""
    from mdfactory.analysis.submit import normalize_slurm_time as nslt

    assert nslt("2h") == "02:00:00"
