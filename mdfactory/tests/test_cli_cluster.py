# ABOUTME: Tests for the `mdfactory config cluster` CLI command, verifying
# ABOUTME: SLURM autodiscovery output in human-readable and JSON formats.
# ABOUTME: Also covers the SLURM config-building glue in analysis_run / analysis_artifacts_run.
"""Tests for mdfactory config cluster CLI command and analysis SLURM glue."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mdfactory import cli
from mdfactory.performance import cluster as cluster_mod
from mdfactory.performance.slurm_config import SlurmConfig


class TestConfigClusterCommand:
    """Tests for the config cluster CLI command."""

    def test_config_cluster_no_slurm(self, monkeypatch, capsys):
        """Test graceful message when SLURM is not available."""
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: None)

        cli.config_cluster(json_output=False)

        captured = capsys.readouterr()
        assert "SLURM cluster not detected" in captured.out
        assert "login node" in captured.out

    def test_config_cluster_no_slurm_json(self, monkeypatch, capsys):
        """Test JSON output when SLURM is not available."""
        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: None)

        cli.config_cluster(json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["cluster"] is None
        assert "error" in data

    def test_config_cluster_with_slurm(self, monkeypatch, capsys):
        """Test human-readable output with SLURM cluster."""
        mock_partition = cluster_mod.Partition(
            name="compute",
            state="up",
            max_time="7-00:00:00",
            default_time="1:00:00",
            node_types=[
                cluster_mod.NodeType(cpus=64, memory_mb=256 * 1024, gpu_specs=(), count=100),
            ],
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
                    cpus=32, memory_mb=128 * 1024, gpu_specs=((4, "a100"),), count=20
                ),
            ],
            total_nodes=20,
            is_default=False,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[mock_partition, gpu_partition],
            accounts=["myaccount", "shared"],
            qos_policies=["normal", "high"],
            default_account="myaccount",
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        cli.config_cluster(json_output=False)

        captured = capsys.readouterr()
        assert "SLURM Cluster Information" in captured.out
        assert "Default Account: myaccount" in captured.out
        assert "compute" in captured.out
        assert "(default)" in captured.out
        assert "gpu" in captured.out
        assert "a100" in captured.out
        assert "64 CPUs" in captured.out
        assert "4x a100" in captured.out

    def test_config_cluster_with_slurm_json(self, monkeypatch, capsys):
        """Test JSON output with SLURM cluster."""
        mock_partition = cluster_mod.Partition(
            name="compute",
            state="up",
            max_time="7-00:00:00",
            default_time="1:00:00",
            node_types=[
                cluster_mod.NodeType(
                    cpus=64,
                    memory_mb=256 * 1024,
                    gpu_specs=(),
                    features=("avx512", "intel"),
                    count=100,
                ),
            ],
            total_nodes=100,
            is_default=True,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[mock_partition],
            accounts=["myaccount"],
            qos_policies=["normal"],
            default_account="myaccount",
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        cli.config_cluster(json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["default_account"] == "myaccount"
        assert data["accounts"] == ["myaccount"]
        assert data["qos_policies"] == ["normal"]
        assert len(data["partitions"]) == 1

        part = data["partitions"][0]
        assert part["name"] == "compute"
        assert part["is_default"] is True
        assert part["total_nodes"] == 100
        assert len(part["node_types"]) == 1

        nt = part["node_types"][0]
        assert nt["cpus"] == 64
        assert nt["memory_mb"] == 256 * 1024
        assert nt["gpu_specs"] == []  # No GPUs
        assert nt["features"] == ["avx512", "intel"]
        assert nt["count"] == 100

    def test_config_cluster_down_partition(self, monkeypatch, capsys):
        """Test display of partition with down state."""
        mock_partition = cluster_mod.Partition(
            name="maintenance",
            state="drained",
            max_time="1:00:00",
            default_time="0:30:00",
            node_types=[cluster_mod.NodeType(cpus=16, memory_mb=32 * 1024, count=5)],
            total_nodes=5,
            is_default=False,
        )
        mock_cluster = cluster_mod.ClusterInfo(
            partitions=[mock_partition],
            accounts=["myaccount"],
            qos_policies=[],
            default_account="myaccount",
        )

        monkeypatch.setattr(cluster_mod, "discover_cluster", lambda: mock_cluster)

        cli.config_cluster(json_output=False)

        captured = capsys.readouterr()
        assert "maintenance" in captured.out
        assert "[drained]" in captured.out


# ---------------------------------------------------------------------------
# Helpers shared by analysis_run / analysis_artifacts_run glue tests
# ---------------------------------------------------------------------------

_FAKE_CFG = SlurmConfig(account="auto-account", partition="auto-part")
_EMPTY_DF = pd.DataFrame()


@pytest.fixture()
def fake_sim_paths(tmp_path: Path) -> list[Path]:
    """Return a non-empty list of paths without touching the filesystem."""
    return [tmp_path / "sim1"]


# ---------------------------------------------------------------------------
# Tests: analysis_run SLURM config-building glue
# ---------------------------------------------------------------------------


class TestAnalysisRunSlurmGlue:
    """Tests for the autodiscovery glue inside analysis_run (--slurm mode)."""

    def test_autodiscovery_used_when_no_account(self, monkeypatch, fake_sim_paths):
        """account=None triggers SlurmConfig.from_cluster(); result is forwarded."""
        captured = {}

        def fake_submit(sim_paths, analysis_names, *, slurm, **kwargs):
            captured["slurm"] = slurm
            return _EMPTY_DF

        with (
            patch("mdfactory.cli._resolve_sim_paths", return_value=fake_sim_paths),
            patch("mdfactory.cli.SlurmConfig.from_cluster", return_value=_FAKE_CFG) as mock_fc,
            patch("mdfactory.cli.submit_analyses_slurm", side_effect=fake_submit),
            patch("mdfactory.cli.determine_log_dir", return_value=fake_sim_paths[0].parent),
        ):
            cli.analysis_run(
                source=fake_sim_paths[0].parent,
                slurm=True,
                account=None,
            )

        mock_fc.assert_called_once()
        assert captured["slurm"] is _FAKE_CFG

    def test_autodiscovery_error_reraised_as_value_error(self, monkeypatch, fake_sim_paths):
        """RuntimeError from from_cluster() is re-raised as ValueError with guidance."""
        with (
            patch("mdfactory.cli._resolve_sim_paths", return_value=fake_sim_paths),
            patch(
                "mdfactory.cli.SlurmConfig.from_cluster",
                side_effect=RuntimeError("no account"),
            ),
        ):
            with pytest.raises(ValueError, match="Please specify --account explicitly"):
                cli.analysis_run(
                    source=fake_sim_paths[0].parent,
                    slurm=True,
                    account=None,
                )

    def test_explicit_account_skips_autodiscovery(self, monkeypatch, fake_sim_paths):
        """When account is provided, from_cluster() is never called."""
        captured = {}

        def fake_submit(sim_paths, analysis_names, *, slurm, **kwargs):
            captured["slurm"] = slurm
            return _EMPTY_DF

        with (
            patch("mdfactory.cli._resolve_sim_paths", return_value=fake_sim_paths),
            patch("mdfactory.cli.SlurmConfig.from_cluster") as mock_fc,
            patch("mdfactory.cli.submit_analyses_slurm", side_effect=fake_submit),
            patch("mdfactory.cli.determine_log_dir", return_value=fake_sim_paths[0].parent),
        ):
            cli.analysis_run(
                source=fake_sim_paths[0].parent,
                slurm=True,
                account="explicit-account",
            )

        mock_fc.assert_not_called()
        assert captured["slurm"].account == "explicit-account"


# ---------------------------------------------------------------------------
# Tests: analysis_artifacts_run SLURM config-building glue
# ---------------------------------------------------------------------------


class TestAnalysisArtifactsRunSlurmGlue:
    """Tests for the autodiscovery glue inside analysis_artifacts_run (--slurm mode)."""

    def test_autodiscovery_used_when_no_account(self, monkeypatch, fake_sim_paths):
        """account=None triggers SlurmConfig.from_cluster(); result is forwarded."""
        captured = {}

        def fake_submit(sim_paths, artifact_names, *, slurm, **kwargs):
            captured["slurm"] = slurm
            return _EMPTY_DF

        with (
            patch("mdfactory.cli._resolve_sim_paths", return_value=fake_sim_paths),
            patch("mdfactory.cli.SlurmConfig.from_cluster", return_value=_FAKE_CFG) as mock_fc,
            patch("mdfactory.cli.submit_artifacts_slurm", side_effect=fake_submit),
            patch("mdfactory.cli.determine_log_dir", return_value=fake_sim_paths[0].parent),
        ):
            cli.analysis_artifacts_run(
                source=fake_sim_paths[0].parent,
                slurm=True,
                account=None,
            )

        mock_fc.assert_called_once()
        assert captured["slurm"] is _FAKE_CFG

    def test_autodiscovery_error_reraised_as_value_error(self, monkeypatch, fake_sim_paths):
        """RuntimeError from from_cluster() is re-raised as ValueError with guidance."""
        with (
            patch("mdfactory.cli._resolve_sim_paths", return_value=fake_sim_paths),
            patch(
                "mdfactory.cli.SlurmConfig.from_cluster",
                side_effect=RuntimeError("no account"),
            ),
        ):
            with pytest.raises(ValueError, match="Please specify --account explicitly"):
                cli.analysis_artifacts_run(
                    source=fake_sim_paths[0].parent,
                    slurm=True,
                    account=None,
                )

    def test_explicit_account_skips_autodiscovery(self, monkeypatch, fake_sim_paths):
        """When account is provided, from_cluster() is never called."""
        captured = {}

        def fake_submit(sim_paths, artifact_names, *, slurm, **kwargs):
            captured["slurm"] = slurm
            return _EMPTY_DF

        with (
            patch("mdfactory.cli._resolve_sim_paths", return_value=fake_sim_paths),
            patch("mdfactory.cli.SlurmConfig.from_cluster") as mock_fc,
            patch("mdfactory.cli.submit_artifacts_slurm", side_effect=fake_submit),
            patch("mdfactory.cli.determine_log_dir", return_value=fake_sim_paths[0].parent),
        ):
            cli.analysis_artifacts_run(
                source=fake_sim_paths[0].parent,
                slurm=True,
                account="explicit-account",
            )

        mock_fc.assert_not_called()
        assert captured["slurm"].account == "explicit-account"
