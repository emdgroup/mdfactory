# ABOUTME: Tests for the `mdfactory config cluster` CLI command, verifying
# ABOUTME: SLURM autodiscovery output in human-readable and JSON formats.
"""Tests for mdfactory config cluster CLI command."""

import json

from mdfactory import cli
from mdfactory.performance import cluster as cluster_mod


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
                cluster_mod.NodeType(cpus=64, memory_mb=256 * 1024, gpus=0),
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
                cluster_mod.NodeType(cpus=32, memory_mb=128 * 1024, gpus=4, gpu_type="a100"),
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
                    gpus=0,
                    features=("avx512", "intel"),
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
        assert nt["gpus"] == 0
        assert nt["features"] == ["avx512", "intel"]

    def test_config_cluster_down_partition(self, monkeypatch, capsys):
        """Test display of partition with down state."""
        mock_partition = cluster_mod.Partition(
            name="maintenance",
            state="drained",
            max_time="1:00:00",
            default_time="0:30:00",
            node_types=[cluster_mod.NodeType(cpus=16, memory_mb=32 * 1024)],
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
