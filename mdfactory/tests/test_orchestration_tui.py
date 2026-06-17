# ABOUTME: Tests for the interactive SLURM configuration TUI wizard
# ABOUTME: Uses mocked questionary to verify prompt flow and config generation
"""Tests for orchestration TUI wizard."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from mdfactory.orchestration.config import SlurmExecutorConfig
from mdfactory.orchestration.tui import (
    UserCancelledError,
    _require,
    configure_slurm_interactive,
    save_slurm_config_yaml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster():
    """Create a mock ClusterInfo for TUI tests."""
    nt = MagicMock()
    nt.cpus = 96
    nt.memory_mb = 512 * 1024
    nt.gpu_specs = ((4, "a100"),)
    nt.features = ("avx512",)
    nt.count = 10

    partition = MagicMock()
    partition.name = "gpu"
    partition.state = "up"
    partition.total_nodes = 10
    partition.node_types = [nt]
    partition.is_default = True
    partition.max_time = "24:00:00"

    cluster = MagicMock()
    cluster.partitions = [partition]
    cluster.accounts = ["hpc_chem", "hpc_bio"]
    cluster.qos_policies = ["normal", "high"]
    cluster.default_account = "hpc_chem"
    return cluster


# ---------------------------------------------------------------------------
# _require tests
# ---------------------------------------------------------------------------


class TestRequire:
    def test_returns_value(self):
        assert _require("hello", "test") == "hello"

    def test_raises_on_none(self):
        with pytest.raises(UserCancelledError):
            _require(None, "test")


# ---------------------------------------------------------------------------
# configure_slurm_interactive tests
# ---------------------------------------------------------------------------


class TestConfigureManualFallback:
    """When discover_cluster returns None, wizard uses manual entry."""

    @patch("mdfactory.orchestration.tui.discover_cluster", return_value=None)
    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_configure_manual_fallback(self, mock_iq, _discover):
        mock_q = mock_iq.return_value
        # confirm → proceed with manual config
        mock_q.confirm.return_value.ask.return_value = True
        # text prompts in order: account, partition, walltime, cpus,
        # gres, mem, qos, max_blocks, worker_init
        mock_q.text.return_value.ask.side_effect = [
            "manual_acc",
            "gpu",
            "4:00:00",
            "24",
            "gpu:a100:1",
            "64G",
            "",
            "4",
            "",
        ]

        cfg = configure_slurm_interactive()
        assert cfg.account == "manual_acc"
        assert cfg.partition == "gpu"
        assert cfg.walltime == "4:00:00"
        assert cfg.cpus_per_node == 24
        assert cfg.gres == "gpu:a100:1"
        assert cfg.mem == "64G"


class TestConfigureWithCluster:
    """When cluster is discovered, wizard uses select menus."""

    @patch("mdfactory.orchestration.tui.discover_cluster")
    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_configure_with_cluster(self, mock_iq, mock_discover):
        mock_q = mock_iq.return_value
        mock_discover.return_value = _make_cluster()

        # select prompts: account, partition, walltime, cpus, mem, max_blocks
        mock_q.select.return_value.ask.side_effect = [
            "hpc_chem",
            "gpu",
            "2h",
            "16",
            "50G",
            "4",
        ]
        # text prompts: gres, worker_init
        mock_q.text.return_value.ask.side_effect = [
            "gpu:a100:1",
            "",
        ]
        # confirm for QOS
        mock_q.confirm.return_value.ask.return_value = False

        cfg = configure_slurm_interactive()
        assert cfg.account == "hpc_chem"
        assert cfg.partition == "gpu"
        assert cfg.walltime == "02:00:00"
        assert cfg.cpus_per_node == 16
        assert cfg.mem == "50G"
        assert cfg.max_blocks == 4


class TestUserCancellation:
    """If a questionary prompt returns None, UserCancelledError is raised."""

    @patch("mdfactory.orchestration.tui.discover_cluster", return_value=None)
    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_user_cancellation(self, mock_iq, _discover):
        mock_q = mock_iq.return_value
        # confirm manual? → None (cancelled)
        mock_q.confirm.return_value.ask.return_value = None

        with pytest.raises(UserCancelledError):
            configure_slurm_interactive()


# ---------------------------------------------------------------------------
# save_slurm_config_yaml tests
# ---------------------------------------------------------------------------


class TestSaveYaml:
    def test_save_slurm_config_yaml(self, tmp_path):
        cfg = SlurmExecutorConfig(
            account="test_acc",
            partition="cpu",
            walltime="1h",
            cpus_per_node=16,
            gres="gpu:v100:1",
            max_blocks=3,
        )
        out = tmp_path / "slurm.yaml"
        save_slurm_config_yaml(cfg, out)

        data = yaml.safe_load(out.read_text())
        assert data["account"] == "test_acc"
        assert data["partition"] == "cpu"
        assert data["walltime"] == "01:00:00"
        assert data["cpus_per_node"] == 16
        assert data["gres"] == "gpu:v100:1"
        assert data["max_blocks"] == 3
        assert data["provider"] == "slurm"
