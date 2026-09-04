# ABOUTME: Tests for SlurmExecutorConfig.from_cluster() and walltime validation
# ABOUTME: Verifies 3-tier autodiscovery precedence and time normalization
"""Tests for SlurmExecutorConfig cluster autodiscovery and walltime."""

from unittest.mock import MagicMock, patch

import pytest

from mdfactory.orchestration.config import SlurmExecutorConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(**overrides):
    s = MagicMock()
    s.slurm_account = overrides.get("slurm_account", None)
    s.slurm_partition_cpu = overrides.get("slurm_partition_cpu", None)
    s.slurm_partition_gpu = overrides.get("slurm_partition_gpu", None)
    s.slurm_qos = overrides.get("slurm_qos", None)
    return s


def _mock_cluster(default_account="test_acc", accounts=None):
    c = MagicMock()
    c.default_account = default_account
    c.accounts = accounts or [default_account]
    return c


# ---------------------------------------------------------------------------
# Walltime normalisation
# ---------------------------------------------------------------------------


class TestWalltimeNormalization:
    """Walltime shorthand is expanded on construction."""

    def test_walltime_normalization(self):
        assert SlurmExecutorConfig(account="x", walltime="2h").walltime == "02:00:00"
        assert SlurmExecutorConfig(account="x", walltime="30m").walltime == "00:30:00"
        assert SlurmExecutorConfig(account="x", walltime="1d").walltime == "1-00:00:00"
        assert SlurmExecutorConfig(account="x", walltime="01:30:00").walltime == "01:30:00"

    def test_walltime_default_normalized(self):
        cfg = SlurmExecutorConfig(account="x")
        assert cfg.walltime == "02:00:00"


# ---------------------------------------------------------------------------
# from_cluster() tests
# ---------------------------------------------------------------------------

# Patch targets — from_cluster imports these inside the method body, so we
# patch the canonical module locations.
_DISCOVER = "mdfactory.performance.cluster.discover_cluster"
_SELECT = "mdfactory.performance.cluster.select_partition"
_SETTINGS = "mdfactory.settings.settings"


class TestFromCluster:
    """Tests for the three-tier autodiscovery in from_cluster()."""

    @patch(_DISCOVER, return_value=None)
    @patch(_SETTINGS, _mock_settings())
    def test_explicit_kwargs(self, _discover):
        cfg = SlurmExecutorConfig.from_cluster(account="myacc", partition="gpu")
        assert cfg.account == "myacc"
        assert cfg.partition == "gpu"

    @patch(_DISCOVER, return_value=None)
    @patch(
        _SETTINGS,
        _mock_settings(slurm_account="cfg_account", slurm_partition_cpu="cfg_partition"),
    )
    def test_config_ini_fallback(self, _discover):
        cfg = SlurmExecutorConfig.from_cluster()
        assert cfg.account == "cfg_account"
        assert cfg.partition == "cfg_partition"

    @patch(_SELECT)
    @patch(_DISCOVER)
    @patch(_SETTINGS, _mock_settings())
    def test_autodiscovery(self, mock_discover, mock_select):
        mock_discover.return_value = _mock_cluster(default_account="discovered_acc")
        part = MagicMock()
        part.name = "auto_part"
        mock_select.return_value = part

        cfg = SlurmExecutorConfig.from_cluster()
        assert cfg.account == "discovered_acc"
        assert cfg.partition == "auto_part"

    @patch(_SELECT)
    @patch(_DISCOVER)
    @patch(_SETTINGS, _mock_settings(slurm_account="cfg_acc"))
    def test_precedence_explicit_wins(self, mock_discover, mock_select):
        mock_discover.return_value = _mock_cluster(default_account="disc_acc")
        part = MagicMock()
        part.name = "disc_part"
        mock_select.return_value = part

        cfg = SlurmExecutorConfig.from_cluster(account="explicit")
        assert cfg.account == "explicit"

    @patch(_DISCOVER, return_value=None)
    @patch(_SETTINGS, _mock_settings())
    def test_no_account_raises(self, _discover):
        with pytest.raises(RuntimeError, match="account"):
            SlurmExecutorConfig.from_cluster()

    @patch(_DISCOVER, return_value=None)
    @patch(_SETTINGS, _mock_settings(slurm_account="has_acc"))
    def test_no_partition_raises(self, _discover):
        with pytest.raises(RuntimeError, match="partition"):
            SlurmExecutorConfig.from_cluster()

    @patch(_SELECT)
    @patch(_DISCOVER)
    @patch(_SETTINGS, _mock_settings())
    def test_extra_fields_forwarded(self, mock_discover, mock_select):
        mock_discover.return_value = _mock_cluster()
        part = MagicMock()
        part.name = "cpu"
        mock_select.return_value = part

        cfg = SlurmExecutorConfig.from_cluster(walltime="4h", cpus_per_node=24, gres="gpu:a100:1")
        assert cfg.walltime == "04:00:00"
        assert cfg.cpus_per_node == 24
        assert cfg.gres == "gpu:a100:1"

    @patch(_DISCOVER, return_value=None)
    @patch(
        _SETTINGS,
        _mock_settings(slurm_account="acc", slurm_partition_gpu="gpu-big"),
    )
    def test_needs_gpu_selects_gpu_partition(self, _discover):
        cfg = SlurmExecutorConfig.from_cluster(needs_gpu=True)
        assert cfg.partition == "gpu-big"
