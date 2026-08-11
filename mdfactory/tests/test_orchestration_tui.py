# ABOUTME: Tests for the interactive SLURM configuration TUI wizard
# ABOUTME: Uses mocked questionary to verify prompt flow and config generation
"""Tests for orchestration TUI wizard."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from mdfactory.orchestration.config import SlurmExecutorConfig
from mdfactory.orchestration.tui import (
    UserCancelledError,
    _default_worker_init,
    _detect_gromacs_modules,
    _prompt_stage_overrides,
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
# _detect_gromacs_modules tests
# ---------------------------------------------------------------------------


class TestDetectGromacsModules:
    """Tests for GROMACS module detection from $LOADEDMODULES."""

    def test_no_env_var(self, monkeypatch):
        monkeypatch.delenv("LOADEDMODULES", raising=False)
        assert _detect_gromacs_modules() == []

    def test_empty_env_var(self, monkeypatch):
        monkeypatch.setenv("LOADEDMODULES", "")
        assert _detect_gromacs_modules() == []

    def test_single_gromacs_module(self, monkeypatch):
        monkeypatch.setenv("LOADEDMODULES", "cuda/12.2:gromacs/2024.3-gpu:openmpi/4.1")
        assert _detect_gromacs_modules() == ["gromacs/2024.3-gpu"]

    def test_multiple_gromacs_modules(self, monkeypatch):
        monkeypatch.setenv("LOADEDMODULES", "gromacs/2023.1:gromacs/2024.3-gpu")
        assert _detect_gromacs_modules() == ["gromacs/2023.1", "gromacs/2024.3-gpu"]

    def test_gmx_prefix(self, monkeypatch):
        monkeypatch.setenv("LOADEDMODULES", "gmx/2024.3:openmpi/4.1")
        assert _detect_gromacs_modules() == ["gmx/2024.3"]

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LOADEDMODULES", "GROMACS/2024.3:Gmx/2023.1")
        assert _detect_gromacs_modules() == ["GROMACS/2024.3", "Gmx/2023.1"]

    def test_no_gromacs_modules(self, monkeypatch):
        monkeypatch.setenv("LOADEDMODULES", "cuda/12.2:openmpi/4.1:python/3.11")
        assert _detect_gromacs_modules() == []


# ---------------------------------------------------------------------------
# _default_worker_init tests
# ---------------------------------------------------------------------------


class TestDefaultWorkerInit:
    """Tests for _default_worker_init with for_simulate flag."""

    def test_without_simulate_no_modules(self, monkeypatch):
        """Build context: no module load even if GROMACS modules present."""
        monkeypatch.setenv("LOADEDMODULES", "gromacs/2024.3-gpu")
        result = _default_worker_init(for_simulate=False)
        assert "module load" not in result

    def test_with_simulate_includes_modules(self, monkeypatch, tmp_path):
        """Simulate context: module load commands are prepended."""
        monkeypatch.setenv("LOADEDMODULES", "gromacs/2024.3-gpu")
        # Ensure pixi env detection doesn't interfere
        monkeypatch.setattr(
            "mdfactory.orchestration.tui.Path",
            lambda *a, **kw: tmp_path / "nonexistent",
        )
        result = _default_worker_init(for_simulate=True)
        assert "module load gromacs/2024.3-gpu" in result

    def test_with_simulate_no_modules(self, monkeypatch):
        """Simulate context but no GROMACS modules → no module load."""
        monkeypatch.delenv("LOADEDMODULES", raising=False)
        result = _default_worker_init(for_simulate=True)
        assert "module load" not in result

    def test_with_simulate_multiple_modules(self, monkeypatch, tmp_path):
        """Multiple GROMACS modules → multiple module load commands."""
        monkeypatch.setenv("LOADEDMODULES", "gromacs/2023.1:gromacs/2024.3-gpu")
        monkeypatch.setattr(
            "mdfactory.orchestration.tui.Path",
            lambda *a, **kw: tmp_path / "nonexistent",
        )
        result = _default_worker_init(for_simulate=True)
        assert "module load gromacs/2023.1" in result
        assert "module load gromacs/2024.3-gpu" in result
        # Should be semicolon-separated
        assert "; " in result


# ---------------------------------------------------------------------------
# configure_slurm_interactive tests
# ---------------------------------------------------------------------------


class TestConfigureManualFallback:
    """When discover_cluster returns None, wizard uses manual entry."""

    @patch("mdfactory.orchestration.tui.discover_cluster", return_value=None)
    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_configure_manual_fallback(self, mock_iq, _discover):
        mock_q = mock_iq.return_value
        # confirm prompts: proceed with manual? → True, stage overrides? → False
        mock_q.confirm.return_value.ask.side_effect = [True, False]
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
        assert cfg.stage_overrides == {}


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
        # text prompts: gres, worker_init, constraint
        mock_q.text.return_value.ask.side_effect = [
            "gpu:a100:1",
            "",
            "a100",
        ]
        # confirm: QOS → False, stage overrides → False
        mock_q.confirm.return_value.ask.side_effect = [False, False]

        cfg = configure_slurm_interactive()
        assert cfg.account == "hpc_chem"
        assert cfg.partition == "gpu"
        assert cfg.walltime == "02:00:00"
        assert cfg.cpus_per_node == 16
        assert cfg.mem == "50G"
        assert cfg.max_blocks == 4
        assert cfg.constraint == "a100"
        assert cfg.stage_overrides == {}


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
# _prompt_stage_overrides tests
# ---------------------------------------------------------------------------


class TestPromptStageOverrides:
    """Tests for the per-stage override prompt helper."""

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_declined(self, mock_iq):
        """User declines stage overrides → empty dict."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = False

        result = _prompt_stage_overrides(
            common_cpus=16, common_gres="gpu:a100:1", common_gmx="auto"
        )
        assert result == {}

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_em_cpu_only(self, mock_iq):
        """GPU config + accept + EM → gres=None and higher cpus."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.checkbox.return_value.ask.return_value = ["EM"]
        # text prompts for EM: cpus (default 32=16*2), gres (default ""), gmx_binary
        mock_q.text.return_value.ask.side_effect = ["32", "", "auto"]

        result = _prompt_stage_overrides(
            common_cpus=16, common_gres="gpu:a100:1", common_gmx="auto"
        )
        assert "EM" in result
        assert result["EM"]["cpus_per_node"] == 32
        assert result["EM"]["gres"] is None
        # gmx_binary same as common → not stored
        assert "gmx_binary" not in result["EM"]

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_multiple_stages(self, mock_iq):
        """Multiple stages selected → each gets correct overrides."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.checkbox.return_value.ask.return_value = ["EM", "Production"]
        # text prompts: EM(cpus, gres, gmx), Production(cpus, gres, gmx)
        mock_q.text.return_value.ask.side_effect = [
            "8",  # EM cpus
            "",  # EM gres (empty → None, differs from common None → no diff)
            "auto",  # EM gmx
            "32",  # Production cpus
            "gpu:a100:2",  # Production gres
            "auto",  # Production gmx
        ]

        result = _prompt_stage_overrides(common_cpus=16, common_gres=None, common_gmx="auto")
        assert result["EM"] == {"cpus_per_node": 8}
        assert result["Production"] == {"cpus_per_node": 32, "gres": "gpu:a100:2"}

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_no_diff_excluded(self, mock_iq):
        """Override values same as common → not stored in overrides."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.checkbox.return_value.ask.return_value = ["NVT"]
        # text prompts: all same as common values
        mock_q.text.return_value.ask.side_effect = ["16", "gpu:a100:1", "auto"]

        result = _prompt_stage_overrides(
            common_cpus=16, common_gres="gpu:a100:1", common_gmx="auto"
        )
        # No actual deviations → NVT not in result
        assert result == {}

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_empty_selection(self, mock_iq):
        """User confirms but selects no stages → empty dict."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.checkbox.return_value.ask.return_value = []

        result = _prompt_stage_overrides(
            common_cpus=16, common_gres="gpu:a100:1", common_gmx="auto"
        )
        assert result == {}

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_cancellation_at_confirm(self, mock_iq):
        """User cancels at confirm prompt → raises UserCancelledError."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = None

        with pytest.raises(UserCancelledError):
            _prompt_stage_overrides(common_cpus=16, common_gres=None, common_gmx="auto")

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_cancellation_at_checkbox(self, mock_iq):
        """User cancels at stage selection → raises UserCancelledError."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.checkbox.return_value.ask.return_value = None

        with pytest.raises(UserCancelledError):
            _prompt_stage_overrides(common_cpus=16, common_gres=None, common_gmx="auto")

    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_gmx_binary_override(self, mock_iq):
        """Override gmx_binary for a stage → stored in overrides."""
        mock_q = mock_iq.return_value
        mock_q.confirm.return_value.ask.return_value = True
        mock_q.checkbox.return_value.ask.return_value = ["Production"]
        mock_q.text.return_value.ask.side_effect = ["16", "gpu:a100:1", "gmx_mpi"]

        result = _prompt_stage_overrides(
            common_cpus=16, common_gres="gpu:a100:1", common_gmx="auto"
        )
        assert result["Production"] == {"gmx_binary": "gmx_mpi"}


class TestStageOverridesIntegration:
    """Tests that stage overrides flow through the full wizard paths."""

    @patch("mdfactory.orchestration.tui.discover_cluster", return_value=None)
    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_manual_path(self, mock_iq, _discover):
        """Manual fallback path produces config with stage overrides."""
        mock_q = mock_iq.return_value
        # confirm: proceed with manual → True, stage overrides → True
        mock_q.confirm.return_value.ask.side_effect = [True, True]
        # text prompts: account, partition, walltime, cpus, gres, mem, qos,
        # max_blocks, worker_init, then stage override prompts for EM
        mock_q.text.return_value.ask.side_effect = [
            "acc",
            "gpu",
            "2:00:00",
            "16",
            "gpu:a100:1",
            "32G",
            "",
            "4",
            "",
            # EM stage prompts: cpus, gres, gmx
            "32",
            "",
            "auto",
        ]
        mock_q.checkbox.return_value.ask.return_value = ["EM"]

        cfg = configure_slurm_interactive()
        assert cfg.stage_overrides == {
            "EM": {"cpus_per_node": 32, "gres": None},
        }

    @patch("mdfactory.orchestration.tui.discover_cluster")
    @patch("mdfactory.orchestration.tui._import_questionary")
    def test_stage_overrides_cluster_path(self, mock_iq, mock_discover):
        """Cluster-assisted path produces config with stage overrides."""
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
        # text prompts: gres, worker_init, constraint, then stage prompts
        mock_q.text.return_value.ask.side_effect = [
            "gpu:a100:1",
            "",
            "",
            # EM stage prompts: cpus, gres, gmx
            "32",
            "",
            "auto",
        ]
        # confirm: QOS → False, stage overrides → True
        mock_q.confirm.return_value.ask.side_effect = [False, True]
        mock_q.checkbox.return_value.ask.return_value = ["EM"]

        cfg = configure_slurm_interactive()
        assert cfg.stage_overrides == {
            "EM": {"cpus_per_node": 32, "gres": None},
        }


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

    def test_save_yaml_excludes_empty_overrides(self, tmp_path):
        """Empty stage_overrides should not appear in the YAML output."""
        cfg = SlurmExecutorConfig(
            account="acc",
            partition="cpu",
            walltime="1h",
            cpus_per_node=8,
        )
        out = tmp_path / "slurm.yaml"
        save_slurm_config_yaml(cfg, out)

        data = yaml.safe_load(out.read_text())
        assert "stage_overrides" not in data

    def test_save_yaml_includes_populated_overrides(self, tmp_path):
        """Non-empty stage_overrides should be written to the YAML output."""
        cfg = SlurmExecutorConfig(
            account="acc",
            partition="gpu",
            walltime="2h",
            cpus_per_node=16,
            gres="gpu:a100:1",
            stage_overrides={
                "EM": {"cpus_per_node": 32, "gres": None},
                "Production": {"cpus_per_node": 64},
            },
        )
        out = tmp_path / "slurm.yaml"
        save_slurm_config_yaml(cfg, out)

        data = yaml.safe_load(out.read_text())
        assert "stage_overrides" in data
        assert data["stage_overrides"]["EM"]["cpus_per_node"] == 32
        assert data["stage_overrides"]["EM"]["gres"] is None
        assert data["stage_overrides"]["Production"]["cpus_per_node"] == 64
