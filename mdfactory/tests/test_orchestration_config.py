# ABOUTME: Tests for orchestration executor configuration models
# ABOUTME: Validates config construction, serialization, and Parsl config generation
"""Tests for orchestration executor configuration."""

import pytest
import yaml

pytest.importorskip("parsl", reason="parsl not installed")

from mdfactory.orchestration.config import ExecutorConfig, SlurmExecutorConfig


def test_executor_config_defaults():
    """ExecutorConfig has sensible defaults."""
    cfg = ExecutorConfig()
    assert cfg.provider == "local"
    assert cfg.worker_init == ""
    assert cfg.working_directory is None
    assert cfg.max_workers_per_node == 1


def test_slurm_executor_config_requires_account():
    """SlurmExecutorConfig requires account field."""
    with pytest.raises(Exception):
        SlurmExecutorConfig()


def test_slurm_executor_config_defaults():
    """SlurmExecutorConfig has expected SLURM defaults."""
    cfg = SlurmExecutorConfig(account="my_account")
    assert cfg.provider == "slurm"
    assert cfg.account == "my_account"
    assert cfg.partition == "gpu"
    assert cfg.walltime == "2:00:00"
    assert cfg.nodes == 1
    assert cfg.cpus_per_node == 12
    assert cfg.gres is None
    assert cfg.mem is None


def test_executor_config_from_yaml_local(tmp_path):
    """from_yaml loads a local executor config."""
    cfg_data = {"provider": "local", "max_workers_per_node": 4}
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert isinstance(loaded, ExecutorConfig)
    assert not isinstance(loaded, SlurmExecutorConfig)
    assert loaded.max_workers_per_node == 4


def test_executor_config_from_yaml_slurm(tmp_path):
    """from_yaml loads a SLURM executor config."""
    cfg_data = {
        "provider": "slurm",
        "account": "hpc_team",
        "partition": "gpu-large",
        "walltime": "4:00:00",
        "cpus_per_node": 24,
        "gres": "gpu:l40s:2",
    }
    cfg_path = tmp_path / "slurm.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert isinstance(loaded, SlurmExecutorConfig)
    assert loaded.account == "hpc_team"
    assert loaded.partition == "gpu-large"
    assert loaded.cpus_per_node == 24
    assert loaded.gres == "gpu:l40s:2"


def test_executor_config_to_parsl_config():
    """to_parsl_config produces a valid Parsl Config for local provider."""
    cfg = ExecutorConfig(max_workers_per_node=2, worker_init="module load cuda/12.0")
    result = cfg.to_parsl_config()

    import parsl

    assert isinstance(result, parsl.Config)
    assert len(result.executors) == 1
    assert result.executors[0].label == "local"


def test_slurm_executor_config_to_parsl_config():
    """to_parsl_config produces a valid Parsl Config for SLURM provider."""
    cfg = SlurmExecutorConfig(account="my_account", gres="gpu:l40s:1")
    result = cfg.to_parsl_config()

    import parsl

    assert isinstance(result, parsl.Config)
    assert len(result.executors) == 1
    assert result.executors[0].label == "slurm"


def test_worker_init_propagation():
    """worker_init string is passed to the SLURM provider."""
    cfg = SlurmExecutorConfig(account="acc", worker_init="module load cuda/12.0")
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "module load cuda/12.0" in provider.worker_init


def test_gres_in_scheduler_options():
    """gres field becomes --gres in scheduler_options."""
    cfg = SlurmExecutorConfig(account="acc", gres="gpu:l40s:2")
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "#SBATCH --gres=gpu:l40s:2" in provider.scheduler_options


def test_mem_and_qos_in_scheduler_options():
    """mem and qos fields are included in scheduler_options."""
    cfg = SlurmExecutorConfig(account="acc", mem="64G", qos="high")
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "#SBATCH --mem=64G" in provider.scheduler_options
    assert "#SBATCH --qos=high" in provider.scheduler_options


def test_constraint_in_scheduler_options():
    """constraint field becomes --constraint in scheduler_options."""
    cfg = SlurmExecutorConfig(account="acc", constraint="a100")
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "#SBATCH --constraint=a100" in provider.scheduler_options


def test_raw_scheduler_options_appended():
    """Raw scheduler_options are appended after structured fields."""
    cfg = SlurmExecutorConfig(account="acc", gres="gpu:1", scheduler_options="#SBATCH --exclusive")
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "#SBATCH --gres=gpu:1" in provider.scheduler_options
    assert "#SBATCH --exclusive" in provider.scheduler_options


def test_config_yaml_roundtrip(tmp_path):
    """Config can be written to YAML and loaded back identically."""
    original = SlurmExecutorConfig(
        account="test_account",
        partition="gpu-dev",
        walltime="1:00:00",
        gres="gpu:l40s:1",
        worker_init="source activate env",
    )
    cfg_path = tmp_path / "roundtrip.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(original.model_dump(), f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert isinstance(loaded, SlurmExecutorConfig)
    assert loaded.account == original.account
    assert loaded.partition == original.partition
    assert loaded.walltime == original.walltime
    assert loaded.gres == original.gres
    assert loaded.worker_init == original.worker_init
