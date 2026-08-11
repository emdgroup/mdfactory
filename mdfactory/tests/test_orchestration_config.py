# ABOUTME: Tests for orchestration executor configuration models
# ABOUTME: Validates config construction, serialization, and Parsl config generation
"""Tests for orchestration executor configuration."""

import pytest
import yaml

pytest.importorskip("parsl", reason="parsl not installed")

from mdfactory.orchestration.config import ExecutorConfig, SlurmExecutorConfig
from mdfactory.orchestration.environment import EnvironmentConfig


def test_executor_config_defaults():
    """ExecutorConfig has sensible defaults."""
    cfg = ExecutorConfig()
    assert cfg.provider == "local"
    assert cfg.environment == EnvironmentConfig()
    assert cfg.working_directory is None
    assert cfg.max_workers_per_node == 1


def test_slurm_executor_config_account_optional():
    """SlurmExecutorConfig works without an account for clusters that don't need one.

    account defaults to "" so that clusters without account-based scheduling
    (or users relying on their default SLURM account) can write a YAML config
    without an account field.  Parsl omits --account from the sbatch script
    when the string is empty.
    """
    cfg = SlurmExecutorConfig()
    assert cfg.account == ""


def test_slurm_executor_config_defaults():
    """SlurmExecutorConfig has expected SLURM defaults."""
    cfg = SlurmExecutorConfig(account="my_account")
    assert cfg.provider == "slurm"
    assert cfg.account == "my_account"
    assert cfg.partition == "cpu"
    assert cfg.walltime == "02:00:00"
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
    env = EnvironmentConfig(modules=["cuda/12.0"])
    cfg = ExecutorConfig(max_workers_per_node=2, environment=env)
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


def test_environment_propagation():
    """environment.compose_worker_init() is passed to the SLURM provider."""
    env = EnvironmentConfig(modules=["cuda/12.0"])
    cfg = SlurmExecutorConfig(account="acc", environment=env)
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "module load cuda/12.0" in provider.worker_init


def test_environment_combined_propagation():
    """Multiple environment fields are composed into the provider worker_init."""
    env = EnvironmentConfig(
        modules=["gromacs/2024.3-gpu"],
        pixi_manifest="/project",
    )
    cfg = SlurmExecutorConfig(account="acc", environment=env)
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "module load gromacs/2024.3-gpu" in provider.worker_init
    assert "pixi shell-hook" in provider.worker_init


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


def test_slurm_executor_config_inherits_base_slurm_config():
    """SlurmExecutorConfig unifies with the shared BaseSlurmConfig hierarchy.

    Guards against re-introducing a divergent third copy of the SLURM fields
    (issue #20): account/partition/qos/constraint and from_cluster() must come
    from BaseSlurmConfig, not be redeclared here.
    """
    from mdfactory.performance.slurm_config import BaseSlurmConfig

    assert issubclass(SlurmExecutorConfig, BaseSlurmConfig)
    # from_cluster is inherited, not reimplemented.
    assert SlurmExecutorConfig.from_cluster.__func__ is BaseSlurmConfig.from_cluster.__func__
    # The shared SLURM fields are not redeclared on the subclass itself.
    own_fields = set(vars(SlurmExecutorConfig).get("__annotations__", {}))
    assert not ({"account", "qos", "constraint"} & own_fields)


def test_slurm_executor_config_is_mutable():
    """Despite BaseSlurmConfig being frozen, the executor config stays mutable."""
    cfg = SlurmExecutorConfig(account="acc")
    cfg.partition = "gpu"  # must not raise (frozen=False)
    assert cfg.partition == "gpu"


def test_raw_scheduler_options_appended():
    """Raw scheduler_options are appended after structured fields."""
    cfg = SlurmExecutorConfig(account="acc", gres="gpu:1", scheduler_options="#SBATCH --exclusive")
    result = cfg.to_parsl_config()

    provider = result.executors[0].provider
    assert "#SBATCH --gres=gpu:1" in provider.scheduler_options
    assert "#SBATCH --exclusive" in provider.scheduler_options


def test_run_dir_default_and_expanduser():
    """run_dir defaults under the home dir and expands ``~`` in supplied values."""
    cfg = ExecutorConfig()
    assert str(cfg.run_dir).endswith("/.parsl/mdfactory")
    assert "~" not in str(cfg.run_dir)

    cfg2 = ExecutorConfig(run_dir="~/scratch/parsl")
    assert "~" not in str(cfg2.run_dir)
    assert str(cfg2.run_dir).endswith("/scratch/parsl")

    # run_dir flows into the Parsl Config
    assert str(cfg.run_dir) == cfg.to_parsl_config().run_dir


def test_paths_serialize_as_strings():
    """run_dir / working_directory serialize as plain strings for YAML."""
    cfg = ExecutorConfig(working_directory="/tmp/work")
    dumped = cfg.model_dump()
    assert isinstance(dumped["run_dir"], str)
    assert isinstance(dumped["working_directory"], str)
    # Round-trips through yaml without RepresenterError
    yaml.safe_dump(dumped)


def test_available_accelerators_wired_local():
    """available_accelerators is forwarded to the local HighThroughputExecutor."""
    cfg = ExecutorConfig(max_workers_per_node=2, available_accelerators=2)
    executor = cfg.to_parsl_config().executors[0]
    # Parsl normalises an int count into a list of device IDs.
    assert executor.available_accelerators == ["0", "1"]


def test_available_accelerators_wired_slurm():
    """available_accelerators is forwarded to the SLURM HighThroughputExecutor."""
    cfg = SlurmExecutorConfig(account="acc", available_accelerators=["0", "1"])
    executor = cfg.to_parsl_config().executors[0]
    assert executor.available_accelerators == ["0", "1"]


def test_launch_options_sets_srun_launcher():
    """launch_options wires a SrunLauncher with the given overrides."""
    from parsl.launchers import SrunLauncher

    cfg = SlurmExecutorConfig(
        account="acc",
        launch_options="--cpu-bind=cores --distribution=block:block",
    )
    provider = cfg.to_parsl_config().executors[0].provider
    assert isinstance(provider.launcher, SrunLauncher)
    assert provider.launcher.overrides == "--cpu-bind=cores --distribution=block:block"


def test_no_launch_options_keeps_default_launcher():
    """Without launch_options, Parsl's default launcher is left untouched."""
    from parsl.launchers import SrunLauncher

    cfg = SlurmExecutorConfig(account="acc")
    provider = cfg.to_parsl_config().executors[0].provider
    assert not isinstance(provider.launcher, SrunLauncher)


def test_launch_options_roundtrip_yaml(tmp_path):
    """launch_options survives a YAML round-trip."""
    original = SlurmExecutorConfig(account="acc", launch_options="--cpu-bind=cores")
    cfg_path = tmp_path / "launch.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(original.model_dump(), f)
    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert loaded.launch_options == "--cpu-bind=cores"


def test_config_yaml_roundtrip(tmp_path):
    """Config can be written to YAML and loaded back identically."""
    env = EnvironmentConfig(modules=["gromacs/2024"], extra_init="source activate env")
    original = SlurmExecutorConfig(
        account="test_account",
        partition="gpu-dev",
        walltime="1:00:00",
        gres="gpu:l40s:1",
        environment=env,
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
    assert loaded.environment.modules == ["gromacs/2024"]
    assert loaded.environment.extra_init == "source activate env"


# === EnvironmentConfig integration tests ===


def test_environment_field_defaults_to_empty():
    """ExecutorConfig defaults environment to an empty EnvironmentConfig."""
    cfg = ExecutorConfig()
    assert cfg.environment == EnvironmentConfig()
    assert cfg.environment.compose_worker_init() == ""


def test_to_parsl_config_uses_environment():
    """to_parsl_config passes environment-composed string to the provider."""
    env = EnvironmentConfig(modules=["gromacs/2024.3-gpu"])
    cfg = ExecutorConfig(environment=env)
    result = cfg.to_parsl_config()
    provider = result.executors[0].provider
    assert "module load gromacs/2024.3-gpu" in provider.worker_init


def test_from_yaml_environment_section(tmp_path):
    """YAML with environment section loads correctly."""
    cfg_data = {
        "provider": "slurm",
        "account": "acc",
        "environment": {
            "modules": ["gromacs/2024.3-gpu"],
            "pixi_manifest": "/project/root",
            "extra_init": "ulimit -s unlimited",
        },
    }
    cfg_path = tmp_path / "new.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert isinstance(loaded, SlurmExecutorConfig)
    assert loaded.environment.modules == ["gromacs/2024.3-gpu"]
    assert str(loaded.environment.pixi_manifest) == "/project/root"
    assert loaded.environment.extra_init == "ulimit -s unlimited"
    worker_init = loaded.environment.compose_worker_init()
    assert "gromacs" in worker_init
    assert "pixi shell-hook" in worker_init


def test_from_yaml_no_environment(tmp_path, monkeypatch):
    """YAML without environment section gets empty EnvironmentConfig when no global exists."""
    # Point config dir to an empty tmp dir so no global file is found
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path / "config"))
    cfg_data = {"provider": "slurm", "account": "acc"}
    cfg_path = tmp_path / "minimal.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert loaded.environment == EnvironmentConfig()
    assert loaded.environment.compose_worker_init() == ""


def test_from_yaml_auto_loads_global_environment(tmp_path, monkeypatch):
    """YAML without environment section auto-loads from global config."""
    # Create a global environment config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(config_dir))
    global_env = EnvironmentConfig(
        modules=["gromacs/2024"],
        conda_env="md",
    )
    global_env.save_yaml(config_dir / "environment.yaml")

    # Load a SLURM YAML with no environment section
    cfg_data = {"provider": "slurm", "account": "acc"}
    cfg_path = tmp_path / "slurm.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert loaded.environment.modules == ["gromacs/2024"]
    assert loaded.environment.conda_env == "md"
    assert "gromacs/2024" in loaded.environment.compose_worker_init()


def test_from_yaml_explicit_environment_overrides_global(tmp_path, monkeypatch):
    """YAML with explicit environment section ignores global config."""
    # Create a global environment config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(config_dir))
    global_env = EnvironmentConfig(modules=["gromacs/2024"])
    global_env.save_yaml(config_dir / "environment.yaml")

    # Load a YAML that has its own environment section
    cfg_data = {
        "provider": "slurm",
        "account": "acc",
        "environment": {"modules": ["gromacs/2025"], "conda_env": "custom"},
    }
    cfg_path = tmp_path / "slurm.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert loaded.environment.modules == ["gromacs/2025"]
    assert loaded.environment.conda_env == "custom"


# === Decision 6: Per-stage resource config tests ===


def test_stage_overrides_empty_by_default():
    """SlurmExecutorConfig has empty stage_overrides by default."""
    cfg = SlurmExecutorConfig(account="acct")
    assert cfg.stage_overrides == {}


def test_stage_overrides_loaded_from_yaml(tmp_path):
    """stage_overrides field is preserved through YAML roundtrip."""
    cfg = SlurmExecutorConfig(
        account="acct",
        cpus_per_node=12,
        stage_overrides={"EM": {"cpus_per_node": 4, "gres": None}},
    )
    yaml_path = tmp_path / "cfg.yaml"
    with open(yaml_path, "w") as f:
        import yaml

        yaml.safe_dump(cfg.model_dump(), f)
    loaded = SlurmExecutorConfig.from_yaml(yaml_path)
    assert loaded.stage_overrides == {"EM": {"cpus_per_node": 4, "gres": None}}


def test_get_stage_config_no_override_returns_self():
    """get_stage_config returns self when no override defined for stage."""
    cfg = SlurmExecutorConfig(account="acct", cpus_per_node=12)
    result = cfg.get_stage_config("NVT")
    assert result is cfg


def test_get_stage_config_applies_override():
    """get_stage_config merges override fields into a copy."""
    cfg = SlurmExecutorConfig(
        account="acct",
        cpus_per_node=12,
        gres="gpu:l40s:1",
        stage_overrides={"EM": {"cpus_per_node": 4, "gres": None}},
    )
    em_cfg = cfg.get_stage_config("EM")
    # Override applied
    assert em_cfg.cpus_per_node == 4
    assert em_cfg.gres is None
    # Original unchanged
    assert cfg.cpus_per_node == 12
    assert cfg.gres == "gpu:l40s:1"


def test_get_stage_config_inherits_non_overridden_fields():
    """get_stage_config preserves base fields not in the override dict."""
    cfg = SlurmExecutorConfig(
        account="my_account",
        partition="gpu",
        cpus_per_node=12,
        gres="gpu:l40s:1",
        stage_overrides={"EM": {"cpus_per_node": 4}},
    )
    em_cfg = cfg.get_stage_config("EM")
    # Overridden
    assert em_cfg.cpus_per_node == 4
    # Inherited from base
    assert em_cfg.account == "my_account"
    assert em_cfg.partition == "gpu"
    assert em_cfg.gres == "gpu:l40s:1"


def test_get_stage_config_does_not_mutate_original():
    """get_stage_config returns a distinct object and leaves original intact."""
    cfg = SlurmExecutorConfig(
        account="acct",
        cpus_per_node=12,
        stage_overrides={"Production": {"cpus_per_node": 24}},
    )
    prod_cfg = cfg.get_stage_config("Production")
    assert prod_cfg is not cfg
    assert prod_cfg.cpus_per_node == 24
    assert cfg.cpus_per_node == 12


def test_get_stage_config_multiple_stages():
    """Each stage can have independent overrides."""
    cfg = SlurmExecutorConfig(
        account="acct",
        cpus_per_node=12,
        gres="gpu:l40s:1",
        stage_overrides={
            "EM": {"cpus_per_node": 4, "gres": None},
            "Production": {"cpus_per_node": 24},
        },
    )
    em = cfg.get_stage_config("EM")
    nvt = cfg.get_stage_config("NVT")
    prod = cfg.get_stage_config("Production")

    assert em.cpus_per_node == 4
    assert em.gres is None
    assert nvt is cfg  # no override, returns self
    assert prod.cpus_per_node == 24
    assert prod.gres == "gpu:l40s:1"  # inherited


# === gmx_binary field tests ===


def test_executor_config_gmx_binary_default():
    """gmx_binary defaults to 'auto' on both local and SLURM configs."""
    local = ExecutorConfig()
    assert local.gmx_binary == "auto"

    slurm = SlurmExecutorConfig(account="acct")
    assert slurm.gmx_binary == "auto"


def test_executor_config_gmx_binary_explicit_gmx():
    """gmx_binary accepts 'gmx' for thread-MPI builds."""
    cfg = ExecutorConfig(gmx_binary="gmx")
    assert cfg.gmx_binary == "gmx"


def test_executor_config_gmx_binary_explicit_gmx_mpi():
    """gmx_binary accepts 'gmx_mpi' for pure-MPI builds."""
    cfg = ExecutorConfig(gmx_binary="gmx_mpi")
    assert cfg.gmx_binary == "gmx_mpi"


def test_executor_config_gmx_binary_invalid():
    """Invalid gmx_binary value raises a Pydantic validation error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExecutorConfig(gmx_binary="gmx_openmp")  # not a valid literal


def test_slurm_executor_config_gmx_binary_stage_override():
    """gmx_binary can be overridden per-stage via stage_overrides."""
    cfg = SlurmExecutorConfig(
        account="acct",
        gmx_binary="auto",
        stage_overrides={"EM": {"gmx_binary": "gmx_mpi"}},
    )
    em_cfg = cfg.get_stage_config("EM")
    nvt_cfg = cfg.get_stage_config("NVT")

    assert em_cfg.gmx_binary == "gmx_mpi"
    assert nvt_cfg.gmx_binary == "auto"  # inherited, no override


def test_executor_config_gmx_binary_yaml_roundtrip(tmp_path):
    """gmx_binary survives a YAML round-trip."""
    original = SlurmExecutorConfig(account="acct", gmx_binary="gmx_mpi")
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    with open(cfg_path, "w") as f:
        yaml.safe_dump(original.model_dump(), f)

    loaded = ExecutorConfig.from_yaml(cfg_path)
    assert loaded.gmx_binary == "gmx_mpi"
