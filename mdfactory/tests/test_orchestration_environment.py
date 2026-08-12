# ABOUTME: Tests for EnvironmentConfig model
# ABOUTME: Validates compose_worker_init(), detect(), and YAML serialization
"""Tests for the EnvironmentConfig structured environment model."""

from pathlib import Path

import pytest
import yaml

from mdfactory.orchestration.environment import EnvironmentConfig


class TestComposeWorkerInit:
    """Tests for EnvironmentConfig.compose_worker_init()."""

    def test_empty_produces_empty_string(self):
        """All defaults produce an empty worker_init string."""
        env = EnvironmentConfig()
        assert env.compose_worker_init() == ""

    def test_modules_only(self):
        """Module list produces 'module load' commands."""
        env = EnvironmentConfig(modules=["gromacs/2024.3-gpu"])
        assert env.compose_worker_init() == "module load gromacs/2024.3-gpu"

    def test_multiple_modules(self):
        """Multiple modules are joined with semicolons."""
        env = EnvironmentConfig(modules=["cuda/12.2", "gromacs/2024.3-gpu"])
        result = env.compose_worker_init()
        assert result == "module load cuda/12.2; module load gromacs/2024.3-gpu"

    def test_pixi_manifest(self):
        """pixi_manifest produces a pixi shell-hook command."""
        env = EnvironmentConfig(pixi_manifest=Path("/project/root"))
        result = env.compose_worker_init()
        assert result == 'eval "$(pixi shell-hook --manifest-path /project/root -e default)"'

    def test_conda_env(self):
        """conda_env produces a conda activate command."""
        env = EnvironmentConfig(conda_env="myenv")
        assert env.compose_worker_init() == "conda activate myenv"

    def test_venv_path(self):
        """venv_path produces a source activate command."""
        env = EnvironmentConfig(venv_path=Path("/home/user/.venvs/md"))
        assert env.compose_worker_init() == "source /home/user/.venvs/md/bin/activate"

    def test_pixi_takes_precedence_over_conda(self):
        """When pixi_manifest and conda_env are both set, pixi wins."""
        env = EnvironmentConfig(
            pixi_manifest=Path("/project"),
            conda_env="myenv",
        )
        result = env.compose_worker_init()
        assert "pixi shell-hook" in result
        assert "conda" not in result

    def test_pixi_takes_precedence_over_venv(self):
        """When pixi_manifest and venv_path are both set, pixi wins."""
        env = EnvironmentConfig(
            pixi_manifest=Path("/project"),
            venv_path=Path("/venv"),
        )
        result = env.compose_worker_init()
        assert "pixi shell-hook" in result
        assert "source" not in result

    def test_conda_takes_precedence_over_venv(self):
        """When conda_env and venv_path are both set, conda wins."""
        env = EnvironmentConfig(
            conda_env="myenv",
            venv_path=Path("/venv"),
        )
        result = env.compose_worker_init()
        assert "conda activate myenv" in result
        assert "source" not in result

    def test_extra_init_only(self):
        """extra_init alone is returned verbatim."""
        env = EnvironmentConfig(extra_init="export OMP_NUM_THREADS=4")
        assert env.compose_worker_init() == "export OMP_NUM_THREADS=4"

    def test_combined_modules_pixi_extra(self):
        """All components are joined with semicolons in correct order."""
        env = EnvironmentConfig(
            modules=["gromacs/2024.3-gpu"],
            pixi_manifest=Path("/project"),
            extra_init="export OMP_NUM_THREADS=4",
        )
        result = env.compose_worker_init()
        parts = result.split("; ")
        assert parts[0] == "module load gromacs/2024.3-gpu"
        assert "pixi shell-hook" in parts[1]
        assert parts[2] == "export OMP_NUM_THREADS=4"

    def test_combined_modules_conda(self):
        """Modules + conda are correctly combined."""
        env = EnvironmentConfig(
            modules=["cuda/12.2"],
            conda_env="gromacs_env",
        )
        result = env.compose_worker_init()
        assert result == "module load cuda/12.2; conda activate gromacs_env"


class TestDetect:
    """Tests for EnvironmentConfig.detect() auto-detection."""

    def test_detect_pixi_from_env_var(self, monkeypatch, tmp_path):
        """PIXI_PROJECT_MANIFEST env var is detected."""
        manifest = tmp_path / "pixi.toml"
        manifest.touch()
        monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(manifest))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        env = EnvironmentConfig.detect()
        assert env.pixi_manifest == tmp_path
        assert env.conda_env is None
        assert env.venv_path is None

    def test_detect_pixi_from_filesystem(self, monkeypatch, tmp_path):
        """Pixi is detected from mdfactory's project root .pixi/ dir."""
        monkeypatch.delenv("PIXI_PROJECT_MANIFEST", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        # Create a fake mdfactory package structure
        fake_pkg = tmp_path / "mdfactory"
        fake_pkg.mkdir()
        (fake_pkg / "__init__.py").write_text("__file__ = __file__\n")
        pixi_env = tmp_path / ".pixi" / "envs" / "default"
        pixi_env.mkdir(parents=True)

        # Patch mdfactory.__file__ to point to our fake
        import mdfactory

        monkeypatch.setattr(mdfactory, "__file__", str(fake_pkg / "__init__.py"))

        env = EnvironmentConfig.detect()
        assert env.pixi_manifest == tmp_path

    def test_detect_conda_from_env(self, monkeypatch):
        """CONDA_PREFIX env var is detected when pixi is absent."""
        monkeypatch.delenv("PIXI_PROJECT_MANIFEST", raising=False)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/gromacs")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        # Ensure pixi filesystem detection also fails
        import mdfactory

        monkeypatch.setattr(mdfactory, "__file__", "/nonexistent/mdfactory/__init__.py")

        env = EnvironmentConfig.detect()
        assert env.pixi_manifest is None
        assert env.conda_env == "gromacs"
        assert env.venv_path is None

    def test_detect_venv_from_env(self, monkeypatch):
        """VIRTUAL_ENV env var is detected when pixi and conda are absent."""
        monkeypatch.delenv("PIXI_PROJECT_MANIFEST", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("VIRTUAL_ENV", "/home/user/.venvs/md")

        import mdfactory

        monkeypatch.setattr(mdfactory, "__file__", "/nonexistent/mdfactory/__init__.py")

        env = EnvironmentConfig.detect()
        assert env.pixi_manifest is None
        assert env.conda_env is None
        assert env.venv_path == Path("/home/user/.venvs/md")

    def test_detect_nothing(self, monkeypatch):
        """Clean environment produces empty EnvironmentConfig."""
        monkeypatch.delenv("PIXI_PROJECT_MANIFEST", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        import mdfactory

        monkeypatch.setattr(mdfactory, "__file__", "/nonexistent/mdfactory/__init__.py")

        env = EnvironmentConfig.detect()
        assert env.pixi_manifest is None
        assert env.conda_env is None
        assert env.venv_path is None
        assert env.modules == []
        assert env.extra_init == ""
        assert env.compose_worker_init() == ""

    def test_detect_pixi_takes_precedence_over_conda(self, monkeypatch, tmp_path):
        """When both PIXI_PROJECT_MANIFEST and CONDA_PREFIX are set, pixi wins."""
        manifest = tmp_path / "pixi.toml"
        manifest.touch()
        monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(manifest))
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/other")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        env = EnvironmentConfig.detect()
        assert env.pixi_manifest == tmp_path
        assert env.conda_env is None

    def test_detect_with_overrides(self, monkeypatch, tmp_path):
        """Keyword overrides take precedence over auto-detection."""
        manifest = tmp_path / "pixi.toml"
        manifest.touch()
        monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(manifest))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        env = EnvironmentConfig.detect(modules=["gromacs/2024"], extra_init="ulimit -s unlimited")
        assert env.pixi_manifest == tmp_path
        assert env.modules == ["gromacs/2024"]
        assert env.extra_init == "ulimit -s unlimited"


class TestSerialization:
    """Tests for EnvironmentConfig YAML serialization."""

    def test_paths_serialize_as_strings(self):
        """Path fields serialize as plain strings for YAML compatibility."""
        env = EnvironmentConfig(
            pixi_manifest=Path("/project/root"),
            venv_path=Path("/home/user/.venvs/md"),
        )
        dumped = env.model_dump()
        assert isinstance(dumped["pixi_manifest"], str)
        assert isinstance(dumped["venv_path"], str)
        # Round-trips through yaml without RepresenterError
        yaml.safe_dump(dumped)

    def test_yaml_roundtrip(self):
        """EnvironmentConfig survives YAML round-trip."""
        env = EnvironmentConfig(
            modules=["gromacs/2024.3-gpu", "cuda/12.2"],
            pixi_manifest=Path("/project"),
            extra_init="export OMP_NUM_THREADS=4",
        )
        dumped = yaml.safe_dump(env.model_dump())
        loaded = yaml.safe_load(dumped)
        restored = EnvironmentConfig(**loaded)
        assert restored.modules == env.modules
        assert restored.pixi_manifest == env.pixi_manifest
        assert restored.extra_init == env.extra_init
        assert restored.compose_worker_init() == env.compose_worker_init()

    def test_none_paths_excluded_with_exclude_none(self):
        """None-valued path fields are excluded with exclude_none=True."""
        env = EnvironmentConfig(modules=["gromacs/2024"])
        dumped = env.model_dump(exclude_none=True)
        assert "pixi_manifest" not in dumped
        assert "venv_path" not in dumped
        assert "conda_env" not in dumped


class TestSaveLoadYaml:
    """Tests for EnvironmentConfig.save_yaml() and from_yaml()."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_yaml then from_yaml produces equivalent config."""
        env = EnvironmentConfig(
            modules=["gromacs/2024.3-gpu"],
            pixi_manifest=Path("/project"),
            extra_init="export OMP_NUM_THREADS=4",
        )
        path = tmp_path / "env.yaml"
        env.save_yaml(path)
        loaded = EnvironmentConfig.from_yaml(path)
        assert loaded.modules == env.modules
        assert loaded.pixi_manifest == env.pixi_manifest
        assert loaded.extra_init == env.extra_init
        assert loaded.compose_worker_init() == env.compose_worker_init()

    def test_save_omits_empty_fields(self, tmp_path):
        """Empty modules and extra_init are not written to YAML."""
        env = EnvironmentConfig(conda_env="gromacs")
        path = tmp_path / "env.yaml"
        env.save_yaml(path)
        data = yaml.safe_load(path.read_text())
        assert "modules" not in data
        assert "extra_init" not in data
        assert data["conda_env"] == "gromacs"

    def test_save_creates_parent_directories(self, tmp_path):
        """save_yaml creates parent directories if they don't exist."""
        path = tmp_path / "deep" / "nested" / "env.yaml"
        env = EnvironmentConfig(modules=["mod1"])
        env.save_yaml(path)
        assert path.is_file()

    def test_from_yaml_file_not_found(self, tmp_path):
        """from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            EnvironmentConfig.from_yaml(tmp_path / "nope.yaml")

    def test_from_yaml_invalid_content(self, tmp_path):
        """from_yaml raises ValueError for non-mapping YAML."""
        path = tmp_path / "bad.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="empty or invalid"):
            EnvironmentConfig.from_yaml(path)


class TestLoadGlobal:
    """Tests for EnvironmentConfig.load_global()."""

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        """load_global returns None when no global config exists."""
        monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path))
        result = EnvironmentConfig.load_global()
        assert result is None

    def test_loads_existing_global_config(self, tmp_path, monkeypatch):
        """load_global returns config when global file exists."""
        monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path))
        env = EnvironmentConfig(
            modules=["gromacs/2024"],
            conda_env="md",
        )
        env.save_yaml(tmp_path / "environment.yaml")
        result = EnvironmentConfig.load_global()
        assert result is not None
        assert result.modules == ["gromacs/2024"]
        assert result.conda_env == "md"

    def test_raises_on_corrupt_file(self, tmp_path, monkeypatch):
        """load_global raises ValueError if file exists but cannot be parsed."""
        monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path))
        (tmp_path / "environment.yaml").write_text("- not a mapping\n")
        with pytest.raises(ValueError, match="empty or invalid"):
            EnvironmentConfig.load_global()
