# ABOUTME: Tests for the unified settings module
# ABOUTME: Covers defaults, env overrides, reload, DB path methods, and import safety
"""Tests for the unified settings module."""

from pathlib import Path

import pytest

from mdfactory.settings import Settings, get_config_dir, get_data_dir


def test_import_does_not_crash():
    """Importing settings should not raise or do file I/O beyond reading config."""
    from mdfactory.settings import settings  # noqa: F811

    assert settings is not None
    assert hasattr(settings, "config")


def test_defaults_loaded():
    """Settings defaults should include all expected sections."""
    s = Settings()
    assert "database" in s.config
    assert "sqlite" in s.config
    assert "csv" in s.config
    assert "foundry" in s.config
    assert "cgenff" in s.config
    assert "gromacs" in s.config
    assert "storage" in s.config
    assert s.config["database"]["TYPE"] == "sqlite"


def test_parameter_store_default():
    """Default parameter_store should point into the data directory."""
    s = Settings()
    data_dir = get_data_dir()
    assert str(data_dir) in str(s.parameter_store)


def test_parameter_store_setter():
    """parameter_store setter should override the default."""
    s = Settings()
    override = Path("/tmp/test_params")
    s.parameter_store = override
    assert s.parameter_store == override
    # Reset does not affect the class-level default
    s._parameter_store_override = None


def test_cgenff_dir_default():
    """Default cgenff_dir resolves from empty SILCSBIODIR."""
    s = Settings()
    # Empty SILCSBIODIR resolves to cwd (or .)
    assert s.cgenff_dir is not None


def test_gromacs_env_prepends_forcefield_dir(tmp_path, monkeypatch):
    """gromacs_env should add the configured force field directory to GMXLIB."""
    monkeypatch.setenv("MDFACTORY_DATA_DIR", str(tmp_path / "data"))
    s = Settings()
    env = s.gromacs_env({"GMXLIB": "/existing/path"})
    assert env["GMXLIB"].split(":") == [
        str(tmp_path / "data" / "forcefields"),
        "/existing/path",
    ]


def test_config_template_documents_gromacs_section():
    """The sample config should expose all user-facing GROMACS settings."""
    template = Path("config_templates/config.ini").read_text()
    assert "[gromacs]" in template
    assert "GMX_PATH =" in template
    assert "FORCEFIELD_DIR =" in template


def test_reload():
    """reload() should re-read config from disk."""
    s = Settings()
    s.parameter_store = Path("/tmp/override")
    assert s.parameter_store == Path("/tmp/override")
    s.reload()
    assert s.parameter_store != Path("/tmp/override")


def test_is_configured_false_by_default(tmp_path, monkeypatch):
    """is_configured returns False when no user config file exists."""
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path / "nonexistent"))
    s = Settings()
    assert s.is_configured() is False


def test_is_configured_true_when_file_exists(tmp_path, monkeypatch):
    """is_configured returns True when user config file exists."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.ini").write_text("[database]\nTYPE = sqlite\n")
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(config_dir))
    s = Settings()
    assert s.is_configured() is True


def test_env_override_config_dir(tmp_path, monkeypatch):
    """MDFACTORY_CONFIG_DIR env var should override the config directory."""
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path))
    assert get_config_dir() == tmp_path


def test_env_override_data_dir(tmp_path, monkeypatch):
    """MDFACTORY_DATA_DIR env var should override the data directory."""
    monkeypatch.setenv("MDFACTORY_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path


def test_env_override_config_file(tmp_path, monkeypatch):
    """MDFACTORY_CONFIG env var should load config from that file."""
    config_file = tmp_path / "custom.ini"
    config_file.write_text("[database]\nTYPE = csv\n")
    monkeypatch.setenv("MDFACTORY_CONFIG", str(config_file))
    s = Settings()
    assert s.config["database"]["TYPE"] == "csv"


def test_get_db_path_run_database():
    """get_db_path should return a path for RUN_DATABASE."""
    s = Settings()
    path = s.get_db_path("RUN_DATABASE", "sqlite")
    assert "runs.db" in path


def test_get_db_path_analysis_table():
    """get_db_path for ANALYSIS_* returns the analysis db path."""
    s = Settings()
    path = s.get_db_path("ANALYSIS_AREA_PER_LIPID", "sqlite")
    assert "analysis.db" in path


def test_get_db_path_unsupported_type():
    """get_db_path raises for unsupported db_type."""
    s = Settings()
    with pytest.raises(ValueError, match="Unsupported db_type"):
        s.get_db_path("RUN_DATABASE", "postgres")


def test_get_csv_path_run_database():
    """get_csv_path returns a CSV path for static tables."""
    s = Settings()
    path = s.get_csv_path("RUN_DATABASE")
    assert path.endswith("runs.csv")


def test_get_csv_path_dynamic_analysis():
    """get_csv_path returns directory-based path for analysis tables."""
    s = Settings()
    path = s.get_csv_path("ANALYSIS_OVERVIEW")
    assert path.endswith("ANALYSIS_OVERVIEW.csv")


def test_get_foundry_path_analysis():
    """get_foundry_path returns correct path for analysis tables."""
    s = Settings()
    path = s.get_foundry_path("ANALYSIS_AREA_PER_LIPID")
    assert "area_per_lipid" in path


def test_get_foundry_path_artifact():
    """get_foundry_path returns correct path for artifact tables."""
    s = Settings()
    path = s.get_foundry_path("ARTIFACT_BILAYER_MAPS")
    assert "bilayer_maps" in path


def test_resolve_local_path_absolute():
    """_resolve_local_path returns absolute paths unchanged."""
    s = Settings()
    assert s._resolve_local_path("/absolute/path") == "/absolute/path"


def test_resolve_local_path_relative(tmp_path):
    """_resolve_local_path resolves relative paths against config base dir."""
    s = Settings()
    s._config_base_dir = tmp_path
    result = s._resolve_local_path("relative/path")
    assert result == str((tmp_path / "relative/path").resolve())


def test_resolve_local_path_empty():
    """_resolve_local_path returns empty string for empty input."""
    s = Settings()
    assert s._resolve_local_path("") == ""
