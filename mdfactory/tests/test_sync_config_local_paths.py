# ABOUTME: Tests for local path normalization and resolution in config wizard
# ABOUTME: Covers relative path handling, sqlite/CSV path config, and wizard db init

import configparser

from mdfactory.settings import Settings
from mdfactory.utils.sync_config import (
    configure_sqlite_paths,
    normalize_local_path,
    run_config_wizard,
)


def test_normalize_local_path_converts_relative_to_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    normalized = normalize_local_path("runs.db")
    assert normalized == str((tmp_path / "runs.db").resolve())


def test_configure_sqlite_paths_writes_normalized_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "sqlite": {
                "RUN_DB_PATH": "~/.mdfactory/data/runs.db",
                "ANALYSIS_DB_PATH": "~/.mdfactory/data/analysis.db",
            }
        }
    )

    answers = iter(["runs.db", "analysis.db"])
    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.text",
        lambda *a, **kw: type("Q", (), {"ask": lambda self: next(answers)})(),
    )

    configure_sqlite_paths(config)

    assert config["sqlite"]["RUN_DB_PATH"] == str((tmp_path / "runs.db").resolve())
    assert config["sqlite"]["ANALYSIS_DB_PATH"] == str((tmp_path / "analysis.db").resolve())


def test_config_resolves_relative_local_paths_against_config_directory(monkeypatch, tmp_path):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    def patched_load_config(self):
        self.config = configparser.ConfigParser()
        for section, options in self.DEFAULT_CONFIG.items():
            self.config[section] = {}
            for key, value in options.items():
                self.config[section][key] = str(value)

        self._config_base_dir = config_dir
        self.config["sqlite"]["RUN_DB_PATH"] = "runs.db"
        self.config["csv"]["RUN_DB_PATH"] = "runs.csv"
        self.config["csv"]["ANALYSIS_DB_PATH"] = "analysis_tables"

    monkeypatch.setattr(Settings, "_load_config", patched_load_config)
    cfg = Settings()

    assert cfg.get_db_path("RUN_DATABASE", "sqlite") == str((config_dir / "runs.db").resolve())
    assert cfg.get_csv_path("RUN_DATABASE") == str((config_dir / "runs.csv").resolve())
    assert cfg.get_csv_path("ANALYSIS_AREA_PER_LIPID") == str(
        (config_dir / "analysis_tables" / "ANALYSIS_AREA_PER_LIPID.csv").resolve()
    )


def test_run_config_wizard_can_initialize_databases(monkeypatch, tmp_path):
    monkeypatch.setenv("MDFACTORY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MDFACTORY_DATA_DIR", str(tmp_path / "data"))
    called = {"value": False}

    def fake_initialize():
        called["value"] = True
        return {
            "systems": {"RUN_DATABASE": True},
            "analysis": {"ANALYSIS_OVERVIEW": True},
            "artifacts": {"ARTIFACT_LAST_FRAME_PDB": False},
        }

    monkeypatch.setattr(
        "mdfactory.utils.sync_config.initialize_configured_databases",
        fake_initialize,
    )

    # Mock questionary prompts in order:
    # 1. confirm GROMACS -> True
    # 2. text GMX_PATH -> ""
    # 3. text FORCEFIELD_DIR -> default
    # 4. confirm force field download -> False
    # 5. confirm CGenFF -> False
    # 6. text parameter store -> default
    # 7. select backend -> "sqlite"
    # 8. text RUN_DB_PATH -> "runs.db"
    # 9. text ANALYSIS_DB_PATH -> "analysis.db"
    # 10. confirm initialize -> True
    class FakeQuestion:
        def __init__(self, value):
            self._value = value

        def ask(self):
            return self._value

    answers_confirm = iter([True, False, False, True])
    answers_text = iter(
        [
            "",
            str(tmp_path / "data" / "forcefields"),
            str(tmp_path / "data" / "parameters"),
            "runs.db",
            "analysis.db",
        ]
    )
    answers_select = iter(["sqlite"])

    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.confirm",
        lambda *a, **kw: FakeQuestion(next(answers_confirm)),
    )
    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.text",
        lambda *a, **kw: FakeQuestion(next(answers_text)),
    )
    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.select",
        lambda *a, **kw: FakeQuestion(next(answers_select)),
    )

    run_config_wizard()

    assert called["value"] is True
    config_path = tmp_path / "config.ini"
    assert config_path.exists()

    config = configparser.ConfigParser()
    config.read(config_path)
    assert config["gromacs"]["GMX_PATH"] == ""
    assert config["gromacs"]["FORCEFIELD_DIR"] == str(tmp_path / "data" / "forcefields")
