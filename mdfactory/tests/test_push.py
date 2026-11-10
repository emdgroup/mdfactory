# ABOUTME: Tests for the push workflow including simulation folder discovery,
# ABOUTME: status determination, database initialization, and record upload.
"""Tests for the push workflow including simulation folder discovery,."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mdfactory.cli import sync_init_systems, sync_push_systems
from mdfactory.models.input import BuildInput
from mdfactory.settings import Settings
from mdfactory.utils.data_manager import DataManager
from mdfactory.utils.db_operations import query_existing_hashes
from mdfactory.utils.push import (
    determine_simulation_status,
    discover_simulation_folders,
    find_yaml_in_folder,
    init_systems_database,
    load_models_from_csv,
    prepare_upload_data,
    push_systems,
    search_folders_for_hash,
    upload_simulations,
)

# Import shared helpers from conftest
from .conftest import _create_simulation_folder, _write_csv_with_models, make_test_record


def _record(hash_value: str, status: str = "build") -> dict[str, str]:
    """Create a test record. Delegates to shared helper."""
    return make_test_record(hash_value, status)


def _create_folder_with_yaml(tmp_path: Path, folder_name: str, yaml_name: str, build_input):
    """Create a folder with a YAML file containing BuildInput data."""
    folder = tmp_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    yaml_path = folder / yaml_name
    with open(yaml_path, "w") as handle:
        yaml.safe_dump(build_input.model_dump(), handle)
    return folder


def _create_summary_yaml(tmp_path: Path, directories: list[Path]) -> Path:
    """Create a summary YAML file with system_directory list."""
    summary_path = tmp_path / "summary.yaml"
    summary_data = {"system_directory": [str(d) for d in directories]}
    with open(summary_path, "w") as handle:
        yaml.safe_dump(summary_data, handle)
    return summary_path


def test_discover_simulation_folders_csv_root(tmp_path):
    csv_path, models = _write_csv_with_models(tmp_path)
    root_dir = tmp_path / "hashes"
    root_dir.mkdir()

    for model in models:
        folder = root_dir / model.hash
        folder.mkdir()
        yaml_path = folder / f"{model.hash}.yaml"
        with open(yaml_path, "w") as handle:
            yaml.safe_dump(model.model_dump(), handle)

    simulations = discover_simulation_folders(csv=csv_path, csv_root=root_dir)
    discovered_hashes = sorted(build_input.hash for _, build_input in simulations)
    expected_hashes = sorted(model.hash for model in models)
    assert discovered_hashes == expected_hashes


def test_discover_simulation_folders_csv_root_missing(tmp_path):
    csv_path, _ = _write_csv_with_models(tmp_path)
    missing_root = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        discover_simulation_folders(csv=csv_path, csv_root=missing_root)


def test_upload_simulations_force_overwrites(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    existing = _record("HASH1", status="build")
    dm.save_data(existing)

    replacement = _record("HASH1", status="completed")
    upload_simulations([replacement], force=True)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"
    assert df.iloc[0]["status"] == "completed"


def test_upload_simulations_diff_skips_existing(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    existing = _record("HASH_EXISTING", status="production")
    new_record = _record("HASH_NEW", status="build")
    dm.save_data(existing)

    upload_simulations([existing, new_record], diff=True)

    df = dm.load_data()
    assert len(df) == 2
    hashes = set(df["hash"].values)
    assert hashes == {"HASH_EXISTING", "HASH_NEW"}


# GROUP 1: find_yaml_in_folder() tests


def test_find_yaml_hash_priority(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]

    folder = _create_folder_with_yaml(tmp_path, "ABC123", "ABC123.yaml", build_input)
    (folder / "build.yaml").write_text("dummy: data")
    (folder / "other.yaml").write_text("dummy: data")

    result = find_yaml_in_folder(folder)
    assert result is not None
    assert result.name == "ABC123.yaml"


def test_find_yaml_build_priority(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]

    folder = _create_folder_with_yaml(tmp_path, "ABC123", "build.yaml", build_input)
    (folder / "other.yaml").write_text("dummy: data")
    (folder / "zzz.yaml").write_text("dummy: data")

    result = find_yaml_in_folder(folder)
    assert result is not None
    assert result.name == "build.yaml"


def test_find_yaml_alphabetical_fallback(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]

    folder = tmp_path / "test_folder"
    folder.mkdir()
    (folder / "zzz.yaml").write_text("dummy: data")
    _create_folder_with_yaml(tmp_path, "test_folder", "aaa.yaml", build_input)
    (folder / "mmm.yaml").write_text("dummy: data")

    result = find_yaml_in_folder(folder)
    assert result is not None
    assert result.name == "aaa.yaml"


def test_find_yaml_no_yaml_files(tmp_path):
    folder = tmp_path / "no_yaml"
    folder.mkdir()
    (folder / "test.txt").write_text("not a yaml")
    (folder / "data.csv").write_text("also not yaml")

    result = find_yaml_in_folder(folder)
    assert result is None


# GROUP 2: determine_simulation_status() tests


def test_status_completed(tmp_path):
    folder = tmp_path / "completed_sim"
    folder.mkdir()
    (folder / "prod.gro").touch()

    status = determine_simulation_status(folder)
    assert status == "completed"


def test_status_production(tmp_path):
    folder = tmp_path / "production_sim"
    folder.mkdir()
    (folder / "prod.xtc").touch()

    status = determine_simulation_status(folder)
    assert status == "production"


def test_status_equilibrated(tmp_path):
    folder = tmp_path / "equilibrated_sim"
    folder.mkdir()
    (folder / "min.gro").touch()
    (folder / "nvt.gro").touch()
    (folder / "npt.gro").touch()

    status = determine_simulation_status(folder)
    assert status == "equilibrated"


def test_status_hierarchy_prod_gro_wins(tmp_path):
    folder = tmp_path / "all_files"
    folder.mkdir()
    (folder / "prod.gro").touch()
    (folder / "prod.xtc").touch()
    (folder / "min.gro").touch()
    (folder / "nvt.gro").touch()
    (folder / "npt.gro").touch()

    status = determine_simulation_status(folder)
    assert status == "completed"


# GROUP 3: load_models_from_csv() tests


def test_load_models_valid_csv(tmp_path):
    csv_path, expected_models = _write_csv_with_models(tmp_path)

    models, errors = load_models_from_csv(csv_path)

    assert len(models) == 2
    assert len(errors) == 0
    assert all(isinstance(m, BuildInput) for m in models)
    assert sorted(m.hash for m in models) == sorted(m.hash for m in expected_models)


def test_load_models_csv_with_errors(tmp_path):
    csv_path = tmp_path / "invalid.csv"
    df = pd.DataFrame(
        [
            {
                "simulation_type": "mixedbox",
                "engine": "gromacs",
                "parametrization": "charmm",
                "system.total_count": 1000,
                "system.species.ABC.smiles": "CCC",
                "system.species.ABC.fraction": 0.5,
                "system.species.DEF.smiles": "CCO",
                "system.species.DEF.fraction": 0.5,
            },
            {
                "simulation_type": "mixedbox",
                "engine": "gromacs",
                "parametrization": "charmm",
                "system.total_count": 1000,
                "system.species.ABC.smiles": "INVALID_SMILES_XYZ",
                "system.species.ABC.fraction": 0.5,
                "system.species.DEF.smiles": "CCO",
                "system.species.DEF.fraction": 0.5,
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    models, errors = load_models_from_csv(csv_path)

    assert len(models) == 1
    assert len(errors) == 1
    assert 1 in errors


def test_load_models_nonexistent_csv(tmp_path):
    nonexistent_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_models_from_csv(nonexistent_path)


# GROUP 4: search_folders_for_hash() tests


def test_search_hash_found_shallow(tmp_path):
    hash_value = "HASH123"
    (tmp_path / hash_value).mkdir()

    result = search_folders_for_hash(hash_value, base_path=tmp_path)

    assert result is not None
    assert result.name == hash_value


def test_search_hash_found_nested(tmp_path):
    hash_value = "HASH456"
    nested_path = tmp_path / "systems" / "simulations" / hash_value
    nested_path.mkdir(parents=True)

    result = search_folders_for_hash(hash_value, base_path=tmp_path)

    assert result is not None
    assert result.name == hash_value


def test_search_hash_not_found(tmp_path):
    (tmp_path / "OTHER_HASH").mkdir()
    (tmp_path / "DIFFERENT_NAME").mkdir()

    result = search_folders_for_hash("NONEXISTENT", base_path=tmp_path)

    assert result is None


# GROUP 5: discover_simulation_folders() tests


def test_discover_path_single_folder(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_folder_with_yaml(tmp_path, "test_sim", f"{build_input.hash}.yaml", build_input)

    simulations = discover_simulation_folders(source=folder)

    assert len(simulations) == 1
    assert simulations[0][0] == folder
    assert simulations[0][1].hash == build_input.hash


def test_discover_path_single_folder_no_yaml(tmp_path):
    folder = tmp_path / "no_yaml_folder"
    folder.mkdir()
    (folder / "some_file.txt").write_text("not yaml")

    simulations = discover_simulation_folders(source=folder)

    assert len(simulations) == 0


def test_discover_path_single_folder_invalid_yaml(tmp_path):
    folder = tmp_path / "invalid_yaml"
    folder.mkdir()
    (folder / "build.yaml").write_text("invalid: yaml\nthat: is\nnot: a BuildInput")

    simulations = discover_simulation_folders(source=folder)

    assert len(simulations) == 0


def test_discover_path_root_with_subfolders(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]

    root = tmp_path / "root"
    root.mkdir()
    folder = _create_folder_with_yaml(
        root, build_input.hash, f"{build_input.hash}.yaml", build_input
    )

    simulations = discover_simulation_folders(source=root)

    assert len(simulations) == 1
    assert simulations[0][0] == folder
    assert simulations[0][1].hash == build_input.hash


def test_discover_path_root_with_summary_yaml_and_subfolders(tmp_path):
    _, models = _write_csv_with_models(tmp_path)

    root = tmp_path / "root"
    root.mkdir()

    folders = []
    for model in models:
        folder = _create_folder_with_yaml(root, model.hash, f"{model.hash}.yaml", model)
        folders.append(folder)

    _create_summary_yaml(root, folders)

    simulations = discover_simulation_folders(source=root)

    assert len(simulations) == len(models)
    discovered_hashes = sorted(sim[1].hash for sim in simulations)
    expected_hashes = sorted(model.hash for model in models)
    assert discovered_hashes == expected_hashes


def test_discover_path_glob_multiple_folders(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()

    folders = []
    for i, model in enumerate(models):
        folder = _create_folder_with_yaml(systems_dir, f"hash{i}", f"{model.hash}.yaml", model)
        folders.append(folder)

    glob_pattern = systems_dir / "*"
    simulations = discover_simulation_folders(source=glob_pattern)

    assert len(simulations) == 2
    discovered_hashes = sorted(sim[1].hash for sim in simulations)
    expected_hashes = sorted(model.hash for model in models)
    assert discovered_hashes == expected_hashes


def test_discover_yaml_mode_valid(tmp_path):
    _, models = _write_csv_with_models(tmp_path)

    folders = []
    for i, model in enumerate(models):
        folder = _create_folder_with_yaml(tmp_path, f"sim{i}", "build.yaml", model)
        folders.append(folder)

    summary_yaml = _create_summary_yaml(tmp_path, folders)
    simulations = discover_simulation_folders(source=summary_yaml)

    assert len(simulations) == 2
    discovered_hashes = sorted(sim[1].hash for sim in simulations)
    expected_hashes = sorted(model.hash for model in models)
    assert discovered_hashes == expected_hashes


def test_discover_yaml_mode_missing_file(tmp_path):
    nonexistent_yaml = tmp_path / "does_not_exist.yaml"

    simulations = discover_simulation_folders(source=nonexistent_yaml)
    assert len(simulations) == 0


def test_discover_yaml_mode_missing_system_directory_field(tmp_path):
    invalid_yaml = tmp_path / "invalid.yaml"
    with open(invalid_yaml, "w") as handle:
        yaml.safe_dump({"some_other_field": "value"}, handle)

    with pytest.raises(ValueError, match="missing 'system_directory'"):
        discover_simulation_folders(source=invalid_yaml)


def test_discover_csv_mode_missing_csv_file(tmp_path):
    nonexistent_csv = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        discover_simulation_folders(csv=nonexistent_csv)


def test_discover_multiple_inputs_error(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("dummy,data")

    with pytest.raises(ValueError, match="Exactly one"):
        discover_simulation_folders(source=folder, csv=csv_file)


def test_discover_no_inputs_error():
    with pytest.raises(ValueError, match="Exactly one"):
        discover_simulation_folders()


# GROUP 6: query_existing_hashes() tests


def test_query_hashes_empty_database(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    hashes = query_existing_hashes("RUN_DATABASE")

    assert hashes == set()
    assert len(hashes) == 0


def test_query_hashes_multiple_records(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", status="build"))
    dm.save_data(_record("HASH2", status="production"))
    dm.save_data(_record("HASH3", status="completed"))

    hashes = query_existing_hashes("RUN_DATABASE")

    assert len(hashes) == 3
    assert hashes == {"HASH1", "HASH2", "HASH3"}


def test_query_hashes_explicit_table_name(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH_EXPLICIT", status="build"))

    hashes = query_existing_hashes("RUN_DATABASE")

    assert len(hashes) == 1
    assert "HASH_EXPLICIT" in hashes


# GROUP 7: prepare_upload_data() tests


def test_prepare_single_simulation(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder(tmp_path, build_input.hash, "completed", build_input)

    simulations = [(folder, build_input)]
    records = prepare_upload_data(simulations)

    assert len(records) == 1
    assert "hash" in records[0]
    assert records[0]["hash"] == build_input.hash
    assert "status" in records[0]
    assert records[0]["status"] == "completed"
    assert "directory" in records[0]
    assert str(folder) in records[0]["directory"]
    assert "timestamp_utc" in records[0]
    assert "engine" in records[0]
    assert "parametrization" in records[0]


def test_prepare_multiple_simulations(tmp_path):
    _, models = _write_csv_with_models(tmp_path)

    simulations = []
    for i, model in enumerate(models):
        folder = _create_simulation_folder(tmp_path, f"hash_{i}", "build", model)
        simulations.append((folder, model))

    records = prepare_upload_data(simulations)

    assert len(records) == 2
    assert all("hash" in r for r in records)
    assert all("status" in r for r in records)
    assert all("directory" in r for r in records)


def test_prepare_status_correctly_determined(tmp_path):
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]

    statuses = ["build", "equilibrated", "production", "completed"]
    simulations = []
    for status in statuses:
        folder = _create_simulation_folder(tmp_path, f"{status}_folder", status, build_input)
        simulations.append((folder, build_input))

    records = prepare_upload_data(simulations)

    assert len(records) == 4
    record_statuses = {r["status"] for r in records}
    assert record_statuses == {"build", "equilibrated", "production", "completed"}


# GROUP 8: upload_simulations() tests


def test_upload_default_mode_success(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    record1 = _record("HASH_NEW1", status="build")
    record2 = _record("HASH_NEW2", status="production")
    upload_simulations([record1, record2])

    df = dm.load_data()
    assert len(df) == 2
    assert set(df["hash"].values) == {"HASH_NEW1", "HASH_NEW2"}


def test_upload_explicit_db_type(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    record = _record("HASH_EXPLICIT_DB", status="build")
    upload_simulations([record], db_type="RUN_DATABASE")

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH_EXPLICIT_DB"


def test_upload_default_mode_duplicate_error(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    existing = _record("HASH_DUP", status="build")
    dm.save_data(existing)

    duplicate = _record("HASH_DUP", status="completed")

    with pytest.raises(ValueError, match="already exist"):
        upload_simulations([duplicate])


def test_upload_force_multiple_records(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", status="build"))
    dm.save_data(_record("HASH2", status="build"))

    updated_records = [
        _record("HASH1", status="completed"),
        _record("HASH2", status="production"),
        _record("HASH3", status="build"),
    ]
    upload_simulations(updated_records, force=True)

    df = dm.load_data()
    assert len(df) == 3
    assert set(df["hash"].values) == {"HASH1", "HASH2", "HASH3"}
    hash1_status = df[df["hash"] == "HASH1"].iloc[0]["status"]
    assert hash1_status == "completed"


def test_upload_simulations_dedupes_input(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    record1 = _record("HASH_DUP", status="build")
    record2 = _record("HASH_DUP", status="completed")

    upload_simulations([record1, record2])

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


def test_upload_diff_skips_all_existing(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    record1 = _record("HASH_EXIST1", status="build")
    record2 = _record("HASH_EXIST2", status="production")
    dm.save_data(record1)
    dm.save_data(record2)

    upload_simulations([record1, record2], diff=True)

    df = dm.load_data()
    assert len(df) == 2


# GROUP 9: Integration tests


def test_integration_path_to_upload(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder(tmp_path, "test_sim", "completed", build_input)

    simulations = discover_simulation_folders(source=folder)
    records = prepare_upload_data(simulations)
    upload_simulations(records)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == build_input.hash
    assert df.iloc[0]["status"] == "completed"


def test_integration_yaml_to_upload(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    folders = []
    for i, model in enumerate(models):
        folder = _create_simulation_folder(tmp_path, f"sim_{i}", "production", model)
        folders.append(folder)

    summary_yaml = _create_summary_yaml(tmp_path, folders)

    simulations = discover_simulation_folders(source=summary_yaml)
    records = prepare_upload_data(simulations)
    upload_simulations(records)

    df = dm.load_data()
    assert len(df) == 2
    assert all(status == "production" for status in df["status"])


def test_integration_csv_to_upload(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    csv_path, models = _write_csv_with_models(tmp_path)
    root_dir = tmp_path / "hashes"
    root_dir.mkdir()

    for model in models:
        _create_simulation_folder(root_dir, model.hash, "build", model)

    simulations = discover_simulation_folders(csv=csv_path, csv_root=root_dir)
    records = prepare_upload_data(simulations)
    upload_simulations(records)

    df = dm.load_data()
    assert len(df) == 2
    expected_hashes = sorted(model.hash for model in models)
    actual_hashes = sorted(df["hash"].values)
    assert actual_hashes == expected_hashes


def test_integration_duplicate_handling(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder(tmp_path, "dup_sim", "build", build_input)

    simulations = discover_simulation_folders(source=folder)
    records = prepare_upload_data(simulations)
    upload_simulations(records)

    df = dm.load_data()
    assert len(df) == 1

    with pytest.raises(ValueError, match="already exist"):
        upload_simulations(records)


# GROUP 10: push_systems() convenience function tests


def test_push_systems_path_mode(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder(tmp_path, "test_sim", "completed", build_input)

    count = push_systems(source=folder)

    assert count == 1
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == build_input.hash
    assert df.iloc[0]["status"] == "completed"


def test_push_systems_yaml_mode(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    folders = []
    for i, model in enumerate(models):
        folder = _create_simulation_folder(tmp_path, f"sim_{i}", "production", model)
        folders.append(folder)

    summary_yaml = _create_summary_yaml(tmp_path, folders)
    count = push_systems(source=summary_yaml)

    assert count == 2
    df = dm.load_data()
    assert len(df) == 2


def test_push_systems_with_force(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder(tmp_path, "force_test", "build", build_input)

    push_systems(source=folder)
    df = dm.load_data()
    assert df.iloc[0]["status"] == "build"

    (folder / "prod.gro").touch()
    push_systems(source=folder, force=True)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


def test_push_systems_with_diff(tmp_path, temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    folders = []
    for i, model in enumerate(models):
        folder = _create_simulation_folder(tmp_path, f"diff_{i}", "build", model)
        folders.append(folder)

    push_systems(source=folders[0])
    df = dm.load_data()
    assert len(df) == 1

    summary_yaml = _create_summary_yaml(tmp_path, folders)
    push_systems(source=summary_yaml, diff=True)

    df = dm.load_data()
    assert len(df) == 2


# GROUP 11: init systems tests


def test_init_systems_database_creates_file(tmp_path, monkeypatch):
    db_path = tmp_path / "runs.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name == "RUN_DATABASE" and db_type == "sqlite":
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    def patched_init(self):
        for section, options in Settings.DEFAULT_CONFIG.items():
            self.config[section] = dict(options)
        self.config["database"]["TYPE"] = "sqlite"

    monkeypatch.setattr(Settings, "_load_config", patched_init)

    results = init_systems_database()

    assert results["RUN_DATABASE"] is True
    assert db_path.exists()

    import sqlite3

    with sqlite3.connect(db_path, autocommit=True) as con:
        user_version = con.execute("PRAGMA user_version").fetchone()[0]
        # Verify the table was actually created
        table_count = con.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='RUN_DATABASE'"
        ).fetchone()[0]
    assert user_version == 1
    assert table_count == 1


def test_init_systems_database_already_exists(tmp_path, monkeypatch):
    db_path = tmp_path / "runs.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name == "RUN_DATABASE" and db_type == "sqlite":
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    def patched_init(self):
        for section, options in Settings.DEFAULT_CONFIG.items():
            self.config[section] = dict(options)
        self.config["database"]["TYPE"] = "sqlite"

    monkeypatch.setattr(Settings, "_load_config", patched_init)

    results_first = init_systems_database()
    results_second = init_systems_database()

    assert results_first["RUN_DATABASE"] is True
    assert results_second["RUN_DATABASE"] is False


def test_init_systems_database_reset_recreates(tmp_path, monkeypatch):
    """reset=True drops and recreates the table, clearing all data."""
    db_path = tmp_path / "runs.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name == "RUN_DATABASE" and db_type == "sqlite":
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    def patched_init(self):
        for section, options in Settings.DEFAULT_CONFIG.items():
            self.config[section] = dict(options)
        self.config["database"]["TYPE"] = "sqlite"

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", patched_init)

    # Init and insert a row
    init_systems_database()
    record = _record("FORCE_HASH")

    dm = DataManager("RUN_DATABASE")
    dm.save_data(record)
    assert len(dm.load_data()) == 1

    # Reset should clear data
    results = init_systems_database(reset=True)
    assert results["RUN_DATABASE"] is True

    dm2 = DataManager("RUN_DATABASE")
    assert len(dm2.load_data()) == 0


def test_cli_sync_init_systems(tmp_path, monkeypatch):
    db_path = tmp_path / "runs.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name == "RUN_DATABASE" and db_type == "sqlite":
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    def patched_init(self):
        for section, options in Settings.DEFAULT_CONFIG.items():
            self.config[section] = dict(options)
        self.config["database"]["TYPE"] = "sqlite"

    monkeypatch.setattr(Settings, "_load_config", patched_init)
    monkeypatch.chdir(tmp_path)

    sync_init_systems()

    assert db_path.exists()


# GROUP 12: CLI sync push systems tests


def test_cli_sync_push_systems_no_inputs(monkeypatch):
    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_systems()

    assert exit_code == 1


def test_cli_sync_push_systems_both_inputs(tmp_path, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("dummy,data")

    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_systems(source=folder, csv=csv_file)

    assert exit_code == 1


def test_cli_sync_push_systems_force_and_diff_together(tmp_path, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()

    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_systems(source=folder, force=True, diff=True)

    assert exit_code == 1


def test_cli_sync_push_systems_path_success(tmp_path, temp_run_db, monkeypatch):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder(tmp_path, "cli_test", "build", build_input)

    monkeypatch.chdir(tmp_path)
    sync_push_systems(source=folder)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == build_input.hash


def test_cli_sync_push_systems_yaml_success(tmp_path, temp_run_db, monkeypatch):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    _, models = _write_csv_with_models(tmp_path)
    folders = []
    for i, model in enumerate(models):
        folder = _create_simulation_folder(tmp_path, f"yaml_test_{i}", "equilibrated", model)
        folders.append(folder)

    summary_yaml = _create_summary_yaml(tmp_path, folders)

    monkeypatch.chdir(tmp_path)
    sync_push_systems(source=summary_yaml)

    df = dm.load_data()
    assert len(df) == 2
    assert all(status == "equilibrated" for status in df["status"])


def test_cli_sync_push_systems_duplicate_exits_cleanly(tmp_path, monkeypatch):
    """CLI should exit cleanly (no re-raise traceback) on duplicate-key errors."""
    folder = tmp_path / "folder"
    folder.mkdir()

    monkeypatch.setattr("mdfactory.cli._ensure_sync_target_initialized", lambda *_: None)
    monkeypatch.setattr(
        "mdfactory.cli.push_systems",
        lambda **_: (_ for _ in ()).throw(ValueError("2 key(s) already exist in RUN_DATABASE")),
    )

    with pytest.raises(SystemExit) as exc:
        sync_push_systems(source=folder)

    assert exc.value.code == 1
