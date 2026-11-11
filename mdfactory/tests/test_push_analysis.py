# ABOUTME: Tests for analysis sync push functionality
# ABOUTME: Tests discovery, preparation, upload, and CLI commands for analysis data

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mdfactory.cli import sync_init_analysis, sync_push_analysis
from mdfactory.settings import Settings
from mdfactory.utils.data_manager import DataManager
from mdfactory.utils.db_operations import query_existing_hashes
from mdfactory.utils.push_analysis import (
    deserialize_csv_to_dataframe,
    discover_and_prepare_analysis_data,
    get_all_analysis_names,
    get_all_artifact_names,
    get_analyses_for_simulation_type,
    get_analysis_table_name,
    get_artifact_table_name,
    get_artifacts_for_simulation_type,
    init_analysis_database,
    prepare_overview_record,
    push_analysis,
    serialize_dataframe_to_csv,
    update_overview_records,
    upload_analysis_data,
)

# Import shared helpers from conftest (temp_analysis_db fixture is auto-discovered)
from .conftest import _write_csv_with_models, default_settings_loader


def _create_simulation_folder_analysis(tmp_path: Path, hash_value: str, build_input) -> Path:
    """Create a simulation folder with YAML and structure file for analysis tests."""
    folder = tmp_path / hash_value
    folder.mkdir(parents=True, exist_ok=True)

    # Write BuildInput YAML
    yaml_path = folder / f"{hash_value}.yaml"
    with open(yaml_path, "w") as handle:
        yaml.safe_dump(build_input.model_dump(), handle)

    # Create minimal structure file
    (folder / "system.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"
    )

    return folder


def _create_analysis_data(folder: Path, analysis_name: str, df: pd.DataFrame) -> None:
    """Create mock analysis data in a simulation folder."""
    analysis_dir = folder / ".analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Save parquet file (requires pyarrow or fastparquet)
    parquet_path = analysis_dir / f"{analysis_name}.parquet"
    try:
        df.to_parquet(parquet_path)
    except ImportError:
        pytest.skip("pyarrow or fastparquet required for this test")

    # Update metadata.json (registry)
    metadata_path = analysis_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"schema_version": "1.0", "analyses": {}, "artifacts": {}}

    metadata["analyses"][analysis_name] = {
        "filename": f"{analysis_name}.parquet",
        "row_count": len(df),
        "columns": df.columns.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extras": {},
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _sample_analysis_df(analysis_name: str) -> pd.DataFrame:
    """Create sample analysis DataFrame based on analysis type."""
    if analysis_name == "area_per_lipid":
        return pd.DataFrame(
            {
                "time_ns": [0.0, 1.0, 2.0],
                "frame": [0, 1, 2],
                "leaflet": ["top", "top", "top"],
                "species": ["POPC", "POPC", "POPC"],
                "apl": [65.0, 64.5, 65.2],
                "n_lipids": [100, 100, 100],
            }
        )
    elif analysis_name == "box_size_timeseries":
        return pd.DataFrame(
            {
                "time_ns": [0.0, 1.0, 2.0],
                "frame": [0, 1, 2],
                "x": [10.0, 10.1, 10.0],
                "y": [10.0, 9.9, 10.0],
                "z": [15.0, 15.0, 15.1],
                "volume": [1500.0, 1499.0, 1501.0],
            }
        )
    else:
        # Generic DataFrame
        return pd.DataFrame(
            {
                "time_ns": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        )


def _analysis_record(hash_value: str, analysis_name: str = "area_per_lipid") -> dict:
    """Create a sample analysis record."""
    df = _sample_analysis_df(analysis_name)
    return {
        "hash": hash_value,
        "directory": f"/tmp/{hash_value}",
        "simulation_type": "bilayer",
        "row_count": len(df),
        "columns": json.dumps(df.columns.tolist()),
        "data_csv": serialize_dataframe_to_csv(df),
        "data_path": f".analysis/{analysis_name}.parquet",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _overview_record(
    hash_value: str,
    item_name: str,
    status: str = "completed",
    item_type: str = "analysis",
) -> dict:
    """Create a sample overview record.

    Note: Uses empty string instead of None to avoid SQLite NULL constraint issues
    when the first record sets the schema.
    """
    return {
        "hash": hash_value,
        "simulation_type": "bilayer",
        "directory": f"/tmp/{hash_value}",
        "item_type": item_type,
        "item_name": item_name,
        "status": status,
        "row_count": 10 if status == "completed" and item_type == "analysis" else "",
        "file_count": 2 if status == "completed" and item_type == "artifact" else "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# temp_analysis_db fixture is auto-discovered from conftest.py


# GROUP 1: Naming convention tests


def test_get_analysis_table_name():
    assert get_analysis_table_name("area_per_lipid") == "ANALYSIS_AREA_PER_LIPID"
    assert get_analysis_table_name("box_size_timeseries") == "ANALYSIS_BOX_SIZE_TIMESERIES"


def test_get_artifact_table_name():
    assert get_artifact_table_name("bilayer_snapshot") == "ARTIFACT_BILAYER_SNAPSHOT"
    assert get_artifact_table_name("last_frame_pdb") == "ARTIFACT_LAST_FRAME_PDB"


def test_get_all_analysis_names():
    names = get_all_analysis_names()
    assert isinstance(names, list)
    assert len(names) > 0
    # Check some known analyses exist
    assert "area_per_lipid" in names
    assert "density_distribution" in names


def test_get_all_artifact_names():
    names = get_all_artifact_names()
    assert isinstance(names, list)
    assert len(names) > 0
    assert "bilayer_snapshot" in names or "last_frame_pdb" in names


def test_get_analyses_for_simulation_type():
    bilayer_analyses = get_analyses_for_simulation_type("bilayer")
    assert "area_per_lipid" in bilayer_analyses

    mixedbox_analyses = get_analyses_for_simulation_type("mixedbox")
    assert isinstance(mixedbox_analyses, list)

    unknown_analyses = get_analyses_for_simulation_type("unknown_type")
    assert unknown_analyses == []


def test_get_artifacts_for_simulation_type():
    bilayer_artifacts = get_artifacts_for_simulation_type("bilayer")
    assert isinstance(bilayer_artifacts, list)

    unknown_artifacts = get_artifacts_for_simulation_type("unknown_type")
    assert unknown_artifacts == []


# GROUP 2: CSV serialization tests


def test_serialize_dataframe_to_csv():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_string = serialize_dataframe_to_csv(df)
    assert isinstance(csv_string, str)
    assert "a,b" in csv_string
    assert "1,3" in csv_string


def test_deserialize_csv_to_dataframe():
    csv_string = "a,b\n1,3\n2,4\n"
    df = deserialize_csv_to_dataframe(csv_string)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_csv_roundtrip():
    original = pd.DataFrame(
        {
            "time_ns": [0.0, 1.0, 2.0],
            "value": [10.5, 20.3, 15.7],
            "name": ["a", "b", "c"],
        }
    )
    csv_string = serialize_dataframe_to_csv(original)
    restored = deserialize_csv_to_dataframe(csv_string)

    assert list(restored.columns) == list(original.columns)
    assert len(restored) == len(original)


# GROUP 3: Database initialization tests


def test_init_analysis_database_creates_file(tmp_path, monkeypatch):
    db_path = tmp_path / "test_analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    results = init_analysis_database()

    assert db_path.exists()
    assert all(v is True for v in results.values())


def test_init_analysis_database_already_exists(temp_analysis_db):
    # Second init should not recreate
    results = init_analysis_database()
    assert all(v is False for v in results.values())


# GROUP 4: Query existing hashes tests


def test_query_existing_hashes_empty_table(temp_analysis_db):
    hashes = query_existing_hashes("ANALYSIS_AREA_PER_LIPID")
    assert hashes == set()


def test_query_existing_hashes_with_data(temp_analysis_db):
    table_name = "ANALYSIS_AREA_PER_LIPID"
    dm = DataManager(table_name)
    dm.save_data(_analysis_record("HASH1", "area_per_lipid"))
    dm.save_data(_analysis_record("HASH2", "area_per_lipid"))

    hashes = query_existing_hashes(table_name)
    assert hashes == {"HASH1", "HASH2"}


# GROUP 5: Upload analysis data tests


def test_upload_analysis_data_success(temp_analysis_db):
    table_name = "ANALYSIS_AREA_PER_LIPID"
    records = [
        _analysis_record("NEW_HASH1", "area_per_lipid"),
        _analysis_record("NEW_HASH2", "area_per_lipid"),
    ]

    count = upload_analysis_data(records, table_name)

    assert count == 2
    dm = DataManager(table_name)
    df = dm.load_data()
    assert len(df) == 2


def test_upload_analysis_data_diff_mode(temp_analysis_db):
    table_name = "ANALYSIS_AREA_PER_LIPID"
    dm = DataManager(table_name)

    # Pre-existing record
    dm.save_data(_analysis_record("EXISTING", "area_per_lipid"))

    records = [
        _analysis_record("EXISTING", "area_per_lipid"),
        _analysis_record("NEW", "area_per_lipid"),
    ]

    count = upload_analysis_data(records, table_name, diff=True)

    assert count == 1  # Only NEW should be uploaded
    df = dm.load_data()
    assert len(df) == 2


def test_upload_analysis_data_force_mode(temp_analysis_db):
    table_name = "ANALYSIS_AREA_PER_LIPID"
    dm = DataManager(table_name)

    # Pre-existing record
    old_record = _analysis_record("HASH1", "area_per_lipid")
    old_record["row_count"] = 5
    dm.save_data(old_record)

    # New record with same hash but different data
    new_record = _analysis_record("HASH1", "area_per_lipid")
    new_record["row_count"] = 10

    count = upload_analysis_data([new_record], table_name, force=True)

    assert count == 1
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["row_count"] == 10


def test_upload_analysis_data_dedupes_input(temp_analysis_db):
    table_name = "ANALYSIS_AREA_PER_LIPID"
    dm = DataManager(table_name)
    dm.delete_data({})

    record1 = _analysis_record("HASH_DUP", "area_per_lipid")
    record1["row_count"] = 1
    record2 = _analysis_record("HASH_DUP", "area_per_lipid")
    record2["row_count"] = 2

    upload_analysis_data([record1, record2], table_name)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["row_count"] == 2


def test_upload_analysis_data_duplicate_error(temp_analysis_db):
    table_name = "ANALYSIS_AREA_PER_LIPID"
    dm = DataManager(table_name)

    dm.save_data(_analysis_record("DUP_HASH", "area_per_lipid"))

    with pytest.raises(ValueError, match="already exist"):
        upload_analysis_data([_analysis_record("DUP_HASH", "area_per_lipid")], table_name)


# GROUP 6: Overview table tests


def test_update_overview_records(temp_analysis_db):
    records = [
        _overview_record("HASH1", "area_per_lipid", "completed"),
        _overview_record("HASH1", "density_distribution", "not_yet_run"),
        _overview_record("HASH2", "area_per_lipid", "not_yet_run"),
    ]

    count = update_overview_records(records)

    assert count == 3
    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    assert len(df) == 3


def test_update_overview_records_updates_existing(temp_analysis_db):
    # First insert
    records1 = [_overview_record("HASH1", "area_per_lipid", "not_yet_run")]
    update_overview_records(records1)

    # Default mode should allow upgrade to completed
    records2 = [_overview_record("HASH1", "area_per_lipid", "completed")]
    update_overview_records(records2)

    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


def test_update_overview_records_diff_no_downgrade(temp_analysis_db):
    """Diff mode must not overwrite completed with not_yet_run."""
    completed = [_overview_record("HASH1", "area_per_lipid", "completed")]
    update_overview_records(completed)

    downgrade = [_overview_record("HASH1", "area_per_lipid", "not_yet_run")]
    update_overview_records(downgrade, diff=True)

    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    row = df[df["hash"] == "HASH1"]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "completed"


def test_update_overview_records_diff_upgrades_status(temp_analysis_db):
    """Diff mode upgrades not_yet_run to completed."""
    initial = [_overview_record("HASH1", "area_per_lipid", "not_yet_run")]
    update_overview_records(initial)

    upgrade = [_overview_record("HASH1", "area_per_lipid", "completed")]
    update_overview_records(upgrade, diff=True)

    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    row = df[df["hash"] == "HASH1"]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "completed"


def test_update_overview_records_diff_skips_same_status(temp_analysis_db):
    """Diff mode skips records with unchanged status."""
    initial = [_overview_record("HASH1", "area_per_lipid", "completed")]
    update_overview_records(initial)

    count = update_overview_records(
        [_overview_record("HASH1", "area_per_lipid", "completed")], diff=True
    )
    assert count == 0

    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    row = df[df["hash"] == "HASH1"]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "completed"


def test_update_overview_records_diff_mixed_new_and_upgrade(temp_analysis_db):
    """Diff mode handles mix of new records and upgrades."""
    initial = [_overview_record("HASH1", "area_per_lipid", "not_yet_run")]
    update_overview_records(initial)

    records = [
        _overview_record("HASH1", "area_per_lipid", "completed"),  # upgrade
        _overview_record("HASH2", "area_per_lipid", "completed"),  # new
    ]
    count = update_overview_records(records, diff=True)
    assert count == 2

    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    h1 = df[df["hash"] == "HASH1"]
    assert h1.iloc[0]["status"] == "completed"
    h2 = df[df["hash"] == "HASH2"]
    assert len(h2) == 1


def test_update_overview_records_force_allows_upgrade(temp_analysis_db):
    """Force mode must overwrite existing keys."""
    initial = [_overview_record("HASH1", "area_per_lipid", "not_yet_run")]
    update_overview_records(initial)

    upgrade = [_overview_record("HASH1", "area_per_lipid", "completed")]
    update_overview_records(upgrade, force=True)

    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    row = df[df["hash"] == "HASH1"]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "completed"


def test_push_analysis_legacy_imports():
    """Symbols historically importable from push_analysis must still resolve."""
    from mdfactory.utils.push_analysis import (
        get_all_analysis_names,
        get_analysis_table_name,
        query_existing_hashes,
    )

    assert callable(query_existing_hashes)
    assert callable(get_analysis_table_name)
    assert callable(get_all_analysis_names)


# GROUP 7: Prepare record tests


def test_prepare_overview_record():
    # Create a mock simulation for testing
    class MockSimulation:
        class build_input:
            hash = "TEST_HASH"
            simulation_type = "bilayer"

        path = Path("/tmp/test")

    record = prepare_overview_record(
        MockSimulation(),
        "analysis",
        "area_per_lipid",
        "completed",
        row_count=100,
    )

    assert record["hash"] == "TEST_HASH"
    assert record["item_type"] == "analysis"
    assert record["item_name"] == "area_per_lipid"
    assert record["status"] == "completed"
    assert record["row_count"] == 100


# GROUP 8: Discovery tests


def test_discover_and_prepare_with_analyses(tmp_path, monkeypatch):
    """Test discovering simulations with completed analyses."""
    # Setup temp database
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()

    # Create simulation folder with analysis data
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_analysis(tmp_path, build_input.hash, build_input)
    _create_analysis_data(folder, "area_per_lipid", _sample_analysis_df("area_per_lipid"))

    simulations = [(folder, build_input)]
    data_records, overview_records = discover_and_prepare_analysis_data(
        simulations, analysis_name="area_per_lipid"
    )

    # Should have one completed analysis record
    assert "ANALYSIS_AREA_PER_LIPID" in data_records
    assert len(data_records["ANALYSIS_AREA_PER_LIPID"]) == 1

    # Overview should have at least one entry
    assert len(overview_records) >= 1


# GROUP 9: Integration tests


def test_push_analysis_integration(tmp_path, monkeypatch):
    """Test full push_analysis flow."""
    # Setup temp database
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()

    # Create simulation folder with analysis data
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_analysis(tmp_path, build_input.hash, build_input)
    _create_analysis_data(folder, "area_per_lipid", _sample_analysis_df("area_per_lipid"))

    # Push analysis
    results = push_analysis(source=folder, analysis_name="area_per_lipid")

    # Verify results
    assert "ANALYSIS_AREA_PER_LIPID" in results
    assert results["ANALYSIS_AREA_PER_LIPID"] == 1
    assert "ANALYSIS_OVERVIEW" in results


def test_push_analysis_diff_mode(tmp_path, monkeypatch):
    """Test push_analysis with diff mode."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()

    # Create simulation folder with analysis
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_analysis(tmp_path, build_input.hash, build_input)
    _create_analysis_data(folder, "area_per_lipid", _sample_analysis_df("area_per_lipid"))

    # First push
    results1 = push_analysis(source=folder, analysis_name="area_per_lipid")
    assert results1["ANALYSIS_AREA_PER_LIPID"] == 1

    # Second push with diff - should skip
    results2 = push_analysis(source=folder, analysis_name="area_per_lipid", diff=True)
    assert results2["ANALYSIS_AREA_PER_LIPID"] == 0


def test_push_analysis_default_upgrades_existing_not_yet_run(tmp_path, monkeypatch):
    """Default mode should upgrade overview status without failing."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_analysis(tmp_path, build_input.hash, build_input)
    _create_analysis_data(folder, "area_per_lipid", _sample_analysis_df("area_per_lipid"))

    # Simulate prior discovery run that only inserted not_yet_run
    update_overview_records([_overview_record(build_input.hash, "area_per_lipid", "not_yet_run")])

    results = push_analysis(source=folder, analysis_name="area_per_lipid")
    assert results["ANALYSIS_AREA_PER_LIPID"] == 1
    assert results["ANALYSIS_OVERVIEW"] == 1

    overview = DataManager("ANALYSIS_OVERVIEW").load_data()
    row = overview[overview["hash"] == build_input.hash]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "completed"


# GROUP 10: CLI tests


def test_cli_sync_init_analysis(tmp_path, monkeypatch):
    """Test CLI sync init analysis command."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    monkeypatch.chdir(tmp_path)

    sync_init_analysis()

    assert db_path.exists()


def test_cli_sync_push_analysis_no_inputs(monkeypatch):
    """Test CLI sync push analysis with no inputs."""
    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_analysis()

    assert exit_code == 1


def test_cli_sync_push_analysis_force_and_diff_together(tmp_path, monkeypatch):
    """Test CLI rejects force and diff together."""
    folder = tmp_path / "folder"
    folder.mkdir()

    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_analysis(source=folder, force=True, diff=True)

    assert exit_code == 1


def test_cli_sync_push_analysis_csv_root_without_csv(tmp_path, monkeypatch):
    """Test CLI rejects csv_root without csv."""
    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_analysis(csv_root=tmp_path)

    assert exit_code == 1


def test_cli_sync_push_analysis_path_success(tmp_path, monkeypatch):
    """Test CLI sync push analysis with path input."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_analysis(tmp_path, build_input.hash, build_input)
    _create_analysis_data(folder, "area_per_lipid", _sample_analysis_df("area_per_lipid"))

    monkeypatch.chdir(tmp_path)
    sync_push_analysis(source=folder, analysis_name="area_per_lipid")

    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == build_input.hash


def test_cli_sync_push_analysis_duplicate_exits_cleanly(tmp_path, monkeypatch):
    """CLI should exit cleanly (no re-raise traceback) on duplicate-key errors."""
    folder = tmp_path / "folder"
    folder.mkdir()

    monkeypatch.setattr("mdfactory.cli._ensure_sync_target_initialized", lambda *_: None)
    monkeypatch.setattr(
        "mdfactory.utils.push_analysis.push_analysis",
        lambda **_: (_ for _ in ()).throw(
            ValueError("40 key(s) already exist in ANALYSIS_BOX_SIZE_TIMESERIES")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        sync_push_analysis(source=folder)

    assert exc.value.code == 1


# GROUP 11: Error propagation and guard tests


def test_push_analysis_duplicate_raises(tmp_path, monkeypatch):
    """Test push_analysis raises ValueError on duplicate data in default mode."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_analysis(tmp_path, build_input.hash, build_input)
    _create_analysis_data(folder, "area_per_lipid", _sample_analysis_df("area_per_lipid"))

    # First push succeeds
    push_analysis(source=folder, analysis_name="area_per_lipid")

    # Second push without force/diff raises
    with pytest.raises(ValueError, match="already exist"):
        push_analysis(source=folder, analysis_name="area_per_lipid")


def test_update_overview_records_force_and_diff_rejected(temp_analysis_db):
    """Test update_overview_records rejects force=True and diff=True together."""
    records = [_overview_record("HASH1", "area_per_lipid", "completed")]

    with pytest.raises(ValueError, match="force and diff"):
        update_overview_records(records, force=True, diff=True)


# GROUP 12: SQLite table initialization tests


def test_init_analysis_database_creates_tables(tmp_path, monkeypatch):
    """init_analysis_database should create SQLite tables for all registered analyses."""
    import sqlite3

    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    results = init_analysis_database()

    # All tables should be created
    assert all(v is True for v in results.values()), f"Some tables not created: {results}"

    # Verify tables exist in the database
    with sqlite3.connect(db_path, autocommit=True) as con:
        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

    # Must have ANALYSIS_OVERVIEW and at least one ANALYSIS_ table
    assert "ANALYSIS_OVERVIEW" in tables

    # All analysis tables should be present
    for name in get_all_analysis_names():
        table_name = get_analysis_table_name(name)
        assert table_name in tables, f"Missing table: {table_name}"

    # Tables should be empty (placeholder was removed)
    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    assert len(df) == 0


def test_init_analysis_database_reset_recreates(tmp_path, monkeypatch):
    """reset=True should clear only analysis overview rows."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    # Init and insert data
    init_analysis_database()
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(
        [
            _overview_record("HASH1", "area_per_lipid", "completed", item_type="analysis"),
            _overview_record("HASH2", "last_frame_pdb", "completed", item_type="artifact"),
        ]
    )
    assert len(dm.load_data()) == 2

    # Reset should clear analysis rows and keep artifact rows
    results = init_analysis_database(reset=True)
    analysis_results = {k: v for k, v in results.items() if k != "ANALYSIS_OVERVIEW"}
    assert all(v is True for v in analysis_results.values())
    assert results["ANALYSIS_OVERVIEW"] is False

    dm2 = DataManager("ANALYSIS_OVERVIEW")
    df = dm2.load_data()
    assert len(df) == 1
    assert df.iloc[0]["item_type"] == "artifact"
    assert df.iloc[0]["hash"] == "HASH2"


def test_init_analysis_database_idempotent(tmp_path, monkeypatch):
    """Second init without reset should return all False."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    results_first = init_analysis_database()
    results_second = init_analysis_database()

    assert all(v is True for v in results_first.values())
    assert all(v is False for v in results_second.values())


# GROUP 13: Backend force+diff guard tests


def test_upload_records_rejects_force_and_diff(temp_analysis_db):
    """upload_records must reject force=True and diff=True together."""
    from mdfactory.utils.db_operations import upload_records

    records = [{"hash": "H1", "data": "test"}]
    with pytest.raises(ValueError, match="force and diff"):
        upload_records(records, "ANALYSIS_OVERVIEW", ["hash"], force=True, diff=True)
