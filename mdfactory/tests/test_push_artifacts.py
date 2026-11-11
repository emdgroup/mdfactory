# ABOUTME: Tests for artifact sync push/pull functionality
# ABOUTME: Tests discovery, preparation, upload, and CLI commands for artifact metadata

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mdfactory.cli import sync_init_artifacts, sync_pull_artifacts, sync_push_artifacts
from mdfactory.settings import Settings
from mdfactory.utils.data_manager import DataManager
from mdfactory.utils.pull_analysis import pull_artifact
from mdfactory.utils.pull_artifacts import format_artifact_summary, pull_artifact_overview
from mdfactory.utils.push_analysis import (
    init_analysis_database,
    init_artifact_database,
    update_overview_records,
    upload_analysis_data,
)
from mdfactory.utils.push_artifacts import (
    discover_and_prepare_artifact_data,
    push_artifacts,
)

# Import shared helpers from conftest (temp_analysis_db-like fixture will be temp_artifact_db)
from .conftest import _write_csv_with_models, default_settings_loader


def _create_simulation_folder_artifact(tmp_path: Path, hash_value: str, build_input) -> Path:
    """Create a simulation folder with YAML and structure file for artifact tests."""
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


def _create_artifact_data(
    folder: Path, artifact_name: str, files: list[str], checksums: dict[str, str]
) -> None:
    """Create mock artifact data in a simulation folder."""
    analysis_dir = folder / ".analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Create artifact directory
    artifact_dir = analysis_dir / "artifacts" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy artifact files
    for file_path in files:
        full_path = analysis_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"dummy content for {file_path}")

    # Update metadata.json (registry)
    metadata_path = analysis_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"schema_version": "1.0", "analyses": {}, "artifacts": {}}

    metadata["artifacts"][artifact_name] = {
        "files": files,
        "checksums": checksums,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "extras": {},
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _artifact_record(hash_value: str, artifact_name: str = "bilayer_snapshot") -> dict:
    """Create a sample artifact record."""
    files = [
        f"artifacts/{artifact_name}/snapshot_top.png",
        f"artifacts/{artifact_name}/snapshot_side.png",
    ]
    checksums = {f: f"sha256_{f.replace('/', '_')}" for f in files}
    return {
        "hash": hash_value,
        "directory": f"/tmp/{hash_value}",
        "simulation_type": "mixedbox",
        "file_count": len(files),
        "files": json.dumps(files),
        "checksums": json.dumps(checksums),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _overview_record(
    hash_value: str,
    item_name: str,
    status: str = "completed",
    item_type: str = "artifact",
) -> dict:
    """Create a sample overview record for artifacts."""
    return {
        "hash": hash_value,
        "simulation_type": "mixedbox",
        "directory": f"/tmp/{hash_value}",
        "item_type": item_type,
        "item_name": item_name,
        "status": status,
        "row_count": "",
        "file_count": 2 if status == "completed" else "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture()
def temp_artifact_db(tmp_path, monkeypatch):
    """Force artifact database to use a temporary sqlite file."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    # Initialize both analysis and artifact database
    init_analysis_database()
    init_artifact_database()
    return db_path


# GROUP 1: Database initialization tests


def test_init_artifact_database_creates_file(tmp_path, monkeypatch):
    """Test that init_artifact_database creates the database file."""
    db_path = tmp_path / "test_artifact.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    results = init_artifact_database()

    assert db_path.exists()
    assert "ANALYSIS_OVERVIEW" not in results
    assert len(results) > 0
    assert all(name.startswith("ARTIFACT_") for name in results)
    assert all(v is True for v in results.values())


def test_init_artifact_database_already_exists(temp_artifact_db):
    """Test that init_artifact_database reports when database already exists."""
    results = init_artifact_database()
    assert "ANALYSIS_OVERVIEW" not in results
    assert len(results) > 0
    assert all(name.startswith("ARTIFACT_") for name in results)
    assert all(v is False for v in results.values())


def test_init_artifact_database_reset_preserves_overview(tmp_path, monkeypatch):
    """Artifact reset should clear only artifact rows in ANALYSIS_OVERVIEW."""
    db_path = tmp_path / "test_artifact.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))

    init_analysis_database()
    init_artifact_database()
    update_overview_records(
        [
            _overview_record("HASH1", "bilayer_snapshot", "completed"),
            _overview_record("HASH2", "area_per_lipid", "completed", item_type="analysis"),
        ]
    )
    upload_analysis_data(
        [_artifact_record("HASH1", "bilayer_snapshot")], "ARTIFACT_BILAYER_SNAPSHOT"
    )

    overview_before = DataManager("ANALYSIS_OVERVIEW").load_data()
    assert len(overview_before) == 2
    artifact_before = DataManager("ARTIFACT_BILAYER_SNAPSHOT").load_data()
    assert len(artifact_before) == 1

    results = init_artifact_database(reset=True)
    assert "ANALYSIS_OVERVIEW" not in results
    assert results["ARTIFACT_BILAYER_SNAPSHOT"] is True

    overview_after = DataManager("ANALYSIS_OVERVIEW").load_data()
    assert len(overview_after) == 1
    assert overview_after.iloc[0]["item_type"] == "analysis"
    assert overview_after.iloc[0]["hash"] == "HASH2"
    artifact_after = DataManager("ARTIFACT_BILAYER_SNAPSHOT").load_data()
    assert len(artifact_after) == 0


# GROUP 2: Upload artifact data tests


def test_upload_artifact_data_success(temp_artifact_db):
    """Test uploading artifact records."""
    table_name = "ARTIFACT_BILAYER_SNAPSHOT"
    records = [
        _artifact_record("HASH1", "bilayer_snapshot"),
        _artifact_record("HASH2", "bilayer_snapshot"),
    ]

    count = upload_analysis_data(records, table_name)

    assert count == 2
    dm = DataManager(table_name)
    df = dm.load_data()
    assert len(df) == 2


def test_upload_artifact_data_diff_mode(temp_artifact_db):
    """Test uploading artifacts in diff mode skips existing."""
    table_name = "ARTIFACT_BILAYER_SNAPSHOT"
    dm = DataManager(table_name)

    # Pre-existing record
    dm.save_data(_artifact_record("EXISTING", "bilayer_snapshot"))

    records = [
        _artifact_record("EXISTING", "bilayer_snapshot"),
        _artifact_record("NEW", "bilayer_snapshot"),
    ]

    count = upload_analysis_data(records, table_name, diff=True)

    assert count == 1  # Only NEW should be uploaded
    df = dm.load_data()
    assert len(df) == 2


def test_upload_artifact_data_force_mode(temp_artifact_db):
    """Test uploading artifacts with force mode overwrites existing."""
    table_name = "ARTIFACT_BILAYER_SNAPSHOT"
    dm = DataManager(table_name)

    # Pre-existing record
    old_record = _artifact_record("HASH1", "bilayer_snapshot")
    old_record["file_count"] = 1
    dm.save_data(old_record)

    # New record with same hash but different data
    new_record = _artifact_record("HASH1", "bilayer_snapshot")
    new_record["file_count"] = 5

    count = upload_analysis_data([new_record], table_name, force=True)

    assert count == 1
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["file_count"] == 5


# GROUP 3: Overview table tests for artifacts


def test_update_overview_records_artifacts(temp_artifact_db):
    """Test updating overview records with artifact entries."""
    records = [
        _overview_record("HASH1", "bilayer_snapshot", "completed"),
        _overview_record("HASH1", "bilayer_movie", "not_yet_run"),
        _overview_record("HASH2", "last_frame_pdb", "completed"),
    ]

    count = update_overview_records(records)

    assert count == 3
    dm = DataManager("ANALYSIS_OVERVIEW")
    df = dm.load_data()
    artifact_entries = df[df["item_type"] == "artifact"]
    assert len(artifact_entries) == 3


# GROUP 4: Discovery tests


def test_discover_and_prepare_artifact_data(tmp_path, monkeypatch):
    """Test discovering simulations with completed artifacts."""
    # Setup temp database
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_artifact_database()

    # Create simulation folder with artifact data
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_artifact(tmp_path, build_input.hash, build_input)

    # Create mock artifact
    files = ["artifacts/last_frame_pdb/last_frame.pdb"]
    checksums = {"artifacts/last_frame_pdb/last_frame.pdb": "sha256_mock"}
    _create_artifact_data(folder, "last_frame_pdb", files, checksums)

    simulations = [(folder, build_input)]
    artifact_records, overview_records = discover_and_prepare_artifact_data(
        simulations, artifact_name="last_frame_pdb"
    )

    # Should have one completed artifact record
    assert "ARTIFACT_LAST_FRAME_PDB" in artifact_records
    assert len(artifact_records["ARTIFACT_LAST_FRAME_PDB"]) == 1

    # Overview should have at least one entry
    assert len(overview_records) >= 1


# GROUP 5: Integration tests


def test_push_artifacts_integration(tmp_path, monkeypatch):
    """Test full push_artifacts flow."""
    # Setup temp database
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()
    init_artifact_database()

    # Create simulation folder with artifact data
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_artifact(tmp_path, build_input.hash, build_input)

    # Create mock artifact
    files = ["artifacts/last_frame_pdb/last_frame.pdb"]
    checksums = {"artifacts/last_frame_pdb/last_frame.pdb": "sha256_mock"}
    _create_artifact_data(folder, "last_frame_pdb", files, checksums)

    # Push artifacts
    results = push_artifacts(source=folder, artifact_name="last_frame_pdb")

    # Verify results
    assert "ARTIFACT_LAST_FRAME_PDB" in results
    assert results["ARTIFACT_LAST_FRAME_PDB"] == 1
    assert "ANALYSIS_OVERVIEW" in results


def test_push_artifacts_diff_mode(tmp_path, monkeypatch):
    """Test push_artifacts with diff mode."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()
    init_artifact_database()

    # Create simulation folder with artifact
    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_artifact(tmp_path, build_input.hash, build_input)
    files = ["artifacts/last_frame_pdb/last_frame.pdb"]
    checksums = {"artifacts/last_frame_pdb/last_frame.pdb": "sha256_mock"}
    _create_artifact_data(folder, "last_frame_pdb", files, checksums)

    # First push
    results1 = push_artifacts(source=folder, artifact_name="last_frame_pdb")
    assert results1["ARTIFACT_LAST_FRAME_PDB"] == 1

    # Second push with diff - should skip
    results2 = push_artifacts(source=folder, artifact_name="last_frame_pdb", diff=True)
    assert results2["ARTIFACT_LAST_FRAME_PDB"] == 0


# GROUP 6: Pull tests


def test_pull_artifact(temp_artifact_db):
    """Test pulling artifact metadata."""
    table_name = "ARTIFACT_BILAYER_SNAPSHOT"
    dm = DataManager(table_name)
    dm.save_data(_artifact_record("PULL_HASH", "bilayer_snapshot"))

    df = pull_artifact("bilayer_snapshot", hash="PULL_HASH")

    assert len(df) == 1
    assert df.iloc[0]["hash"] == "PULL_HASH"


def test_pull_artifact_overview(temp_artifact_db):
    """Test pulling artifact overview."""
    # Add some artifact overview records
    records = [
        _overview_record("HASH1", "bilayer_snapshot", "completed"),
        _overview_record("HASH2", "last_frame_pdb", "not_yet_run"),
    ]
    update_overview_records(records)

    df = pull_artifact_overview()

    assert len(df) == 2
    assert set(df["item_type"]) == {"artifact"}


def test_format_artifact_summary():
    """Test formatting artifact summary for display."""
    df = pd.DataFrame([_artifact_record("HASH1", "bilayer_snapshot")])

    formatted = format_artifact_summary(df)

    # Check that preferred columns are first
    assert list(formatted.columns)[0] == "hash"
    assert "simulation_type" in formatted.columns


# GROUP 7: CLI tests


def test_cli_sync_init_artifacts(tmp_path, monkeypatch):
    """Test CLI sync init artifacts command."""
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

    sync_init_artifacts()

    assert db_path.exists()


def test_cli_sync_push_artifacts_no_inputs(monkeypatch):
    """Test CLI sync push artifacts with no inputs."""
    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_push_artifacts()

    assert exit_code == 1


def test_cli_sync_push_artifacts_force_and_diff_together(tmp_path, monkeypatch):
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
        sync_push_artifacts(source=folder, force=True, diff=True)

    assert exit_code == 1


def test_cli_sync_pull_artifacts_no_artifact_name(monkeypatch):
    """Test CLI sync pull artifacts requires artifact_name unless --overview."""
    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_pull_artifacts()

    assert exit_code == 1


def test_cli_sync_push_artifacts_path_success(tmp_path, monkeypatch):
    """Test CLI sync push artifacts with path input."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)

    # Force sqlite backend
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()
    init_artifact_database()

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_artifact(tmp_path, build_input.hash, build_input)

    # Create mock artifact
    files = ["artifacts/last_frame_pdb/last_frame.pdb"]
    checksums = {"artifacts/last_frame_pdb/last_frame.pdb": "sha256_mock"}
    _create_artifact_data(folder, "last_frame_pdb", files, checksums)

    monkeypatch.chdir(tmp_path)
    sync_push_artifacts(source=folder, artifact_name="last_frame_pdb")

    dm = DataManager("ARTIFACT_LAST_FRAME_PDB")
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == build_input.hash


def test_cli_sync_push_artifacts_duplicate_exits_cleanly(tmp_path, monkeypatch):
    """CLI should exit cleanly (no re-raise traceback) on duplicate-key errors."""
    folder = tmp_path / "folder"
    folder.mkdir()

    monkeypatch.setattr("mdfactory.cli._ensure_sync_target_initialized", lambda *_: None)
    monkeypatch.setattr(
        "mdfactory.utils.push_artifacts.push_artifacts",
        lambda **_: (_ for _ in ()).throw(
            ValueError("40 key(s) already exist in ARTIFACT_LAST_FRAME_PDB")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        sync_push_artifacts(source=folder)

    assert exc.value.code == 1


# GROUP 8: Error propagation tests


def test_push_artifacts_duplicate_raises(tmp_path, monkeypatch):
    """Test push_artifacts raises ValueError on duplicate data in default mode."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("sqlite"))
    init_analysis_database()
    init_artifact_database()

    _, models = _write_csv_with_models(tmp_path)
    build_input = models[0]
    folder = _create_simulation_folder_artifact(tmp_path, build_input.hash, build_input)
    files = ["artifacts/last_frame_pdb/last_frame.pdb"]
    checksums = {"artifacts/last_frame_pdb/last_frame.pdb": "sha256_mock"}
    _create_artifact_data(folder, "last_frame_pdb", files, checksums)

    # First push succeeds
    push_artifacts(source=folder, artifact_name="last_frame_pdb")

    # Second push without force/diff raises
    with pytest.raises(ValueError, match="already exist"):
        push_artifacts(source=folder, artifact_name="last_frame_pdb")
