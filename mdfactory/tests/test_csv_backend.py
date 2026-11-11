# ABOUTME: Tests for the CSV file database backend
# ABOUTME: Covers CsvDataSource, init_csv_tables, and upload modes via CSV

from __future__ import annotations

import pandas as pd
import pytest

from mdfactory.analysis.constants import OVERVIEW_COLUMNS, RUN_DATABASE_COLUMNS
from mdfactory.settings import Settings
from mdfactory.utils.data_manager import DataManager
from mdfactory.utils.db_operations import local_upload_with_modes

from .conftest import default_settings_loader, make_test_record

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_csv_run_db(tmp_path, monkeypatch):
    """Force RUN_DATABASE to use a temporary CSV file."""
    csv_path = tmp_path / "runs.csv"

    original_get_csv_path = Settings.get_csv_path

    def patched_get_csv_path(self, db_name: str) -> str:
        if db_name == "RUN_DATABASE":
            return str(csv_path)
        return original_get_csv_path(self, db_name)

    monkeypatch.setattr(Settings, "get_csv_path", patched_get_csv_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("csv"))

    from mdfactory.utils.push import init_systems_database

    init_systems_database()
    return csv_path


@pytest.fixture()
def temp_csv_analysis_db(tmp_path, monkeypatch):
    """Force analysis database to use temporary CSV files in a directory."""
    analysis_dir = tmp_path / "analysis"

    def patched_get_csv_path(self, db_name: str) -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(analysis_dir / f"{db_name}.csv")
        return Settings.get_csv_path(self, db_name)

    monkeypatch.setattr(Settings, "get_csv_path", patched_get_csv_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("csv"))

    from mdfactory.utils.push_analysis import init_analysis_database

    init_analysis_database()
    return analysis_dir


# ---------------------------------------------------------------------------
# GROUP 1: init_csv_tables
# ---------------------------------------------------------------------------


def test_csv_init_creates_file(temp_csv_run_db):
    """init_systems_database creates a CSV file with correct headers."""
    assert temp_csv_run_db.exists()
    df = pd.read_csv(temp_csv_run_db)
    assert list(df.columns) == RUN_DATABASE_COLUMNS
    assert len(df) == 0


def test_csv_init_already_exists(temp_csv_run_db, monkeypatch):
    """Second init returns {table: False} when file already exists."""
    from mdfactory.utils.push import init_systems_database

    results = init_systems_database()
    assert results["RUN_DATABASE"] is False


def test_csv_init_reset_recreates(temp_csv_run_db, monkeypatch):
    """reset=True recreates CSV even when data exists."""
    # Write some data first
    dm = DataManager("RUN_DATABASE")
    dm.save_data(make_test_record("HASH1"))
    assert len(dm.load_data()) == 1

    from mdfactory.utils.push import init_systems_database

    results = init_systems_database(reset=True)
    assert results["RUN_DATABASE"] is True

    # Data should be gone
    dm2 = DataManager("RUN_DATABASE")
    assert len(dm2.load_data()) == 0


def test_csv_init_creates_directories(tmp_path, monkeypatch):
    """init_csv_tables creates parent directories automatically."""
    nested_path = tmp_path / "deep" / "nested" / "runs.csv"

    def patched_get_csv_path(self, db_name: str) -> str:
        if db_name == "RUN_DATABASE":
            return str(nested_path)
        return str(tmp_path / f"{db_name}.csv")

    monkeypatch.setattr(Settings, "get_csv_path", patched_get_csv_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("csv"))

    from mdfactory.utils.push import init_systems_database

    init_systems_database()

    assert nested_path.exists()
    assert nested_path.parent.name == "nested"


def test_csv_analysis_init_creates_files_in_directory(temp_csv_analysis_db):
    """init_analysis_database creates per-table CSV files in the analysis dir."""
    assert temp_csv_analysis_db.exists()
    csv_files = list(temp_csv_analysis_db.glob("*.csv"))
    assert len(csv_files) > 0

    # ANALYSIS_OVERVIEW should always be created
    overview_path = temp_csv_analysis_db / "ANALYSIS_OVERVIEW.csv"
    assert overview_path.exists()
    df = pd.read_csv(overview_path)
    assert list(df.columns) == OVERVIEW_COLUMNS


# ---------------------------------------------------------------------------
# GROUP 2: CsvDataSource save and load
# ---------------------------------------------------------------------------


def test_csv_save_and_load(temp_csv_run_db):
    """Round-trip save and load via DataManager."""
    dm = DataManager("RUN_DATABASE")
    record = make_test_record("HASH1")
    dm.save_data(record)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"


def test_csv_save_multiple_records(temp_csv_run_db):
    """Save multiple records as a list."""
    dm = DataManager("RUN_DATABASE")
    records = [make_test_record("HASH1"), make_test_record("HASH2")]
    dm.save_data(records)

    df = dm.load_data()
    assert len(df) == 2
    assert set(df["hash"]) == {"HASH1", "HASH2"}


def test_csv_save_deduplicates(temp_csv_run_db):
    """Saving duplicate records skips conflicts (existing wins)."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data(make_test_record("HASH1", status="build"))
    dm.save_data(make_test_record("HASH1", status="completed"))

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "build"


def test_csv_save_overwrite(temp_csv_run_db):
    """save_data with overwrite=True replaces all data."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data(make_test_record("HASH1"))

    new_df = pd.DataFrame([make_test_record("HASH2")])
    dm.save_data(new_df, overwrite=True)

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH2"


# ---------------------------------------------------------------------------
# GROUP 3: CsvDataSource query and delete
# ---------------------------------------------------------------------------


def test_csv_query_data(temp_csv_run_db):
    """query_data filters by conditions."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data(
        [
            make_test_record("HASH1", status="build"),
            make_test_record("HASH2", status="completed"),
        ]
    )

    result = dm.query_data({"status": "completed"})
    assert len(result) == 1
    assert result.iloc[0]["hash"] == "HASH2"


def test_csv_query_data_empty_conditions(temp_csv_run_db):
    """query_data with empty conditions returns all rows."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data([make_test_record("HASH1"), make_test_record("HASH2")])

    result = dm.query_data({})
    assert len(result) == 2


def test_csv_query_unknown_column_raises(temp_csv_run_db):
    """query_data fails fast when condition columns do not exist."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data([make_test_record("HASH1"), make_test_record("HASH2")])

    with pytest.raises(ValueError, match="Unknown condition column"):
        dm.query_data({"unknown_field": "value"})


def test_csv_delete_specific(temp_csv_run_db):
    """delete_data removes only matching rows."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data([make_test_record("HASH1"), make_test_record("HASH2")])

    dm.delete_data({"hash": "HASH1"})

    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH2"


def test_csv_delete_all(temp_csv_run_db):
    """delete_data with empty conditions clears all rows, preserves headers."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data([make_test_record("HASH1"), make_test_record("HASH2")])

    dm.delete_data({})

    df = dm.load_data()
    assert len(df) == 0
    # File should still exist (headers preserved)
    assert temp_csv_run_db.exists()


def test_csv_delete_unknown_column_raises(temp_csv_run_db):
    """delete_data fails fast when condition columns do not exist."""
    dm = DataManager("RUN_DATABASE")
    dm.save_data([make_test_record("HASH1"), make_test_record("HASH2")])

    with pytest.raises(ValueError, match="Unknown condition column"):
        dm.delete_data({"unknown_field": "value"})

    # Existing data remains untouched
    assert len(dm.load_data()) == 2


# ---------------------------------------------------------------------------
# GROUP 4: Upload modes via local_upload_with_modes (reused for CSV)
# ---------------------------------------------------------------------------


def test_csv_upload_default_insert(temp_csv_run_db):
    """Default mode inserts new records."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    records = [make_test_record("HASH1"), make_test_record("HASH2")]

    count = local_upload_with_modes(
        dm=dm,
        records=records,
        key_fields=["hash"],
        table_name="RUN_DATABASE",
        force=False,
        diff=False,
    )

    assert count == 2
    assert len(dm.load_data()) == 2


def test_csv_upload_default_duplicate_error(temp_csv_run_db):
    """Default mode raises ValueError when duplicates exist."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(make_test_record("EXISTING"))

    with pytest.raises(ValueError, match="already exist"):
        local_upload_with_modes(
            dm=dm,
            records=[make_test_record("EXISTING")],
            key_fields=["hash"],
            table_name="RUN_DATABASE",
            force=False,
            diff=False,
        )


def test_csv_upload_diff_skips_existing(temp_csv_run_db):
    """Diff mode skips records that already exist."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(make_test_record("EXISTING"))

    count = local_upload_with_modes(
        dm=dm,
        records=[make_test_record("EXISTING"), make_test_record("NEW")],
        key_fields=["hash"],
        table_name="RUN_DATABASE",
        force=False,
        diff=True,
    )

    assert count == 1
    assert set(dm.load_data()["hash"]) == {"EXISTING", "NEW"}


def test_csv_upload_force_overwrites(temp_csv_run_db):
    """Force mode overwrites existing records."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(make_test_record("HASH1", status="build"))

    count = local_upload_with_modes(
        dm=dm,
        records=[make_test_record("HASH1", status="completed")],
        key_fields=["hash"],
        table_name="RUN_DATABASE",
        force=True,
        diff=False,
    )

    assert count == 1
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# GROUP 5: DataManager.database_exists
# ---------------------------------------------------------------------------


def test_csv_database_exists(temp_csv_run_db):
    """database_exists returns True for initialized CSV."""
    exists, path = DataManager.database_exists("RUN_DATABASE")
    assert exists is True
    assert path == str(temp_csv_run_db)


def test_csv_database_not_exists(tmp_path, monkeypatch):
    """database_exists returns False for non-existent CSV."""
    csv_path = tmp_path / "nonexistent.csv"

    def patched_get_csv_path(self, db_name: str) -> str:
        return str(csv_path)

    monkeypatch.setattr(Settings, "get_csv_path", patched_get_csv_path)
    monkeypatch.setattr(Settings, "_load_config", default_settings_loader("csv"))

    exists, path = DataManager.database_exists("RUN_DATABASE")
    assert exists is False


# ---------------------------------------------------------------------------
# GROUP 6: Settings.get_csv_path
# ---------------------------------------------------------------------------


def test_get_csv_path_static_table():
    """get_csv_path returns file path for static tables."""
    config = Settings()
    path = config.get_csv_path("RUN_DATABASE")
    assert path.endswith("runs.csv")


def test_get_csv_path_static_analysis_database():
    """get_csv_path returns configured analysis base path for ANALYSIS_DATABASE."""
    config = Settings()
    path = config.get_csv_path("ANALYSIS_DATABASE")
    assert "analysis" in path
    assert not path.endswith(".csv")


def test_get_csv_path_dynamic_analysis_table():
    """get_csv_path returns directory-based path for analysis tables."""
    config = Settings()
    path = config.get_csv_path("ANALYSIS_AREA_PER_LIPID")
    assert path.endswith("ANALYSIS_AREA_PER_LIPID.csv")
    assert "analysis" in path


def test_get_csv_path_dynamic_artifact_table():
    """get_csv_path returns directory-based path for artifact tables."""
    config = Settings()
    path = config.get_csv_path("ARTIFACT_BILAYER_MAPS")
    assert path.endswith("ARTIFACT_BILAYER_MAPS.csv")
    assert "analysis" in path
