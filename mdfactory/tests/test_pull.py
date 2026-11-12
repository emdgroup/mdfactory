# ABOUTME: Tests for pull functionality (mdfactory sync pull systems)
# ABOUTME: Tests pull_systems(), format_systems_summary(), format_systems_full()

from __future__ import annotations

import pandas as pd

from mdfactory.analysis.constants import SUMMARY_COLUMNS
from mdfactory.cli import sync_pull_systems
from mdfactory.utils.data_manager import DataManager
from mdfactory.utils.pull import (
    format_systems_full,
    format_systems_summary,
    pull_systems,
)

# Import shared helpers from conftest
from .conftest import make_test_record


def _record(hash_value: str, status: str = "build", **overrides) -> dict[str, str]:
    """Create a test record. Delegates to shared helper."""
    return make_test_record(hash_value, status, **overrides)


# GROUP 1: pull_systems() tests


def test_pull_systems_empty_database(temp_run_db):
    """pull_systems returns empty DataFrame when database is empty."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    df = pull_systems()

    assert df.empty
    assert len(df) == 0


def test_pull_systems_returns_all_records(temp_run_db):
    """pull_systems returns all records when no filters are provided."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", status="build"))
    dm.save_data(_record("HASH2", status="completed"))
    dm.save_data(_record("HASH3", status="production"))

    df = pull_systems()

    assert len(df) == 3
    assert set(df["hash"].values) == {"HASH1", "HASH2", "HASH3"}


def test_pull_systems_filter_by_status(temp_run_db):
    """pull_systems filters by status correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", status="build"))
    dm.save_data(_record("HASH2", status="completed"))
    dm.save_data(_record("HASH3", status="completed"))

    df = pull_systems(status="completed")

    assert len(df) == 2
    assert set(df["hash"].values) == {"HASH2", "HASH3"}
    assert all(df["status"] == "completed")


def test_pull_systems_filter_by_simulation_type(temp_run_db):
    """pull_systems filters by simulation_type correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", simulation_type="mixedbox"))
    dm.save_data(_record("HASH2", simulation_type="bilayer"))
    dm.save_data(_record("HASH3", simulation_type="bilayer"))

    df = pull_systems(simulation_type="bilayer")

    assert len(df) == 2
    assert set(df["hash"].values) == {"HASH2", "HASH3"}
    assert all(df["simulation_type"] == "bilayer")


def test_pull_systems_filter_by_parametrization(temp_run_db):
    """pull_systems filters by parametrization correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", parametrization="charmm"))
    dm.save_data(_record("HASH2", parametrization="openff"))

    df = pull_systems(parametrization="openff")

    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH2"
    assert df.iloc[0]["parametrization"] == "openff"


def test_pull_systems_filter_by_engine(temp_run_db):
    """pull_systems filters by engine correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", engine="gromacs"))
    dm.save_data(_record("HASH2", engine="gromacs"))

    df = pull_systems(engine="gromacs")

    assert len(df) == 2
    assert all(df["engine"] == "gromacs")


def test_pull_systems_multiple_filters(temp_run_db):
    """pull_systems applies multiple filters correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", status="completed", simulation_type="bilayer"))
    dm.save_data(_record("HASH2", status="completed", simulation_type="mixedbox"))
    dm.save_data(_record("HASH3", status="build", simulation_type="bilayer"))

    df = pull_systems(status="completed", simulation_type="bilayer")

    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"


def test_pull_systems_no_matches(temp_run_db):
    """pull_systems returns empty DataFrame when no records match filters."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    dm.save_data(_record("HASH1", status="build"))
    dm.save_data(_record("HASH2", status="production"))

    df = pull_systems(status="completed")

    assert df.empty


# GROUP 2: format_systems_summary() tests


def test_format_systems_summary_returns_correct_columns():
    """format_systems_summary returns only the summary columns."""
    df = pd.DataFrame([_record("HASH1"), _record("HASH2")])

    result = format_systems_summary(df)

    assert list(result.columns) == SUMMARY_COLUMNS
    assert "input_data" not in result.columns
    assert "input_data_type" not in result.columns
    assert "timestamp_utc" not in result.columns


def test_format_systems_summary_empty_df():
    """format_systems_summary handles empty DataFrame."""
    df = pd.DataFrame()

    result = format_systems_summary(df)

    assert result.empty


def test_format_systems_summary_preserves_data():
    """format_systems_summary preserves the data values."""
    df = pd.DataFrame([_record("HASH1", status="completed")])

    result = format_systems_summary(df)

    assert result.iloc[0]["hash"] == "HASH1"
    assert result.iloc[0]["status"] == "completed"


def test_format_systems_summary_handles_missing_columns():
    """format_systems_summary gracefully handles missing columns."""
    df = pd.DataFrame([{"hash": "HASH1", "status": "build"}])

    result = format_systems_summary(df)

    assert "hash" in result.columns
    assert "status" in result.columns
    # Missing columns are not included
    assert len(result.columns) == 2


# GROUP 3: format_systems_full() tests


def test_format_systems_full_excludes_json_columns():
    """format_systems_full excludes input_data and input_data_type."""
    df = pd.DataFrame([_record("HASH1")])

    result = format_systems_full(df)

    assert "input_data" not in result.columns
    assert "input_data_type" not in result.columns
    assert "hash" in result.columns
    assert "status" in result.columns
    assert "directory" in result.columns
    assert "timestamp_utc" in result.columns


def test_format_systems_full_empty_df():
    """format_systems_full handles empty DataFrame."""
    df = pd.DataFrame()

    result = format_systems_full(df)

    assert result.empty


def test_format_systems_full_preserves_data():
    """format_systems_full preserves data values."""
    record = _record("HASH1", status="production")
    df = pd.DataFrame([record])

    result = format_systems_full(df)

    assert result.iloc[0]["hash"] == "HASH1"
    assert result.iloc[0]["status"] == "production"
    assert result.iloc[0]["engine"] == "gromacs"


# GROUP 4: CLI sync_pull_systems() tests


def test_cli_pull_systems_empty_database(temp_run_db, capsys):
    """CLI pull_systems handles empty database gracefully."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})

    sync_pull_systems()

    # Should not crash, warning is logged


def test_cli_pull_systems_displays_records(temp_run_db, capsys):
    """CLI pull_systems displays records to stdout."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(_record("HASH1", status="completed"))

    sync_pull_systems()

    captured = capsys.readouterr()
    assert "1 simulation" in captured.out
    assert "HASH1" in captured.out


def test_cli_pull_systems_with_filter(temp_run_db, capsys):
    """CLI pull_systems respects filter arguments."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(_record("HASH1", status="build"))
    dm.save_data(_record("HASH2", status="completed"))

    sync_pull_systems(status="completed")

    captured = capsys.readouterr()
    assert "1 simulation" in captured.out
    assert "HASH2" in captured.out
    assert "HASH1" not in captured.out


def test_cli_pull_systems_output_csv(temp_run_db, tmp_path):
    """CLI pull_systems writes CSV file correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(_record("HASH1", status="completed"))
    dm.save_data(_record("HASH2", status="build"))

    output_path = tmp_path / "output.csv"
    sync_pull_systems(output=output_path)

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 2
    assert "hash" in df.columns
    assert "input_data" in df.columns  # Full data in file output


def test_cli_pull_systems_output_json(temp_run_db, tmp_path):
    """CLI pull_systems writes JSON file correctly."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(_record("HASH1", status="completed"))

    output_path = tmp_path / "output.json"
    sync_pull_systems(output=output_path)

    assert output_path.exists()
    df = pd.read_json(output_path, lines=True)
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"


def test_cli_pull_systems_full_flag(temp_run_db, capsys):
    """CLI pull_systems --full shows more columns."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(_record("HASH1", status="completed"))

    sync_pull_systems(full=True)

    captured = capsys.readouterr()
    assert "timestamp_utc" in captured.out or "engine" in captured.out


def test_cli_pull_systems_combined_filters(temp_run_db, capsys):
    """CLI pull_systems with multiple filters."""
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(_record("HASH1", status="completed", simulation_type="bilayer"))
    dm.save_data(_record("HASH2", status="completed", simulation_type="mixedbox"))
    dm.save_data(_record("HASH3", status="build", simulation_type="bilayer"))

    sync_pull_systems(status="completed", simulation_type="bilayer")

    captured = capsys.readouterr()
    assert "1 simulation" in captured.out
    assert "HASH1" in captured.out


def test_check_run_exists_matches_on_family_name(temp_run_db):
    """check_run_exists queries the parametrization column with the family name."""
    from mdfactory.models.input import BuildInput
    from mdfactory.utils.data_manager import check_run_exists

    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    inp = BuildInput(
        simulation_type="mixedbox",
        parametrization={"family": "openff", "water_model": "tip3p"},
        system={"species": [{"smiles": "O", "resname": "SOL", "count": 10}]},
    )
    assert not check_run_exists(inp)

    dm.save_data(_record(inp.hash, parametrization="charmm"))
    assert not check_run_exists(inp)

    dm.delete_data({})
    dm.save_data(_record(inp.hash, parametrization="openff"))
    assert check_run_exists(inp)
    assert inp.to_data_row()["parametrization"] == "openff"


def test_placeholder_record_uses_valid_family():
    from mdfactory.models.parametrization import FAMILY_SPECS
    from mdfactory.utils.push import get_placeholder_record

    assert get_placeholder_record()["parametrization"] in FAMILY_SPECS
