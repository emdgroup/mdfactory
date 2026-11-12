# ABOUTME: Tests for the sync clear-all command that removes records from
# ABOUTME: both run and analysis databases.
"""Tests for the sync clear-all command that removes records from."""

from __future__ import annotations

from datetime import datetime, timezone

from mdfactory.cli import sync_clear_all
from mdfactory.utils.data_manager import DataManager

# temp_run_db and temp_analysis_db fixtures are auto-discovered from conftest.py


def _run_record(hash_value: str) -> dict[str, object]:
    return {
        "hash": hash_value,
        "engine": "gromacs",
        "parametrization": "charmm",
        "simulation_type": "mixedbox",
        "input_data": "{}",
        "input_data_type": "BuildInput",
        "directory": f"/tmp/{hash_value}",
        "status": "build",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _analysis_record(hash_value: str) -> dict[str, object]:
    return {
        "hash": hash_value,
        "directory": f"/tmp/{hash_value}",
        "simulation_type": "bilayer",
        "row_count": 1,
        "columns": "[]",
        "data_csv": "col1\n1\n",
        "data_path": ".analysis/test.parquet",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _overview_record(hash_value: str) -> dict[str, object]:
    return {
        "hash": hash_value,
        "simulation_type": "bilayer",
        "directory": f"/tmp/{hash_value}",
        "item_type": "analysis",
        "item_name": "area_per_lipid",
        "status": "completed",
        "row_count": 1,
        "file_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def test_sync_clear_all(monkeypatch, temp_run_db, temp_analysis_db):
    dm_run = DataManager("RUN_DATABASE")
    dm_run.save_data(_run_record("HASH_RUN"))

    dm_analysis = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm_analysis.save_data(_analysis_record("HASH_ANALYSIS"))

    dm_overview = DataManager("ANALYSIS_OVERVIEW")
    dm_overview.save_data(_overview_record("HASH_ANALYSIS"))

    class _FakeQuestion:
        def ask(self):
            return True

    monkeypatch.setattr("mdfactory.cli.questionary.confirm", lambda *a, **kw: _FakeQuestion())

    sync_clear_all()

    assert dm_run.load_data().empty
    assert dm_analysis.load_data().empty
    assert dm_overview.load_data().empty
