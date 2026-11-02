# ABOUTME: Shared pytest fixtures for test modules
# ABOUTME: Contains database fixtures and helper functions for simulation folder creation

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mdfactory.prepare import df_to_build_input_models
from mdfactory.settings import Settings


def make_test_record(hash_value: str, status: str = "build", **overrides: Any) -> dict[str, str]:
    """Create a test database record for RUN_DATABASE.

    Parameters
    ----------
    hash_value : str
        Simulation hash
    status : str
        Simulation status (default: "build")
    **overrides
        Additional fields to override defaults

    Returns
    -------
    dict[str, str]
        Record dict suitable for DataManager.save_data()
    """
    record = {
        "hash": hash_value,
        "engine": "gromacs",
        "parametrization": "cgenff",
        "simulation_type": "mixedbox",
        "input_data": "{}",
        "input_data_type": "BuildInput",
        "directory": f"/tmp/{hash_value}",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    record.update(overrides)
    return record


def _sample_rows() -> list[dict[str, object]]:
    """Sample CSV rows for creating test BuildInput models.

    Uses simple 3-character residue names and simple SMILES to avoid
    validation issues in tests. Returns 2 rows with different total_count
    to generate unique hashes.
    """
    return [
        {
            "simulation_type": "mixedbox",
            "engine": "gromacs",
            "parametrization": "cgenff",
            "system.total_count": 1000,
            "system.species.ABC.smiles": "CCC",
            "system.species.ABC.fraction": 0.4,
            "system.species.DEF.smiles": "CCO",
            "system.species.DEF.fraction": 0.6,
        },
        {
            "simulation_type": "mixedbox",
            "engine": "gromacs",
            "parametrization": "cgenff",
            "system.total_count": 1500,
            "system.species.ABC.smiles": "CCC",
            "system.species.ABC.fraction": 0.5,
            "system.species.DEF.smiles": "CCO",
            "system.species.DEF.fraction": 0.5,
        },
    ]


def _write_csv_with_models(tmp_path: Path) -> tuple[Path, list]:
    """Create a CSV file and return path and BuildInput models."""
    df = pd.DataFrame(_sample_rows())
    csv_path = tmp_path / "systems.csv"
    df.to_csv(csv_path, index=False)
    models, errors = df_to_build_input_models(df)
    assert not errors
    return csv_path, models


def _create_simulation_folder(tmp_path: Path, hash_value: str, status: str, build_input) -> Path:
    """Create a complete simulation folder with appropriate status files.

    Status files created based on status parameter:
    - "completed": prod.gro
    - "production": prod.xtc (but NOT prod.gro)
    - "equilibrated": min.gro, nvt.gro, npt.gro (but NOT prod.xtc or prod.gro)
    - "build": no simulation output files

    Also creates system.pdb structure file for Simulation class compatibility.
    """
    folder = tmp_path / hash_value
    folder.mkdir(parents=True, exist_ok=True)

    yaml_path = folder / f"{hash_value}.yaml"
    with open(yaml_path, "w") as handle:
        yaml.safe_dump(build_input.model_dump(), handle)

    # Create minimal structure file for Simulation class
    (folder / "system.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"
    )

    if status == "completed":
        (folder / "prod.gro").touch()
    elif status == "production":
        (folder / "prod.xtc").touch()
    elif status == "equilibrated":
        (folder / "min.gro").touch()
        (folder / "nvt.gro").touch()
        (folder / "npt.gro").touch()

    return folder


@pytest.fixture()
def temp_run_db(tmp_path, monkeypatch):
    """Force RUN_DATABASE to use a temporary sqlite file."""
    db_path = tmp_path / "runs.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name == "RUN_DATABASE" and db_type == "sqlite":
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    # Force sqlite backend
    def patched_init(self):
        for section, options in Settings.DEFAULT_CONFIG.items():
            self.config[section] = dict(options)
        self.config["database"]["TYPE"] = "sqlite"

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", patched_init)

    # Initialize the database via the public init path
    from mdfactory.utils.push import init_systems_database

    init_systems_database()
    return db_path


@pytest.fixture()
def temp_analysis_db(tmp_path, monkeypatch):
    """Force analysis database to use a temporary sqlite file."""
    db_path = tmp_path / "analysis.db"
    original_get_db_path = Settings.get_db_path

    def patched_get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            return str(db_path)
        return original_get_db_path(self, db_name, db_type)

    # Force sqlite backend
    def patched_init(self):
        for section, options in Settings.DEFAULT_CONFIG.items():
            self.config[section] = dict(options)
        self.config["database"]["TYPE"] = "sqlite"

    monkeypatch.setattr(Settings, "get_db_path", patched_get_db_path)
    monkeypatch.setattr(Settings, "_load_config", patched_init)

    # Initialize the database
    from mdfactory.utils.push_analysis import init_analysis_database

    init_analysis_database()
    return db_path


@pytest.fixture()
def protein_pdb(tmp_path) -> Path:
    """One-atom ALA PDB so ProteinSpecies has an existing (unvalidated) pdb_path."""
    path = tmp_path / "test.pdb"
    path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
    return path


def protein_input_data(
    sim_type: str,
    pdb: Path,
    *,
    config: dict[str, Any] | None = None,
    protein: dict[str, Any] | None = None,
    **top_level: Any,
) -> dict[str, Any]:
    """Return minimal valid BuildInput data for a protein simulation type.

    Parameters
    ----------
    sim_type : str
        ``"proteinbox"`` or ``"protein_mixedbox"``.
    pdb : Path
        Path to an existing PDB file.
    config : dict, optional
        Entries for ``parametrization_config`` (its ``type`` defaults to the parametrization).
    protein : dict, optional
        Overrides merged into the ``system.protein`` mapping.
    **top_level
        Overrides replacing top-level keys (e.g. ``parametrization="cgenff"``).
    """
    parametrization = "pdb2gmx" if sim_type == "proteinbox" else "charmm"
    system: dict[str, Any] = {
        "protein": {"resname": "LYZ", "pdb_path": str(pdb), **(protein or {})},
        "padding": 12.0,
    }
    if sim_type == "protein_mixedbox":
        system["species"] = [{"smiles": "O", "resname": "SOL", "count": 10}]
        system["sizing"] = {"type": "fixed_box", "box_size": 72.0}
    data: dict[str, Any] = {
        "simulation_type": sim_type,
        "parametrization": parametrization,
        "system": system,
    }
    if config is not None:
        data["parametrization_config"] = {"type": parametrization, **config}
    data.update(top_level)
    return data
