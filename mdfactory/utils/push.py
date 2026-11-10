# ABOUTME: Helper functions for pushing simulation metadata to database
# ABOUTME: Discovers folders, validates builds, determines status, uploads to DB

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError

from ..analysis.constants import (
    EQUILIBRATION_FILES,
    RUN_DATABASE_COLUMNS,
    SimulationStatus,
)
from ..models.input import BuildInput
from ..prepare import load_build_input_csv
from .data_manager import PLACEHOLDER_HASH
from .db_operations import (
    init_csv_tables,
    init_foundry_tables,
    init_sqlite_tables,
    upload_records,
)
from .utilities import load_yaml_file


def find_yaml_in_folder(folder: Path) -> Path | None:
    """Find BuildInput YAML file in folder with priority ordering.

    Priority order:
    1. <hash>.yaml (try to match folder name if it looks like a hash)
    2. build.yaml (common convention)
    3. Any other .yaml file (alphabetically sorted)

    Parameters
    ----------
    folder : Path
        Directory to search for YAML files

    Returns
    -------
    Path | None
        Path to YAML file if found, None otherwise
    """
    if not folder.is_dir():
        return None

    # Try hash-named YAML (folder name might be the hash)
    folder_name = folder.name
    hash_yaml = folder / f"{folder_name}.yaml"
    if hash_yaml.exists():
        return hash_yaml

    # Try build.yaml
    build_yaml = folder / "build.yaml"
    if build_yaml.exists():
        return build_yaml

    # Try any .yaml file (alphabetically sorted)
    yaml_files = sorted(folder.glob("*.yaml"))
    if yaml_files:
        return yaml_files[0]

    return None


def determine_simulation_status(folder: Path) -> str:
    """Determine simulation status based on output files.

    Status hierarchy (checked in this order):
    - "completed": prod.gro exists
    - "production": prod.xtc exists (but no prod.gro)
    - "equilibrated": min.gro AND nvt.gro AND npt.gro all exist
    - "build": valid build folder but no simulation outputs

    Parameters
    ----------
    folder : Path
        Simulation directory to check

    Returns
    -------
    str
        Status string ("completed", "production", "equilibrated", or "build")

    Note
    ----
    This function delegates to Simulation.status for consistency.
    For new code, prefer using Simulation(folder).status directly.
    """
    from ..analysis.simulation import Simulation

    try:
        sim = Simulation(folder, trajectory_file=None)
        return sim.status
    except (FileNotFoundError, NotADirectoryError, ValueError):
        # Fallback for folders without structure file - use SimulationStatus enum
        # NOTE: prod.gro/prod.xtc are Gromacs defaults; user-configurable via Simulation class
        if (folder / "prod.gro").exists():
            return SimulationStatus.COMPLETED.value
        if (folder / "prod.xtc").exists():
            return SimulationStatus.PRODUCTION.value
        if all((folder / f).exists() for f in EQUILIBRATION_FILES):
            return SimulationStatus.EQUILIBRATED.value
        return SimulationStatus.BUILD.value


def load_models_from_csv(csv_path: Path) -> tuple[list[BuildInput], dict[int, str]]:
    """Load BuildInput models from CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    tuple[list[BuildInput], dict[int, str]]
        Tuple of (models, errors) where errors is a dict mapping row indices to error messages
    """
    # Same loader as prepare-build/build, so relative protein PDB paths resolve
    # identically and the hashes match the built folders.
    _, models, errors = load_build_input_csv(csv_path)
    return models, errors


def search_folders_for_hash(hash_value: str, base_path: Path = Path(".")) -> Path | None:
    """Recursively search for a folder matching the given hash.

    Searches for directories named exactly as the hash value.

    Parameters
    ----------
    hash_value : str
        Hash string to search for (folder name)
    base_path : Path, optional
        Starting directory for recursive search, by default Path(".")

    Returns
    -------
    Path | None
        Path to matching folder if found, None otherwise
    """
    # Search recursively for folder with matching name
    for path in base_path.rglob(f"{hash_value}"):
        if path.is_dir():
            return path
    return None


def discover_simulation_folders(
    source: Path | None = None,
    csv: Path | None = None,
    csv_root: Path | None = None,
) -> list[tuple[Path, BuildInput]]:
    """Discover and validate simulation folders from various input modes.

    Exactly one of source or csv must be provided.

    Parameters
    ----------
    source : Path, optional
        Directory path, glob pattern, or summary YAML file (auto-detected)
    csv : Path, optional
        CSV file with build specifications
    csv_root : Path, optional
        Root directory to search for hash folders when csv is provided

    Returns
    -------
    list[tuple[Path, BuildInput]]
        List of (folder_path, build_input) tuples for valid simulations

    Raises
    ------
    ValueError
        If multiple or no inputs are provided
    """
    # Validate exactly one input provided
    inputs = [source, csv]
    if sum(x is not None for x in inputs) != 1:
        raise ValueError("Exactly one of source or csv must be provided")

    candidate_folders = []

    # Mode 1: Source (directory, glob pattern, or YAML file)
    if source is not None:
        source = Path(source).resolve()

        # Check if it's a glob pattern (contains wildcards)
        if "*" in str(source) or "?" in str(source):
            # Glob pattern mode
            parent = Path(str(source).split("*")[0]).parent
            pattern = str(source.relative_to(parent) if source.is_relative_to(parent) else source)
            candidate_folders = [p for p in parent.glob(pattern) if p.is_dir()]
            logger.info(f"Glob pattern matched {len(candidate_folders)} folder(s)")
        elif source.is_dir():
            # Single folder mode only if YAML is a valid BuildInput; otherwise scan subdirs
            yaml_file = find_yaml_in_folder(source)
            if yaml_file is not None:
                try:
                    yaml_data = load_yaml_file(str(yaml_file))
                    BuildInput(**yaml_data)
                    candidate_folders = [source]
                except (yaml.YAMLError, ValidationError, KeyError, TypeError):
                    logger.debug(f"YAML in source directory is not a valid BuildInput: {yaml_file}")

            if not candidate_folders:
                subdirs = [p for p in source.iterdir() if p.is_dir()]
                candidate_folders = [p for p in subdirs if find_yaml_in_folder(p) is not None]
                if candidate_folders:
                    logger.info(
                        "No valid BuildInput YAML in source directory; "
                        f"found {len(candidate_folders)} simulation folder(s) in subdirectories"
                    )
                else:
                    logger.warning(
                        f"No YAML file found in folder or immediate subdirectories: {source}"
                    )
                    return []
        elif source.is_file() and source.suffix in (".yaml", ".yml"):
            # Summary YAML mode
            logger.info(f"Loading summary YAML: {source}")
            summary_data = load_yaml_file(str(source))

            if "system_directory" not in summary_data:
                raise ValueError(f"Summary YAML missing 'system_directory' field: {source}")

            system_dirs = summary_data["system_directory"]
            candidate_folders = [Path(d) for d in system_dirs]
            logger.info(f"Found {len(candidate_folders)} system directories in YAML")
        else:
            logger.error(f"Source is not a directory, glob pattern, or YAML file: {source}")
            return []

    # Mode 2: CSV
    elif csv is not None:
        csv = Path(csv).resolve()
        if not csv.exists():
            raise FileNotFoundError(f"CSV file not found: {csv}")

        logger.info(f"Loading CSV: {csv}")
        models, errors = load_models_from_csv(csv)

        if errors:
            logger.warning(f"Found {len(errors)} error(s) in CSV:")
            for row_idx, error_msg in errors.items():
                logger.warning(f"  Row {row_idx}: {error_msg}")

        # Extract hashes and search for matching folders
        base_path = Path(csv_root).resolve() if csv_root else Path(".").resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"CSV root path not found: {base_path}")
        logger.info(f"Searching for {len(models)} hash folder(s) recursively from {base_path}...")
        for model in models:
            hash_folder = search_folders_for_hash(model.hash, base_path=base_path)
            if hash_folder:
                candidate_folders.append(hash_folder)
            else:
                logger.warning(f"No folder found for hash: {model.hash}")

        logger.info(f"Found {len(candidate_folders)} matching folder(s) for CSV hashes")

    # Validate each candidate folder
    valid_simulations = []
    skipped_count = 0

    for folder in candidate_folders:
        resolved_folder = folder.resolve()

        # Find YAML file in folder
        yaml_file = find_yaml_in_folder(resolved_folder)
        if yaml_file is None:
            logger.warning(f"No YAML file found in folder: {resolved_folder}")
            skipped_count += 1
            continue

        # Try to load and validate as BuildInput
        try:
            yaml_data = load_yaml_file(str(yaml_file))
            build_input = BuildInput(**yaml_data)
            valid_simulations.append((resolved_folder, build_input))
            logger.debug(
                f"Validated simulation folder: {resolved_folder} (hash: {build_input.hash})"
            )
        except (yaml.YAMLError, ValidationError, KeyError, TypeError, FileNotFoundError) as e:
            logger.warning(f"Failed to validate {resolved_folder}: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} invalid folder(s)")

    return valid_simulations


def prepare_upload_data(simulations: list[tuple[Path, BuildInput]]) -> list[dict[str, Any]]:
    """Convert simulation list to database records.

    Parameters
    ----------
    simulations : list[tuple[Path, BuildInput]]
        List of (folder_path, build_input) tuples

    Returns
    -------
    list[dict[str, Any]]
        List of database records ready for upload
    """
    records = []

    for folder, build_input in simulations:
        # Get base data from BuildInput
        record = build_input.to_data_row()

        # Add additional fields
        record["directory"] = str(folder.resolve())
        record["status"] = determine_simulation_status(folder)
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        records.append(record)

    return records


def upload_simulations(
    records: list[dict[str, Any]],
    db_type: str = "RUN_DATABASE",
    force: bool = False,
    diff: bool = False,
) -> int:
    """Upload records to database with duplicate handling.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Database records to upload
    db_type : str, optional
        Database type to upload to, by default "RUN_DATABASE"
    force : bool, optional
        Delete existing records before uploading, by default False
    diff : bool, optional
        Only upload records not already in database, by default False

    Raises
    ------
    ValueError
        If duplicates exist in default mode (no force/diff flags)
    """
    return upload_records(records, db_type, ["hash"], force=force, diff=diff)


def push_systems(
    source: Path | None = None,
    csv: Path | None = None,
    csv_root: Path | None = None,
    force: bool = False,
    diff: bool = False,
) -> int:
    """Push simulation metadata to RUN_DATABASE.

    Discovers simulation folders and uploads their metadata to the database.

    Parameters
    ----------
    source : Path, optional
        Directory path, glob pattern, or summary YAML file (auto-detected)
    csv : Path, optional
        Input CSV file (hashes will be extracted and folders searched)
    csv_root : Path, optional
        Root directory to search for hash folders when using --csv mode
    force : bool, optional
        Delete existing records before uploading, by default False
    diff : bool, optional
        Only upload records not already in database, by default False

    Returns
    -------
    int
        Number of records uploaded
    """
    simulations = discover_simulation_folders(source=source, csv=csv, csv_root=csv_root)
    records = prepare_upload_data(simulations)
    return upload_simulations(records, db_type="RUN_DATABASE", force=force, diff=diff)


def get_placeholder_record() -> dict[str, Any]:
    """Create a placeholder record with correct schema for initialization."""
    return {
        "hash": PLACEHOLDER_HASH,
        "engine": "gromacs",
        "parametrization": "cgenff",
        "simulation_type": "mixedbox",
        "input_data": "{}",
        "input_data_type": "BuildInput",
        "directory": "/placeholder",
        "status": SimulationStatus.BUILD.value,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def init_systems_database(
    reset: bool = False,
) -> dict[str, bool]:
    """Initialize RUN_DATABASE.

    Creates the database table (SQLite) or Foundry dataset based on config.

    Parameters
    ----------
    reset : bool, optional
        If True, drop and recreate even if it exists. By default False.

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}
    """
    from ..settings import Settings

    config = Settings()
    database_type = config.config["database"]["TYPE"]
    tables = [("RUN_DATABASE", get_placeholder_record(), RUN_DATABASE_COLUMNS)]

    if database_type == "sqlite":
        return init_sqlite_tables(tables, reset=reset)
    elif database_type == "csv":
        return init_csv_tables(tables, reset=reset)
    elif database_type == "foundry":
        return init_foundry_tables(tables, reset=reset)
    else:
        raise ValueError(f"Unsupported database type: {database_type}")
