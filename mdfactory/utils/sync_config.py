# ABOUTME: Configuration wizard for database settings
# ABOUTME: Interactive setup for sqlite, CSV, and Foundry backends using questionary
"""Configuration wizard for database settings."""

import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

import questionary
from loguru import logger

from ..settings import Settings, get_data_dir, get_user_config_path


def normalize_local_path(path_value: str) -> str:
    """Normalize local filesystem paths for persisted config values.

    Expands environment variables and ``~``. Relative paths are converted to
    absolute paths relative to the current working directory.

    Parameters
    ----------
    path_value : str
        Raw path string, possibly containing ``~`` or env vars

    Returns
    -------
    str
        Absolute, expanded path string

    """
    expanded = os.path.expandvars(os.path.expanduser(path_value.strip()))
    if not expanded:
        return expanded

    path = Path(expanded)
    if path.is_absolute():
        return str(path)

    resolved = (Path.cwd() / path).resolve()
    logger.warning(f"Converted relative path '{path_value}' to absolute '{resolved}'.")
    return str(resolved)


def validate_foundry_folder(ctx, path: str) -> bool:
    """Check if a Foundry folder exists.

    Parameters
    ----------
    ctx : FoundryContext
        Foundry context instance
    path : str
        Foundry path to check

    Returns
    -------
    bool
        True if folder exists

    """
    try:
        response = ctx.compass.api_get_resource_by_path(path)
        return response.status_code == 200
    except Exception:
        return False


def configure_sqlite_paths(config: configparser.ConfigParser) -> None:
    """Configure SQLite database paths interactively.

    Parameters
    ----------
    config : configparser.ConfigParser
        Config parser to update with paths

    """
    logger.info("Configuring sqlite paths")
    data_dir = get_data_dir()
    run_db = normalize_local_path(
        questionary.text(
            "RUN_DATABASE path:",
            default=config["sqlite"].get("RUN_DB_PATH", str(data_dir / "runs.db")),
        ).ask()
    )
    analysis_db = normalize_local_path(
        questionary.text(
            "ANALYSIS_DATABASE path:",
            default=config["sqlite"].get("ANALYSIS_DB_PATH", str(data_dir / "analysis.db")),
        ).ask()
    )
    config["sqlite"]["RUN_DB_PATH"] = run_db
    config["sqlite"]["ANALYSIS_DB_PATH"] = analysis_db


def configure_csv_paths(config: configparser.ConfigParser) -> None:
    """Configure CSV file paths interactively.

    Parameters
    ----------
    config : configparser.ConfigParser
        Config parser to update with paths

    """
    logger.info("Configuring CSV file paths")
    data_dir = get_data_dir()
    if "csv" not in config:
        config["csv"] = {}
    run_db = normalize_local_path(
        questionary.text(
            "RUN_DATABASE CSV path:",
            default=config["csv"].get("RUN_DB_PATH", str(data_dir / "runs.csv")),
        ).ask()
    )
    analysis_db = normalize_local_path(
        questionary.text(
            "ANALYSIS_DATABASE directory (for per-table CSV files):",
            default=config["csv"].get("ANALYSIS_DB_PATH", str(data_dir / "analysis")),
        ).ask()
    )
    config["csv"]["RUN_DB_PATH"] = run_db
    config["csv"]["ANALYSIS_DB_PATH"] = analysis_db


def configure_foundry_paths(config: configparser.ConfigParser) -> None:
    """Configure Foundry dataset paths interactively.

    Validates Foundry connectivity and guides user through path setup.

    Parameters
    ----------
    config : configparser.ConfigParser
        Config parser to update with paths

    """
    logger.info("Foundry backend selected. Running `fdt config` to validate setup.")
    if shutil.which("fdt") is None:
        logger.error("`fdt` command not found. Install Foundry Dev Tools and retry.")
        sys.exit(1)
    result = subprocess.run(["fdt", "config"], check=False)
    if result.returncode != 0:
        logger.error("`fdt config` failed. Fix Foundry Dev Tools setup and retry.")
        sys.exit(1)

    fdt_paths = [
        Path("/etc/xdg/foundry-dev-tools/config.toml"),
        Path("~/.foundry-dev-tools/config.toml").expanduser(),
        Path("~/.config/foundry-dev-tools/config.toml").expanduser(),
        Path(".") / ".foundry_dev_tools.toml",
    ]
    fdt_status = {str(p): p.exists() for p in fdt_paths}
    existing_fdt = [p for p, exists in fdt_status.items() if exists]
    if existing_fdt:
        logger.info(f"Detected Foundry Dev Tools config files: {existing_fdt}")
    else:
        logger.info(
            f"No Foundry Dev Tools config files detected in standard locations: {fdt_status}"
        )

    # Validate Foundry connectivity before configuring dataset paths
    try:
        from foundry_dev_tools import FoundryContext

        ctx = FoundryContext()
        _ = ctx.multipass.get_user_info()
        logger.success("Foundry connectivity OK")
    except Exception as exc:
        logger.error(f"Foundry connectivity check failed: {exc}")
        sys.exit(1)

    logger.info("Configuring Foundry dataset paths")

    # Step 1: Ask for base directory
    base_path = questionary.text(
        "Foundry base directory (must already exist in Foundry):",
        default=config["foundry"].get("BASE_PATH", "/Group Functions/mdfactory"),
    ).ask()

    if not base_path.startswith("/"):
        logger.error("Foundry dataset paths must be absolute (start with '/').")
        sys.exit(1)

    # Step 2: Validate that base path exists in Foundry
    if not validate_foundry_folder(ctx, base_path):
        logger.warning(f"Base folder does not exist in Foundry: {base_path}")
        logger.warning("Create this folder in Foundry before running 'sync init analysis'")
        if not questionary.confirm("Continue anyway?", default=False).ask():
            sys.exit(1)

    # Step 3: Ask for analysis directory name
    analysis_name = questionary.text(
        "Analysis directory name:",
        default=config["foundry"].get("ANALYSIS_NAME", "analysis"),
    ).ask()

    # Validate analysis_name
    if "/" in analysis_name or analysis_name.startswith("."):
        logger.error("Analysis name must be a simple directory name (no slashes or leading dots)")
        sys.exit(1)

    # Step 4: Derive paths
    analysis_db_path = f"{base_path}/{analysis_name}"
    run_db_path = f"{base_path}/runs"
    artifact_db_path = f"{base_path}/artifacts"

    # Display derived paths
    logger.info("Derived dataset paths:")
    logger.info(f"  Analysis:  {analysis_db_path}")
    logger.info(f"  Artifacts: {artifact_db_path}")
    logger.info(f"  Runs:      {run_db_path}")

    # Optional override
    if not questionary.confirm("Use these paths?", default=True).ask():
        analysis_db_path = questionary.text(
            "Analysis dataset path:", default=analysis_db_path
        ).ask()
        artifact_db_path = questionary.text(
            "Artifacts dataset path:", default=artifact_db_path
        ).ask()
        run_db_path = questionary.text("Runs dataset path:", default=run_db_path).ask()

    # Store all values
    config["foundry"]["BASE_PATH"] = base_path
    config["foundry"]["ANALYSIS_NAME"] = analysis_name
    config["foundry"]["ANALYSIS_DB_PATH"] = analysis_db_path
    config["foundry"]["ARTIFACT_DB_PATH"] = artifact_db_path
    config["foundry"]["RUN_DB_PATH"] = run_db_path


def initialize_configured_databases() -> dict[str, dict[str, bool]]:
    """Initialize configured systems/analysis/artifact backends.

    Returns
    -------
    dict[str, dict[str, bool]]
        Nested dict mapping resource name ("systems", "analysis", "artifacts")
        to {table_name: was_created} results

    """
    from .push import init_systems_database
    from .push_analysis import init_analysis_database, init_artifact_database

    return {
        "systems": init_systems_database(reset=False),
        "analysis": init_analysis_database(reset=False),
        "artifacts": init_artifact_database(reset=False),
    }


def run_config_wizard() -> None:
    """Run the interactive configuration wizard.

    Guides user through database backend selection and path configuration.
    Writes user config to the platformdirs config location.
    """
    user_config_path = get_user_config_path()
    user_config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Settings()
    data_dir = get_data_dir()

    logger.info("Starting config wizard for mdfactory database settings")

    if user_config_path.exists():
        existing = configparser.ConfigParser()
        existing.read(user_config_path)
        lines = [
            "",
            "Existing user config:",
            f"  Path: {user_config_path}",
        ]
        for section in existing.sections():
            lines.append(f"[{section}]")
            for key, value in existing[section].items():
                lines.append(f"  {key} = {value}")
            lines.append("")
        print("\n".join(lines))

        proceed = questionary.confirm(
            f"Config already exists at {user_config_path}. Overwrite?", default=False
        ).ask()
        if not proceed:
            logger.info("Using existing user config. No changes made.")
            return

    # GROMACS setup (optional)
    use_gromacs = questionary.confirm("Do you use GROMACS (pdb2gmx)?", default=True).ask()
    gmx_path = ""
    forcefield_dir = str(data_dir / "forcefields")
    if use_gromacs:
        gmx_path = questionary.text(
            "Path to gmx binary (leave empty to use PATH):",
            default="",
        ).ask()
        forcefield_dir = questionary.text(
            "Force field download directory:",
            default=str(data_dir / "forcefields"),
        ).ask()
        download_ffs = questionary.confirm(
            "Download CHARMM36m force field now?", default=True
        ).ask()

    # CGenFF setup (optional)
    use_cgenff = questionary.confirm("Do you use CGenFF (SILCSBIO)?", default=False).ask()
    silcsbiodir = ""
    if use_cgenff:
        silcsbiodir = questionary.text("Path to SILCSBIO installation:").ask()

    # Parameter store
    param_dir = questionary.text(
        "Parameter store directory:",
        default=str(data_dir / "parameters"),
    ).ask()

    # Database backend
    backend = questionary.select(
        "Database backend:",
        choices=["sqlite", "csv", "foundry"],
    ).ask()

    config = configparser.ConfigParser()

    # Preserve defaults as baseline
    defaults = cfg._get_defaults()
    for section, options in defaults.items():
        config[section] = {}
        for key, value in options.items():
            config[section][key] = str(value)

    config["gromacs"]["GMX_PATH"] = gmx_path
    config["gromacs"]["FORCEFIELD_DIR"] = normalize_local_path(forcefield_dir)
    config["cgenff"]["SILCSBIODIR"] = silcsbiodir
    config["storage"]["PARAMETERS"] = normalize_local_path(param_dir)
    config["database"]["TYPE"] = backend

    if backend == "sqlite":
        configure_sqlite_paths(config)
    elif backend == "csv":
        configure_csv_paths(config)
    elif backend == "foundry":
        configure_foundry_paths(config)

    with user_config_path.open("w") as handle:
        config.write(handle)
    logger.success(f"Wrote user config to {user_config_path}")

    cfg.ensure_dirs()
    cfg.reload()

    # Download force fields if requested
    if use_gromacs and download_ffs:
        from ..setup.protein import download_all_forcefields

        logger.info("Downloading registered force fields...")
        download_all_forcefields()
        logger.success("Force fields downloaded.")

    init_default = backend in {"sqlite", "csv"}
    if not questionary.confirm(
        "Initialize databases now?",
        default=init_default,
    ).ask():
        logger.info("Skipped database initialization.")
        return

    try:
        init_results = initialize_configured_databases()
    except Exception as exc:
        logger.error(f"Initialization failed: {exc}")
        logger.info("Run one of these commands manually:")
        logger.info("  mdfactory sync init systems")
        logger.info("  mdfactory sync init analysis")
        logger.info("  mdfactory sync init artifacts")
        return

    for name, results in init_results.items():
        created = sum(results.values())
        existing = len(results) - created
        logger.info(f"{name}: {created} created, {existing} already existed")
    logger.success("Database initialization finished.")
