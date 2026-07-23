# ABOUTME: Command-line interface for mdfactory using cyclopts
# ABOUTME: Provides build, sync, and analysis commands for MD simulation management
"""Command-line interface for mdfactory using cyclopts."""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import pandas as pd
import questionary
import yaml
from cyclopts import App, Parameter
from loguru import logger

from . import __version__
from .analysis.store import SimulationStore
from .analysis.submit import (
    SlurmConfig,
    determine_log_dir,
    filter_paths_by_hash,
    resolve_simulation_paths,
    resolve_simulation_paths_from_yaml,
    run_analyses_local,
    submit_analyses_slurm,
    submit_artifacts_slurm,
)
from .analysis.utils import get_chemistry_extractor
from .prepare import df_to_build_input_models
from .utils.data_manager import check_run_exists
from .utils.push import push_systems
from .utils.utilities import working_directory
from .workflows import run_build_from_dict, run_build_from_file

if TYPE_CHECKING:
    from .orchestration import ExecutorConfig

app = App(name="MDFactory", version=__version__)


@app.command(group="Build")
def prepare_build(input: Path, output: Path = Path(".")):
    """Prepare YAML files for system build from a CSV data frame.

    Parameters
    ----------
    input : Path
        Path to the CSV file with the compositions
    output : Path, optional
        Output directory, by default Path(".")

    """
    input = input.resolve()
    output = output.resolve()
    output.mkdir(exist_ok=True)

    logger.info(f"Building YAML files from CSV {input} and output directory: {output}.")

    yml_build_path = output / f"{input.stem}.yaml"
    if yml_build_path.is_file():
        glob_backup = list(output.glob(f"#{input.stem}.yaml.bak.*#"))
        nbak = len(glob_backup)
        yml_bak = output / f"#{input.stem}.yaml.bak.{nbak}#"
        logger.warning(
            f"Back off! I will back up the info YAML file: {yml_build_path} -> {yml_bak}"
        )
        shutil.copy(yml_build_path, yml_bak)

    df, models, errors = df_models_from_input_csv(input)
    for m in models:
        if check_run_exists(m):
            raise ValueError("This system has already been built and is in the database:")
    if errors:
        logger.error("Error occured in building models from CSV data.")
        for k, v in errors.items():
            logger.error(f"\t Row {k} -- {v}")
        sys.exit(1)

    logger.info("Successfully created PyDantic models for all rows.")
    df["model"] = models
    df["hash"] = df.apply(lambda x: x.model.hash, axis=1)

    logger.info("Created DataFrame for systems:\n{}", df)

    _prepare_system_directories(models, input, output, exist_ok=False)


def df_models_from_input_csv(input):
    df = pd.read_csv(input)
    # remove 'unnamed' CSV columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    models, errors = df_to_build_input_models(df)
    return df, models, errors


def _prepare_system_directories(
    models: list,
    input_path: Path,
    output: Path,
    *,
    exist_ok: bool = True,
) -> tuple[list[str], Path]:
    """Create per-hash build directories, write system YAMLs, and generate summary.

    Parameters
    ----------
    models : list
        List of BuildInput models.
    input_path : Path
        Original input file path (used for summary YAML naming).
    output : Path
        Base output directory.
    exist_ok : bool
        Whether to allow existing directories.

    Returns
    -------
    tuple[list[str], Path]
        List of system directory paths, and path to the summary YAML.

    """
    dirs = []
    for model in models:
        build_dir = output / model.hash
        build_dir.mkdir(parents=True, exist_ok=exist_ok)
        yml_path = build_dir / f"{model.hash}.yaml"
        if not yml_path.exists():
            with open(yml_path, "w") as f:
                yaml.safe_dump(model.model_dump(), f)
        dirs.append(str(build_dir.resolve()))

    summary_path = output / f"{input_path.stem}.yaml"
    summary = {
        "n_systems": len(models),
        "input": str(input_path),
        "output": str(output),
        "hash": [m.hash for m in models],
        "simulation_type": [m.simulation_type for m in models],
        "system_directory": dirs,
        "date": datetime.now(),
    }
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f)
    return dirs, summary_path


def _resolve_slurm_flag(slurm: str | None) -> "ExecutorConfig | Path | None":
    """Resolve the ``--slurm`` CLI flag to an executor config or YAML path.

    Parameters
    ----------
    slurm : str or None
        Raw value from the ``--slurm`` flag.  ``"tui"`` launches the
        interactive wizard (which returns a config object directly).
        Any other non-None value is treated as a file path.

    Returns
    -------
    ExecutorConfig, Path, or None
        An ExecutorConfig when using TUI mode, a Path to executor config
        YAML, or None if no SLURM config requested.

    """
    if slurm is None:
        return None

    if slurm.strip().lower() == "tui":
        from mdfactory.orchestration.tui import UserCancelledError, configure_slurm_interactive

        try:
            return configure_slurm_interactive()
        except UserCancelledError:
            logger.info("SLURM configuration cancelled.")
            sys.exit(0)

    path = Path(slurm)
    if not path.exists():
        logger.error(f"SLURM config file not found: {path}")
        sys.exit(1)
    return path


def _load_executor_config(config: "ExecutorConfig | Path | None") -> "ExecutorConfig":
    """Load executor config from path or return directly if already loaded.

    Parameters
    ----------
    config : ExecutorConfig, Path, or None
        An already-loaded config object, a path to a YAML file, or None
        for default local execution.

    Returns
    -------
    ExecutorConfig
        The resolved executor configuration.

    """
    if config is None:
        from mdfactory.orchestration import ExecutorConfig

        return ExecutorConfig()
    if isinstance(config, Path):
        from mdfactory.orchestration import ExecutorConfig

        return ExecutorConfig.from_yaml(config)
    return config  # already an ExecutorConfig instance


@app.command(name="build", group="Build")
def build_system(
    input: Path,
    output: Path = Path("."),
    slurm: Annotated[
        str | None,
        Parameter(
            help=("SLURM executor config: path to YAML file, or 'tui' for interactive setup.")
        ),
    ] = None,
    dry_run: Annotated[
        bool, Parameter(help="Print what would be built without executing.")
    ] = False,
):
    """Build MD system(s) from YAML, CSV, or summary input.

    Supports three input modes:

    - Single YAML: builds one system locally (original behavior)
    - CSV file: builds systems from each row (parallel with --slurm, sequential without)
    - Summary YAML: dispatches builds for previously prepared systems

    Parameters
    ----------
    input : Path
        Path to input file (YAML for single build, CSV for batch, summary YAML for prepared).
    output : Path, optional
        Output directory, by default Path(".").
    slurm : str, optional
        Path to SLURM executor config YAML, or ``"tui"`` to launch the
        interactive configuration wizard.
    dry_run : bool, optional
        Print what would be built without executing.

    """
    input = input.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Resolve --slurm flag: "tui" launches the interactive wizard,
    # anything else is treated as a path to a YAML config file.
    config: "ExecutorConfig | Path | None" = _resolve_slurm_flag(slurm)

    suffix = input.suffix.lower()

    if suffix == ".csv":
        _build_from_csv(input, output, config=config, dry_run=dry_run)
    elif suffix in (".yaml", ".yml"):
        _build_from_yaml(input, output, config=config, dry_run=dry_run)
    else:
        logger.error(f"Unsupported input file format: {suffix}. Use .csv, .yaml, or .yml.")
        sys.exit(1)


def _build_from_csv(
    input: Path,
    output: Path,
    *,
    config: "ExecutorConfig | Path | None" = None,
    dry_run: bool = False,
):
    """Handle CSV input for the build command."""
    df, models, errors = df_models_from_input_csv(input)
    if errors:
        logger.error("Errors building models from CSV:")
        for k, v in errors.items():
            logger.error(f"  Row {k}: {v}")
        sys.exit(1)

    if dry_run:
        from mdfactory.orchestration import build_systems

        executor_config = _load_executor_config(config)
        build_systems(models, executor_config, output_dir=output, dry_run=True)
        return

    # Create per-hash directories, write per-system YAMLs, and generate summary
    dirs, summary_path = _prepare_system_directories(models, input, output, exist_ok=True)
    logger.info(f"Summary YAML written to {summary_path}")

    if config is not None:
        # Parallel builds via Parsl
        from mdfactory.orchestration import build_systems

        executor_config = _load_executor_config(config)
        results = build_systems(models, executor_config, output_dir=output)
        _report_build_results(results, output_dir=output)
    else:
        # Sequential local builds
        logger.info(f"Building {len(models)} system(s) sequentially.")
        for model in models:
            build_dir = output / model.hash
            logger.info(f"Building {model.hash} ({model.simulation_type})...")
            with working_directory(build_dir, create=True):
                run_build_from_dict(model)


def _build_from_yaml(
    input: Path,
    output: Path,
    *,
    config: "ExecutorConfig | Path | None" = None,
    dry_run: bool = False,
):
    """Handle YAML input for the build command."""
    with open(input) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        logger.error(f"Invalid YAML input file (empty or not a mapping): {input}")
        sys.exit(1)

    if "system_directory" in data:
        # Summary YAML -- dispatch builds for prepared systems
        _build_from_summary_yaml(data, output, config=config, dry_run=dry_run)
    else:
        # Single build YAML
        from mdfactory.models.input import BuildInput

        model = BuildInput(**data)

        if config is not None or dry_run:
            from mdfactory.orchestration import build_systems

            executor_config = _load_executor_config(config)
            results = build_systems([model], executor_config, output_dir=output, dry_run=dry_run)
            if not dry_run:
                _report_build_results(results, output_dir=output)
        else:
            build_dir = output / model.hash
            logger.info(f"Building {model.hash} ({model.simulation_type}) -> {build_dir}")
            with working_directory(build_dir, create=True):
                run_build_from_file(input)
            logger.info(f"Output: {output}")
            logger.info("Next step — run simulations:")
            logger.info(f"  mdfactory simulate {output}")
            logger.info(f"  # with SLURM: mdfactory simulate {output} --slurm examples/slurm_gpu.yaml")


def _build_from_summary_yaml(
    data: dict,
    output: Path,
    *,
    config: "ExecutorConfig | Path | None" = None,
    dry_run: bool = False,
):
    """Handle summary YAML (from prepare-build) for the build command."""
    dirs = data.get("system_directory", [])
    hashes = data.get("hash", [])

    if not dirs:
        logger.error("Summary YAML contains no system_directory entries.")
        sys.exit(1)

    if len(dirs) != len(hashes):
        logger.error(
            f"Summary YAML is malformed: system_directory has {len(dirs)} entries "
            f"but hash has {len(hashes)}."
        )
        sys.exit(1)

    from mdfactory.models.input import BuildInput

    models = []
    for sim_dir, sim_hash in zip(dirs, hashes):
        yml_path = Path(sim_dir) / f"{sim_hash}.yaml"
        if not yml_path.exists():
            logger.error(f"Build YAML not found: {yml_path}")
            sys.exit(1)
        with open(yml_path) as f:
            model_data = yaml.safe_load(f)
        models.append(BuildInput(**model_data))

    if config is not None or dry_run:
        from mdfactory.orchestration import build_systems

        executor_config = _load_executor_config(config)
        results = build_systems(models, executor_config, output_dir=output, dry_run=dry_run)
        if not dry_run:
            _report_build_results(results, output_dir=output)
    else:
        # Sequential local builds
        logger.info(f"Building {len(models)} prepared system(s) sequentially.")
        for model, sim_dir in zip(models, dirs):
            logger.info(f"Building {model.hash} ({model.simulation_type})...")
            with working_directory(Path(sim_dir), create=True):
                run_build_from_dict(model)
        logger.info(f"Output: {output}")
        logger.info("Next step — run simulations:")
        logger.info(f"  mdfactory simulate {output}")
        logger.info(f"  # with SLURM: mdfactory simulate {output} --slurm examples/slurm_gpu.yaml")


def _report_build_results(results: list[dict], output_dir: Path | None = None):
    """Report build results to the user.

    Parameters
    ----------
    results : list[dict]
        Build result dicts from build_systems().
    output_dir : Path, optional
        Output directory used for the build.  When provided, a "next steps"
        hint pointing to ``mdfactory simulate`` is printed so users know
        exactly what to run next.

    """
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    logger.info(f"Build complete: {succeeded} succeeded, {failed} failed.")
    for r in results:
        if r.get("status") == "failed":
            logger.error(f"  {r.get('hash', 'unknown')}: {r.get('error', 'unknown error')}")
    if succeeded and output_dir is not None:
        logger.info(f"Output: {output_dir}")
        logger.info("Next step — run simulations:")
        logger.info(f"  mdfactory simulate {output_dir}")
        logger.info(f"  # with SLURM: mdfactory simulate {output_dir} --slurm examples/slurm_gpu.yaml")


def _filter_sim_paths_by_hash_prefix(
    sim_paths: list[Path], prefixes: list[str]
) -> list[Path]:
    """Filter simulation paths by hash prefix without requiring completed trajectories.

    Unlike :func:`~mdfactory.analysis.submit.filter_paths_by_hash`, this
    helper works purely on directory names and therefore works on freshly
    built directories that have not yet been simulated.

    Hash matching is case-insensitive so short prefixes typed in lower-case
    still resolve correctly.

    Parameters
    ----------
    sim_paths : list[Path]
        Candidate simulation directories (already validated by
        :func:`_discover_sim_dirs`).
    prefixes : list[str]
        Hash strings or prefix fragments to match against directory names.

    Returns
    -------
    list[Path]
        Subset of *sim_paths* whose directory names start with at least one
        requested prefix.

    Raises
    ------
    ValueError
        If none of the requested prefixes match any discovered directory.

    """
    upper_prefixes = [p.upper() for p in prefixes]
    matched = [p for p in sim_paths if any(p.name.upper().startswith(pref) for pref in upper_prefixes)]
    if not matched:
        available = sorted(p.name for p in sim_paths)
        raise ValueError(
            f"No simulation directories match {prefixes}.\n"
            f"Available: {available}"
        )
    return sorted(matched)


def _discover_sim_dirs(root: Path) -> list[Path]:
    """Find simulation-ready directories under *root*.

    A directory is considered simulation-ready when it contains ``system.pdb``.
    Unlike :func:`~mdfactory.analysis.utils.discover_simulations` (which is
    designed for post-run analysis), this helper does **not** require a
    ``BuildInput`` YAML or a completed trajectory — it works on freshly
    prepared directories.

    Rules
    -----
    - If *root* itself contains ``system.pdb`` it is returned as a single-item
      list (direct simulation directory).
    - Otherwise the immediate children of *root* that contain ``system.pdb``
      are returned (build-output layout where each hash subdirectory is one
      simulation).

    Raises
    ------
    ValueError
        If no simulation-ready directories are found under *root*.

    """
    if (root / "system.pdb").exists():
        return [root]

    sims = sorted(d for d in root.iterdir() if d.is_dir() and (d / "system.pdb").exists())
    if not sims:
        raise ValueError(
            f"No simulation directories found under {root}.\n"
            "Each simulation directory must contain system.pdb.  "
            "Run 'mdfactory build' first, or point directly to a prepared "
            "simulation directory."
        )
    return sims


@app.command(name="simulate", group="Simulation")
def simulate_systems(
    source: Annotated[Path, Parameter(help="Simulation directory or summary YAML")],
    slurm: Annotated[
        str | None,
        Parameter(help="SLURM config YAML or 'tui' for interactive setup"),
    ] = None,
    stages: Annotated[
        list[str] | None,
        Parameter(help="Stages to run (EM, NVT, NPT, Production)"),
    ] = None,
    checkpoint: Annotated[
        str,
        Parameter(help="Checkpoint mode: auto (default), skip, force"),
    ] = "auto",
    hash: Annotated[
        list[str] | None,
        Parameter(help="Filter by hash prefix"),
    ] = None,
    dry_run: Annotated[
        bool,
        Parameter(help="Preview plan without executing"),
    ] = False,
):
    """Run GROMACS MD simulations via Parsl.

    Examples
    --------
    Local execution (all stages)::

        mdfactory simulate output_dir/

    SLURM with interactive config::

        mdfactory simulate output_dir/ --slurm tui

    Resume from checkpoint::

        mdfactory simulate output_dir/ --slurm gpu.yaml --checkpoint auto

    Equilibration only::

        mdfactory simulate output_dir/ --stages EM NVT NPT

    Filter specific simulations::

        mdfactory simulate output_dir/ --hash abc123 def456

    Dry-run preview::

        mdfactory simulate output_dir/ --slurm gpu.yaml --dry-run

    """
    source = source.resolve()

    # Resolve simulation paths
    if source.is_dir():
        try:
            sim_paths = _discover_sim_dirs(source)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)
    elif source.suffix in (".yaml", ".yml"):
        # Load from summary YAML
        with open(source) as f:
            summary = yaml.safe_load(f)
        sim_paths = [Path(p) for p in summary.get("system_directory", [])]
    else:
        logger.error(f"Invalid source: {source}. Provide directory or YAML.")
        sys.exit(1)

    # Filter by hash if provided.
    # Use a simple directory-name prefix match rather than SimulationStore.discover()
    # so this works on freshly built directories that have no trajectory yet.
    if hash:
        try:
            sim_paths = _filter_sim_paths_by_hash_prefix(sim_paths, list(hash))
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)

    if not sim_paths:
        logger.error("No simulations found.")
        sys.exit(1)

    logger.info(f"Found {len(sim_paths)} simulation(s)")

    # Validate stages
    if stages:
        all_stages = ["EM", "NVT", "NPT", "Production"]
        invalid = [s for s in stages if s not in all_stages]
        if invalid:
            logger.error(f"Invalid stages: {invalid}. Valid: {all_stages}")
            sys.exit(1)

    # Validate checkpoint mode
    if checkpoint not in ["auto", "skip", "force"]:
        logger.error(f"Invalid checkpoint mode: {checkpoint}. Valid: auto, skip, force")
        sys.exit(1)

    # Resolve SLURM config
    config = _resolve_slurm_flag(slurm)
    executor_config = _load_executor_config(config)

    # Run simulations
    from mdfactory.orchestration import run_simulations

    results = run_simulations(
        sim_paths,
        executor_config,
        stages=stages,
        checkpoint_mode=checkpoint,
        dry_run=dry_run,
    )

    # Report results
    if not dry_run:
        _report_simulation_results(results)


def _report_simulation_results(results: list[dict]):
    """Report simulation results to user."""
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    logger.info(f"Simulation complete: {succeeded} succeeded, {failed} failed")

    for r in results:
        if r.get("status") == "failed":
            logger.error(f"  {r.get('hash', 'unknown')}: {r.get('error', 'unknown error')}")


@app.command(name="clean")
def clean(parameters: bool = True, database: bool = True):
    import shutil

    from mdfactory.settings import settings
    from mdfactory.utils.data_manager import DataManager

    parameter_store = settings.parameter_store

    if parameters:
        if questionary.confirm(
            "This will remove all stored parameter sets. Are you sure?",
            default=False,
        ).ask():
            logger.warning(f"Removing parameter store at {parameter_store}.")
            shutil.rmtree(parameter_store, ignore_errors=True)
    if database:
        if questionary.confirm(
            "This will delete all entries in the run database. Are you sure?",
            default=False,
        ).ask():
            dm = DataManager("RUN_DATABASE")
            logger.warning("Deleting all entries in the run database.")
            dm.delete_data(conditions={})


@app.command(name="showdb")
def show_db(name: Literal["run"] = "run"):
    """Show information about the specified database."""
    from mdfactory.utils.data_manager import DataManager

    dbs = {
        "run": "RUN_DATABASE",
    }
    db = dbs.get(name, None)
    if db is None:
        raise KeyError(
            f"Database {name} not recognized. Available databases are: {list(dbs.keys())}"
        )
    dm = DataManager(db)
    df = dm.load_data()
    print(f"Database: {db}")
    print(f"Number of entries: {len(df)}")
    print(f"Columns: {list(df.columns)}")


def info():
    # NOTE: print system info?
    pass


@app.command(group="Build")
def check_csv(input: Path):
    """Check if CSV is valid by building `BuildInput` models from each row.

    Parameters
    ----------
    input : Path
        Input CSV file to check.

    """
    input = input.resolve()
    _, models, errors = df_models_from_input_csv(input)
    if errors:
        logger.error("Error occured in building models from CSV data.")
        for k, v in errors.items():
            logger.error(f"\t Row {k} -- {v}")
        sys.exit(1)
    build_errors = {}
    for im, m in enumerate(models):
        if check_run_exists(m):
            build_errors[im] = (
                f"This system has already been built and is in the database: {m.hash}"
            )
            continue
        if m.simulation_type == "bilayer" and not errors:
            from .check import check_bilayer_buildable

            try:
                check_bilayer_buildable(m.system)  # type: ignore[arg-type]
                logger.info(f"Bilayer system {m.hash} / Row {im} looks good ✅.")
            except Exception as e:
                logger.error(f"Bilayer system {m.hash} / Row {im} cannot be built ❌.\n Error: {e}")
                build_errors[im] = str(e)
    if build_errors:
        logger.error("Error occured in checking buildability of bilayer systems.")
        for k, v in build_errors.items():
            logger.error(f"\t Row {k} -- {v}")
        sys.exit(1)
    logger.info("🍔 Everything fine with your CSV.")


sync_app = App(help="Synchronize system and analysis metadata with configured backend.")
app.command(sync_app, name="sync")


def _validate_sync_push_inputs(
    source: Path | None,
    csv: Path | None,
    csv_root: Path | None,
    force: bool,
    diff: bool,
) -> None:
    """Validate sync push command inputs.

    Parameters
    ----------
    source : Path | None
        Source directory or glob pattern
    csv : Path | None
        CSV input file
    csv_root : Path | None
        Root for CSV hash folder search
    force : bool
        Overwrite mode flag
    diff : bool
        Skip-existing mode flag

    Raises
    ------
    SystemExit
        If validation fails (exactly one input required, incompatible flags)

    """
    inputs = [source, csv]
    if sum(x is not None for x in inputs) != 1:
        logger.error("Exactly one of --source or --csv must be provided")
        sys.exit(1)
    if csv_root is not None and csv is None:
        logger.error("--csv-root can only be used together with --csv")
        sys.exit(1)
    if force and diff:
        logger.error("Cannot use --force and --diff together")
        sys.exit(1)


def _ensure_sync_target_initialized(table_name: str, init_command: str) -> None:
    """Ensure local backend database/file exists and is initialized.

    Parameters
    ----------
    table_name : str
        Table name to check (e.g., "RUN_DATABASE")
    init_command : str
        CLI command to show in error message if not initialized

    Raises
    ------
    SystemExit
        If the database/file does not exist or is not initialized

    """
    from .settings import Settings
    from .utils.data_manager import DataManager

    backend = Settings().config["database"].get("TYPE", "sqlite").lower()
    if backend == "foundry":
        return

    try:
        DataManager(table_name)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        logger.error(f"Initialize it first with: `{init_command}`")
        sys.exit(1)


def _exit_sync_push_error(exc: Exception) -> None:
    """Log push errors and exit without Python traceback.

    Parameters
    ----------
    exc : Exception
        The exception that caused the push failure

    Raises
    ------
    SystemExit
        Always exits with code 1

    """
    logger.error(f"Push failed: {exc}")
    if isinstance(exc, ValueError) and "already exist" in str(exc):
        logger.info("Use `--diff` to skip existing records or `--force` to overwrite.")
    sys.exit(1)


sync_push_app = App(help="Push local systems or analyses into the database.")
sync_app.command(sync_push_app, name="push")


@sync_push_app.command(name="systems")
def sync_push_systems(
    source: Annotated[
        Path | None, Parameter(help="Directory, glob pattern, or summary YAML file.")
    ] = None,
    csv: Annotated[Path | None, Parameter(help="Input CSV to locate hash folders.")] = None,
    csv_root: Annotated[
        Path | None, Parameter(help="Root directory to search when using --csv.")
    ] = None,
    force: Annotated[
        bool, Parameter(help="Overwrite existing records for matching hashes.")
    ] = False,
    diff: Annotated[
        bool, Parameter(help="Skip hashes that already exist in the database.")
    ] = False,
):
    """Push simulation system metadata to database.

    Discovers simulation folders and uploads their metadata (hash, status,
    directory, build input) to the runs database.

    Exactly one of --source or --csv must be provided.

    Parameters
    ----------
    source : Path, optional
        Directory, glob pattern (e.g., "systems/*/"), or summary YAML file
    csv : Path, optional
        Input CSV file (hashes will be extracted and folders searched)
    csv_root : Path, optional
        Root directory to search for hash folders when using --csv mode
    force : bool, default False
        Delete existing records and re-insert (overwrite duplicates)
    diff : bool, default False
        Only upload new hashes that don't exist in database yet

    """
    logger.info("Starting mdfactory sync push systems")
    logger.info(f"Input: source={source}, csv={csv}, csv_root={csv_root}")
    logger.info(f"Flags: force={force}, diff={diff}")

    _validate_sync_push_inputs(source, csv, csv_root, force, diff)
    _ensure_sync_target_initialized("RUN_DATABASE", "mdfactory sync init systems")

    try:
        count = push_systems(
            source=source,
            csv=csv,
            csv_root=csv_root,
            force=force,
            diff=diff,
        )
        logger.success(f"Push complete: {count} simulation(s)")
    except Exception as e:
        _exit_sync_push_error(e)


sync_pull_app = App(help="Pull systems or analyses from the database.")
sync_app.command(sync_pull_app, name="pull")


@sync_pull_app.command(name="systems")
def sync_pull_systems(
    status: Annotated[
        str | None, Parameter(help="Filter by status (build/equilibrated/production/completed).")
    ] = None,
    simulation_type: Annotated[
        str | None, Parameter(help="Filter by simulation type (mixedbox/bilayer).")
    ] = None,
    parametrization: Annotated[
        str | None, Parameter(help="Filter by parametrization (cgenff/smirnoff).")
    ] = None,
    engine: Annotated[str | None, Parameter(help="Filter by engine (e.g., gromacs).")] = None,
    output: Annotated[
        Path | None, Parameter(help="Write results to .csv or .json instead of stdout.")
    ] = None,
    full: Annotated[bool, Parameter(help="Show all columns in CLI output.")] = False,
):
    """Pull simulation system metadata from database.

    Retrieves simulation records from the runs database and outputs
    them to CLI or file.

    Parameters
    ----------
    status : str, optional
        Filter by status (build, equilibrated, production, completed)
    simulation_type : str, optional
        Filter by simulation type (mixedbox, bilayer)
    parametrization : str, optional
        Filter by parametrization (cgenff, smirnoff)
    engine : str, optional
        Filter by engine (gromacs)
    output : Path, optional
        Output file path (.csv or .json). If not provided, prints to CLI.
        File output always includes all columns.
    full : bool, default False
        Show all columns in CLI output (excluding only JSON blob).
        Default shows summary columns: hash, simulation_type, parametrization, status, directory

    """
    from .utils.pull import format_systems_full, format_systems_summary, pull_systems

    logger.info("Starting mdfactory sync pull systems")

    df = pull_systems(
        status=status,
        simulation_type=simulation_type,
        parametrization=parametrization,
        engine=engine,
    )

    if df.empty:
        logger.warning("No records found matching criteria.")
        return

    # File output - always includes all columns
    if output is not None:
        output = output.resolve()
        suffix = output.suffix.lower()

        if suffix == ".json":
            df.to_json(output, orient="records", lines=True)
        else:
            # Default to CSV
            df.to_csv(output, index=False)

        logger.success(f"Wrote {len(df)} record(s) to {output}")
        return

    # CLI output
    if full:
        display_df = format_systems_full(df)
    else:
        display_df = format_systems_summary(df)

    print(f"\nFound {len(df)} simulation(s):\n")
    print(display_df.to_string(index=False))


sync_init_app = App(help="Initialize databases or datasets for systems or analyses.")
sync_app.command(sync_init_app, name="init")


@sync_init_app.command(name="systems")
def sync_init_systems(
    reset: Annotated[
        bool,
        Parameter(alias="--force", help="Reset and recreate existing systems dataset/table."),
    ] = False,
):
    """Initialize the systems database (RUN_DATABASE).

    Creates the database or Foundry dataset if it doesn't exist.
    For Foundry, validates schema compatibility if dataset exists.

    The backend type (sqlite or foundry) is determined by config.

    Parameters
    ----------
    reset : bool, default False
        Reset and recreate existing dataset/table. `--force` is an alias for `--reset`.

    """
    from .utils.push import init_systems_database

    logger.info("Initializing systems database (RUN_DATABASE)")

    try:
        results = init_systems_database(reset=reset)
        _log_init_results(results, "Systems")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


def _log_init_results(results: dict[str, bool], resource_type: str) -> None:
    """Log initialization results for tables or datasets.

    Parameters
    ----------
    results : dict[str, bool]
        Mapping of table name to whether it was created
    resource_type : str
        Label for log messages (e.g., "Systems", "Analysis")

    """
    created_count = sum(results.values())
    if created_count > 0:
        logger.success(f"{resource_type} initialized: {created_count} table(s) created")
    else:
        logger.info("No action taken (all tables already exist)")


@sync_init_app.command(name="analysis")
def sync_init_analysis(
    reset: Annotated[
        bool,
        Parameter(alias="--force", help="Reset and recreate existing analysis datasets/tables."),
    ] = False,
):
    """Initialize the analysis database.

    Creates the database file/tables (SQLite), CSV files, or datasets
    (Foundry) for storing analysis results.

    Parameters
    ----------
    reset : bool, default False
        Reset and recreate existing datasets/tables. `--force` is an alias for `--reset`.

    """
    from .utils.push_analysis import init_analysis_database

    logger.info("Initializing analysis database")

    try:
        results = init_analysis_database(reset=reset)
        _log_init_results(results, "Analysis")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


@sync_init_app.command(name="artifacts")
def sync_init_artifacts(
    reset: Annotated[
        bool,
        Parameter(alias="--force", help="Reset and recreate existing artifact datasets/tables."),
    ] = False,
):
    """Initialize the artifact database tables.

    Creates the database file/tables (SQLite), CSV files, or datasets
    (Foundry) for storing artifact metadata.

    Parameters
    ----------
    reset : bool, default False
        Reset and recreate existing datasets/tables. `--force` is an alias for `--reset`.

    """
    from .utils.push_analysis import init_artifact_database

    logger.info("Initializing artifact database")

    try:
        results = init_artifact_database(reset=reset)
        _log_init_results(results, "Artifact")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


@sync_init_app.command(name="check")
def sync_init_check():
    """Validate Foundry connection and configured paths.

    Checks that:
    1. Foundry connection can be established
    2. All configured directories (BASE_PATH, analysis, artifacts, runs) exist
    """
    from .settings import Settings

    config = Settings()
    db_type = config.config["database"].get("TYPE", "sqlite")

    if db_type != "foundry":
        logger.info(f"Database type is '{db_type}', not 'foundry'. Nothing to check.")
        return

    logger.info("Checking Foundry connection and paths...")

    # Step 1: Check Foundry connection
    try:
        from foundry_dev_tools import FoundryContext

        ctx = FoundryContext()
        user_info = ctx.multipass.get_user_info()
        logger.success(f"Foundry connection OK (user: {user_info.get('username', 'unknown')})")
    except Exception as e:
        logger.error(f"Foundry connection failed: {e}")
        sys.exit(1)

    # Step 2: Check configured paths
    paths_to_check = {
        "BASE_PATH": config.config["foundry"].get("BASE_PATH"),
        "ANALYSIS_DB_PATH": config.config["foundry"].get("ANALYSIS_DB_PATH"),
        "ARTIFACT_DB_PATH": config.config["foundry"].get("ARTIFACT_DB_PATH"),
        "RUN_DB_PATH": config.config["foundry"].get("RUN_DB_PATH"),
    }

    all_ok = True
    for name, path in paths_to_check.items():
        if not path:
            logger.warning(f"{name}: not configured")
            continue

        try:
            response = ctx.compass.api_get_resource_by_path(path)
            if response.status_code == 200:
                resource = response.json()
                rid = resource.get("rid", "unknown")
                logger.success(f"{name}: {path} (RID: {rid})")
            else:
                logger.error(f"{name}: {path} - not found (status {response.status_code})")
                all_ok = False
        except Exception as e:
            logger.error(f"{name}: {path} - error: {e}")
            all_ok = False

    if all_ok:
        logger.success("All paths validated successfully")
    else:
        logger.warning(
            "Some paths could not be validated. Run 'mdfactory config init' to reconfigure."
        )
        sys.exit(1)


sync_clear_app = App(help="Clear sync datasets (destructive).")
sync_app.command(sync_clear_app, name="clear")


def _clear_tables_with_confirmation(tables: list[str]) -> None:
    """Prompt for confirmation and delete all rows from the given tables.

    Parameters
    ----------
    tables : list[str]
        Table names to clear (e.g., ["RUN_DATABASE", "ANALYSIS_OVERVIEW"])

    """
    from .utils.data_manager import DataManager

    if not tables:
        logger.error("No datasets selected for clearing.")
        sys.exit(1)

    prompt = (
        "This will delete all entries in the following datasets:\n"
        + "\n".join(f"  - {t}" for t in tables)
        + "\nAre you sure?"
    )
    if not questionary.confirm(prompt, default=False).ask():
        logger.info("Clear aborted by user.")
        return

    for table_name in tables:
        try:
            exists, _ = DataManager.database_exists(table_name)
            if not exists:
                logger.warning(f"Dataset not found for {table_name}; skipping.")
                continue
            dm = DataManager(table_name)
            dm.delete_data(conditions={})
            logger.success(f"Cleared {table_name}")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {table_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to clear {table_name}: {e}")


@sync_clear_app.command(name="systems")
def sync_clear_systems():
    """Clear all records from systems database (RUN_DATABASE)."""
    _clear_tables_with_confirmation(["RUN_DATABASE"])


@sync_clear_app.command(name="analysis")
def sync_clear_analysis(
    analysis_name: Annotated[
        str | None, Parameter(help="Specific analysis table (e.g., area_per_lipid).")
    ] = None,
    artifact_name: Annotated[
        str | None, Parameter(help="Specific artifact table (e.g., bilayer_snapshot).")
    ] = None,
    overview: Annotated[bool, Parameter(help="Clear overview table.")] = False,
    analyses: Annotated[bool, Parameter(help="Clear all analysis tables.")] = False,
    artifacts: Annotated[bool, Parameter(help="Clear all artifact tables.")] = False,
    all: Annotated[
        bool, Parameter(help="Clear all analysis/artifact tables and overview.")
    ] = False,
):
    """Clear analysis/artifact datasets with confirmation."""
    from .utils.push_analysis import (
        get_all_analysis_names,
        get_all_artifact_names,
        get_analysis_table_name,
        get_artifact_table_name,
    )

    tables: list[str] = []

    if all:
        overview = True
        analyses = True
        artifacts = True

    if analysis_name:
        tables.append(get_analysis_table_name(analysis_name))
    if artifact_name:
        tables.append(get_artifact_table_name(artifact_name))
    if overview:
        tables.append("ANALYSIS_OVERVIEW")

    if analyses:
        for name in get_all_analysis_names():
            tables.append(get_analysis_table_name(name))
    if artifacts:
        for name in get_all_artifact_names():
            tables.append(get_artifact_table_name(name))

    # Deduplicate while preserving order
    tables = list(dict.fromkeys(tables))
    _clear_tables_with_confirmation(tables)


@sync_clear_app.command(name="all")
def sync_clear_all():
    """Clear all sync datasets (runs, analyses, artifacts, overview)."""
    from .utils.push_analysis import (
        get_all_analysis_names,
        get_all_artifact_names,
        get_analysis_table_name,
        get_artifact_table_name,
    )

    tables = ["RUN_DATABASE", "ANALYSIS_OVERVIEW"]
    for name in get_all_analysis_names():
        tables.append(get_analysis_table_name(name))
    for name in get_all_artifact_names():
        tables.append(get_artifact_table_name(name))
    _clear_tables_with_confirmation(tables)


@sync_push_app.command(name="analysis")
def sync_push_analysis(
    source: Annotated[
        Path | None, Parameter(help="Directory, glob pattern, or summary YAML file.")
    ] = None,
    csv: Annotated[Path | None, Parameter(help="Input CSV to locate hash folders.")] = None,
    csv_root: Annotated[
        Path | None, Parameter(help="Root directory to search when using --csv.")
    ] = None,
    analysis_name: Annotated[
        str | None, Parameter(help="Only push a specific analysis table.")
    ] = None,
    force: Annotated[
        bool, Parameter(help="Overwrite existing records for matching hashes.")
    ] = False,
    diff: Annotated[
        bool, Parameter(help="Skip hashes that already exist in the database.")
    ] = False,
):
    """Push analysis results to database.

    Discovers simulation folders, loads their analysis data (parquet files
    from .analysis/), and uploads to the analysis database.

    Exactly one of --source or --csv must be provided.

    Parameters
    ----------
    source : Path, optional
        Directory, glob pattern (e.g., "systems/*/"), or summary YAML file
    csv : Path, optional
        Input CSV file (hashes will be extracted and folders searched)
    csv_root : Path, optional
        Root directory to search for hash folders when using --csv mode
    analysis_name : str, optional
        Push only this specific analysis (e.g., "area_per_lipid").
        If not provided, pushes all available analyses.
    force : bool, default False
        Delete existing records and re-insert (overwrite duplicates)
    diff : bool, default False
        Only upload new hashes that don't exist in database yet

    """
    from .utils.push_analysis import push_analysis

    logger.info("Starting mdfactory sync push analysis")
    logger.info(f"Input: source={source}, csv={csv}, csv_root={csv_root}")
    logger.info(f"Flags: force={force}, diff={diff}")
    if analysis_name:
        logger.info(f"Filtering to analysis: {analysis_name}")

    _validate_sync_push_inputs(source, csv, csv_root, force, diff)
    _ensure_sync_target_initialized("ANALYSIS_OVERVIEW", "mdfactory sync init analysis")

    try:
        results = push_analysis(
            source=source,
            csv=csv,
            csv_root=csv_root,
            analysis_name=analysis_name,
            force=force,
            diff=diff,
        )
        total = sum(results.values())
        logger.success(f"Push complete: {total} total record(s)")
        for table_name, count in sorted(results.items()):
            logger.info(f"  {table_name}: {count}")
    except Exception as e:
        _exit_sync_push_error(e)


@sync_push_app.command(name="artifacts")
def sync_push_artifacts(
    source: Annotated[
        Path | None, Parameter(help="Directory, glob pattern, or summary YAML file.")
    ] = None,
    csv: Annotated[Path | None, Parameter(help="Input CSV to locate hash folders.")] = None,
    csv_root: Annotated[
        Path | None, Parameter(help="Root directory to search when using --csv.")
    ] = None,
    artifact_name: Annotated[
        str | None, Parameter(help="Only push a specific artifact table.")
    ] = None,
    force: Annotated[
        bool, Parameter(help="Overwrite existing records for matching hashes.")
    ] = False,
    diff: Annotated[
        bool, Parameter(help="Skip hashes that already exist in the database.")
    ] = False,
):
    """Push artifact metadata to database.

    Discovers simulation folders, loads their artifact metadata (file paths
    and checksums from .analysis/artifacts/), and uploads to the artifact database.

    Exactly one of --source or --csv must be provided.

    Parameters
    ----------
    source : Path, optional
        Directory, glob pattern (e.g., "systems/*/"), or summary YAML file
    csv : Path, optional
        Input CSV file (hashes will be extracted and folders searched)
    csv_root : Path, optional
        Root directory to search for hash folders when using --csv mode
    artifact_name : str, optional
        Push only this specific artifact (e.g., "bilayer_snapshot").
        If not provided, pushes all available artifacts.
    force : bool, default False
        Delete existing records and re-insert (overwrite duplicates)
    diff : bool, default False
        Only upload new hashes that don't exist in database yet

    """
    from .utils.push_artifacts import push_artifacts

    logger.info("Starting mdfactory sync push artifacts")
    logger.info(f"Input: source={source}, csv={csv}, csv_root={csv_root}")
    logger.info(f"Flags: force={force}, diff={diff}")
    if artifact_name:
        logger.info(f"Filtering to artifact: {artifact_name}")

    _validate_sync_push_inputs(source, csv, csv_root, force, diff)
    _ensure_sync_target_initialized("ANALYSIS_OVERVIEW", "mdfactory sync init artifacts")

    try:
        results = push_artifacts(
            source=source,
            csv=csv,
            csv_root=csv_root,
            artifact_name=artifact_name,
            force=force,
            diff=diff,
        )
        total = sum(results.values())
        logger.success(f"Push complete: {total} total record(s)")
        for table_name, count in sorted(results.items()):
            logger.info(f"  {table_name}: {count}")
    except Exception as e:
        _exit_sync_push_error(e)


@sync_pull_app.command(name="analysis")
def sync_pull_analysis(
    analysis_name: Annotated[
        str | None, Parameter(help="Analysis name to pull (required unless --overview).")
    ] = None,
    hash: Annotated[str | None, Parameter(help="Filter by simulation hash.")] = None,
    simulation_type: Annotated[
        str | None, Parameter(help="Filter by simulation type (mixedbox/bilayer).")
    ] = None,
    output: Annotated[
        Path | None, Parameter(help="Write results to .csv or .json instead of stdout.")
    ] = None,
    full: Annotated[bool, Parameter(help="Show all columns in CLI output.")] = False,
    overview: Annotated[
        bool, Parameter(help="Pull from overview table instead of a specific analysis.")
    ] = False,
):
    """Pull analysis results from database.

    Retrieves analysis records from the analysis database and outputs
    them to CLI or file.

    Parameters
    ----------
    analysis_name : str, optional
        Pull from specific analysis table (e.g., "area_per_lipid").
        Required unless --overview is specified.
    hash : str, optional
        Filter by simulation hash
    simulation_type : str, optional
        Filter by simulation type (mixedbox, bilayer)
    output : Path, optional
        Output file path (.csv or .json). If not provided, prints to CLI.
    full : bool, default False
        Show all columns in CLI output including data_csv (can be very long)
    overview : bool, default False
        Pull from the overview table instead of a specific analysis table

    """
    from .utils.pull_analysis import (
        format_analysis_summary,
        format_overview_summary,
        pull_overview,
    )
    from .utils.pull_analysis import (
        pull_analysis as pull_analysis_data,
    )

    logger.info("Starting mdfactory sync pull analysis")

    if overview:
        # Pull from overview table
        df = pull_overview(
            hash=hash,
            simulation_type=simulation_type,
            item_name=analysis_name,
        )
        format_fn = format_overview_summary
    else:
        # Pull from specific analysis table
        if analysis_name is None:
            logger.error("--analysis-name is required unless --overview is specified")
            sys.exit(1)
        df = pull_analysis_data(
            analysis_name=analysis_name,
            hash=hash,
            simulation_type=simulation_type,
        )
        format_fn = format_analysis_summary

    if df.empty:
        logger.warning("No records found matching criteria.")
        return

    # File output
    if output is not None:
        output = output.resolve()
        suffix = output.suffix.lower()

        if suffix == ".json":
            df.to_json(output, orient="records", lines=True)
        else:
            # Default to CSV
            df.to_csv(output, index=False)

        logger.success(f"Wrote {len(df)} record(s) to {output}")
        return

    # CLI output
    if full:
        display_df = df
    else:
        display_df = format_fn(df)

    print(f"\nFound {len(df)} record(s):\n")
    print(display_df.to_string(index=False))


@sync_pull_app.command(name="artifacts")
def sync_pull_artifacts(
    artifact_name: Annotated[
        str | None, Parameter(help="Artifact name to pull (required unless --overview).")
    ] = None,
    hash: Annotated[str | None, Parameter(help="Filter by simulation hash.")] = None,
    simulation_type: Annotated[
        str | None, Parameter(help="Filter by simulation type (mixedbox/bilayer).")
    ] = None,
    output: Annotated[
        Path | None, Parameter(help="Write results to .csv or .json instead of stdout.")
    ] = None,
    full: Annotated[bool, Parameter(help="Show all columns in CLI output.")] = False,
    overview: Annotated[
        bool, Parameter(help="Pull from overview table instead of a specific artifact.")
    ] = False,
):
    """Pull artifact metadata from database.

    Retrieves artifact records from the artifact database and outputs
    them to CLI or file.

    Parameters
    ----------
    artifact_name : str, optional
        Pull from specific artifact table (e.g., "bilayer_snapshot").
        Required unless --overview is specified.
    hash : str, optional
        Filter by simulation hash
    simulation_type : str, optional
        Filter by simulation type (mixedbox, bilayer)
    output : Path, optional
        Output file path (.csv or .json). If not provided, prints to CLI.
    full : bool, default False
        Show all columns in CLI output
    overview : bool, default False
        Pull from the overview table instead of a specific artifact table

    """
    from .utils.pull_analysis import pull_artifact
    from .utils.pull_artifacts import format_artifact_summary, pull_artifact_overview

    logger.info("Starting mdfactory sync pull artifacts")

    if overview:
        # Pull from overview table filtered to artifacts
        df = pull_artifact_overview(
            artifact_name=artifact_name,
            hash=hash,
            simulation_type=simulation_type,
        )

        def format_fn(x):
            return x  # Overview already has preferred column order
    else:
        # Pull from specific artifact table
        if artifact_name is None:
            logger.error("--artifact-name is required unless --overview is specified")
            sys.exit(1)
        df = pull_artifact(
            artifact_name=artifact_name,
            hash=hash,
            simulation_type=simulation_type,
        )
        format_fn = format_artifact_summary

    if df.empty:
        logger.warning("No records found matching criteria.")
        return

    # File output
    if output is not None:
        output = output.resolve()
        suffix = output.suffix.lower()

        if suffix == ".json":
            df.to_json(output, orient="records", lines=True)
        else:
            # Default to CSV
            df.to_csv(output, index=False)

        logger.success(f"Wrote {len(df)} record(s) to {output}")
        return

    # CLI output
    if full:
        display_df = df
    else:
        display_df = format_fn(df)

    print(f"\nFound {len(df)} record(s):\n")
    print(display_df.to_string(index=False))


analysis_app = App()
app.command(analysis_app, name="analysis")

analysis_artifacts_app = App()
analysis_app.command(analysis_artifacts_app, name="artifacts")


def _resolve_sim_paths(
    source: Path,
    *,
    trajectory_file: str = "prod.xtc",
    structure_file: str = "system.pdb",
    simulation_type: str | None = None,
    hashes: list[str] | None = None,
) -> list[Path]:
    """Resolve source to simulation paths, applying optional filters."""
    source = source.resolve()
    if source.is_dir():
        sim_paths = resolve_simulation_paths(
            [source],
            trajectory_file=trajectory_file,
            structure_file=structure_file,
        )
    elif source.is_file():
        sim_paths = resolve_simulation_paths_from_yaml(source)
    else:
        raise ValueError(f"Source path is not valid: {source}")

    if simulation_type is not None:
        store = SimulationStore(
            [str(p) for p in sim_paths],
            trajectory_file=trajectory_file,
            structure_file=structure_file,
        )
        df = store.discover()
        df = df[
            df["simulation"].apply(lambda sim: sim.build_input.simulation_type == simulation_type)
        ]
        sim_paths = [Path(p) for p in df["path"].tolist()]

    if hashes:
        sim_paths = filter_paths_by_hash(
            sim_paths,
            hashes,
            trajectory_file=trajectory_file,
            structure_file=structure_file,
        )

    return sim_paths


@analysis_app.command(name="run")
def analysis_run(
    source: Path | None = None,
    analysis: list[str] | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
    skip_existing: bool = True,
    slurm: bool = False,
    account: str | None = None,
    partition: str | None = None,
    time: str = "2h",
    cpus: int = 4,
    mem_gb: int = 8,
    analysis_backend: str = "multiprocessing",
    analysis_workers: int | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    log_dir: Path | None = None,
    job_name_prefix: str = "mdfactory-analysis",
    start_ns: float | None = None,
    end_ns: float | None = None,
    last_ns: float | None = None,
    stride: int | None = None,
    max_residues: int | None = None,
):
    """Run analyses locally or via submitit/SLURM.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).

    Notes
    -----
    For local runs (without ``--slurm``), ``analysis_backend`` and
    ``analysis_workers`` are passed through to analysis execution.

    Analysis parameters (--start-ns, --end-ns, --last-ns, --stride,
    --max-residues) are forwarded to each analysis function. Analyses
    that do not accept a given parameter will ignore it.

    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")

    if analysis_workers is not None and analysis_workers < 1:
        raise ValueError("--analysis-workers must be >= 1.")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    analysis_names = None
    if analysis and "all" not in analysis:
        analysis_names = []
        for entry in analysis:
            analysis_names.extend([name.strip() for name in entry.split(",") if name.strip()])

    analysis_kwargs: dict[str, object] = {}
    if start_ns is not None:
        analysis_kwargs["start_ns"] = start_ns
    if end_ns is not None:
        analysis_kwargs["end_ns"] = end_ns
    if last_ns is not None:
        analysis_kwargs["last_ns"] = last_ns
    if stride is not None:
        analysis_kwargs["stride"] = stride
    if max_residues is not None:
        analysis_kwargs["max_residues"] = max_residues

    if not slurm:
        local_workers = analysis_workers or 1
        result_df = run_analyses_local(
            sim_paths,
            analysis_names,
            structure_file=structure_file,
            trajectory_file=trajectory_file,
            backend=analysis_backend,
            n_workers=local_workers,
            skip_existing=skip_existing,
            analysis_kwargs=analysis_kwargs or None,
        )
        print(result_df)
        return

    # Use autodiscovery if account not provided
    if account is None:
        try:
            slurm_cfg = SlurmConfig.from_cluster(
                needs_gpu=False,
                min_cpus=cpus,
                min_mem_gb=mem_gb,
                time=time,
                cpus_per_task=cpus,
                mem_gb=mem_gb,
                qos=qos,
                constraint=constraint,
                job_name_prefix=job_name_prefix,
                **({"partition": partition} if partition is not None else {}),
            )
            logger.info(
                f"Using autodiscovered SLURM config: account={slurm_cfg.account}, "
                f"partition={slurm_cfg.partition}"
            )
        except RuntimeError as e:
            raise ValueError(
                f"SLURM autodiscovery failed: {e}\nPlease specify --account explicitly."
            ) from e
    else:
        slurm_cfg = SlurmConfig(
            account=account,
            partition=partition or "cpu",
            time=time,
            cpus_per_task=cpus,
            mem_gb=mem_gb,
            qos=qos,
            constraint=constraint,
            job_name_prefix=job_name_prefix,
        )
    if log_dir is None:
        log_dir = determine_log_dir(sim_paths)
    result_df = submit_analyses_slurm(
        sim_paths,
        analysis_names,
        structure_file=structure_file,
        trajectory_file=trajectory_file,
        slurm=slurm_cfg,
        log_dir=log_dir,
        skip_existing=skip_existing,
        wait=True,
        analysis_backend=analysis_backend,
        analysis_workers=analysis_workers,
        analysis_kwargs=analysis_kwargs or None,
    )
    print(result_df)


@analysis_app.command(name="info")
def analysis_info(
    source: Path | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
    output: Path | None = None,
    chemistry_output: Path | None = None,
    chemistry_mode: Literal["all", "lnp"] = "all",
):
    """Show analysis status for simulations.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).

    Options:
        --chemistry-output: Path to write chemistry CSV
        --chemistry-mode: "all" extracts all species from YAML,
                          "lnp" uses LNP-specific grouping (HL, CHL, IL with ILN+ILP merged)
    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    store = SimulationStore(
        [str(p) for p in sim_paths],
        trajectory_file=trajectory_file,
        structure_file=structure_file,
    )
    store.discover()
    df = store.list_analyses_status(simulation_type=simulation_type)
    total_sims = df["hash"].nunique() if not df.empty else 0
    total_rows = len(df)
    completed = int((df["status"] == "completed").sum()) if not df.empty else 0
    pending = int((df["status"] != "completed").sum()) if not df.empty else 0
    print("Analysis Status Summary")
    print(f"Simulations: {total_sims}")
    print(f"Analyses: {total_rows}")
    print(f"Completed: {completed}")
    print(f"Not yet run: {pending}")
    print(df)
    if output is not None:
        output = output.resolve()
        df.to_csv(output, index=False)
        print(f"Wrote analysis status to {output}")

    if chemistry_output is not None:
        chemistry_output = chemistry_output.resolve()
        extractor = get_chemistry_extractor(mode=chemistry_mode)
        chemistry_rows = []
        for hash_val in store.list_simulations():
            sim = store.get_simulation(hash_val)
            build_input = sim.build_input
            if simulation_type is not None and build_input.simulation_type != simulation_type:
                continue
            try:
                chemistry = extractor(build_input)
                chemistry["hash"] = hash_val
                chemistry_rows.append(chemistry)
            except Exception as e:
                logger.warning(f"Failed to extract chemistry for {hash_val}: {e}")
                chemistry_rows.append({"hash": hash_val, "error": str(e)})
        chemistry_df = pd.DataFrame(chemistry_rows)
        # Reorder columns with hash first
        cols = ["hash"] + [c for c in chemistry_df.columns if c != "hash"]
        chemistry_df = chemistry_df[cols]
        chemistry_df.to_csv(chemistry_output, index=False)
        print(f"Wrote chemistry data ({chemistry_mode} mode) to {chemistry_output}")


@analysis_app.command(name="preprocess")
def analysis_preprocess(
    source: Path | None = None,
    script: Path | None = None,
    output: str | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
    dry_run: bool = False,
):
    """Run a preprocessing script across simulations.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).
    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")
    if script is None:
        raise ValueError("Provide --script pointing to an executable script.")
    output_arg = output or ""

    source = source.resolve()
    script = script.resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    if not os.access(script, os.X_OK):
        raise PermissionError(f"Script is not executable: {script}")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    store = SimulationStore(
        [str(p) for p in sim_paths],
        trajectory_file=trajectory_file,
        structure_file=structure_file,
    )
    df = store.discover()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    for _, row in df.iterrows():
        sim = row["simulation"]
        sim_path = Path(str(row["path"]))
        sim_hash = sim.build_input.hash  # type: ignore[attr-defined]
        struct_path = Path(sim_path) / structure_file
        traj_path = Path(sim_path) / trajectory_file
        output_path = Path(output_arg) if output_arg else None
        if output_path is not None and not output_path.is_absolute():
            output_path = sim_path / output_path
        log_dir = sim_path / ".analysis" / "logs" / "preprocess"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"preprocess_{timestamp}.out"
        stderr_path = log_dir / f"preprocess_{timestamp}.err"
        cmd = [
            str(script),
            str(struct_path),
            str(traj_path),
            str(output_path) if output_path is not None else "",
            str(sim_path),
        ]

        start_time = datetime.now()
        status = "success"
        error = None
        exit_code = None

        if dry_run:
            status = "dry-run"
        else:
            with open(stdout_path, "w") as stdout_handle, open(stderr_path, "w") as stderr_handle:
                proc = subprocess.run(
                    cmd,
                    cwd=sim_path,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
                exit_code = proc.returncode
                if proc.returncode != 0:
                    status = "failed"
                    error = f"Non-zero exit status: {proc.returncode}"
            if stdout_path.exists():
                stdout_text = stdout_path.read_text().strip()
                if stdout_text:
                    print(f"[{sim_hash}] stdout: start")
                    print(stdout_text)
                    print(f"[{sim_hash}] stdout: end")
            if stderr_path.exists():
                stderr_text = stderr_path.read_text().strip()
                if stderr_text:
                    print(f"[{sim_hash}] stderr: start")
                    print(stderr_text)
                    print(f"[{sim_hash}] stderr: end")

        duration = (datetime.now() - start_time).total_seconds()
        rows.append(
            {
                "hash": sim_hash,
                "path": str(sim_path),
                "status": status,
                "exit_code": exit_code,
                "error": error,
                "output": str(output_path) if output_path is not None else "",
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "duration_seconds": round(duration, 3),
            }
        )

    summary = pd.DataFrame(rows)
    print(summary)


@analysis_artifacts_app.command(name="run")
def analysis_artifacts_run(
    source: Path | None = None,
    artifact: list[str] | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
    output_prefix: str | None = None,
    vmd_path: str | None = None,
    ffmpeg_path: str | None = None,
    skip_existing: bool = True,
    slurm: bool = False,
    account: str | None = None,
    partition: str | None = None,
    time: str = "2h",
    cpus: int = 4,
    mem_gb: int = 8,
    qos: str | None = None,
    constraint: str | None = None,
    log_dir: Path | None = None,
    job_name_prefix: str = "mdfactory-artifacts",
):
    """Run artifacts locally or via submitit/SLURM.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).
    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    artifact_names = None
    if artifact and "all" not in artifact:
        artifact_names = []
        for entry in artifact:
            artifact_names.extend([name.strip() for name in entry.split(",") if name.strip()])

    if not slurm:
        store = SimulationStore(
            [str(p) for p in sim_paths],
            trajectory_file=trajectory_file,
            structure_file=structure_file,
        )
        summary = store.run_artifacts_batch(
            artifact_names=artifact_names,
            output_prefix=output_prefix,
            vmd_path=vmd_path,
            ffmpeg_path=ffmpeg_path,
            skip_existing=skip_existing,
        )
        print(summary)
        return

    # Use autodiscovery if account not provided
    if account is None:
        try:
            slurm_cfg = SlurmConfig.from_cluster(
                needs_gpu=False,
                min_cpus=cpus,
                min_mem_gb=mem_gb,
                time=time,
                cpus_per_task=cpus,
                mem_gb=mem_gb,
                qos=qos,
                constraint=constraint,
                job_name_prefix=job_name_prefix,
                **({"partition": partition} if partition is not None else {}),
            )
            logger.info(
                f"Using autodiscovered SLURM config: account={slurm_cfg.account}, "
                f"partition={slurm_cfg.partition}"
            )
        except RuntimeError as e:
            raise ValueError(
                f"SLURM autodiscovery failed: {e}\nPlease specify --account explicitly."
            ) from e
    else:
        slurm_cfg = SlurmConfig(
            account=account,
            partition=partition or "cpu",
            time=time,
            cpus_per_task=cpus,
            mem_gb=mem_gb,
            qos=qos,
            constraint=constraint,
            job_name_prefix=job_name_prefix,
        )

    if log_dir is None:
        log_dir = determine_log_dir(sim_paths)
    result_df = submit_artifacts_slurm(
        sim_paths,
        artifact_names,
        structure_file=structure_file,
        trajectory_file=trajectory_file,
        output_prefix=output_prefix,
        vmd_path=vmd_path,
        ffmpeg_path=ffmpeg_path,
        slurm=slurm_cfg,
        log_dir=log_dir,
        skip_existing=skip_existing,
        wait=True,
    )
    print(result_df)


@analysis_artifacts_app.command(name="info")
def analysis_artifacts_info(
    source: Path | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
):
    """Show artifact status for simulations.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).
    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    store = SimulationStore(
        [str(p) for p in sim_paths],
        trajectory_file=trajectory_file,
        structure_file=structure_file,
    )
    store.discover()
    df = store.list_artifacts_status(simulation_type=simulation_type)
    print(df)


@analysis_artifacts_app.command(name="remove")
def analysis_artifacts_remove(
    source: Path | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
):
    """Remove artifacts for simulations.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).
    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    store = SimulationStore(
        [str(p) for p in sim_paths],
        trajectory_file=trajectory_file,
        structure_file=structure_file,
    )
    store.discover()
    info_df = store.list_artifacts_status(simulation_type=simulation_type)
    completed_df = info_df[info_df["status"] == "completed"]
    total_sims = int(completed_df["hash"].nunique()) if not completed_df.empty else 0  # type: ignore[call-arg]
    total_rows = len(completed_df)
    print("Completed Artifacts Summary")
    print(f"Simulations: {total_sims}")
    print(f"Artifacts: {total_rows}")
    print(completed_df)
    if not questionary.confirm(
        "This will remove artifacts for selected simulations. Continue?", default=False
    ).ask():
        logger.info("Artifact removal cancelled by user.")
        return
    df = store.remove_all_artifacts(simulation_type=simulation_type)
    print(df)


@analysis_app.command(name="remove")
def analysis_remove(
    source: Path | None = None,
    simulation_type: str | None = None,
    hash: list[str] | None = None,
    structure_file: str = "system.pdb",
    trajectory_file: str = "prod.xtc",
):
    """Remove all analyses for simulations.

    Provide SOURCE as either a simulation directory or a build summary YAML.
    Use --hash to filter to specific simulations (comma-separated, prefix matching supported).
    """
    if source is None:
        raise ValueError("Provide SOURCE as a simulation directory or build summary YAML.")

    sim_paths = _resolve_sim_paths(
        source,
        trajectory_file=trajectory_file,
        structure_file=structure_file,
        simulation_type=simulation_type,
        hashes=hash,
    )

    store = SimulationStore(
        [str(p) for p in sim_paths],
        trajectory_file=trajectory_file,
        structure_file=structure_file,
    )
    store.discover()
    info_df = store.list_analyses_status(simulation_type=simulation_type)
    completed_df = info_df[info_df["status"] == "completed"]
    total_sims = int(completed_df["hash"].nunique()) if not completed_df.empty else 0  # type: ignore[call-arg]
    total_rows = len(completed_df)
    print("Completed Analyses Summary")
    print(f"Simulations: {total_sims}")
    print(f"Analyses: {total_rows}")
    print(completed_df)
    if not questionary.confirm(
        "This will remove all analyses for selected simulations. Continue?", default=False
    ).ask():
        logger.info("Analysis removal cancelled by user.")
        return
    if simulation_type is None:
        df = store.remove_all_analyses()
    else:
        df = store.remove_all_analyses(simulation_type=simulation_type)
    print(df)


config_app = App(help="Manage mdfactory configuration.")
app.command(config_app, name="config")


@config_app.command(name="init")
def config_init():
    """Interactive wizard to set up mdfactory configuration."""
    from .utils.sync_config import run_config_wizard

    run_config_wizard()


@config_app.command(name="show")
def config_show():
    """Display current active configuration."""
    from .settings import Settings, get_user_config_path

    s = Settings()
    config_path = get_user_config_path()
    print(f"Config file: {config_path}")
    print(f"Exists: {config_path.is_file()}")
    print()
    for section in s.config.sections():
        print(f"[{section}]")
        for key, value in s.config[section].items():
            print(f"  {key} = {value}")
        print()


@config_app.command(name="path")
def config_path():
    """Print the configuration file path."""
    from .settings import get_user_config_path

    print(get_user_config_path())


@config_app.command(name="edit")
def config_edit():
    """Open the configuration file in a terminal text editor.

    Uses $EDITOR if set, otherwise falls back to vi.
    Creates the config file with defaults if it does not exist.
    """
    import os
    import subprocess

    from .settings import Settings, get_user_config_path

    config_path = get_user_config_path()
    if not config_path.is_file():
        logger.info(f"No config file found. Creating defaults at {config_path}")
        s = Settings()
        s.ensure_dirs()
        import configparser

        config = configparser.ConfigParser()
        for section, options in s._get_defaults().items():
            config[section] = {}
            for key, value in options.items():
                config[section][key] = str(value)
        with config_path.open("w") as f:
            config.write(f)

    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(config_path)], check=False)


@config_app.command(name="slurm")
def config_slurm():
    """Interactive wizard to configure a SLURM executor and save to YAML.

    Queries the local SLURM scheduler for available accounts, partitions,
    and hardware, then walks through resource selection interactively.
    The result is saved to a YAML file that can be reused with
    ``mdfactory build --slurm <file>``.

    On non-SLURM machines, falls back to manual text entry.
    """
    from mdfactory.orchestration.tui import UserCancelledError, configure_and_save_slurm

    try:
        configure_and_save_slurm()
    except UserCancelledError:
        logger.info("SLURM configuration cancelled.")


@config_app.command(name="cluster")
def config_cluster(
    json_output: Annotated[bool, Parameter("--json", help="Output in JSON format.")] = False,
):
    """Show discovered SLURM cluster information.

    Queries the local SLURM scheduler and displays available partitions,
    accounts, and QOS policies. Useful for verifying autodiscovery works
    and understanding cluster resources before submitting jobs.

    On non-SLURM machines, prints a helpful message instead of failing.
    """
    import json as json_module

    from mdfactory.performance.cluster import discover_cluster

    cluster = discover_cluster()

    if cluster is None:
        if json_output:
            print(json_module.dumps({"error": "SLURM not available", "cluster": None}))
        else:
            print("SLURM cluster not detected.")
            print()
            print("This machine does not appear to be a SLURM cluster node,")
            print("or SLURM commands (sinfo, sacctmgr) are not in PATH.")
            print()
            print("To use SLURM submission, run this command on a cluster login node.")
        return

    if json_output:
        # Build JSON-serializable structure
        data = {
            "default_account": cluster.default_account,
            "accounts": cluster.accounts,
            "qos_policies": cluster.qos_policies,
            "partitions": [
                {
                    "name": p.name,
                    "state": p.state,
                    "is_default": p.is_default,
                    "max_time": p.max_time,
                    "default_time": p.default_time,
                    "total_nodes": p.total_nodes,
                    "node_types": [
                        {
                            "cpus": nt.cpus,
                            "memory_mb": nt.memory_mb,
                            "gpu_specs": [
                                {"count": count, "type": gtype} for count, gtype in nt.gpu_specs
                            ],
                            "features": list(nt.features),
                            "count": nt.count,
                        }
                        for nt in p.node_types
                    ],
                }
                for p in cluster.partitions
            ],
        }
        print(json_module.dumps(data, indent=2))
        return

    # Human-readable output
    print("SLURM Cluster Information")
    print("=" * 50)
    print()

    # Account info
    print(f"Default Account: {cluster.default_account or '(none)'}")
    if cluster.accounts:
        print(f"Available Accounts: {', '.join(cluster.accounts)}")
    print()

    # QOS info
    if cluster.qos_policies:
        print(f"QOS Policies: {', '.join(cluster.qos_policies)}")
        print()

    # Partition info
    print("Partitions:")
    print("-" * 50)
    for partition in cluster.partitions:
        default_marker = " (default)" if partition.is_default else ""
        state_marker = f" [{partition.state}]" if partition.state != "up" else ""
        print(f"\n  {partition.name}{default_marker}{state_marker}")
        print(f"    Nodes: {partition.total_nodes}")
        print(f"    Max Time: {partition.max_time}")
        if partition.default_time != partition.max_time:
            print(f"    Default Time: {partition.default_time}")

        # Summarize node types
        for nt in partition.node_types:
            gpu_info = ""
            if nt.gpu_specs:
                # Format multiple GPU types like "7x b200, 7x 1g.23gb"
                gpu_parts = [f"{count}x {gtype or 'unknown'}" for count, gtype in nt.gpu_specs]
                gpu_info = f", {', '.join(gpu_parts)}"
            mem_gb = nt.memory_mb // 1024
            features_str = ""
            if nt.features:
                features_str = f" [{', '.join(nt.features)}]"
            count_str = f" ({nt.count} node{'s' if nt.count != 1 else ''})"
            print(f"      - {nt.cpus} CPUs, {mem_gb} GB{gpu_info}{features_str}{count_str}")


def main():
    app()


if __name__ == "__main__":
    app()
