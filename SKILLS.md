# MDFactory — Agent Quick Reference

High-throughput MD simulation setup and analysis library. Builds mixedbox and bilayer systems, runs analyses, and syncs results to SQLite/CSV/Foundry backends.

## Project Layout

```
mdfactory/
  cli.py              # CLI entry point (cyclopts App)
  config.py            # Reads config_templates/config.ini, sets env vars
  build.py             # build_mixedbox(), build_bilayer()
  workflows.py         # run_build_from_file() — dispatches to build.py
  prepare.py           # df_to_build_input_models() — CSV row -> BuildInput
  check.py             # check_bilayer_buildable()
  parametrize.py       # Parametrization dispatch (cgenff/openff), OpenFF cache, resolve_parametrizer
  models/
    input.py           # BuildInput (central model, has .hash, .metadata)
    species.py         # Species, SingleMoleculeSpecies, LipidSpecies
    parametrization.py # charmm/amber/openff family specs, parameter sets
    simulation_settings.py # SimulationSettings (T/P = identity; lengths, outputs, overrides)
    composition.py     # MixedBoxComposition, BilayerComposition, IonizationConfig
  analysis/
    simulation.py      # Simulation class, ANALYSIS_REGISTRY
    store.py           # SimulationStore — multi-simulation discovery/aggregation
    registry.py        # AnalysisRegistry — per-simulation analysis metadata
    artifacts.py       # ARTIFACT_REGISTRY, artifact generators
    bilayer/           # Bilayer-specific analysis functions
    submit.py          # SlurmConfig, submitit integration
    utils.py           # discover_simulations()
    constants.py       # SimulationStatus enum, schema columns
  utils/
    data_manager.py    # DataManager facade, DataSource (SQLite/CSV/Foundry)
    db_operations.py   # dedupe_records(), foundry_merge_upload()
    push.py            # push_systems(), discover_simulation_folders()
    push_analysis.py   # push_analysis(), init_analysis_database()
    push_artifacts.py  # push_artifacts()
    pull.py            # pull_systems()
    pull_analysis.py   # pull_analysis(), pull_overview()
    pull_artifacts.py  # pull_artifact_overview()
    sync_config.py     # run_config_wizard()
    utilities.py       # working_directory(), load_yaml_file(), lock_local_folder()
    setup_utilities.py # create_bilayer_from_model(), create_mixed_box_universe()
    topology_utilities.py   # run_cgenff_to_gmx(), merge_extra_parameter_itps()
    chemistry_utilities.py  # detect_lipid_parts_from_smiles_modified()
  run_schedules/       # Run schedules generated at build time (no mdp templates)
    __init__.py        # write_mdp_files(), apply_simulation_settings() re-render API
    gromacs.py         # stage dicts + type/nonbonded-scheme overlays, mdp writer
  tests/               # pytest suite
config_templates/
  config.ini           # Default config (cgenff, storage, database, foundry paths)
```

## CLI Commands

Entry point: `mdfactory` (defined in pyproject.toml `[project.scripts]`). Uses cyclopts.

### Build Group

```
mdfactory prepare-build INPUT [--output DIR] [--simulation-settings FILE]
```
CSV -> BuildInput YAML files. Creates one directory per system hash.

```
mdfactory build INPUT [--output DIR] [--simulation-settings FILE]
```
YAML -> built MD system (system.pdb + topology + generated em/nvt/npt/md.mdp).
The settings file fills simulation_settings fields the input leaves out (row > file > defaults).

```
mdfactory apply-settings SETTINGS SOURCE [--dry-run]
```
Re-render the mdps of existing builds; refuses T/P changes and changed stages with outputs.

```
mdfactory check-csv INPUT
```
Validate CSV: builds models, checks for duplicates, checks bilayer buildability.

### Sync Group

#### Init (setup databases)
```
mdfactory sync init config          # Interactive config wizard
mdfactory sync init check           # Validate Foundry connection + paths
mdfactory sync init systems [--force]    # Create RUN_DATABASE
mdfactory sync init analysis [--force]   # Create analysis tables
mdfactory sync init artifacts [--force]  # Create artifact tables
```

#### Push (local -> database)
All push commands require exactly one of `--source` or `--csv`.
```
mdfactory sync push systems --source DIR|GLOB|YAML [--force] [--diff]
mdfactory sync push systems --csv FILE [--csv-root DIR] [--force] [--diff]

mdfactory sync push analysis --source DIR [--analysis-name NAME] [--force] [--diff]
mdfactory sync push artifacts --source DIR [--artifact-name NAME] [--force] [--diff]
```
- `--force`: delete + re-insert matching hashes
- `--diff`: skip hashes already in database
- `--source`: directory, glob pattern (e.g. `systems/*/`), or build summary YAML
- `--csv`: input CSV (hashes extracted, folders searched under `--csv-root`)

#### Pull (database -> local/stdout)
```
mdfactory sync pull systems [--status S] [--simulation-type T] [--parametrization P] [--engine E] [--output FILE] [--full]

mdfactory sync pull analysis --analysis-name NAME [--hash H] [--simulation-type T] [--output FILE] [--full]
mdfactory sync pull analysis --overview [--hash H] [--simulation-type T]

mdfactory sync pull artifacts --artifact-name NAME [--hash H] [--simulation-type T] [--output FILE]
mdfactory sync pull artifacts --overview
```

#### Clear (destructive)
```
mdfactory sync clear systems
mdfactory sync clear analysis [--analysis-name N] [--artifact-name N] [--analyses] [--artifacts] [--overview] [--all]
mdfactory sync clear all
```

### Analysis Group

```
mdfactory analysis run --source DIR|YAML [--analysis NAME,...] [--simulation-type T] [--hash H] [--skip-existing] [--slurm --account ACC ...]
mdfactory analysis info --source DIR|YAML [--simulation-type T] [--hash H] [--output FILE] [--chemistry-output FILE] [--chemistry-mode all|lnp]
mdfactory analysis preprocess --source DIR|YAML --script SCRIPT [--output NAME] [--simulation-type T] [--hash H] [--dry-run]
mdfactory analysis remove --source DIR|YAML [--simulation-type T] [--hash H]
```

#### Artifacts subgroup
```
mdfactory analysis artifacts run --source DIR|YAML [--artifact NAME,...] [--simulation-type T] [--hash H] [--slurm ...]
mdfactory analysis artifacts info --source DIR|YAML [--simulation-type T] [--hash H]
mdfactory analysis artifacts remove --source DIR|YAML [--simulation-type T] [--hash H]
```

### Database Utilities
```
mdfactory showdb [--name mol|run]
mdfactory clean [--molecules] [--database]
```

## Key Python API

### Building Systems
```python
from mdfactory.prepare import df_to_build_input_models
from mdfactory.workflows import run_build_from_file
from mdfactory.build import build_mixedbox, build_bilayer

# CSV -> models
models, errors = df_to_build_input_models(df)

# YAML -> build
run_build_from_file(Path("hash.yaml"))
```

### Models
```python
from mdfactory.models.input import BuildInput
from mdfactory.models.species import SingleMoleculeSpecies, LipidSpecies
from mdfactory.models.composition import MixedBoxComposition, BilayerComposition

bi = BuildInput(simulation_type="bilayer", system=composition, parametrization="charmm", engine="gromacs")
bi.hash          # deterministic content hash
bi.metadata      # dict summary
bi.to_data_row() # serialize for database
BuildInput.from_data_row(row)  # deserialize
```

### Analysis
```python
from mdfactory.analysis.simulation import Simulation, ANALYSIS_REGISTRY
from mdfactory.analysis.store import SimulationStore
from mdfactory.analysis.artifacts import ARTIFACT_REGISTRY

# Single simulation
sim = Simulation(path=Path("hash_dir"), structure_file="system.pdb", trajectory_file="prod.xtc")
sim.status          # SimulationStatus enum: BUILD < EQUILIBRATED < PRODUCTION < COMPLETED
sim.build_input     # lazy-loaded BuildInput from YAML
sim.universe        # lazy-loaded MDAnalysis Universe
sim.run_analysis("area_per_lipid")
sim.load_analysis("area_per_lipid")  # returns DataFrame

# Multiple simulations
store = SimulationStore(roots=["path1", "path2"])
df = store.discover()  # DataFrame with [hash, path, simulation]
store.list_analyses_status()
store.list_artifacts_status()
```

### Data Management
```python
from mdfactory.utils.data_manager import DataManager

dm = DataManager("RUN_DATABASE")   # or ANALYSIS_OVERVIEW, ANALYSIS_*, etc.
dm.load_data()                     # -> DataFrame
dm.save_data(df)
dm.query_data(conditions={"hash": "abc123"})
dm.delete_data(conditions={"hash": "abc123"})
DataManager.database_exists("RUN_DATABASE")  # -> (bool, str|None)
```

### Sync Operations
```python
from mdfactory.utils.push import push_systems, discover_simulation_folders
from mdfactory.utils.push_analysis import push_analysis, init_analysis_database
from mdfactory.utils.push_artifacts import push_artifacts
from mdfactory.utils.pull import pull_systems
from mdfactory.utils.pull_analysis import pull_analysis, pull_overview
```

## Registered Analyses & Artifacts

### Bilayer Analyses
`area_per_lipid`, `density_distribution`, `cholesterol_tilt`, `tail_end_to_end`, `headgroup_hydration`, `interdigitation`, `leaflet_distribution`, `tail_order_parameter`, `bilayer_thickness_map`, `box_size_timeseries`, `lipid_rg`

### Bilayer Artifacts
`last_frame_pdb`, `bilayer_snapshot`, `bilayer_movie`, `conformational_density`

### Mixedbox Artifacts
`last_frame_pdb`

### Mixedbox Analyses
_(none registered yet)_

## Database Schema

### RUN_DATABASE
`hash` (PK), `engine`, `parametrization` (family name: charmm/amber/openff), `simulation_type`, `input_data` (JSON), `input_data_type`, `directory`, `status`, `timestamp_utc`

### ANALYSIS tables (one per analysis type)
Per-analysis columns auto-inferred from parquet. All include `hash` as foreign key.

### ARTIFACT tables (one per artifact type)
`hash`, `directory`, `simulation_type`, `file_count`, `files`, `checksums`, `timestamp_utc`

## Simulation Status Hierarchy

`BUILD` -> `EQUILIBRATED` -> `PRODUCTION` -> `COMPLETED`

Detection: checks for `prod.gro` (completed), `prod.xtc` (production), equilibration files `min.gro`+`nvt.gro`+`npt.gro` (equilibrated), else build.

## Configuration

Config loaded from `config_templates/config.ini` (repo default) with user override at `~/.mdfactory/config.ini`.

Key sections: `[cgenff]` (SILCSBIODIR), `[storage]` (parameter store path), `[database]` (TYPE: sqlite|csv|foundry), `[sqlite]`/`[csv]`/`[foundry]` (per-backend paths).

## Dispatch Tables

Build: `DISPATCH_BUILD = {"mixedbox": build_mixedbox, "bilayer": build_bilayer}`

Parametrize: `DISPATCH_ENGINE_PARAMETRIZE = {"gromacs": {"cgenff": ..., "openff": ...}}`

## Testing

```
pytest mdfactory/tests/
```
Key test files: `test_build.py`, `test_push.py`, `test_push_analysis.py`, `test_push_artifacts.py`, `test_simulation.py`, `test_simulation_store.py`, `test_db_operations.py`, `test_sql.py`, `test_models.py`, `test_parametrization.py`, `test_parametrization_schema.py`
