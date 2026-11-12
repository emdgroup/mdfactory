# ABOUTME: Input preparation for the build pipeline
# ABOUTME: Converts CSV/dict data into validated BuildInput models
"""Input preparation for the build pipeline."""

from collections import defaultdict
from collections.abc import Mapping
from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

from .models.input import BuildInput
from .models.parametrization import PROTEIN_SIMULATION_TYPES
from .models.simulation_settings import SimulationSettings, merge_simulation_settings

# CSV column holding the protein PDB path (flat, dotted notation).
PROTEIN_PDB_PATH_COLUMN = "system.protein.pdb_path"

# Fields with defaults that a CSV row may leave blank: protein lists/dicts and
# preparation options, and the simulation settings (a blank cell falls back to
# the protocol file or the defaults). They are excluded from the strict
# all-columns NaN check and dropped per row when empty, so heterogeneous rows
# can share one CSV.
OPTIONAL_FIELD_PREFIXES = (
    "system.protein.chains",
    "system.protein.disulfide_bonds",
    "system.protein.protonation_states",
    "system.protein.ignore_hydrogens",
    "system.protein.merge_all",
    "simulation_settings.",
)

# CSV columns holding raw GROMACS mdp overrides.
GROMACS_OVERRIDES_PREFIX = "simulation_settings.gromacs_overrides."


def _row_cells(row: pd.Series) -> dict[str, Any]:
    """Return a CSV row's non-blank cells, with integer mdp overrides restored.

    pandas stores a numeric column that has a blank cell as float, so an
    integer override such as ``nstcalcenergy`` would be written as ``100.0``,
    which grompp rejects for integer parameters. Whole-number override values
    are therefore converted back to ``int``; GROMACS reads an integer for a
    real-valued parameter as well.
    """
    cells = row.dropna().to_dict()
    for column, value in cells.items():
        if (
            column.startswith(GROMACS_OVERRIDES_PREFIX)
            and isinstance(value, float)
            and value.is_integer()
        ):
            cells[column] = int(value)
    return cells


def _parse_delimited_protein_fields(nested: dict) -> None:
    """Parse a proteinbox row's delimited list cells into lists in place.

    ``chains`` is a ``;``-separated list of chain ids ("A;B"). ``disulfide_bonds``
    is a ``;``-separated list of ``RESNAME<id>-RESNAME<id>`` pairs
    ("CYS6-CYS127;CYS30-CYS115"). ``protonation_states`` is expressed as nested
    dotted columns (system.protein.protonation_states.HIS15) and needs no parsing.
    """
    protein = nested.get("system", {}).get("protein")
    if not isinstance(protein, dict):
        return

    chains = protein.get("chains")
    if isinstance(chains, str):
        protein["chains"] = [c.strip() for c in chains.split(";") if c.strip()]

    bonds = protein.get("disulfide_bonds")
    if isinstance(bonds, str):
        parsed = []
        for raw_token in bonds.split(";"):
            token = raw_token.strip()
            if not token:
                continue
            first, sep, second = token.partition("-")
            if not sep:
                raise ValueError(
                    f"Invalid disulfide bond '{token}'. Expected "
                    "'RESNAME<id>-RESNAME<id>', e.g. 'CYS6-CYS127'."
                )
            parsed.append((first.strip(), second.strip()))
        protein["disulfide_bonds"] = parsed


def dict_to_nested_dict_with_species_prefix(dct: dict) -> dict:
    """Convert a flat dot-separated key dict into a nested dict with species grouping.

    Parameters
    ----------
    dct : dict
        Flat dictionary with dot-separated keys (e.g., "system.species.POPC.count")

    Returns
    -------
    dict
        Nested dictionary suitable for BuildInput construction

    """
    result = defaultdict(dict)
    species_data = defaultdict(dict)
    system_keys = []

    for key, value in dct.items():
        if pd.isna(value):
            raise ValueError("Cannot process NaN values in dict.")

        keys = key.split(".")

        if len(keys) > 1 and keys[1].lower() == "species":
            system_keys.append(keys[0])
            keys_spec = keys[1:]
            if len(keys_spec) != 3:
                raise ValueError(
                    f"Need exactly 3 levels for specifying species.<resname>.<property>. "
                    f"Got '{key}'."
                )
            resname = keys_spec[1].upper()
            property_path = keys_spec[2:]
            current = species_data[resname]
            for prop_key in property_path[:-1]:
                current = current.setdefault(prop_key, {})
            current[property_path[-1]] = value
        else:
            reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], result)[keys[-1]] = value

    if len(set(system_keys)) > 1:
        raise ValueError("Can only specify species for a single system key.")

    if species_data:
        result[system_keys[0]]["species"] = [
            {"resname": resname, **props} for resname, props in species_data.items() if props
        ]

    nested = dict(result)
    _parse_delimited_protein_fields(nested)
    return nested


def apply_simulation_settings_file(
    data: dict[str, Any],
    simulation_settings: SimulationSettings | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge protocol-file settings under a raw build dict's own settings.

    Precedence is the input's ``simulation_settings`` section (or CSV row),
    then *simulation_settings*, then the per-type defaults.

    Parameters
    ----------
    data : dict
        Raw (nested) build specification; modified in place.
    simulation_settings : SimulationSettings, mapping or None
        Settings shared by every input of a build call.

    Returns
    -------
    dict
        *data*, for chaining.

    """
    if simulation_settings is not None:
        data["simulation_settings"] = merge_simulation_settings(
            simulation_settings, data.get("simulation_settings")
        )
    return data


def df_to_build_input_models(
    df: pd.DataFrame,
    simulation_settings: SimulationSettings | Mapping[str, Any] | None = None,
) -> tuple[list[BuildInput], dict[int, str]]:
    """Validate DataFrame rows into BuildInput models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row is a flat build specification
    simulation_settings : SimulationSettings or mapping, optional
        Settings shared by every row; a row's ``simulation_settings.*``
        columns take precedence.

    Returns
    -------
    tuple[list[BuildInput], dict[int, str]]
        (valid_models, {row_index: error_message} for failed rows)

    """
    # Optional protein list/dict fields may legitimately be blank for some proteins,
    # so exclude them from the strict NaN check and drop them per row when empty.
    required_cols = [c for c in df.columns if not c.startswith(OPTIONAL_FIELD_PREFIXES)]
    if df[required_cols].isnull().values.any():
        raise ValueError("Cannot process data frame with NaN values.")
    if df.duplicated(keep=False).any():
        raise ValueError("Data frame contains duplicate rows.")
    ret = []
    errors = {}
    for ii, row in df.iterrows():
        try:
            nested = dict_to_nested_dict_with_species_prefix(_row_cells(row))
            inp = BuildInput(**apply_simulation_settings_file(nested, simulation_settings))
            ret.append(inp)
        except Exception as e:
            errors[ii] = str(e)
    return ret, errors


def _resolve_relative(path: Any, base_dir: Path) -> Any:
    """Return *path* made absolute against *base_dir*; blanks and absolute paths pass through."""
    if pd.isna(path) or Path(path).is_absolute():
        return path
    return str((Path(base_dir) / path).resolve())


def resolve_protein_pdb_path(dct: Any, base_dir: Path) -> None:
    """Rewrite a relative protein ``pdb_path`` in a build dict to be absolute.

    Applies to protein-carrying system types (proteinbox and protein_mixedbox),
    whose ``system.protein.pdb_path`` may be given relative to the input file.
    The dict is modified in place; other system types and absolute paths are
    left untouched.

    Parameters
    ----------
    dct : Any
        Raw (nested) build specification, typically loaded from YAML
    base_dir : Path
        Directory that relative paths are resolved against

    """
    if not isinstance(dct, dict) or dct.get("simulation_type") not in PROTEIN_SIMULATION_TYPES:
        return
    protein = (dct.get("system") or {}).get("protein")
    if not isinstance(protein, dict):
        return
    if protein.get("pdb_path") is not None:
        protein["pdb_path"] = _resolve_relative(protein["pdb_path"], base_dir)


def load_build_input_csv(
    csv_path: Path,
    simulation_settings: SimulationSettings | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[BuildInput], dict[int, str]]:
    """Read a build CSV and validate its rows into BuildInput models.

    This is the single CSV entry point shared by ``prepare-build``/``build`` and
    ``push``/discovery, so all of them compute identical hashes for a row.
    ``Unnamed`` index columns are dropped and relative protein PDB paths are
    resolved against the CSV's directory.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file
    simulation_settings : SimulationSettings or mapping, optional
        Settings shared by every row (a protocol file); the rows'
        ``simulation_settings.*`` columns take precedence.

    Returns
    -------
    tuple[pd.DataFrame, list[BuildInput], dict[int, str]]
        (cleaned DataFrame, valid models, {row_index: error_message} for failed rows)

    """
    csv_path = Path(csv_path)
    # Only an empty cell is blank. pandas' default NA strings ("None", "NA",
    # "null", ...) would otherwise silently drop legitimate values such as the
    # GROMACS override ``comm-mode = None``.
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if PROTEIN_PDB_PATH_COLUMN in df.columns:
        df[PROTEIN_PDB_PATH_COLUMN] = df[PROTEIN_PDB_PATH_COLUMN].apply(
            _resolve_relative, base_dir=csv_path.parent
        )
    models, errors = df_to_build_input_models(df, simulation_settings)
    return df, models, errors
