# ABOUTME: Input preparation for the build pipeline
# ABOUTME: Converts CSV/dict data into validated BuildInput models
"""Input preparation for the build pipeline."""

from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

from .models.input import BuildInput

# Simulation types whose system carries a protein with a pdb_path to resolve.
PROTEIN_PDB_SIMULATION_TYPES = frozenset({"proteinbox", "protein_mixedbox"})

# CSV column holding the protein PDB path (flat, dotted notation).
PROTEIN_PDB_PATH_COLUMN = "system.protein.pdb_path"

# Protein fields that are lists/dicts and therefore optional: a CSV row may leave
# them blank. They are excluded from the strict all-columns NaN check and dropped
# per row when empty so heterogeneous proteins can share one CSV.
OPTIONAL_PROTEIN_FIELD_PREFIXES = (
    "system.protein.chains",
    "system.protein.disulfide_bonds",
    "system.protein.protonation_states",
)


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


def df_to_build_input_models(
    df: pd.DataFrame,
) -> tuple[list[BuildInput], dict[int, str]]:
    """Validate DataFrame rows into BuildInput models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row is a flat build specification

    Returns
    -------
    tuple[list[BuildInput], dict[int, str]]
        (valid_models, {row_index: error_message} for failed rows)

    """
    # Optional protein list/dict fields may legitimately be blank for some proteins,
    # so exclude them from the strict NaN check and drop them per row when empty.
    required_cols = [c for c in df.columns if not c.startswith(OPTIONAL_PROTEIN_FIELD_PREFIXES)]
    if df[required_cols].isnull().values.any():
        raise ValueError("Cannot process data frame with NaN values.")
    if df.duplicated(keep=False).any():
        raise ValueError("Data frame contains duplicate rows.")
    ret = []
    errors = {}
    for ii, row in df.iterrows():
        try:
            inp = BuildInput(**dict_to_nested_dict_with_species_prefix(row.dropna()))
            ret.append(inp)
        except Exception as e:
            errors[ii] = str(e)
    return ret, errors


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
    if not isinstance(dct, dict) or dct.get("simulation_type") not in PROTEIN_PDB_SIMULATION_TYPES:
        return
    protein = (dct.get("system") or {}).get("protein")
    if not isinstance(protein, dict):
        return
    pdb_path = protein.get("pdb_path")
    if pdb_path is None or Path(pdb_path).is_absolute():
        return
    protein["pdb_path"] = str((Path(base_dir) / pdb_path).resolve())


def load_build_input_csv(
    csv_path: Path,
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

    Returns
    -------
    tuple[pd.DataFrame, list[BuildInput], dict[int, str]]
        (cleaned DataFrame, valid models, {row_index: error_message} for failed rows)

    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if PROTEIN_PDB_PATH_COLUMN in df.columns:
        df[PROTEIN_PDB_PATH_COLUMN] = df[PROTEIN_PDB_PATH_COLUMN].apply(
            lambda p: (
                p if pd.isna(p) or Path(p).is_absolute() else str((csv_path.parent / p).resolve())
            )
        )
    models, errors = df_to_build_input_models(df)
    return df, models, errors
