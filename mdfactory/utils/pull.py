# ABOUTME: Helper functions for pulling simulation metadata from database
# ABOUTME: Retrieves system records from RUN_DATABASE with optional filtering
"""Helper functions for pulling simulation metadata from database."""

from typing import Any

import pandas as pd
from loguru import logger

from ..analysis.constants import SUMMARY_COLUMNS
from .data_manager import DataManager
from .db_operations import drop_placeholder


def pull_systems(
    status: str | None = None,
    simulation_type: str | None = None,
    parametrization: str | None = None,
    engine: str | None = None,
    db_type: str = "RUN_DATABASE",
) -> pd.DataFrame:
    """Pull simulation metadata from RUN_DATABASE.

    Parameters
    ----------
    status : str, optional
        Filter by status ("build", "equilibrated", "production", "completed")
    simulation_type : str, optional
        Filter by simulation type ("mixedbox", "bilayer")
    parametrization : str, optional
        Filter by force-field family ("charmm", "amber", "openff")
    engine : str, optional
        Filter by engine ("gromacs")
    db_type : str, optional
        Database type, by default "RUN_DATABASE"

    Returns
    -------
    pd.DataFrame
        DataFrame with all matching records

    """
    dm = DataManager(db_type)

    # Build conditions dict from non-None filters
    conditions: dict[str, Any] = {}
    if status is not None:
        conditions["status"] = status
    if simulation_type is not None:
        conditions["simulation_type"] = simulation_type
    if parametrization is not None:
        conditions["parametrization"] = parametrization
    if engine is not None:
        conditions["engine"] = engine

    # Query or load all
    if conditions:
        logger.info(f"Querying database with filters: {conditions}")
        df = dm.query_data(conditions)
    else:
        logger.info("Loading all records from database")
        df = dm.load_data()

    df = drop_placeholder(df)

    logger.info(f"Retrieved {len(df)} record(s)")
    return df


def format_systems_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Format systems DataFrame for CLI summary display.

    Returns a view with key columns, excluding verbose JSON data.

    Parameters
    ----------
    df : pd.DataFrame
        Full systems DataFrame from pull_systems

    Returns
    -------
    pd.DataFrame
        DataFrame with only SUMMARY_COLUMNS present in the data

    """
    if df.empty:
        return df

    available_cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    return df[available_cols].copy()


def format_systems_full(df: pd.DataFrame) -> pd.DataFrame:
    """Format systems DataFrame showing all columns except JSON blob.

    Returns all columns except input_data and input_data_type.

    Parameters
    ----------
    df : pd.DataFrame
        Full systems DataFrame from pull_systems

    Returns
    -------
    pd.DataFrame
        DataFrame with input_data and input_data_type columns removed

    """
    if df.empty:
        return df

    exclude_cols = {"input_data", "input_data_type"}
    display_cols = [c for c in df.columns if c not in exclude_cols]
    return df[display_cols].copy()
