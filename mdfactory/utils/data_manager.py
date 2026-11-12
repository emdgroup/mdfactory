# ABOUTME: Data source abstraction layer for database backends (SQLite, CSV, Foundry)
# ABOUTME: Provides DataSource ABC, backend implementations, and DataManager facade
"""Data source abstraction layer for database backends (SQLite, CSV, Foundry)."""

import functools
import os
import re
import sqlite3
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, TypeVar, Union
from uuid import uuid4

import pandas as pd

from ..models.input import BuildInput
from ..settings import Settings

T = TypeVar("T", bound=Dict[str, Any])

# Placeholder row inserted for Foundry schema bootstrapping.
PLACEHOLDER_HASH = "__PLACEHOLDER__"

SQLITE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_sqlite_identifier(identifier: str) -> str:
    """Validate and quote a SQLite identifier.

    Parameters
    ----------
    identifier : str
        Table or column identifier

    Returns
    -------
    str
        Safely quoted identifier for SQL interpolation

    """
    if not SQLITE_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid SQLite identifier: {identifier!r}")
    return f'"{identifier}"'


class DataSource(ABC, Generic[T]):  # pragma: no cover
    """Abstract base class for data sources."""

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from the data source."""
        pass

    @abstractmethod
    def save_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        overwrite: bool = False,
    ):
        """Save data to the data source. Accepts single dict, list of dicts, or DataFrame."""
        pass

    @abstractmethod
    def query_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Query data from the data source based on specified conditions."""
        pass

    @abstractmethod
    def update_data(self, conditions: Dict[str, Any], updates: T):
        """Update existing data in the data source based on specified conditions."""
        pass

    @abstractmethod
    def delete_data(self, conditions: Dict[str, Any]):
        """Delete data from the data source based on specified conditions."""
        pass

    @abstractmethod
    def grab_column(self, column_name: str) -> pd.Series:
        """Retrieve a specific column from the data source."""
        pass

    @abstractmethod
    def grab_row(self, index: int) -> pd.Series:
        """Retrieve a specific row from the data source."""
        pass

    @staticmethod
    def _normalize_input(
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame | None:
        """Convert dict, list of dicts, or DataFrame to a DataFrame.

        Parameters
        ----------
        data : dict, list of dict, or pd.DataFrame
            Input data in any supported format

        Returns
        -------
        pd.DataFrame | None
            Normalized DataFrame, or None when the input is empty

        """
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:
                print("No data to save (empty list)")
                return None
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. Use dict, list of dicts, or DataFrame."
            )


def retry_on_error(
    err: list[Exception] = [sqlite3.OperationalError, pd.errors.DatabaseError],
    retries: int = 240,
    wait: float = 0.5,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(retries):
                try:
                    return func(*args, **kwargs)
                except tuple(err):
                    time.sleep(wait)
            raise TimeoutError("Operation timed out after multiple attempts.")

        return wrapper

    return decorator


class SQLiteDataSource(DataSource[T]):
    def __init__(
        self,
        db_path: str | Path,
        table_name: str,
        allow_null: bool = False,
        unique: bool = True,
        unique_columns: list[str] | None = None,
    ):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._quoted_name = quote_sqlite_identifier(table_name)
        self.allow_null = allow_null
        self.unique = unique
        self.unique_columns = unique_columns

    def exists(self) -> bool:
        """Check if the database file exists.

        Returns
        -------
        bool
            True if the SQLite database file exists on disk

        """
        return self.db_path.exists()

    @property
    def table_exists(self):
        """Check if the table exists in the database.

        Returns
        -------
        bool
            True if the table exists

        """
        try:
            return self._table_exists()
        except sqlite3.DatabaseError:
            return False

    @retry_on_error()
    def _table_exists(self):
        with sqlite3.connect(self.db_path, autocommit=True) as con:
            cur = con.cursor()
            cnt = cur.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                (self.table_name,),
            ).fetchone()
            ret = sum(cnt) > 0
            return ret

    @retry_on_error()
    def _columns(self) -> list[str]:
        if not self.table_exists:
            return []
        with sqlite3.connect(self.db_path, autocommit=True) as con:
            info = con.execute(f"pragma table_info({self._quoted_name})").fetchall()
            ret = [col[1] for col in info]
            return ret

    @retry_on_error()
    def _create_if_not_exists(self, columns: list[str]):
        if self.allow_null:
            cols = ",".join(columns)
        else:
            cols = " NOT NULL, ".join(columns)
            cols += " NOT NULL"
        unique = ""
        if self.unique:
            if self.unique_columns is None:
                unique = f"UNIQUE({','.join(columns)})"
            else:
                missing = [col for col in self.unique_columns if col not in columns]
                if missing:
                    raise ValueError(
                        f"Unique columns missing in data: {missing}. Available columns: {columns}"
                    )
                unique = f"UNIQUE({','.join(self.unique_columns)})"
        with sqlite3.connect(self.db_path, autocommit=True) as con:
            con.execute("pragma journal_mode=WAL")
            cur = con.cursor()
            q = f"CREATE TABLE IF NOT EXISTS {self._quoted_name}({cols}, {unique})"
            cur.execute(q)

    @retry_on_error()
    def load_data(self) -> pd.DataFrame:
        """Load data from the data source."""
        if not self.table_exists:
            return pd.DataFrame()
        with sqlite3.connect(self.db_path, autocommit=True) as con:
            return pd.read_sql_query(f"SELECT * from {self._quoted_name}", con=con)

    @retry_on_error()
    def save_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        overwrite: bool = False,
    ):
        """Save data to the data source. Accepts single dict, list of dicts, or DataFrame."""
        if overwrite:
            raise NotImplementedError("Cannot overwrite in SQLiteDataSource.save_data.")

        data_df = self._normalize_input(data)
        if data_df is None or data_df.empty:
            if data_df is not None:
                print("No data to save (empty DataFrame)")
            return

        cols_in_table = sorted(self._columns())
        if cols_in_table and cols_in_table != sorted(data_df.columns.tolist()):
            raise ValueError("Column names do not match with existing table.")

        self._create_if_not_exists(data_df.columns)
        labels = ",".join([":" + col for col in data_df.columns])

        with sqlite3.connect(self.db_path, autocommit=True) as con:
            cur = con.cursor()
            before_count = cur.execute(f"SELECT COUNT(*) FROM {self._quoted_name}").fetchone()[0]
            cur.executemany(
                f"INSERT INTO {self._quoted_name} VALUES({labels}) ON CONFLICT DO NOTHING",
                data_df.to_dict(orient="records"),
            )
            after_count = cur.execute(f"SELECT COUNT(*) FROM {self._quoted_name}").fetchone()[0]
            actual_inserted = after_count - before_count
            if actual_inserted < len(data_df):
                from loguru import logger

                logger.warning(
                    f"Inserted {actual_inserted}/{len(data_df)} records "
                    f"({len(data_df) - actual_inserted} skipped due to conflicts)"
                )

    @retry_on_error()
    def query_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Query data from the data source based on specified conditions."""
        if not self.table_exists:
            return pd.DataFrame()

        query = []
        params: list[Any] = []
        for k, v in conditions.items():
            query.append(f"{k} = ?")
            params.append(v)
        query = " AND ".join(query)
        with sqlite3.connect(self.db_path, autocommit=True) as con:
            ret = pd.read_sql_query(
                f"SELECT * from {self._quoted_name} WHERE {query}", con, params=params
            )
        return ret

    @retry_on_error()
    def update_data(self, conditions: Dict[str, Any], updates: T):
        """Update existing data in the data source based on specified conditions."""
        raise NotImplementedError()

    @retry_on_error()
    def delete_data(self, conditions: Dict[str, Any]):
        """Delete data from the data source based on specified conditions."""
        if not self.table_exists:
            return
        with sqlite3.connect(self.db_path, autocommit=True) as con:
            cur = con.cursor()
            if not conditions:
                cur.execute(f"DELETE FROM {self._quoted_name}")
                return

            query = []
            params: list[Any] = []
            for k, v in conditions.items():
                query.append(f"{k} = ?")
                params.append(v)
            where_clause = " AND ".join(query)
            cur.execute(f"DELETE FROM {self._quoted_name} WHERE {where_clause}", params)

    @retry_on_error()
    def grab_column(self, column_name: str) -> pd.Series:
        """Retrieve a specific column from the data source."""
        raise NotImplementedError()

    @retry_on_error()
    def grab_row(self, index: int) -> pd.Series:
        """Retrieve a specific row from the data source."""
        raise NotImplementedError()


class CsvDataSource(DataSource[T]):
    """File-based CSV data source. Each table is a separate CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file
    unique_columns : list[str] or None
        Column names used for duplicate detection on append

    """

    def __init__(
        self,
        file_path: str | Path,
        unique_columns: list[str] | None = None,
    ):
        self.file_path = Path(file_path)
        self.unique_columns = unique_columns

    def exists(self) -> bool:
        """Check if the CSV file exists.

        Returns
        -------
        bool
            True if the CSV file exists on disk

        """
        return self.file_path.exists()

    @property
    def table_exists(self) -> bool:
        """Check if the CSV file exists and has content.

        Returns
        -------
        bool
            True if file exists and is non-empty

        """
        return self.file_path.exists() and self.file_path.stat().st_size > 0

    def load_data(self) -> pd.DataFrame:
        """Load all data from the CSV file.

        Returns
        -------
        pd.DataFrame
            Contents of the CSV file, or empty DataFrame if file is missing

        """
        if not self.table_exists:
            return pd.DataFrame()
        return pd.read_csv(self.file_path)

    def save_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        overwrite: bool = False,
    ):
        """Save data to CSV. Accepts single dict, list of dicts, or DataFrame."""
        data_df = self._normalize_input(data)
        if data_df is None or data_df.empty:
            if data_df is not None:
                print("No data to save (empty DataFrame)")
            return

        if overwrite:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data_df.to_csv(self.file_path, index=False)
            return

        # Append mode: load existing, concat, drop duplicates (existing wins)
        if self.table_exists:
            existing = pd.read_csv(self.file_path)
            if sorted(existing.columns.tolist()) != sorted(data_df.columns.tolist()):
                raise ValueError("Column names do not match with existing CSV file.")
            combined = pd.concat([existing, data_df], ignore_index=True)
            before_count = len(existing)
            if self.unique_columns:
                combined = combined.drop_duplicates(subset=self.unique_columns, keep="first")
            actual_inserted = len(combined) - before_count
            if actual_inserted < len(data_df):
                from loguru import logger

                logger.warning(
                    f"Inserted {actual_inserted}/{len(data_df)} records "
                    f"({len(data_df) - actual_inserted} skipped due to conflicts)"
                )
            combined.to_csv(self.file_path, index=False)
        else:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data_df.to_csv(self.file_path, index=False)

    def query_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Query data from CSV based on conditions."""
        if not self.table_exists:
            return pd.DataFrame()
        df = pd.read_csv(self.file_path)
        if not conditions:
            return df
        unknown_columns = [k for k in conditions if k not in df.columns]
        if unknown_columns:
            raise ValueError(
                f"Unknown condition column(s): {unknown_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )
        mask = pd.Series(True, index=df.index)
        for k, v in conditions.items():
            mask &= df[k] == v
        return df[mask].copy()

    def update_data(self, conditions: Dict[str, Any], updates: T):
        """Update existing data in CSV based on conditions."""
        raise NotImplementedError()

    def delete_data(self, conditions: Dict[str, Any]):
        """Delete data from CSV based on conditions."""
        if not self.table_exists:
            return
        df = pd.read_csv(self.file_path)
        if not conditions:
            pd.DataFrame(columns=df.columns).to_csv(self.file_path, index=False)
            return
        unknown_columns = [k for k in conditions if k not in df.columns]
        if unknown_columns:
            raise ValueError(
                f"Unknown condition column(s): {unknown_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )
        mask = pd.Series(True, index=df.index)
        for k, v in conditions.items():
            mask &= df[k] == v
        remaining = df[~mask]
        remaining.to_csv(self.file_path, index=False)

    def grab_column(self, column_name: str) -> pd.Series:
        """Retrieve a specific column from CSV."""
        raise NotImplementedError()

    def grab_row(self, index: int) -> pd.Series:
        """Retrieve a specific row from CSV."""
        raise NotImplementedError()


class FoundryDataSource(DataSource[T]):
    """Foundry data source backed by Palantir Foundry datasets."""

    @classmethod
    def dataset_exists(cls, dataset_path: str) -> bool:
        """Check if a Foundry dataset exists without auto-creating.

        Parameters
        ----------
        dataset_path : str
            Foundry dataset path to check

        Returns
        -------
        bool
            True if the dataset exists in Foundry

        """
        try:
            from foundry_dev_tools import FoundryContext

            ctx = FoundryContext()
            ctx.get_dataset_by_path(dataset_path, create_if_not_exist=False)
            return True
        except Exception:
            return False

    def __init__(self, dataset_path: str):
        try:
            from foundry_dev_tools import FoundryContext
        except ImportError:
            raise ImportError(
                "foundry_dev_tools is required for Foundry database support. "
                "Install it with: pip install foundry_dev_tools"
            )

        self.dataset_path = dataset_path
        self.ctx = FoundryContext()
        self.dataset = None
        self.rid = None
        self._initialized = False

        self._init_dataset()

    def _init_dataset(self):
        """Initialize dataset on Foundry."""
        try:
            self.dataset = self.ctx.get_dataset_by_path(
                self.dataset_path,
                create_if_not_exist=True,
                create_branch_if_not_exists=True,
            )
            self.rid = self.dataset.rid
            print(f"Connected to Foundry dataset: {self.dataset_path} (RID: {self.rid})")
        except Exception as e:
            raise ValueError(
                f"Failed to connect to Foundry dataset '{self.dataset_path}'. "
                f"Please check path and permissions. Error: {e}"
            )

    @property
    def table_exists(self):
        return self._check_dataset_exists()

    def _check_dataset_exists(self) -> bool:
        """Check if dataset has schema (may be empty)."""
        try:
            self.dataset.query_foundry_sql("SELECT * LIMIT 1")
            return True
        except Exception:
            return False

    def wait_for_row_count(
        self,
        expected_rows: int,
        timeout_seconds: int = 90,
        interval_seconds: int = 2,
    ) -> bool:
        """Wait for dataset to report at least expected_rows.

        Parameters
        ----------
        expected_rows : int
            Minimum number of rows to wait for
        timeout_seconds : int
            Maximum time to wait in seconds
        interval_seconds : int
            Polling interval in seconds

        Returns
        -------
        bool
            True if expected row count was reached before timeout

        """
        deadline = time.time() + timeout_seconds
        last_seen: int | None = None
        while time.time() < deadline:
            try:
                result = self.dataset.query_foundry_sql("SELECT COUNT(*) as count")
                if not result.empty:
                    last_seen = int(result.iloc[0, 0])
                    if last_seen >= expected_rows:
                        return True
            except Exception:
                pass
            time.sleep(interval_seconds)
        print(
            f"Warning: Foundry dataset row count not visible after wait. "
            f"Expected>={expected_rows}, last_seen={last_seen}"
        )
        return False

    def _empty_schema_frame(self) -> pd.DataFrame:
        """Return an empty DataFrame with the dataset schema (if available).

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with column names from the dataset schema

        """
        try:
            schema_df = self.dataset.query_foundry_sql("SELECT * LIMIT 0")
            return schema_df.iloc[0:0].copy()
        except Exception:
            return pd.DataFrame()

    def init_schema(self):
        """Add schema based on the Foundry inference service. One time execution."""
        dataset = self.ctx.get_dataset(self.rid)

        schema = dataset.infer_schema()
        print(schema)

        dataset.start_transaction(start_transaction_type="UPDATE")
        dataset.upload_schema(dataset.transaction.get("rid"), schema)
        dataset.commit_transaction()

    def load_data(self) -> pd.DataFrame:
        """Load all data from the Foundry dataset."""
        if not self._check_dataset_exists():
            return pd.DataFrame()  # Return empty DataFrame if no schema/data

        try:
            return self.dataset.query_foundry_sql("SELECT *")
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def save_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        overwrite: bool = False,
    ):
        """Save data to the Foundry dataset. Accepts single dict, list of dicts, or DataFrame."""
        data_df = self._normalize_input(data)
        if data_df is None:
            return

        if data_df.empty:
            if not overwrite:
                print("No data to save (empty DataFrame)")
                return
            if data_df.columns.empty:
                schema_df = self._empty_schema_frame()
                if schema_df.columns.empty:
                    raise ValueError(
                        "Cannot overwrite Foundry dataset with empty data and no schema."
                    )
                data_df = schema_df

        # Initialize schema if needed
        if not self._check_dataset_exists():
            # First upload data to enable schema inference
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                data_df.to_parquet(tmp_file.name, index=False)

                self.dataset.upload_file(
                    file_path=Path(tmp_file.name),
                    path_in_foundry_dataset=f"init_{uuid4()}.parquet",
                    transaction_type="SNAPSHOT",
                )

                os.unlink(tmp_file.name)

            # Now initialize schema
            self.init_schema()
            print(f"Schema initialized and saved {len(data_df)} record(s)")
            return

        # Choose transaction type
        transaction_type = "SNAPSHOT" if overwrite else "UPDATE"
        # always initialize if overwrite as things might change

        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                data_df.to_parquet(tmp_file.name, index=False)

                self.dataset.upload_file(
                    file_path=Path(tmp_file.name),
                    path_in_foundry_dataset=f"data_{uuid4()}.parquet",
                    transaction_type=transaction_type,
                )

                # Clean up temp file
                os.unlink(tmp_file.name)

            print(f"Saved {len(data_df)} record(s) to Foundry ({transaction_type} transaction)")
            if overwrite:
                try:
                    self.init_schema()
                except Exception as e:
                    print(f"Warning: schema initialization failed after overwrite: {e}")
        except Exception as e:
            print(f"Error saving data: {e}")
            raise

    def query_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Query data based on conditions."""
        if not self._check_dataset_exists():
            return pd.DataFrame()

        if not conditions:
            return self.load_data()

        try:
            where_clauses = []
            for k, v in conditions.items():
                if not SQLITE_IDENTIFIER_RE.fullmatch(k):
                    raise ValueError(f"Invalid column name: {k!r}")
                if isinstance(v, str):
                    escaped = v.replace("'", "''")
                    where_clauses.append(f"{k} = '{escaped}'")
                elif isinstance(v, (int, float)):
                    where_clauses.append(f"{k} = {v}")
                else:
                    escaped = str(v).replace("'", "''")
                    where_clauses.append(f"{k} = '{escaped}'")

            where_clause = " AND ".join(where_clauses)
            sql_query = f"SELECT * WHERE {where_clause}"

            return self.dataset.query_foundry_sql(sql_query)

        except Exception as e:
            print(f"Error querying data: {e}")
            return pd.DataFrame()

    def update_data(self, conditions: Dict[str, Any], updates: T):
        """Update existing data."""
        data = self.load_data()
        if data.empty:
            print("No data to update")
            return

        # Apply updates
        mask = pd.Series(True, index=data.index)
        for k, v in conditions.items():
            if k in data.columns:
                mask &= data[k] == v

        for k, v in updates.items():
            if k in data.columns:
                data.loc[mask, k] = v

        self.save_data(data, overwrite=True)
        print(f"Updated {mask.sum()} record(s)")

    def delete_data(self, conditions: Dict[str, Any]):
        """Delete data based on conditions."""
        data = self.load_data()
        if data.empty:
            print("No data to delete")
            return

        # Apply deletion
        mask = pd.Series(True, index=data.index)
        for k, v in conditions.items():
            if k in data.columns:
                mask &= data[k] == v

        deleted_count = mask.sum()
        data = data[~mask]
        self.save_data(data, overwrite=True)
        self.wait_for_row_count(len(data))
        print(f"Deleted {deleted_count} record(s)")

    def grab_column(self, column_name: str) -> pd.Series:
        """Retrieve a specific column."""
        if not self._check_dataset_exists():
            return pd.Series(dtype=object)

        try:
            sql_query = f"SELECT {column_name}"
            result = self.dataset.query_foundry_sql(sql_query)
            return result[column_name] if column_name in result.columns else pd.Series(dtype=object)
        except Exception:
            return pd.Series(dtype=object)

    def grab_row(self, index: int) -> pd.Series:
        """Retrieve a specific row."""
        data = self.load_data()
        if data.empty or index >= len(data):
            return pd.Series(dtype=object)

        return data.iloc[index]

    def coalesce_parquets(self):
        """Combine parquets into one file."""
        df = self.load_data()
        if not df.empty:
            self.save_data(df, overwrite=True)
            print("Parquet files coalesced successfully")


class DataManager(Generic[T]):
    """Simplified data manager for handling data operations."""

    @staticmethod
    def database_exists(db_type: str) -> tuple[bool, str]:
        """Check if database exists without initializing it.

        Parameters
        ----------
        db_type : str
            Database type (e.g., "RUN_DATABASE", "ANALYSIS_OVERVIEW")

        Returns
        -------
        tuple[bool, str]
            (exists, path_info) where path_info is the database path or dataset path

        """
        s = Settings()
        database_type = s.config["database"]["TYPE"]

        if database_type == "sqlite":
            db_path = Path(s.get_db_path(db_type, "sqlite"))
            return db_path.exists(), str(db_path)
        elif database_type == "csv":
            csv_path = Path(s.get_csv_path(db_type))
            return csv_path.exists(), str(csv_path)
        elif database_type == "foundry":
            dataset_path = s.get_foundry_path(db_type)
            return FoundryDataSource.dataset_exists(dataset_path), dataset_path
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

    def __init__(self, db_type: str = None):
        if not db_type:
            raise ValueError(
                "db_type must be provided. Specify one of: 'RUN_DATABASE' or 'ANALYSIS_DATABASE'"
            )

        self.data_source = self._initialize_data_source(db_type)

    def _initialize_data_source(self, db_type: str):
        """Initialize the appropriate data source based on configuration."""
        s = Settings()
        database_type = s.config["database"]["TYPE"]

        if database_type == "sqlite":
            db_path = Path(s.get_db_path(db_type, "sqlite"))
            if not db_path.exists():
                raise FileNotFoundError(
                    f"SQLite database file does not exist: {db_path}. "
                    "Run `mdfactory sync init systems` (or the corresponding init command) "
                    "to create it."
                )
            with sqlite3.connect(db_path, autocommit=True) as con:
                user_version = con.execute("PRAGMA user_version").fetchone()[0]
            if user_version < 1:
                raise RuntimeError(
                    f"SQLite database at {db_path} is not initialized for mdfactory. "
                    "Run `mdfactory sync init systems` (or the corresponding init command)."
                )
            unique_columns = self._get_unique_columns(db_type)
            return SQLiteDataSource(
                db_path=db_path,
                table_name=db_type,
                unique_columns=unique_columns,
            )

        elif database_type == "csv":
            csv_path = Path(s.get_csv_path(db_type))
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file does not exist: {csv_path}. "
                    "Run `mdfactory sync init systems` (or the corresponding init command) "
                    "to create it."
                )
            unique_columns = self._get_unique_columns(db_type)
            return CsvDataSource(
                file_path=csv_path,
                unique_columns=unique_columns,
            )

        elif database_type == "foundry":
            dataset_path = s.get_foundry_path(db_type)
            return FoundryDataSource(dataset_path)

        else:
            raise ValueError(f"Unsupported database type: {database_type}")

    @staticmethod
    def _get_unique_columns(db_type: str) -> list[str] | None:
        """Get unique columns for a given database/table type.

        Parameters
        ----------
        db_type : str
            Table name (e.g., "RUN_DATABASE", "ANALYSIS_AREA_PER_LIPID")

        Returns
        -------
        list[str] | None
            Column names that form the unique constraint, or None

        """
        if db_type == "RUN_DATABASE":
            return ["hash"]
        elif db_type == "ANALYSIS_OVERVIEW":
            return ["hash", "item_type", "item_name"]
        elif db_type.startswith("ANALYSIS_"):
            # Per-analysis tables: unique by hash
            return ["hash"]
        elif db_type.startswith("ARTIFACT_"):
            # Per-artifact tables: unique by hash
            return ["hash"]
        return None

    # Delegate all methods to the data source
    def load_data(self) -> pd.DataFrame:
        return self.data_source.load_data()

    def save_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        overwrite: bool = False,
    ):
        """Save data - accepts single dict, list of dicts, or DataFrame."""
        self.data_source.save_data(data, overwrite=overwrite)

    def query_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        return self.data_source.query_data(conditions)

    def update_data(self, conditions: Dict[str, Any], updates: T):
        self.data_source.update_data(conditions, updates)

    def delete_data(self, conditions: Dict[str, Any]):
        self.data_source.delete_data(conditions)

    def grab_column(self, column_name: str) -> pd.Series:
        return self.data_source.grab_column(column_name)

    def grab_row(self, index: int) -> pd.Series:
        return self.data_source.grab_row(index)


def check_run_exists(build_model: BuildInput) -> bool:
    conditions = {
        "hash": build_model.hash,
        "engine": build_model.engine,
        "parametrization": build_model.parametrization.family,
        "simulation_type": build_model.simulation_type,
    }
    dm = DataManager("RUN_DATABASE")
    df = dm.query_data(conditions=conditions)
    return not df.empty
