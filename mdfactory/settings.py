# ABOUTME: Unified configuration module for mdfactory
# ABOUTME: Single source of truth for all config, replacing config.py and Config class
"""Unified configuration module for mdfactory."""

import configparser
import os
from pathlib import Path

import platformdirs


def get_config_dir() -> Path:
    """Return the configuration directory, overridable via MDFACTORY_CONFIG_DIR."""
    override = os.environ.get("MDFACTORY_CONFIG_DIR")
    if override:
        return Path(override)
    return Path(platformdirs.user_config_dir("mdfactory"))


def get_data_dir() -> Path:
    """Return the data directory, overridable via MDFACTORY_DATA_DIR."""
    override = os.environ.get("MDFACTORY_DATA_DIR")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("mdfactory"))


def get_user_config_path() -> Path:
    """Return the user config file path."""
    return get_config_dir() / "config.ini"


class Settings:
    """Unified configuration for mdfactory.

    Absorbs all functionality from the old ``config.py`` module and the
    ``Config`` class from ``data_manager.py``.
    """

    @staticmethod
    def _get_defaults() -> dict[str, dict[str, str]]:
        """Compute all default config values including platform-appropriate paths."""
        data_dir = get_data_dir()
        return {
            "cgenff": {
                "SILCSBIODIR": "",
            },
            "gromacs": {
                "GMX_PATH": "",
                "FORCEFIELD_DIR": str(data_dir / "forcefields"),
            },
            "storage": {
                "PARAMETERS": str(data_dir / "parameters"),
            },
            "database": {
                "TYPE": "sqlite",
            },
            "databases": {
                "USE_RUN_DB": "true",
                "USE_ANALYSIS_DB": "true",
            },
            "sqlite": {
                "RUN_DB_PATH": str(data_dir / "runs.db"),
                "ANALYSIS_DB_PATH": str(data_dir / "analysis.db"),
            },
            "csv": {
                "RUN_DB_PATH": str(data_dir / "runs.csv"),
                "ANALYSIS_DB_PATH": str(data_dir / "analysis"),
            },
            "foundry": {
                "BASE_PATH": "/Group Functions/mdfactory",
                "ANALYSIS_NAME": "analysis",
                "RUN_DB_PATH": "/Group Functions/mdfactory/runs",
                "ANALYSIS_DB_PATH": "/Group Functions/mdfactory/analysis",
                "ARTIFACT_DB_PATH": "/Group Functions/mdfactory/artifacts",
            },
            "slurm": {
                "ACCOUNT": "",
                "PARTITION_CPU": "",
                "PARTITION_GPU": "",
                "DEFAULT_QOS": "",
            },
        }

    # Keep DEFAULT_CONFIG as class attribute for backward compat in tests
    DEFAULT_CONFIG = {
        "cgenff": {"SILCSBIODIR": ""},
        "gromacs": {"GMX_PATH": "", "FORCEFIELD_DIR": ""},
        "storage": {"PARAMETERS": ""},
        "database": {"TYPE": "sqlite"},
        "databases": {
            "USE_RUN_DB": "true",
            "USE_ANALYSIS_DB": "true",
        },
        "sqlite": {
            "RUN_DB_PATH": "",
            "ANALYSIS_DB_PATH": "",
        },
        "csv": {
            "RUN_DB_PATH": "",
            "ANALYSIS_DB_PATH": "",
        },
        "foundry": {
            "BASE_PATH": "/Group Functions/mdfactory",
            "ANALYSIS_NAME": "analysis",
            "RUN_DB_PATH": "/Group Functions/mdfactory/runs",
            "ANALYSIS_DB_PATH": "/Group Functions/mdfactory/analysis",
            "ARTIFACT_DB_PATH": "/Group Functions/mdfactory/artifacts",
        },
        "slurm": {
            "ACCOUNT": "",
            "PARTITION_CPU": "",
            "PARTITION_GPU": "",
            "DEFAULT_QOS": "",
        },
    }

    def __init__(self):
        self.config = configparser.ConfigParser()
        self._config_base_dir = Path.cwd()
        self._parameter_store_override = None
        self._load_config()
        if "cgenff" in self.config:
            os.environ.setdefault("SILCSBIODIR", str(self.cgenff_dir))
        if "gromacs" in self.config:
            env = self.gromacs_env(os.environ)
            if "GMXLIB" in env:
                os.environ["GMXLIB"] = env["GMXLIB"]

    def _load_config(self):
        """Load configuration from defaults, then user config, then env overrides."""
        defaults = self._get_defaults()

        for section, options in defaults.items():
            if section not in self.config:
                self.config[section] = {}
            for key, value in options.items():
                self.config[section][key] = str(value)

        # Load user config if it exists
        user_config = get_user_config_path()
        if user_config.is_file():
            self.config.read(user_config)
            self._config_base_dir = user_config.parent

        # Allow env var override for entire config file
        env_config = os.environ.get("MDFACTORY_CONFIG")
        if env_config:
            env_path = Path(env_config)
            if env_path.is_file():
                self.config.read(env_path)
                self._config_base_dir = env_path.parent

    def reload(self):
        """Reload configuration from disk."""
        self.config = configparser.ConfigParser()
        self._config_base_dir = Path.cwd()
        self._parameter_store_override = None
        self._load_config()

    def is_configured(self) -> bool:
        """Check if a user config file exists."""
        return get_user_config_path().is_file()

    def ensure_dirs(self):
        """Create data and config directories if they don't exist."""
        get_config_dir().mkdir(parents=True, exist_ok=True)
        get_data_dir().mkdir(parents=True, exist_ok=True)
        self.parameter_store.mkdir(parents=True, exist_ok=True)
        self.gromacs_forcefield_dir.mkdir(parents=True, exist_ok=True)

    # --- CGenFF properties ---

    @property
    def cgenff_config(self) -> configparser.SectionProxy:
        """Return the CGenFF config section."""
        return self.config["cgenff"]

    @property
    def cgenff_dir(self) -> Path:
        """Return the CGenFF directory (SILCSBIODIR)."""
        return Path(self.config["cgenff"].get("SILCSBIODIR", "")).resolve()

    # --- GROMACS properties ---

    @property
    def gromacs_gmx_path(self) -> Path | None:
        """Return the configured gmx binary path, or None to use PATH lookup."""
        raw = self.config["gromacs"].get("GMX_PATH", "")
        if not raw:
            return None
        return Path(self._resolve_local_path(raw))

    @property
    def gromacs_forcefield_dir(self) -> Path:
        """Return the directory where downloaded force fields are stored."""
        raw = self.config["gromacs"].get("FORCEFIELD_DIR", "")
        if raw:
            return Path(self._resolve_local_path(raw))
        return get_data_dir() / "forcefields"

    def gromacs_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Return an environment with configured GROMACS force fields on GMXLIB."""
        env = dict(os.environ if base_env is None else base_env)
        forcefield_dir = str(self.gromacs_forcefield_dir)
        if not forcefield_dir:
            return env

        paths = [p for p in env.get("GMXLIB", "").split(":") if p]
        if forcefield_dir not in paths:
            paths.insert(0, forcefield_dir)
        env["GMXLIB"] = ":".join(paths)
        return env

    # --- Storage properties ---

    @property
    def parameter_store(self) -> Path:
        """Return the parameter store directory."""
        if self._parameter_store_override is not None:
            return self._parameter_store_override
        raw = self.config.get("storage", "PARAMETERS")
        return Path(self._resolve_local_path(raw))

    @parameter_store.setter
    def parameter_store(self, value: Path):
        """Override the parameter store directory (used in tests)."""
        self._parameter_store_override = value

    # --- Path resolution ---

    def _resolve_local_path(self, raw_path: str) -> str:
        """Resolve a local filesystem path from config.

        Expand environment variables and ``~``. Resolve relative paths
        against the directory of the active config file.

        Parameters
        ----------
        raw_path : str
            Raw path string from config

        Returns
        -------
        str
            Absolute, expanded path string

        """
        expanded = os.path.expandvars(os.path.expanduser(raw_path))
        if not expanded:
            return ""
        path = Path(expanded)
        if path.is_absolute():
            return str(path)
        return str((self._config_base_dir / path).resolve())

    # --- Database path methods ---

    def get_db_path(self, db_name: str, db_type: str = "sqlite") -> str:
        """Get the path for a specific database.

        Parameters
        ----------
        db_name : str
            Logical database name (e.g., "RUN_DATABASE", "ANALYSIS_AREA_PER_LIPID")
        db_type : str
            Backend type, currently only "sqlite" is supported

        Returns
        -------
        str
            Absolute path to the database file

        """
        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            if db_type == "sqlite":
                raw_path = self.config["sqlite"].get(
                    "ANALYSIS_DB_PATH",
                    str(get_data_dir() / "analysis.db"),
                )
                return self._resolve_local_path(raw_path)
            else:
                raise ValueError(f"Unsupported db_type: {db_type}")

        db_mapping = {
            "RUN_DATABASE": "RUN_DB_PATH",
            "ANALYSIS_DATABASE": "ANALYSIS_DB_PATH",
        }

        path_key = db_mapping.get(db_name, db_name)
        if db_type == "sqlite":
            raw_path = self.config["sqlite"].get(path_key, "")
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")
        return self._resolve_local_path(raw_path)

    def get_foundry_path(self, db_name: str) -> str:
        """Get the path for a specific Foundry dataset.

        Parameters
        ----------
        db_name : str
            Logical database name

        Returns
        -------
        str
            Foundry dataset path

        """
        base_analysis_path = self.config["foundry"].get(
            "ANALYSIS_DB_PATH", "/Group Functions/mdfactory/analysis"
        )

        if db_name.startswith("ANALYSIS_"):
            analysis_name = db_name.replace("ANALYSIS_", "").lower()
            if analysis_name == "overview":
                return f"{base_analysis_path}/overview"
            return f"{base_analysis_path}/{analysis_name}"

        if db_name.startswith("ARTIFACT_"):
            artifact_name = db_name.replace("ARTIFACT_", "").lower()
            base_artifact_path = self.config["foundry"].get(
                "ARTIFACT_DB_PATH", "/Group Functions/mdfactory/artifacts"
            )
            return f"{base_artifact_path}/{artifact_name}"

        db_mapping = {
            "RUN_DATABASE": "RUN_DB_PATH",
            "ANALYSIS_DATABASE": "ANALYSIS_DB_PATH",
            "SUBMITTED_DATABASE": "SUBMITTED_DB_PATH",
        }

        path_key = db_mapping.get(db_name, db_name)
        return self.config["foundry"].get(path_key, "")

    def get_csv_path(self, db_name: str) -> str:
        """Get the CSV file path for a specific database/table.

        Parameters
        ----------
        db_name : str
            Logical database name

        Returns
        -------
        str
            Absolute path to the CSV file

        """
        db_mapping = {
            "RUN_DATABASE": "RUN_DB_PATH",
            "ANALYSIS_DATABASE": "ANALYSIS_DB_PATH",
        }
        if db_name in db_mapping:
            raw_path = self.config["csv"].get(db_mapping[db_name], "")
            return self._resolve_local_path(raw_path)

        if db_name.startswith("ANALYSIS_") or db_name.startswith("ARTIFACT_"):
            base_dir = self.config["csv"].get("ANALYSIS_DB_PATH", str(get_data_dir() / "analysis"))
            base_dir = self._resolve_local_path(base_dir)
            return os.path.join(base_dir, f"{db_name}.csv")

        path_key = db_mapping.get(db_name, db_name)
        raw_path = self.config["csv"].get(path_key, "")
        return self._resolve_local_path(raw_path)

    # --- SLURM properties ---

    @property
    def slurm_account(self) -> str | None:
        """Return configured SLURM account, or None if not set."""
        val = self.config.get("slurm", "ACCOUNT", fallback="").strip()
        return val if val else None

    @property
    def slurm_partition_cpu(self) -> str | None:
        """Return configured CPU partition name, or None if not set.

        Maps to ``[slurm] PARTITION_CPU`` in config.ini.
        Used by ``BaseSlurmConfig.from_cluster()`` when ``needs_gpu=False``.
        """
        val = self.config.get("slurm", "PARTITION_CPU", fallback="").strip()
        return val if val else None

    @property
    def slurm_partition_gpu(self) -> str | None:
        """Return configured GPU partition name, or None if not set.

        Maps to ``[slurm] PARTITION_GPU`` in config.ini.
        Used by ``BaseSlurmConfig.from_cluster()`` when ``needs_gpu=True``.
        """
        val = self.config.get("slurm", "PARTITION_GPU", fallback="").strip()
        return val if val else None

    @property
    def slurm_qos(self) -> str | None:
        """Return configured SLURM default QOS, or None if not set.

        Maps to ``[slurm] DEFAULT_QOS`` in config.ini.
        """
        val = self.config.get("slurm", "DEFAULT_QOS", fallback="").strip()
        return val if val else None


# Module-level singleton
settings = Settings()
