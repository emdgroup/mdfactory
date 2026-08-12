# ABOUTME: Structured execution-environment configuration for compute workers
# ABOUTME: Replaces the opaque worker_init string with typed fields and auto-detection
"""Execution-environment configuration for Parsl compute workers.

Defines :class:`EnvironmentConfig`, a structured replacement for the legacy
``worker_init`` flat string on :class:`~mdfactory.orchestration.config.ExecutorConfig`.
Environment concerns (module loads, pixi/conda activation, MPS lifecycle)
are separated from SLURM allocation and per-stage tuning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, field_serializer


class EnvironmentConfig(BaseModel):
    """Compute-node environment setup — what makes workers capable of running tasks.

    This model captures the structured fields that compose the ``worker_init``
    shell snippet passed to Parsl providers.  Instead of hand-editing a single
    opaque string, users specify modules, activation commands, and custom
    init separately.

    Parameters
    ----------
    modules : list[str]
        Environment modules to load (e.g. ``["gromacs/2024.3-gpu"]``).
        Each entry becomes a ``module load <name>`` command.
    pixi_manifest : Path or None
        Path to a pixi project root (containing ``pixi.toml``).
        When set, a ``pixi shell-hook`` activation command is emitted.
        Auto-detected by :meth:`detect` from the filesystem or
        ``PIXI_PROJECT_MANIFEST`` environment variable.
    conda_env : str or None
        Conda environment name to activate.  Ignored when
        ``pixi_manifest`` is set (pixi takes precedence).
    venv_path : Path or None
        Path to a Python virtualenv.  Ignored when ``pixi_manifest``
        or ``conda_env`` is set.
    extra_init : str
        Additional shell commands appended after all structured fields.
        This is the escape hatch for anything not covered by the typed
        fields (e.g. custom ``export`` statements, ``ulimit`` settings).

    Examples
    --------
    >>> env = EnvironmentConfig(
    ...     modules=["gromacs/2024.3-gpu"],
    ...     pixi_manifest=Path("/project"),
    ... )
    >>> env.compose_worker_init()
    'module load gromacs/2024.3-gpu; eval "$(pixi shell-hook --manifest-path /project -e default)"'

    """

    modules: list[str] = []
    pixi_manifest: Path | None = None
    conda_env: str | None = None
    venv_path: Path | None = None
    extra_init: str = ""

    @field_serializer("pixi_manifest", "venv_path")
    def _serialize_paths(self, v: Path | None) -> str | None:
        """Serialize Path fields as plain strings for YAML compatibility."""
        return str(v) if v is not None else None

    def compose_worker_init(self) -> str:
        """Build the shell snippet from structured fields.

        Produces a semicolon-separated string of shell commands suitable
        for Parsl's ``worker_init`` parameter.  The order is:

        1. ``module load`` commands (one per entry in :attr:`modules`)
        2. Environment activation (pixi > conda > venv, first match wins)
        3. :attr:`extra_init` (appended verbatim)

        Returns
        -------
        str
            Shell snippet, or ``""`` if all fields are empty/default.
        """
        parts: list[str] = []

        for mod in self.modules:
            parts.append(f"module load {mod}")

        if self.pixi_manifest:
            parts.append(
                f'eval "$(pixi shell-hook --manifest-path {self.pixi_manifest} -e default)"'
            )
        elif self.conda_env:
            parts.append(f"conda activate {self.conda_env}")
        elif self.venv_path:
            parts.append(f"source {self.venv_path}/bin/activate")

        if self.extra_init:
            parts.append(self.extra_init)

        return "; ".join(parts)

    @classmethod
    def detect(cls, **overrides: Any) -> EnvironmentConfig:
        """Auto-detect environment from the current shell context.

        Detection priority for Python environment activation:

        1. **pixi** — checks ``PIXI_PROJECT_MANIFEST`` env var, then
           falls back to the ``mdfactory`` package's own project root
           (looks for ``.pixi/envs/default``).
        2. **conda** — checks ``CONDA_PREFIX`` env var.
        3. **venv** — checks ``VIRTUAL_ENV`` env var.

        Only the first match is used; the rest are ignored.

        GROMACS modules are **not** auto-detected here — that requires
        subprocess calls to ``module avail`` which belong in the TUI
        layer (see :func:`~mdfactory.orchestration.tui._detect_gromacs_modules`).

        Parameters
        ----------
        **overrides
            Keyword arguments forwarded to the constructor, overriding
            any auto-detected values.

        Returns
        -------
        EnvironmentConfig
            Environment config with detected fields populated.
        """
        kwargs: dict[str, Any] = {}

        # --- pixi detection ---
        pixi_manifest_env = os.environ.get("PIXI_PROJECT_MANIFEST")
        if pixi_manifest_env:
            # PIXI_PROJECT_MANIFEST points to the manifest file itself
            manifest_path = Path(pixi_manifest_env)
            kwargs["pixi_manifest"] = manifest_path.parent
        else:
            # Fall back to mdfactory's own project root
            try:
                import mdfactory as _mdf

                project_root = Path(_mdf.__file__).parent.parent
                if (project_root / ".pixi" / "envs" / "default").exists():
                    kwargs["pixi_manifest"] = project_root
            except Exception:
                pass

        # --- conda detection (only if pixi not found) ---
        if "pixi_manifest" not in kwargs:
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                # Extract env name from the prefix path
                conda_path = Path(conda_prefix)
                kwargs["conda_env"] = conda_path.name

        # --- venv detection (only if pixi and conda not found) ---
        if "pixi_manifest" not in kwargs and "conda_env" not in kwargs:
            venv_path = os.environ.get("VIRTUAL_ENV")
            if venv_path:
                kwargs["venv_path"] = Path(venv_path)

        # Apply overrides last
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_yaml(self, path: Path) -> None:
        """Write this environment config to a YAML file.

        Empty/default fields are omitted for cleaner output.

        Parameters
        ----------
        path : Path
            Destination file path.  Parent directories are created if
            needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(exclude_none=True)
        if not data.get("modules"):
            data.pop("modules", None)
        if not data.get("extra_init"):
            data.pop("extra_init", None)

        with open(path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> EnvironmentConfig:
        """Load environment config from a YAML file.

        Parameters
        ----------
        path : Path
            Path to a YAML file with environment fields.

        Returns
        -------
        EnvironmentConfig
            Parsed environment configuration.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is empty or not a YAML mapping.
        """
        path = Path(path)
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"Environment config YAML is empty or invalid (expected a mapping): {path}"
            )
        return cls(**data)

    @classmethod
    def load_global(cls) -> EnvironmentConfig | None:
        """Load the global environment config, if it exists.

        The global config lives at
        ``<config-dir>/environment.yaml`` (see
        :func:`get_global_environment_path`).

        Returns
        -------
        EnvironmentConfig or None
            The loaded config, or ``None`` if the file does not exist.

        Raises
        ------
        ValueError
            If the file exists but cannot be parsed.  This fails fast
            so a corrupt config is caught immediately rather than
            silently submitting jobs with no environment.
        """
        path = get_global_environment_path()
        if not path.is_file():
            return None
        env = cls.from_yaml(path)
        logger.debug("Loaded global environment config from %s", path)
        return env


def get_global_environment_path() -> Path:
    """Return the path to the global environment config file.

    Returns
    -------
    Path
        ``<user-config-dir>/environment.yaml``, where the config dir
        is determined by :func:`~mdfactory.settings.get_config_dir`.
    """
    from mdfactory.settings import get_config_dir

    return get_config_dir() / "environment.yaml"
