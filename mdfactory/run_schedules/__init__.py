# ABOUTME: Run schedule management and data directory
# ABOUTME: Contains RunScheduleManager plus YAML schedules and engine-specific MDP templates
"""Run schedule management and data directory."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from loguru import logger


class RunScheduleManager:
    """Utility class to manage and access run files from run_schedules.yaml."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the manager and load run schedules.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to a custom ``run_schedules.yaml`` file. When *None*, the
            default file shipped with ``mdfactory/run_schedules/`` is used.

        """
        if config_path is None:
            run_schedules_dir = self._get_run_schedules_dir()
            self.config_path = run_schedules_dir / "run_schedules.yaml"
        else:
            self.config_path = Path(config_path)

        self.config_dir = self.config_path.parent
        self._schedules = None
        self._load_schedules()

    def _get_run_schedules_dir(self) -> Path:
        """Return the path to the ``run_schedules`` directory inside the package."""
        return Path(__file__).parent

    def _load_schedules(self):
        """Load the run schedules from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                self._schedules = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Run schedules file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _get_config(
        self,
        engine: str,
        system_type: str,
        version: Optional[str] = None,
        parametrization: Optional[str] = None,
    ) -> Dict:
        """Get the configuration for a specific engine/system_type combination.

        Entries may carry an optional ``parametrization`` key. An entry whose key
        matches *parametrization* wins; otherwise the entries without the key
        are used, so callers that pass no parametrization (or one without a
        dedicated schedule) get the generic schedule.
        """
        type_configs = [
            config
            for config in self._schedules
            if config.get("engine") == engine and config.get("system_type") == system_type
        ]
        matching_configs = [
            config
            for config in type_configs
            if parametrization is not None and config.get("parametrization") == parametrization
        ] or [config for config in type_configs if "parametrization" not in config]

        if not matching_configs:
            raise ValueError(
                f"No configuration found for engine='{engine}', system_type='{system_type}'"
                + (f", parametrization='{parametrization}'" if parametrization else "")
            )

        if version is None:
            return max(matching_configs, key=lambda x: float(x.get("version", 0)))
        else:
            version_configs = [c for c in matching_configs if str(c.get("version")) == str(version)]
            if not version_configs:
                available_versions = [str(c.get("version")) for c in matching_configs]
                raise ValueError(
                    f"Version '{version}' not found. Available versions: {available_versions}"
                )
            return version_configs[0]

    def get_settings(
        self,
        engine: str,
        system_type: str,
        version: Optional[str] = None,
        parametrization: Optional[str] = None,
    ) -> Dict:
        """Get the computational settings for a specific configuration."""
        config = self._get_config(engine, system_type, version, parametrization)
        return config.get("settings", {})

    def get_all_run_file_paths(
        self,
        engine: str,
        system_type: str,
        version: Optional[str] = None,
        parametrization: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Get paths to all run files for a specific configuration."""
        config = self._get_config(engine, system_type, version, parametrization)
        run_files = config.get("run_files", [])
        # ``template_dir`` lets a system type reuse another type's run files.
        template_dir = self.config_dir / engine / config.get("template_dir", system_type)

        file_paths = {}
        for filename in run_files:
            file_path = template_dir / filename
            if file_path.exists():
                file_paths[filename] = file_path
            else:
                print(f"Warning: Run file not found on disk: {file_path}")

        return file_paths

    def check_target_folder(
        self,
        engine: str,
        system_type: str,
        target_folder: Union[str, Path],
        version: Optional[str] = None,
        parametrization: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, bool], List[str]]:
        """Check if run files already exist in the target folder."""
        target_folder = Path(target_folder)

        # Get the expected run files for this configuration
        config = self._get_config(engine, system_type, version, parametrization)
        expected_files = config.get("run_files", [])

        file_status = {}
        existing_files = []

        for filename in expected_files:
            target_path = target_folder / filename
            exists = target_path.exists()
            file_status[filename] = exists
            if exists:
                existing_files.append(filename)

        has_existing_files = len(existing_files) > 0

        return has_existing_files, file_status, existing_files

    def copy_run_files_with_check(
        self,
        engine: str,
        system_type: str,
        target_folder: Union[str, Path],
        version: Optional[str] = None,
        force_copy: bool = False,
        parametrization: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Copy run files to the target folder after checking for conflicts.

        Parameters
        ----------
        engine : str
            Simulation engine name (e.g. ``"gromacs"``).
        system_type : str
            System category (e.g. ``"bilayer"``, ``"mixedbox"``).
        target_folder : str or Path
            Destination directory for the copied files.
        version : str, optional
            Configuration version string. Uses the latest when *None*.
        force_copy : bool, optional
            Overwrite existing files if True. Default is False.
        parametrization : str, optional
            Parametrization of the build. Selects an entry with a matching
            ``parametrization`` key when one exists; otherwise the generic entry.

        Returns
        -------
        Dict[str, bool]
            Mapping of filename to copy success. All values are False when
            files exist and ``force_copy`` is False.

        """
        # First check if files already exist
        has_existing, file_status, existing_files = self.check_target_folder(
            engine, system_type, target_folder, version, parametrization
        )

        if has_existing and not force_copy:
            logger.warning(
                f"Files already exist in target folder: {existing_files}. "
                "Use force_copy=True to overwrite existing files."
            )
            # Return the current status without copying
            return {filename: False for filename in file_status.keys()}

        # If no existing files or force_copy=True, proceed with copying
        if has_existing and force_copy:
            logger.warning(f"Overwriting existing files: {existing_files}")
        else:
            logger.debug("No existing files found. Proceeding with copy.")

        return self.copy_run_files(
            engine,
            system_type,
            target_folder,
            version,
            overwrite=force_copy,
            parametrization=parametrization,
        )

    def copy_run_files(
        self,
        engine: str,
        system_type: str,
        target_folder: Union[str, Path],
        version: Optional[str] = None,
        overwrite: bool = False,
        parametrization: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Copy run files to a target folder.

        Parameters
        ----------
        engine : str
            Simulation engine name.
        system_type : str
            System category.
        target_folder : str or Path
            Destination directory. Created if it does not exist.
        version : str, optional
            Configuration version string. Uses the latest when *None*.
        overwrite : bool, optional
            Overwrite existing files if True. Default is False.
        parametrization : str, optional
            Parametrization of the build; see :meth:`copy_run_files_with_check`.

        Returns
        -------
        Dict[str, bool]
            Mapping of filename to whether the copy succeeded.

        """
        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        file_paths = self.get_all_run_file_paths(engine, system_type, version, parametrization)
        copy_results = {}

        for filename, source_path in file_paths.items():
            target_path = target_folder / filename

            # Check if file already exists
            if target_path.exists() and not overwrite:
                logger.warning(f"Skipping {filename} - already exists in target folder")
                copy_results[filename] = False
            else:
                try:
                    # Copy file content using pathlib
                    file_content = source_path.read_bytes()
                    target_path.write_bytes(file_content)

                    action = "Overwritten" if target_path.exists() else "Copied"
                    logger.info(f"{action} {filename} to {target_path}")
                    copy_results[filename] = True
                except Exception as e:
                    logger.error(f"Error copying {filename}: {e}")
                    copy_results[filename] = False

        return copy_results

    def get_run_schedules_path(self) -> Path:
        """Return the path to the ``run_schedules`` directory."""
        return self._get_run_schedules_dir()

    def list_available_configs(self) -> List[Dict[str, str]]:
        """List all available engine/system_type/version combinations."""
        configs = []
        for config in self._schedules:
            entry = {
                "engine": config.get("engine"),
                "system_type": config.get("system_type"),
                "version": str(config.get("version")),
            }
            if "parametrization" in config:
                entry["parametrization"] = str(config["parametrization"])
            configs.append(entry)
        return configs

    def list_engines(self) -> List[str]:
        """List all available engines."""
        engines = set()
        for config in self._schedules:
            engines.add(config.get("engine"))
        return sorted(list(engines))

    def list_system_types(self, engine: Optional[str] = None) -> List[str]:
        """List all available system types, optionally filtered by engine."""
        system_types = set()
        for config in self._schedules:
            if engine is None or config.get("engine") == engine:
                system_types.add(config.get("system_type"))
        return sorted(list(system_types))
