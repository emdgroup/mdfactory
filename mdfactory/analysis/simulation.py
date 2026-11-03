# ABOUTME: Simulation object representing a single MD simulation directory
# ABOUTME: Loads build input, provides trajectory access, and manages analysis dispatch
"""Simulation directory management for analysis storage."""

import inspect
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import MDAnalysis as mda
import pandas as pd

from mdfactory.models.input import BuildInput
from mdfactory.utils.utilities import load_yaml_file

from .artifacts import ARTIFACT_REGISTRY
from .bilayer import (
    area_per_lipid,
    bilayer_thickness_map,
    box_size_timeseries,
    cholesterol_tilt,
    density_distribution,
    headgroup_hydration,
    interdigitation,
    leaflet_distribution,
    lipid_rg,
    tail_end_to_end,
    tail_order_parameter,
)
from .constants import EQUILIBRATION_FILES, SimulationStatus
from .registry import AnalysisRegistry

# Class-level analysis registry mapping simulation types to analysis functions
ANALYSIS_REGISTRY: dict[str, dict[str, Callable]] = {
    "bilayer": {
        "area_per_lipid": area_per_lipid,
        "density_distribution": density_distribution,
        "cholesterol_tilt": cholesterol_tilt,
        # "lipid_clustering": lipid_clustering,
        "tail_end_to_end": tail_end_to_end,
        "headgroup_hydration": headgroup_hydration,
        "interdigitation": interdigitation,
        "leaflet_distribution": leaflet_distribution,
        "tail_order_parameter": tail_order_parameter,
        "bilayer_thickness_map": bilayer_thickness_map,
        "box_size_timeseries": box_size_timeseries,
        "lipid_rg": lipid_rg,
    },
    "mixedbox": {},
    "proteinbox": {},
    "protein_mixedbox": {},
}

# Register system_chemistry for all simulation types
from .utils import system_chemistry  # noqa: E402

for _sim_type, _registry in ANALYSIS_REGISTRY.items():
    _registry["system_chemistry"] = system_chemistry


class Simulation:
    """Manages a single simulation directory and its analyses.

    Provides interfaces for saving/loading per-simulation metadata,
    managing analysis results, and executing registered analyses.

    Attributes
    ----------
    path : Path
        Simulation directory path (absolute)
    build_input : BuildInput
        BuildInput instance loaded from YAML
    _registry : AnalysisRegistry | None
        AnalysisRegistry instance (lazy-loaded)

    """

    METADATA_FILENAME = "metadata.json"
    ANALYSIS_DIR = ".analysis"

    def __init__(
        self,
        path: Path | str,
        build_input: BuildInput | None = None,
        structure_file: Path | str = "system.pdb",
        trajectory_file: Path | str | None = "prod.xtc",
        completed_file: Path | str = "prod.gro",
    ) -> None:
        """Initialize Simulation instance.

        Parameters
        ----------
        path : Path | str
            Simulation directory path
        build_input : BuildInput | None
            Optional BuildInput (if None, will be loaded from YAML)
        structure_file : Path | str
            Structure filename (default: system.pdb)
        trajectory_file : Path | str | None
            Trajectory filename (default: prod.xtc). Set to None for
            simulations without trajectory (e.g., build-only).
        completed_file : Path | str
            File that indicates simulation completion (default: prod.gro).
            Used by the status property to determine if simulation is completed.

        Raises
        ------
        NotADirectoryError
            If path is not a directory
        FileNotFoundError
            If structure file does not exist

        """
        self.path = Path(path).resolve()

        if not self.path.is_dir():
            raise NotADirectoryError(f"{self.path} is not a directory")

        self.structure_file = self.path / Path(structure_file)
        self._trajectory_file = trajectory_file
        self._completed_file = Path(completed_file)

        if not self.structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {self.structure_file}")

        self._build_input = build_input
        self._registry: AnalysisRegistry | None = None

    @property
    def trajectory_file(self) -> Path:
        """Get trajectory file path, validating it exists.

        Returns
        -------
        Path
            Absolute path to trajectory file

        Raises
        ------
        ValueError
            If no trajectory file was configured
        FileNotFoundError
            If trajectory file does not exist

        """
        if self._trajectory_file is None:
            raise ValueError("No trajectory file configured for this simulation")
        traj_path = self.path / Path(self._trajectory_file)
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        return traj_path

    @property
    def status(self) -> str:
        """Determine simulation status based on output files.

        Status hierarchy (checked in order):
        - "completed": completed_file exists (default: prod.gro)
        - "production": trajectory_file exists (default: prod.xtc)
        - "equilibrated": min.gro, nvt.gro, npt.gro all exist
        - "build": system.pdb + valid BuildInput YAML exist

        Returns
        -------
        str
            Status string: "completed", "production", "equilibrated", or "build"

        """
        # Use completed_file parameter (default: prod.gro)
        if (self.path / self._completed_file).exists():
            return SimulationStatus.COMPLETED.value

        # Use trajectory_file parameter if set, otherwise check default prod.xtc
        # This ensures status detection works even when trajectory_file=None
        trajectory_to_check = self._trajectory_file or "prod.xtc"
        if (self.path / trajectory_to_check).exists():
            return SimulationStatus.PRODUCTION.value

        # TODO: Make equilibration_files customizable for non-Gromacs engines.
        # Could add an `equilibration_files` constructor parameter or use
        # engine-specific file registry when multi-engine support is needed.
        if all((self.path / f).exists() for f in EQUILIBRATION_FILES):
            return SimulationStatus.EQUILIBRATED.value

        return SimulationStatus.BUILD.value

    @cached_property
    def universe(self) -> mda.Universe:
        """MDAnalysis Universe for this simulation.

        Requires trajectory file to exist.

        Raises
        ------
        ValueError
            If no trajectory file configured
        FileNotFoundError
            If trajectory file does not exist

        """
        return mda.Universe(self.structure_file, self.trajectory_file)

    @property
    def build_input(self) -> BuildInput:
        """Get BuildInput instance, loading from YAML if needed.

        Returns
        -------
        BuildInput
            BuildInput instance

        Raises
        ------
        ValueError
            If no YAML found or multiple valid YAMLs

        """
        if self._build_input is None:
            self._build_input = self.discover_build_input(self.path)
        return self._build_input

    @property
    def metadata(self) -> dict[str, Any]:
        """Get metadata dict from BuildInput.

        Returns
        -------
        dict
            Simulation metadata including species composition

        """
        metadata = self.build_input.metadata
        build_metadata_path = self.path / "build_metadata.json"
        if build_metadata_path.is_file():
            with open(build_metadata_path) as file_handle:
                metadata["build_metadata"] = json.load(file_handle)
        return metadata

    @property
    def analysis_dir(self) -> Path:
        """Path to .analysis directory."""
        return self.path / self.ANALYSIS_DIR

    @property
    def artifact_dir(self) -> Path:
        """Path to .analysis/artifacts directory."""
        return self.analysis_dir / "artifacts"

    @property
    def registry(self) -> AnalysisRegistry:
        """Lazy-load and return AnalysisRegistry instance.

        Creates .analysis directory if needed.

        Returns
        -------
        AnalysisRegistry
            AnalysisRegistry instance

        """
        if self._registry is None:
            self._registry = AnalysisRegistry(self.analysis_dir)
            # self._registry.load()
        return self._registry

    def save_metadata(self) -> None:
        """Save per-simulation metadata.json.

        Generates metadata from BuildInput and writes to disk.
        """
        from loguru import logger

        metadata_path = self.path / self.METADATA_FILENAME
        metadata = self.metadata

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved metadata to {metadata_path}")

    def load_metadata(self) -> dict[str, Any]:
        """Load per-simulation metadata.json from disk.

        Returns
        -------
        dict[str, Any]
            Metadata dict

        Raises
        ------
        FileNotFoundError
            If metadata file doesn't exist

        """
        from loguru import logger

        metadata_path = self.path / self.METADATA_FILENAME

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.debug(f"Loaded metadata from {metadata_path}")
        return metadata

    def save_analysis(
        self,
        name: str,
        df: pd.DataFrame,
        **metadata_extras,
    ) -> None:
        """Save analysis results to parquet and update registry.

        Atomically updates both file and registry. If registry update fails,
        the parquet file is removed to maintain consistency.

        Parameters
        ----------
        name : str
            Analysis name (without .parquet extension)
        df : pd.DataFrame
            DataFrame to save
        **metadata_extras
            Additional metadata for registry entry

        Raises
        ------
        ValueError
            If name contains invalid characters

        """
        from loguru import logger

        # Ensure analysis directory exists
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet file
        parquet_path = self.analysis_dir / f"{name}.parquet"
        df.to_parquet(parquet_path)
        logger.debug(f"Wrote analysis data to {parquet_path}")

        try:
            # Update registry
            self.registry.update_entry(name, df, **metadata_extras)
            logger.info(f"Saved analysis '{name}' for simulation {self.path.name}")
        except Exception as e:
            # Rollback: remove parquet file
            logger.error(f"Failed to update registry, rolling back: {e}")
            parquet_path.unlink(missing_ok=True)
            raise

    def load_analysis(self, name: str) -> pd.DataFrame:
        """Load analysis results from parquet.

        Parameters
        ----------
        name : str
            Analysis name

        Returns
        -------
        pd.DataFrame
            DataFrame with analysis results

        Raises
        ------
        FileNotFoundError
            If analysis doesn't exist

        """
        from loguru import logger

        parquet_path = self.analysis_dir / f"{name}.parquet"

        if not parquet_path.exists():
            available = self.list_analyses()
            raise FileNotFoundError(f"Analysis '{name}' not found. Available analyses: {available}")

        df = pd.read_parquet(parquet_path)
        logger.debug(f"Loaded analysis '{name}' from {parquet_path}")
        return df

    def list_analyses(self) -> list[str]:
        """List available analyses from registry.

        Returns
        -------
        list[str]
            Sorted list of analysis names (empty if registry doesn't exist yet)

        """
        if not self.analysis_dir.exists():
            return []

        return self.registry.list_analyses()

    def list_artifacts(self) -> list[str]:
        """List available artifacts from registry.

        Returns
        -------
        list[str]
            Sorted list of artifact names (empty if registry doesn't exist yet)

        """
        if not self.analysis_dir.exists():
            return []

        return self.registry.list_artifacts()

    def run_analysis(self, name: str, **kwargs) -> pd.DataFrame:
        """Run analysis for this simulation using registered analysis function.

        Executes the analysis function from ANALYSIS_REGISTRY, saves the result,
        and returns the DataFrame.

        Parameters
        ----------
        name : str
            Analysis name (must be registered for simulation type)
        **kwargs
            Analysis-specific parameters. Unsupported kwargs are ignored
            for analyses that do not declare ``**kwargs``.

        Returns
        -------
        pd.DataFrame
            DataFrame with analysis results

        Raises
        ------
        ValueError
            If analysis not registered for this simulation type
        NotImplementedError
            If analysis function is a stub

        """
        from loguru import logger

        simulation_type = self.build_input.simulation_type

        # Check if analysis is registered for this simulation type
        if simulation_type not in ANALYSIS_REGISTRY:
            raise ValueError(f"No analyses registered for simulation type '{simulation_type}'")

        analyses = ANALYSIS_REGISTRY[simulation_type]
        if name not in analyses:
            available = list(analyses.keys())
            raise ValueError(
                f"Analysis '{name}' not registered for '{simulation_type}'. Available: {available}"
            )

        # Execute analysis function
        logger.info(f"Running analysis '{name}' for simulation {self.path.name}")
        analysis_func = analyses[name]
        signature = inspect.signature(analysis_func)
        parameters = signature.parameters
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        if accepts_var_kwargs:
            analysis_kwargs = dict(kwargs)
        else:
            accepted_names: set[str] = set()
            first_keywordable = None
            for param_name, param in parameters.items():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    if first_keywordable is None:
                        first_keywordable = param_name
                    accepted_names.add(param_name)
            if first_keywordable is not None:
                accepted_names.discard(first_keywordable)
            analysis_kwargs = {k: v for k, v in kwargs.items() if k in accepted_names}
            dropped = sorted(set(kwargs) - set(analysis_kwargs))
            if dropped:
                logger.debug(f"Ignoring unsupported kwargs for analysis '{name}': {dropped}")

        df = analysis_func(self, **analysis_kwargs)

        # Save results
        self.save_analysis(name, df, **analysis_kwargs)

        return df

    def save_artifact(
        self,
        name: str,
        files: Path | list[Path],
        **metadata_extras,
    ) -> list[Path]:
        """Save artifact files and update registry.

        Parameters
        ----------
        name : str
            Artifact name
        files : Path | list[Path]
            File path or list of file paths to move
        **metadata_extras
            Additional metadata for registry entry

        Returns
        -------
        list[Path]
            Paths to artifact files under .analysis

        """
        from loguru import logger

        file_list = [files] if isinstance(files, Path) else list(files)
        if len(file_list) == 0:
            raise ValueError("Artifact file list is empty")
        for file_path in file_list:
            if not file_path.exists() or not file_path.is_file():
                raise FileNotFoundError(f"Artifact file not found: {file_path}")

        destination_dir = self.artifact_dir / name
        destination_dir.mkdir(parents=True, exist_ok=True)

        moved_paths = []
        relative_paths = []
        checksums = {}

        try:
            for file_path in file_list:
                destination_path = destination_dir / file_path.name
                file_path.replace(destination_path)
                moved_paths.append(destination_path)

                relative_path = destination_path.relative_to(self.analysis_dir).as_posix()
                relative_paths.append(relative_path)
                checksums[relative_path] = self.registry._calculate_checksum(destination_path)

            self.registry.update_artifact_entry(
                name,
                relative_paths,
                checksums,
                **metadata_extras,
            )
            logger.info(f"Saved artifact '{name}' for simulation {self.path.name}")

        except Exception as e:
            logger.error(f"Failed to update artifact registry, rolling back: {e}")
            for moved_path in moved_paths:
                moved_path.unlink(missing_ok=True)
            if destination_dir.exists() and not any(destination_dir.iterdir()):
                destination_dir.rmdir()
            raise

        return moved_paths

    def load_artifact(self, name: str) -> list[Path]:
        """Load artifact file paths from registry.

        Parameters
        ----------
        name : str
            Artifact name

        Returns
        -------
        list[Path]
            Paths to artifact files

        """
        entry = self.registry.get_artifact_entry(name)
        files = entry.get("files", [])
        artifact_paths = [self.analysis_dir / rel_path for rel_path in files]

        missing = [path for path in artifact_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Artifact '{name}' missing files: {[path.as_posix() for path in missing]}"
            )

        return artifact_paths

    def run_artifact(self, name: str, **kwargs) -> list[Path]:
        """Run artifact producer for this simulation and save artifacts.

        Parameters
        ----------
        name : str
            Artifact name registered for the simulation type
        **kwargs
            Artifact-specific parameters

        Returns
        -------
        list[Path]
            Paths to saved artifacts

        """
        from loguru import logger

        simulation_type = self.build_input.simulation_type
        if simulation_type not in ARTIFACT_REGISTRY:
            raise ValueError(f"No artifacts registered for simulation type '{simulation_type}'")

        artifacts = ARTIFACT_REGISTRY[simulation_type]
        if name not in artifacts:
            available = list(artifacts.keys())
            raise ValueError(
                f"Artifact '{name}' not registered for '{simulation_type}'. Available: {available}"
            )

        logger.info(f"Running artifact '{name}' for simulation {self.path.name}")
        artifact_func = artifacts[name]
        produced_files = artifact_func(self, **kwargs)

        return self.save_artifact(name, produced_files, **kwargs)

    def remove_artifact(self, name: str) -> None:
        """Remove artifact files and registry entry.

        Parameters
        ----------
        name : str
            Artifact name to remove

        """
        from loguru import logger

        entry = self.registry.get_artifact_entry(name)
        for rel_path in entry.get("files", []):
            artifact_path = self.analysis_dir / rel_path
            artifact_path.unlink(missing_ok=True)
            logger.info(f"Removed artifact file: {artifact_path}")

        artifact_dir = self.artifact_dir / name
        if artifact_dir.exists() and not any(artifact_dir.iterdir()):
            artifact_dir.rmdir()

        self.registry.remove_artifact_entry(name)
        logger.info(f"Removed artifact '{name}' from registry for simulation {self.path.name}")

    def check_integrity(self) -> dict[str, Any]:
        """Check integrity of simulation directory.

        Verifies that required files exist and registry is consistent.

        Returns
        -------
        dict[str, Any]
            Dict with:
            - valid: bool - True if no issues found
            - missing_metadata: bool - Per-simulation metadata.json missing
            - missing_build_input: bool - BuildInput YAML missing
            - registry_issues: dict - Issues from AnalysisRegistry.check_integrity()

        """
        from loguru import logger

        missing_metadata = not (self.path / self.METADATA_FILENAME).exists()
        if missing_metadata:
            logger.warning(f"Metadata file missing in {self.path}")

        # Check for BuildInput YAML
        missing_build_input = False
        try:
            self.discover_build_input(self.path)
        except (ValueError, FileNotFoundError):
            missing_build_input = True
            logger.warning(f"BuildInput YAML missing or invalid in {self.path}")

        # Check registry integrity
        registry_issues = {"valid": True}
        if self.analysis_dir.exists():
            registry_issues = self.registry.check_integrity()

        valid = not missing_metadata and not missing_build_input and registry_issues["valid"]

        return {
            "valid": valid,
            "missing_metadata": missing_metadata,
            "missing_build_input": missing_build_input,
            "registry_issues": registry_issues,
        }

    @staticmethod
    def discover_build_input(sim_dir: Path) -> BuildInput:
        """Load BuildInput from YAML file in simulation directory.

        Reuses logic from discover_simulations() - looks for *.yaml files
        and validates them as BuildInput.

        Parameters
        ----------
        sim_dir : Path
            Simulation directory path

        Returns
        -------
        BuildInput
            BuildInput instance

        Raises
        ------
        ValueError
            If no YAML found or multiple valid YAMLs
        FileNotFoundError
            If directory doesn't exist

        """
        from loguru import logger

        sim_dir = Path(sim_dir)

        if not sim_dir.exists():
            raise FileNotFoundError(f"Directory not found: {sim_dir}")

        if not sim_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {sim_dir}")

        yaml_files = list(sim_dir.glob("*.yaml"))

        if not yaml_files:
            raise ValueError(f"No YAML files found in {sim_dir}")

        # Try to load each YAML as BuildInput
        valid_inputs = []
        for yaml_file in yaml_files:
            try:
                data = load_yaml_file(str(yaml_file))
                build_input = BuildInput(**data)
                valid_inputs.append(build_input)
                logger.debug(f"Found valid BuildInput in {yaml_file}")
            except Exception as e:
                logger.debug(f"Skipping {yaml_file}: {e}")
                continue

        if len(valid_inputs) == 0:
            raise ValueError(f"No valid BuildInput YAML files found in {sim_dir}")
        elif len(valid_inputs) > 1:
            raise ValueError(f"Multiple valid BuildInput YAML files found in {sim_dir}")

        return valid_inputs[0]

    def remove_analysis(self, name: str) -> None:
        """Remove analysis results and registry entry.

        Parameters
        ----------
        name : str
            Analysis name to remove

        Raises
        ------
        FileNotFoundError
            If analysis doesn't exist

        """
        from loguru import logger

        parquet_path = self.analysis_dir / f"{name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Analysis '{name}' not found for removal.")

        # Remove parquet file
        parquet_path.unlink()
        logger.info(f"Removed analysis data file: {parquet_path}")

        # Update registry
        self.registry.remove_entry(name)
        logger.info(f"Removed analysis '{name}' from registry for simulation {self.path.name}")

    def remove_all_analyses(self) -> None:
        """Remove all analysis results and clear registry."""
        from loguru import logger

        if not self.analysis_dir.exists():
            logger.info(f"No analysis directory to remove in {self.path}")
            return

        # Remove all parquet files
        for parquet_file in self.analysis_dir.glob("*.parquet"):
            parquet_file.unlink()
            logger.info(f"Removed analysis data file: {parquet_file}")

        # Clear registry
        for analysis in self.registry.list_analyses():
            self.registry.remove_entry(analysis)
        logger.info(f"Cleared all analyses from registry for simulation {self.path.name}")

    def remove_all_artifacts(self) -> None:
        """Remove all artifact files and clear registry."""
        from loguru import logger

        if not self.artifact_dir.exists():
            logger.info(f"No artifact directory to remove in {self.path}")
            return

        for artifact in self.registry.list_artifacts():
            self.remove_artifact(artifact)

        if self.artifact_dir.exists() and not any(self.artifact_dir.iterdir()):
            self.artifact_dir.rmdir()
            logger.info(f"Removed empty artifact directory for {self.path.name}")
