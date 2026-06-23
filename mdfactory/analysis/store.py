# ABOUTME: SimulationStore for discovering and aggregating analyses across simulations
# ABOUTME: Scans directory trees for simulation folders and collects analysis status
"""Simulation store for discovering and aggregating analysis across simulations."""

from pathlib import Path
from typing import Callable

import pandas as pd

from mdfactory.models.input import BuildInput

from .artifacts import ARTIFACT_REGISTRY
from .simulation import ANALYSIS_REGISTRY, Simulation
from .utils import discover_simulations


class SimulationStore:
    """Discovers and manages multiple simulations across root directories.

    Provides discovery, caching, and aggregation capabilities for working
    with multiple simulation directories.

    Attributes
    ----------
    roots : list[Path]
        List of root paths to search
    _simulations : dict[str, Simulation]
        Dict mapping hash -> Simulation instance (cache)
    _discovery_df : pd.DataFrame | None
        Cached discovery DataFrame
    trajectory_file : str
        Trajectory filename to discover
    structure_file : str
        Structure filename to discover

    """

    def __init__(
        self,
        roots: list[Path | str] | Path | str,
        trajectory_file: str = "prod.xtc",
        structure_file: str = "system.pdb",
        min_status: str = "production",
    ):
        """Initialize store with one or more root paths.

        Parameters
        ----------
        roots : list[Path | str] | Path | str
            Single path or list of paths to search
        trajectory_file : str
            Trajectory filename to discover
        structure_file : str
            Structure filename to discover
        min_status : str
            Minimum simulation status to include in discovery. One of:
            "build", "equilibrated", "production", "completed".

        """
        # Normalize roots to list of Paths
        if isinstance(roots, (str, Path)):
            self.roots = [Path(roots)]
        else:
            self.roots = [Path(r) for r in roots]

        self.trajectory_file = trajectory_file
        self.structure_file = structure_file
        self.min_status = min_status

        self._simulations: dict[str, Simulation] = {}  # hash -> Simulation
        self._discovery_df: pd.DataFrame | None = None

    def discover(self, refresh: bool = False) -> pd.DataFrame:
        """Discover simulations under all roots using existing discover_simulations().

        Returns DataFrame with columns: ["hash", "path", "simulation"]

        Parameters
        ----------
        refresh : bool
            If True, re-discover; if False, use cached result

        Returns
        -------
        pd.DataFrame
            DataFrame with discovered simulations

        """
        from loguru import logger

        if self._discovery_df is not None and not refresh:
            logger.debug("Using cached discovery results")
            return self._discovery_df

        # Clear caches on refresh
        if refresh:
            logger.debug("Refreshing discovery, clearing caches")
            self._simulations.clear()
            self._discovery_df = None

        # Discover simulations in each root
        dfs = []
        for root in self.roots:
            if not root.exists():
                logger.warning(f"Root path does not exist: {root}")
                continue

            logger.debug(f"Discovering simulations in {root}")
            df = discover_simulations(
                root,
                trajectory_file=self.trajectory_file,
                structure_file=self.structure_file,
                min_status=self.min_status,
            )
            dfs.append(df)

        # Concatenate all discoveries
        if dfs:
            self._discovery_df = pd.concat(dfs, ignore_index=True)
            logger.info(
                f"Discovered {len(self._discovery_df)} simulations across {len(self.roots)} roots"
            )
        else:
            self._discovery_df = pd.DataFrame(columns=["hash", "path", "simulation"])
            logger.info("No simulations discovered")

        # Cache Simulation instances by hash
        for _, row in self._discovery_df.iterrows():
            hash_val = row["hash"]
            simulation = row["simulation"]
            self._simulations[hash_val] = simulation

        return self._discovery_df

    def get_simulation(self, hash: str) -> Simulation:
        """Get Simulation instance for given hash.

        Returns cached Simulation instance from discovery DataFrame.

        Parameters
        ----------
        hash : str
            Simulation hash (primary identifier)

        Returns
        -------
        Simulation
            Simulation instance

        Raises
        ------
        ValueError
            If hash not found in discovered simulations

        """
        from loguru import logger

        # Ensure discovery has been run
        self._ensure_discovered()

        # Return cached if available
        if hash in self._simulations:
            logger.debug(f"Returning cached Simulation for hash {hash}")
            return self._simulations[hash]

        # Check if hash is in discovered simulations
        if hash not in self._discovery_df["hash"].values:
            raise ValueError(
                f"Hash {hash} not found in discovered simulations. "
                f"Run discover() or check that simulation is under configured roots."
            )

        # Get Simulation from discovery (already created in discover())
        row = self._discovery_df[self._discovery_df["hash"] == hash].iloc[0]
        simulation = row["simulation"]
        self._simulations[hash] = simulation

        logger.debug(f"Returning Simulation for hash {hash}")
        return simulation

    def list_simulations(self) -> list[str]:
        """Return sorted list of discovered simulation hashes.

        Returns
        -------
        list[str]
            Sorted list of hash strings

        """
        self._ensure_discovered()
        return sorted(self._discovery_df["hash"].tolist())

    def build_metadata_table(
        self,
        flatten_fn: Callable[[BuildInput], dict],
    ) -> pd.DataFrame:
        """Build flattened metadata table across all simulations.

        Applies user-provided flatten function to each BuildInput to
        extract desired fields.

        Parameters
        ----------
        flatten_fn : Callable[[BuildInput], dict]
            Function that takes BuildInput and returns flat dict

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per simulation, including:
            - hash, path (from discovery)
            - flattened fields (from flatten_fn)

        Raises
        ------
        ValueError
            If flatten_fn fails for any simulation

        """
        from loguru import logger

        self._ensure_discovered()

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame()

        # Extract base columns from discovery
        metadata_rows = []

        for _, row in self._discovery_df.iterrows():
            hash_val = row["hash"]
            path = row["path"]
            simulation = row["simulation"]
            build_input = simulation.build_input

            # Apply flatten function
            try:
                flattened = flatten_fn(build_input)
            except Exception as e:
                raise ValueError(f"Flatten function failed for simulation at {path}: {e}") from e

            # Combine with hash and path
            metadata_row = {
                "hash": hash_val,
                "path": path,
                **flattened,
            }

            # Merge tags into metadata row
            if build_input.tags:
                for tag_key, tag_val in build_input.tags.items():
                    metadata_row[f"tag_{tag_key}"] = tag_val

            metadata_rows.append(metadata_row)

        metadata_df = pd.DataFrame(metadata_rows)
        logger.info(f"Built metadata table with {len(metadata_df)} rows")

        return metadata_df

    def search(
        self,
        *,
        simulation_type: str | None = None,
        status: str | None = None,
        hash_prefix: str | None = None,
        tags: dict[str, str] | None = None,
        smiles: str | None = None,
    ) -> pd.DataFrame:
        """Search and filter discovered simulations.

        Applies all provided filters conjunctively (AND logic).

        Parameters
        ----------
        simulation_type : str | None
            Filter by simulation type (exact match).
        status : str | None
            Filter by minimum status threshold.
        hash_prefix : str | None
            Filter by hash prefix (case-insensitive).
        tags : dict[str, str] | None
            Filter by tag key-value pairs (all must match).
        smiles : str | None
            Filter by SMILES substructure match against any species.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with columns: hash, path, simulation_type,
            status, tags (dict or None).

        """
        from loguru import logger

        self._ensure_discovered()

        if len(self._discovery_df) == 0:
            logger.info("No simulations to search")
            return pd.DataFrame(columns=["hash", "path", "simulation_type", "status", "tags"])

        # Build result rows from discovery
        rows = []
        for _, row in self._discovery_df.iterrows():
            sim = row["simulation"]
            bi = sim.build_input
            sim_hash = row["hash"]
            sim_path = row["path"]
            sim_status = sim.status

            # Filter: simulation_type
            if simulation_type is not None and bi.simulation_type != simulation_type:
                continue

            # Filter: status (minimum threshold)
            if status is not None:
                from .constants import STATUS_ORDER

                if status not in STATUS_ORDER:
                    raise ValueError(f"Invalid status '{status}'. Must be one of: {STATUS_ORDER}")
                status_idx = STATUS_ORDER.index(sim_status)
                min_idx = STATUS_ORDER.index(status)
                if status_idx < min_idx:
                    continue

            # Filter: hash prefix
            if hash_prefix is not None:
                if not sim_hash.upper().startswith(hash_prefix.upper()):
                    continue

            # Filter: tags
            if tags is not None:
                if bi.tags is None:
                    continue
                if not all(bi.tags.get(k) == v for k, v in tags.items()):
                    continue

            # Filter: SMILES substructure
            if smiles is not None:
                try:
                    from mdfactory.utils.chemistry_utilities import smiles_substructure_match
                except ImportError:
                    raise ImportError(
                        "RDKit is required for SMILES search. "
                        "Install it via conda: conda install -c conda-forge rdkit"
                    )
                match_found = False
                for species in bi.system.species:
                    species_smiles = getattr(species, "smiles", None)
                    if species_smiles and smiles_substructure_match(smiles, species_smiles):
                        match_found = True
                        break
                if not match_found:
                    continue

            rows.append(
                {
                    "hash": sim_hash,
                    "path": sim_path,
                    "simulation_type": bi.simulation_type,
                    "status": sim_status,
                    "tags": bi.tags,
                }
            )

        logger.info(f"Search returned {len(rows)} results")
        return pd.DataFrame(
            rows,
            columns=["hash", "path", "simulation_type", "status", "tags"],
        )

    def load_analysis_with_metadata(
        self,
        analysis_name: str,
        flatten_fn: Callable[[BuildInput], dict],
        missing_ok: bool = False,
    ) -> pd.DataFrame:
        """Eager load: join analysis data with metadata across simulations.

        Loads the specified analysis from all simulations and joins with
        flattened metadata. Adds 'simulation_hash' column to track source.

        Parameters
        ----------
        analysis_name : str
            Name of analysis to load
        flatten_fn : Callable[[BuildInput], dict]
            Function to flatten BuildInput for each simulation
        missing_ok : bool
            If True, skip simulations without this analysis;
            if False, raise error

        Returns
        -------
        pd.DataFrame
            DataFrame with analysis data joined with metadata.
            Includes 'simulation_hash' column.

        Raises
        ------
        FileNotFoundError
            If missing_ok=False and any simulation lacks analysis

        """
        from loguru import logger

        self._ensure_discovered()

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame()

        # Build metadata table
        metadata_df = self.build_metadata_table(flatten_fn)

        # Load analysis from each simulation
        analysis_dfs = []
        skipped = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                # Load analysis
                df = sim.load_analysis(analysis_name)

                # Add hash column
                df = df.copy()
                df["hash"] = hash_val

                analysis_dfs.append(df)
                logger.debug(f"Loaded '{analysis_name}' from simulation {hash_val}")

            except FileNotFoundError:
                if missing_ok:
                    skipped.append(hash_val)
                    logger.debug(f"Skipping {hash_val}, analysis '{analysis_name}' not found")
                else:
                    raise FileNotFoundError(
                        f"Analysis '{analysis_name}' not found in simulation {hash_val}. "
                        f"Use missing_ok=True to skip simulations without this analysis."
                    )

        if skipped:
            logger.info(f"Skipped {len(skipped)} simulations without '{analysis_name}'")

        if not analysis_dfs:
            logger.warning(f"No simulations have analysis '{analysis_name}'")
            return pd.DataFrame()

        # Concatenate all analysis data
        combined_df = pd.concat(analysis_dfs, ignore_index=True)
        logger.info(
            f"Combined '{analysis_name}' from {len(analysis_dfs)} simulations "
            f"({len(combined_df)} total rows)"
        )

        # Join with metadata
        # Merge on hash (from analysis) = hash (from metadata)
        result_df = combined_df.merge(
            metadata_df,
            left_on="hash",
            right_on="hash",
            how="left",
        )

        logger.info(f"Joined analysis data with metadata ({len(result_df)} rows)")

        return result_df

    def remove_all_analyses(self, simulation_type: str | None = None) -> pd.DataFrame:
        """Remove all analyses from discovered simulations.

        Useful for cleaning up storage or resetting analysis state.

        Parameters
        ----------
        simulation_type : str | None
            If specified, only remove analyses for this simulation type.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with columns:
            - hash: str - Simulation hash
            - simulation_type: str - Type (bilayer, mixedbox)
            - status: str - "success" or "failed"
            - error: str | None - Error message if failed

        """
        from loguru import logger

        self._ensure_discovered()

        columns = ["hash", "simulation_type", "status", "error"]

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, nothing to remove")
            return pd.DataFrame(columns=columns)

        results = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                sim_type = sim.build_input.simulation_type
            except Exception as e:
                logger.warning(f"Failed to get simulation_type for {hash_val}: {e}")
                continue

            if simulation_type is not None and sim_type != simulation_type:
                continue

            try:
                sim.remove_all_analyses()
                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "status": "success",
                        "error": None,
                    }
                )
                logger.debug(f"Removed all analyses from simulation {hash_val}")
            except Exception as e:
                error_msg = str(e)
                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "status": "failed",
                        "error": error_msg,
                    }
                )
                logger.error(f"Failed to remove analyses for {hash_val}: {error_msg}")

        logger.info("Removed all analyses from selected simulations")
        return pd.DataFrame(results, columns=columns)

    def list_analyses_status(
        self,
        simulation_type: str | None = None,
    ) -> pd.DataFrame:
        """List available and completed analyses across all simulations.

        Shows which analyses CAN be run for each simulation type (from ANALYSIS_REGISTRY)
        and which HAVE been run (from AnalysisRegistry).

        Parameters
        ----------
        simulation_type : str | None
            If specified, filter to only this simulation type

        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with columns:
            - hash: str - Simulation hash
            - simulation_type: str - Type (bilayer, mixedbox)
            - analysis_name: str - Name of analysis
            - status: str - "available" or "completed"

        """
        from loguru import logger

        self._ensure_discovered()

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame(columns=["hash", "simulation_type", "analysis_name", "status"])

        # Build rows for each simulation-analysis combination
        rows = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                sim_type = sim.build_input.simulation_type
            except Exception as e:
                logger.warning(f"Failed to get simulation_type for {hash_val}: {e}")
                continue

            # Filter by simulation_type if specified
            if simulation_type is not None and sim_type != simulation_type:
                continue

            # Check if simulation type is in registry
            if sim_type not in ANALYSIS_REGISTRY:
                logger.warning(
                    f"Simulation type '{sim_type}' for {hash_val} not in ANALYSIS_REGISTRY"
                )
                continue

            # Get available analyses for this simulation type
            available_analyses = ANALYSIS_REGISTRY[sim_type].keys()

            # Get completed analyses for this simulation
            try:
                completed_analyses = set(sim.list_analyses())
            except Exception as e:
                logger.warning(f"Failed to list analyses for {hash_val}: {e}")
                completed_analyses = set()

            # Create rows for each available analysis
            for analysis_name in available_analyses:
                status = "completed" if analysis_name in completed_analyses else "not yet run"

                rows.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "analysis_name": analysis_name,
                        "status": status,
                    }
                )

        if not rows:
            logger.info("No analyses found")
            return pd.DataFrame(columns=["hash", "simulation_type", "analysis_name", "status"])

        # Create DataFrame and sort
        status_df = pd.DataFrame(rows)
        status_df = status_df.sort_values(["hash", "analysis_name"]).reset_index(drop=True)

        logger.info(
            f"Listed {len(status_df)} analysis status entries across "
            f"{status_df['hash'].nunique()} simulations"
        )

        return status_df

    def list_artifacts_status(
        self,
        simulation_type: str | None = None,
    ) -> pd.DataFrame:
        """List available and completed artifacts across all simulations.

        Shows which artifacts CAN be run for each simulation type (from ARTIFACT_REGISTRY)
        and which HAVE been run (from AnalysisRegistry).

        Parameters
        ----------
        simulation_type : str | None
            If specified, filter to only this simulation type

        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with columns:
            - hash: str - Simulation hash
            - simulation_type: str - Type (bilayer, mixedbox)
            - artifact_name: str - Name of artifact
            - status: str - "not yet run" or "completed"

        """
        from loguru import logger

        self._ensure_discovered()

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame(columns=["hash", "simulation_type", "artifact_name", "status"])

        rows = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                sim_type = sim.build_input.simulation_type
            except Exception as e:
                logger.warning(f"Failed to get simulation_type for {hash_val}: {e}")
                continue

            if simulation_type is not None and sim_type != simulation_type:
                continue

            if sim_type not in ARTIFACT_REGISTRY:
                logger.warning(
                    f"Simulation type '{sim_type}' for {hash_val} not in ARTIFACT_REGISTRY"
                )
                continue

            available_artifacts = ARTIFACT_REGISTRY[sim_type].keys()

            try:
                completed_artifacts = set(sim.list_artifacts())
            except Exception as e:
                logger.warning(f"Failed to list artifacts for {hash_val}: {e}")
                completed_artifacts = set()

            for artifact_name in available_artifacts:
                status = "completed" if artifact_name in completed_artifacts else "not yet run"
                rows.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "artifact_name": artifact_name,
                        "status": status,
                    }
                )

        if not rows:
            logger.info("No artifacts found")
            return pd.DataFrame(columns=["hash", "simulation_type", "artifact_name", "status"])

        status_df = pd.DataFrame(rows)
        status_df = status_df.sort_values(["hash", "artifact_name"]).reset_index(drop=True)

        logger.info(
            f"Listed {len(status_df)} artifact status entries across "
            f"{status_df['hash'].nunique()} simulations"
        )

        return status_df

    def run_artifacts_batch(
        self,
        artifact_names: list[str] | None = None,
        simulation_type: str | None = None,
        skip_existing: bool = True,
        output_prefix: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run artifacts across multiple simulations in batch mode.

        Executes artifact producers for all (or filtered) simulations, with options
        to skip already-completed artifacts and handle errors gracefully.

        Parameters
        ----------
        artifact_names : list[str] | None
            Specific artifacts to run. If None, run all registered artifacts
            for each simulation's type.
        simulation_type : str | None
            If specified, only run for simulations of this type
        skip_existing : bool
            If True, skip artifacts that have already been run (default: True)
        **kwargs
            Global parameters passed to all artifact functions

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with columns:
            - hash: str - Simulation hash
            - simulation_type: str - Type (bilayer, mixedbox)
            - artifact_name: str - Name of artifact
            - status: str - "success", "skipped", or "failed"
            - error: str | None - Error message if failed
            - files: int | None - Number of files produced (if successful)
            - duration_seconds: float | None - Execution time

        """
        import time

        from loguru import logger

        self._ensure_discovered()

        columns = [
            "hash",
            "simulation_type",
            "artifact_name",
            "status",
            "error",
            "files",
            "duration_seconds",
        ]

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame(columns=columns)

        execution_plan = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                sim_type = sim.build_input.simulation_type
            except Exception as e:
                logger.warning(f"Failed to get simulation_type for {hash_val}: {e}")
                continue

            if simulation_type is not None and sim_type != simulation_type:
                continue

            if sim_type not in ARTIFACT_REGISTRY:
                logger.warning(
                    f"Simulation type '{sim_type}' for {hash_val} not in ARTIFACT_REGISTRY"
                )
                continue

            if artifact_names is not None:
                available = set(ARTIFACT_REGISTRY[sim_type].keys())
                to_run = []
                for name in artifact_names:
                    if name not in available:
                        logger.warning(
                            f"Artifact '{name}' not available for '{sim_type}' "
                            f"(available: {list(available)})"
                        )
                    else:
                        to_run.append(name)
            else:
                to_run = list(ARTIFACT_REGISTRY[sim_type].keys())

            if skip_existing:
                try:
                    completed = set(sim.list_artifacts())
                    to_run = [name for name in to_run if name not in completed]
                except Exception as e:
                    logger.warning(f"Failed to list artifacts for {hash_val}: {e}")

            for artifact_name in to_run:
                execution_plan.append((hash_val, sim_type, sim, artifact_name))

        if not execution_plan:
            logger.info("No artifacts to run (all may be completed or skipped)")
            return pd.DataFrame(columns=columns)

        logger.info(
            f"Batch execution plan: {len(execution_plan)} artifacts across "
            f"{len(set(h for h, _, _, _ in execution_plan))} simulations"
        )

        results = []
        for idx, (hash_val, sim_type, sim, artifact_name) in enumerate(execution_plan, 1):
            logger.debug(
                f"[{idx}/{len(execution_plan)}] Running '{artifact_name}' for {hash_val[:8]}..."
            )

            start_time = time.time()

            try:
                artifact_prefix = output_prefix or artifact_name
                output_paths = sim.run_artifact(
                    artifact_name,
                    output_prefix=artifact_prefix,
                    **kwargs,
                )
                duration = time.time() - start_time

                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "artifact_name": artifact_name,
                        "status": "success",
                        "error": None,
                        "files": len(output_paths),
                        "duration_seconds": round(duration, 3),
                    }
                )

                logger.info(
                    f"Successfully ran '{artifact_name}' for {hash_val[:8]}... "
                    f"({len(output_paths)} files, {duration:.2f}s)"
                )

            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)

                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "artifact_name": artifact_name,
                        "status": "failed",
                        "error": error_msg,
                        "files": None,
                        "duration_seconds": round(duration, 3),
                    }
                )

                logger.error(f"Failed to run '{artifact_name}' for {hash_val[:8]}...: {error_msg}")

        summary_df = pd.DataFrame(results)
        n_success = (summary_df["status"] == "success").sum()
        n_failed = (summary_df["status"] == "failed").sum()
        total_time = summary_df["duration_seconds"].sum()

        logger.info(
            f"Batch execution complete: {n_success} succeeded, {n_failed} failed "
            f"(total time: {total_time:.2f}s)"
        )

        return summary_df

    def remove_all_artifacts(
        self,
        simulation_type: str | None = None,
    ) -> pd.DataFrame:
        """Remove all artifacts across simulations.

        Parameters
        ----------
        simulation_type : str | None
            If specified, only remove artifacts for simulations of this type

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with columns:
            - hash: str - Simulation hash
            - simulation_type: str - Type (bilayer, mixedbox)
            - status: str - "success" or "failed"
            - error: str | None - Error message if failed

        """
        from loguru import logger

        self._ensure_discovered()

        columns = ["hash", "simulation_type", "status", "error"]

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame(columns=columns)

        results = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                sim_type = sim.build_input.simulation_type
            except Exception as e:
                logger.warning(f"Failed to get simulation_type for {hash_val}: {e}")
                continue

            if simulation_type is not None and sim_type != simulation_type:
                continue

            try:
                sim.remove_all_artifacts()
                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as e:
                error_msg = str(e)
                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "status": "failed",
                        "error": error_msg,
                    }
                )
                logger.error(f"Failed to remove artifacts for {hash_val[:8]}...: {error_msg}")

        return pd.DataFrame(results, columns=columns)

    def run_analyses_batch(
        self,
        analysis_names: list[str] | None = None,
        simulation_type: str | None = None,
        skip_existing: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Run analyses across multiple simulations in batch mode.

        Executes analyses for all (or filtered) simulations, with options to
        skip already-completed analyses and handle errors gracefully.

        Parameters
        ----------
        analysis_names : list[str] | None
            Specific analyses to run. If None, run all registered analyses
            for each simulation's type.
        simulation_type : str | None
            If specified, only run for simulations of this type
        skip_existing : bool
            If True, skip analyses that have already been run (default: True)
        **kwargs
            Global parameters passed to all analysis functions
            (e.g., seed=42, n_frames=100)

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with columns:
            - hash: str - Simulation hash
            - simulation_type: str - Type (bilayer, mixedbox)
            - analysis_name: str - Name of analysis
            - status: str - "success", "skipped", or "failed"
            - error: str | None - Error message if failed
            - rows: int | None - Number of rows in result (if successful)
            - duration_seconds: float | None - Execution time

        """
        import time

        from loguru import logger

        self._ensure_discovered()

        columns = [
            "hash",
            "simulation_type",
            "analysis_name",
            "status",
            "error",
            "rows",
            "duration_seconds",
        ]

        if len(self._discovery_df) == 0:
            logger.warning("No simulations discovered, returning empty DataFrame")
            return pd.DataFrame(columns=columns)

        # Build execution plan
        execution_plan = []

        for hash_val in self.list_simulations():
            sim = self.get_simulation(hash_val)

            try:
                sim_type = sim.build_input.simulation_type
            except Exception as e:
                logger.warning(f"Failed to get simulation_type for {hash_val}: {e}")
                continue

            # Filter by simulation_type if specified
            if simulation_type is not None and sim_type != simulation_type:
                continue

            # Check if simulation type is in registry
            if sim_type not in ANALYSIS_REGISTRY:
                logger.warning(
                    f"Simulation type '{sim_type}' for {hash_val} not in ANALYSIS_REGISTRY"
                )
                continue

            # Determine which analyses to run
            if analysis_names is not None:
                # Use specified analyses (validate they're available)
                available = set(ANALYSIS_REGISTRY[sim_type].keys())
                to_run = []
                for name in analysis_names:
                    if name not in available:
                        logger.warning(
                            f"Analysis '{name}' not available for '{sim_type}' "
                            f"(available: {list(available)})"
                        )
                    else:
                        to_run.append(name)
            else:
                # Run all analyses for this simulation type
                to_run = list(ANALYSIS_REGISTRY[sim_type].keys())

            # Filter out existing analyses if skip_existing=True
            if skip_existing:
                try:
                    completed = set(sim.list_analyses())
                    to_run = [name for name in to_run if name not in completed]
                except Exception as e:
                    logger.warning(f"Failed to list analyses for {hash_val}: {e}")

            # Add to execution plan
            for analysis_name in to_run:
                execution_plan.append((hash_val, sim_type, sim, analysis_name))

        if not execution_plan:
            logger.info("No analyses to run (all may be completed or skipped)")
            return pd.DataFrame(columns=columns)

        logger.info(
            f"Batch execution plan: {len(execution_plan)} analyses across "
            f"{len(set(h for h, _, _, _ in execution_plan))} simulations"
        )

        # Execute analyses
        results = []
        for idx, (hash_val, sim_type, sim, analysis_name) in enumerate(execution_plan, 1):
            logger.debug(
                f"[{idx}/{len(execution_plan)}] Running '{analysis_name}' for {hash_val[:8]}..."
            )

            start_time = time.time()

            try:
                result_df = sim.run_analysis(analysis_name, **kwargs)
                duration = time.time() - start_time

                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "analysis_name": analysis_name,
                        "status": "success",
                        "error": None,
                        "rows": len(result_df),
                        "duration_seconds": round(duration, 3),
                    }
                )

                logger.info(
                    f"Successfully ran '{analysis_name}' for {hash_val[:8]}... "
                    f"({len(result_df)} rows, {duration:.2f}s)"
                )

            except Exception as e:
                duration = time.time() - start_time

                error_msg = str(e)
                results.append(
                    {
                        "hash": hash_val,
                        "simulation_type": sim_type,
                        "analysis_name": analysis_name,
                        "status": "failed",
                        "error": error_msg,
                        "rows": None,
                        "duration_seconds": round(duration, 3),
                    }
                )

                logger.error(f"Failed to run '{analysis_name}' for {hash_val[:8]}...: {error_msg}")

        # Build summary DataFrame
        summary_df = pd.DataFrame(results)

        # Log summary
        n_success = (summary_df["status"] == "success").sum()
        n_failed = (summary_df["status"] == "failed").sum()
        total_time = summary_df["duration_seconds"].sum()

        logger.info(
            f"Batch execution complete: {n_success} succeeded, {n_failed} failed "
            f"(total time: {total_time:.2f}s)"
        )

        return summary_df

    def _ensure_discovered(self) -> None:
        """Ensure discovery has been run at least once."""
        if self._discovery_df is None:
            self.discover()

    def build_lnp_chemistry_table(self) -> pd.DataFrame:
        """Build table of LNP chemistry for all discovered simulations.

        Extracts HL, CHL, and IL (ILN+ILP combined) counts, fractions,
        and SMILES for each simulation.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - hash, path
            - HL_count, HL_fraction, HL_smiles
            - CHL_count, CHL_fraction, CHL_smiles
            - IL_count, IL_fraction, ILN_smiles, ILP_smiles

        """
        from .utils import extract_lnp_chemistry

        return self.build_metadata_table(extract_lnp_chemistry)

    def load_analysis_with_lnp_chemistry(
        self,
        analysis_name: str,
        missing_ok: bool = False,
    ) -> pd.DataFrame:
        """Load analysis data joined with LNP chemistry metadata.

        Loads the specified analysis from all simulations and joins with
        LNP chemistry (HL, CHL, IL counts/fractions/SMILES).

        Parameters
        ----------
        analysis_name : str
            Name of analysis to load
        missing_ok : bool
            If True, skip simulations without this analysis;
            if False, raise error

        Returns
        -------
        pd.DataFrame
            DataFrame with analysis data joined with LNP chemistry.
            Includes 'hash' column for tracking source simulation.

        """
        from .utils import extract_lnp_chemistry

        return self.load_analysis_with_metadata(
            analysis_name,
            flatten_fn=extract_lnp_chemistry,
            missing_ok=missing_ok,
        )

    def build_all_species_table(self) -> pd.DataFrame:
        """Build table of all species for all discovered simulations.

        Extracts count, fraction, and SMILES for every species defined
        in each simulation's YAML file. No filtering or grouping.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - hash, path
            - {resname}_count, {resname}_fraction, {resname}_smiles (per species)
            - total_species_count, total_molecule_count

        """
        from .utils import extract_all_species

        return self.build_metadata_table(extract_all_species)

    def load_analysis_with_all_species(
        self,
        analysis_name: str,
        missing_ok: bool = False,
    ) -> pd.DataFrame:
        """Load analysis data joined with all species metadata.

        Loads the specified analysis from all simulations and joins with
        all species data (count, fraction, SMILES for every species).

        Parameters
        ----------
        analysis_name : str
            Name of analysis to load
        missing_ok : bool
            If True, skip simulations without this analysis;
            if False, raise error

        Returns
        -------
        pd.DataFrame
            DataFrame with analysis data joined with all species.
            Includes 'hash' column for tracking source simulation.

        """
        from .utils import extract_all_species

        return self.load_analysis_with_metadata(
            analysis_name,
            flatten_fn=extract_all_species,
            missing_ok=missing_ok,
        )

    def build_chemistry_table(
        self,
        mode: str = "all",
        species_groups: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """Build chemistry table with configurable extraction mode.

        Parameters
        ----------
        mode : str
            Extraction mode:
            - "all": Extract all species from YAML (default)
            - "lnp": Use LNP-specific grouping (HL, CHL, IL)
            - "custom": Use custom species_groups
        species_groups : dict[str, list[str]] | None
            Required when mode="custom". Mapping of group names to resnames.

        Returns
        -------
        pd.DataFrame
            DataFrame with hash, path, and chemistry columns.

        """
        from .utils import get_chemistry_extractor

        extractor = get_chemistry_extractor(mode=mode, species_groups=species_groups)
        return self.build_metadata_table(extractor)

    def load_analysis_with_chemistry(
        self,
        analysis_name: str,
        mode: str = "all",
        species_groups: dict[str, list[str]] | None = None,
        missing_ok: bool = False,
    ) -> pd.DataFrame:
        """Load analysis data joined with chemistry metadata.

        Parameters
        ----------
        analysis_name : str
            Name of analysis to load
        mode : str
            Extraction mode: "all", "lnp", or "custom"
        species_groups : dict[str, list[str]] | None
            Required when mode="custom"
        missing_ok : bool
            If True, skip simulations without this analysis

        Returns
        -------
        pd.DataFrame
            DataFrame with analysis data joined with chemistry.

        """
        from .utils import get_chemistry_extractor

        extractor = get_chemistry_extractor(mode=mode, species_groups=species_groups)
        return self.load_analysis_with_metadata(
            analysis_name,
            flatten_fn=extractor,
            missing_ok=missing_ok,
        )
