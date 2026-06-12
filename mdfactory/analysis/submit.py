# ABOUTME: Analysis execution via local workers or SLURM via submitit
# ABOUTME: Resolves simulation paths, dispatches analyses, and manages parallel execution
"""Submit analyses locally or via submitit."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from mdfactory.analysis.artifacts import ARTIFACT_REGISTRY
from mdfactory.analysis.simulation import ANALYSIS_REGISTRY, Simulation

# SlurmConfig and normalize_slurm_time live in the performance package so that
# every SLURM-facing backend (submitit, Parsl, Nextflow) can share them.
# Re-exported here for backward compatibility:
#   from mdfactory.analysis.submit import SlurmConfig        # still works
#   from mdfactory.analysis.submit import normalize_slurm_time  # still works
from mdfactory.performance.slurm_config import (  # noqa: F401  (re-export)
    SlurmConfig,
    normalize_slurm_time,
)


def resolve_simulation_paths(
    paths: Iterable[Path | str],
    *,
    trajectory_file: str = "prod.xtc",
    structure_file: str = "system.pdb",
) -> list[Path]:
    """Expand simulation paths using discovery rules."""
    from mdfactory.analysis.utils import discover_simulations

    sim_paths: list[Path] = []
    for raw in paths:
        base = Path(raw).resolve()
        if not base.exists():
            raise FileNotFoundError(f"Simulation path does not exist: {base}")
        if base.is_dir():
            df = discover_simulations(
                base,
                trajectory_file=trajectory_file,
                structure_file=structure_file,
            )
            sim_paths.extend([Path(p) for p in df["path"].tolist()])
        else:
            raise NotADirectoryError(f"Simulation path is not a directory: {base}")
    if not sim_paths:
        raise ValueError("No simulations discovered from provided paths.")
    return sorted(set(sim_paths))


def resolve_simulation_paths_from_yaml(build_yaml: Path) -> list[Path]:
    """Resolve simulation paths from the build summary YAML."""
    import yaml

    build_yaml = Path(build_yaml).resolve()
    if not build_yaml.exists():
        raise FileNotFoundError(f"Build YAML not found: {build_yaml}")
    with open(build_yaml, "r") as handle:
        data = yaml.safe_load(handle)
    dirs = data.get("system_directory", [])
    if not dirs:
        raise ValueError("Build YAML does not contain system_directory entries.")
    return [Path(p).resolve() for p in dirs]


def parse_hash_filter(raw_hashes: Iterable[str]) -> set[str]:
    """Parse comma-separated hash values into a flat set."""
    result: set[str] = set()
    for entry in raw_hashes:
        for token in entry.split(","):
            stripped = token.strip()
            if stripped:
                result.add(stripped)
    return result


def resolve_hash_prefix(requested: str, available: set[str]) -> str | None:
    """Resolve a hash string against available hashes, supporting prefix matching.

    Returns the matched full hash, or None if no match is found.
    Raises ValueError if the prefix matches multiple hashes.
    """
    if requested in available:
        return requested
    matches = [h for h in available if h.startswith(requested)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous hash prefix '{requested}' matches {len(matches)} simulations: "
            f"{sorted(matches)}"
        )
    return None


def filter_paths_by_hash(
    sim_paths: list[Path],
    hashes: Iterable[str],
    *,
    trajectory_file: str = "prod.xtc",
    structure_file: str = "system.pdb",
) -> list[Path]:
    """Filter simulation paths to only those matching the requested hashes.

    Supports prefix matching: a short string that uniquely identifies
    a single discovered hash will be expanded to match.
    """
    from loguru import logger

    from mdfactory.analysis.store import SimulationStore

    requested = parse_hash_filter(hashes)
    if not requested:
        return sim_paths

    store = SimulationStore(
        [str(p) for p in sim_paths],
        trajectory_file=trajectory_file,
        structure_file=structure_file,
    )
    df = store.discover()
    available_hashes = set(df["hash"].tolist())

    matched_hashes: set[str] = set()
    for req in requested:
        resolved = resolve_hash_prefix(req, available_hashes)
        if resolved is None:
            logger.warning(f"Hash '{req}' not found in discovered simulations")
        else:
            matched_hashes.add(resolved)

    if not matched_hashes:
        raise ValueError(
            f"None of the requested hashes were found. "
            f"Requested: {sorted(requested)}. "
            f"Available: {sorted(available_hashes)}"
        )

    filtered_df = df[df["hash"].isin(matched_hashes)]
    return sorted(Path(p) for p in filtered_df["path"].tolist())


def determine_log_dir(sim_paths: Iterable[Path]) -> Path:
    """Determine default log dir for submitit."""
    sim_paths = list(sim_paths)
    if not sim_paths:
        raise ValueError("No simulations provided for log directory resolution.")
    root = sim_paths[0].parent
    return root / ".analysis" / "logs"


def resolve_analyses(
    simulation: Simulation,
    analysis_names: Iterable[str] | None,
) -> list[str]:
    """Return analyses to run for a simulation."""
    sim_type = simulation.build_input.simulation_type
    if sim_type not in ANALYSIS_REGISTRY:
        raise ValueError(f"Simulation type '{sim_type}' not in ANALYSIS_REGISTRY")
    available = set(ANALYSIS_REGISTRY[sim_type].keys())
    if analysis_names is None:
        return sorted(available)
    requested = set(analysis_names)
    missing = requested - available
    if missing:
        raise ValueError(
            f"Analyses {sorted(missing)} not available for simulation type '{sim_type}'. "
            f"Possible analyses are: {available}."
        )
    return sorted(requested)


def resolve_artifacts(
    simulation: Simulation,
    artifact_names: Iterable[str] | None,
) -> list[str]:
    """Return artifacts to run for a simulation."""
    sim_type = simulation.build_input.simulation_type
    if sim_type not in ARTIFACT_REGISTRY:
        raise ValueError(f"Simulation type '{sim_type}' not in ARTIFACT_REGISTRY")
    available = set(ARTIFACT_REGISTRY[sim_type].keys())
    if artifact_names is None:
        return sorted(available)
    requested = set(artifact_names)
    missing = requested - available
    if missing:
        raise ValueError(
            f"Artifacts {sorted(missing)} not available for simulation type '{sim_type}'. "
            f"Possible artifacts are: {available}."
        )
    return sorted(requested)


def resolve_tool_paths(
    *,
    vmd_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve tool paths from explicit inputs or PATH."""
    resolved_vmd = vmd_path or shutil.which("vmd")
    resolved_ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if resolved_vmd:
        resolved_vmd = str(Path(resolved_vmd).resolve())
    if resolved_ffmpeg:
        resolved_ffmpeg = str(Path(resolved_ffmpeg).resolve())
    return resolved_vmd, resolved_ffmpeg


def apply_tool_paths(vmd_path: str | None, ffmpeg_path: str | None) -> dict[str, str]:
    """Apply tool paths by prepending to PATH."""
    original_env = os.environ.copy()
    paths = []
    if vmd_path:
        vmd_bin = Path(vmd_path)
        if not vmd_bin.exists():
            raise FileNotFoundError(f"VMD not found: {vmd_bin}")
        paths.append(str(vmd_bin.parent))
    if ffmpeg_path:
        ffmpeg_bin = Path(ffmpeg_path)
        if not ffmpeg_bin.exists():
            raise FileNotFoundError(f"ffmpeg not found: {ffmpeg_bin}")
        paths.append(str(ffmpeg_bin.parent))
    if paths:
        os.environ["PATH"] = os.pathsep.join(paths + [original_env.get("PATH", "")])
    return original_env


def restore_env(original_env: dict[str, str]) -> None:
    """Restore environment variables."""
    os.environ.clear()
    os.environ.update(original_env)


def run_simulation_analyses(
    sim_path: Path,
    analysis_names: Iterable[str] | None,
    *,
    structure_file: str,
    trajectory_file: str,
    backend: str = "serial",
    n_workers: int = 1,
    skip_existing: bool = True,
    analysis_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run analyses for a single simulation path."""
    start_time = datetime.now()
    simulation = Simulation(
        sim_path,
        structure_file=structure_file,
        trajectory_file=trajectory_file,
    )
    analyses = resolve_analyses(simulation, analysis_names)
    if skip_existing:
        completed = set(simulation.list_analyses())
        analyses = [name for name in analyses if name not in completed]
    extra_kwargs = analysis_kwargs or {}
    results = []
    status = "success"
    error = None
    try:
        if not analyses:
            status = "skipped"
        for name in analyses:
            df = simulation.run_analysis(name, backend=backend, n_workers=n_workers, **extra_kwargs)
            results.append({"analysis": name, "rows": len(df)})
    except Exception as exc:
        status = "failed"
        error = str(exc)
    duration = (datetime.now() - start_time).total_seconds()
    return {
        "path": str(sim_path),
        "hash": simulation.build_input.hash,
        "status": status,
        "error": error,
        "analyses": analyses,
        "results": results,
        "duration_seconds": round(duration, 3),
    }


def run_simulation_artifacts(
    sim_path: Path,
    artifact_names: Iterable[str] | None,
    *,
    structure_file: str,
    trajectory_file: str,
    output_prefix: str | None = None,
    vmd_path: str | None = None,
    ffmpeg_path: str | None = None,
    skip_existing: bool = True,
) -> dict[str, object]:
    """Run artifacts for a single simulation path."""
    start_time = datetime.now()
    simulation = Simulation(
        sim_path,
        structure_file=structure_file,
        trajectory_file=trajectory_file,
    )
    artifacts = resolve_artifacts(simulation, artifact_names)
    if skip_existing:
        completed = set(simulation.list_artifacts())
        artifacts = [name for name in artifacts if name not in completed]
    results = []
    status = "success"
    error = None
    resolved_vmd, resolved_ffmpeg = resolve_tool_paths(
        vmd_path=vmd_path,
        ffmpeg_path=ffmpeg_path,
    )
    original_env = apply_tool_paths(resolved_vmd, resolved_ffmpeg)
    try:
        if not artifacts:
            status = "skipped"
        for name in artifacts:
            prefix = output_prefix or name
            files = simulation.run_artifact(name, output_prefix=prefix)
            results.append({"artifact": name, "files": len(files)})
    except Exception as exc:
        status = "failed"
        error = str(exc)
    finally:
        restore_env(original_env)
    duration = (datetime.now() - start_time).total_seconds()
    return {
        "path": str(sim_path),
        "hash": simulation.build_input.hash,
        "status": status,
        "error": error,
        "artifacts": artifacts,
        "results": results,
        "duration_seconds": round(duration, 3),
        "vmd_path": resolved_vmd,
        "ffmpeg_path": resolved_ffmpeg,
    }


def run_analyses_local(
    sim_paths: Iterable[Path],
    analysis_names: Iterable[str] | None,
    *,
    structure_file: str,
    trajectory_file: str,
    backend: str = "serial",
    n_workers: int = 1,
    skip_existing: bool = True,
    analysis_kwargs: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run analyses sequentially across simulations."""
    rows = []
    for sim_path in sim_paths:
        rows.append(
            run_simulation_analyses(
                sim_path,
                analysis_names,
                structure_file=structure_file,
                trajectory_file=trajectory_file,
                backend=backend,
                n_workers=n_workers,
                skip_existing=skip_existing,
                analysis_kwargs=analysis_kwargs,
            )
        )
    return pd.DataFrame(rows)


def submit_analyses_slurm(
    sim_paths: Iterable[Path],
    analysis_names: Iterable[str] | None,
    *,
    structure_file: str,
    trajectory_file: str,
    slurm: SlurmConfig,
    log_dir: Path,
    skip_existing: bool = True,
    wait: bool = True,
    analysis_backend: str = "multiprocessing",
    analysis_workers: int | None = None,
    analysis_kwargs: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Submit analyses via submitit on SLURM."""
    from loguru import logger

    try:
        import submitit  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "submitit is required for SLURM submission. "
            "Install with `pip install 'mdfactory[submitit]'`."
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        slurm_partition=slurm.partition,
        slurm_time=normalize_slurm_time(slurm.time),
        slurm_cpus_per_task=slurm.cpus_per_task,
        slurm_mem=f"{slurm.mem_gb}G",
        slurm_account=slurm.account,
        slurm_job_name=slurm.job_name_prefix,
    )
    if slurm.qos:
        executor.update_parameters(slurm_qos=slurm.qos)
    if slurm.constraint:
        executor.update_parameters(slurm_constraint=slurm.constraint)

    sim_paths = list(sim_paths)
    jobs = []
    logger.info(f"Submitting {len(sim_paths)} analysis jobs to SLURM")
    for sim_path in sim_paths:
        workers = analysis_workers or slurm.cpus_per_task
        job = executor.submit(
            run_simulation_analyses,
            sim_path,
            analysis_names,
            structure_file=structure_file,
            trajectory_file=trajectory_file,
            backend=analysis_backend,
            n_workers=workers,
            skip_existing=skip_existing,
            analysis_kwargs=analysis_kwargs,
        )
        logger.info(f"Submitted analysis job {job.job_id} for {sim_path}")
        jobs.append(job)

    manifest = log_dir / "submitit_manifest.csv"
    rows = []
    for job, sim_path in zip(jobs, sim_paths, strict=False):
        rows.append(
            {
                "path": str(sim_path),
                "job_id": job.job_id,
                "log_dir": str(log_dir),
            }
        )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(manifest, index=False)
    if not wait:
        return manifest_df

    results = []
    for job, sim_path in zip(jobs, sim_paths, strict=False):
        try:
            result = job.result()
            result["job_id"] = job.job_id
            if result.get("status") == "failed" and result.get("error"):
                logger.error(result["error"])
            results.append(result)
        except Exception as exc:
            logger.error(str(exc))
            results.append(
                {
                    "path": str(sim_path),
                    "job_id": job.job_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(results)


def submit_artifacts_slurm(
    sim_paths: Iterable[Path],
    artifact_names: Iterable[str] | None,
    *,
    structure_file: str,
    trajectory_file: str,
    output_prefix: str | None,
    vmd_path: str | None,
    ffmpeg_path: str | None,
    slurm: SlurmConfig,
    log_dir: Path,
    skip_existing: bool = True,
    wait: bool = True,
) -> pd.DataFrame:
    """Submit artifacts via submitit on SLURM."""
    from loguru import logger

    try:
        import submitit  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "submitit is required for SLURM submission. "
            "Install with `pip install 'mdfactory[submitit]'`."
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        slurm_partition=slurm.partition,
        slurm_time=normalize_slurm_time(slurm.time),
        slurm_cpus_per_task=slurm.cpus_per_task,
        slurm_mem=f"{slurm.mem_gb}G",
        slurm_account=slurm.account,
        slurm_job_name=slurm.job_name_prefix,
    )
    if slurm.qos:
        executor.update_parameters(slurm_qos=slurm.qos)
    if slurm.constraint:
        executor.update_parameters(slurm_constraint=slurm.constraint)

    sim_paths = list(sim_paths)
    jobs = []
    resolved_vmd, resolved_ffmpeg = resolve_tool_paths(
        vmd_path=vmd_path,
        ffmpeg_path=ffmpeg_path,
    )
    logger.info(f"Submitting {len(sim_paths)} artifact jobs to SLURM")
    for sim_path in sim_paths:
        job = executor.submit(
            run_simulation_artifacts,
            sim_path,
            artifact_names,
            structure_file=structure_file,
            trajectory_file=trajectory_file,
            output_prefix=output_prefix,
            vmd_path=resolved_vmd,
            ffmpeg_path=resolved_ffmpeg,
            skip_existing=skip_existing,
        )
        logger.info(f"Submitted artifact job {job.job_id} for {sim_path}")
        jobs.append(job)

    manifest = log_dir / "submitit_manifest.csv"
    rows = []
    for job, sim_path in zip(jobs, sim_paths, strict=False):
        rows.append(
            {
                "path": str(sim_path),
                "job_id": job.job_id,
                "log_dir": str(log_dir),
                "vmd_path": resolved_vmd,
                "ffmpeg_path": resolved_ffmpeg,
            }
        )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(manifest, index=False)
    if not wait:
        return manifest_df

    results = []
    for job, sim_path in zip(jobs, sim_paths, strict=False):
        try:
            result = job.result()
            result["job_id"] = job.job_id
            if result.get("status") == "failed" and result.get("error"):
                logger.error(result["error"])
            results.append(result)
        except Exception as exc:
            logger.error(str(exc))
            results.append(
                {
                    "path": str(sim_path),
                    "job_id": job.job_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(results)
