# ABOUTME: Scalability benchmark sweep for optimal resource discovery
# ABOUTME: Runs short GROMACS trials across CPU/GPU configs, parses ns/day, selects optimum
"""Scalability benchmark sweep.

Discovers optimal resource configuration for a given system size by running
short production trials across a sweep of executor configs.  Each sweep point
opens a sequential :func:`~mdfactory.orchestration.session.parsl_session`,
submits a short mdrun via the standard Parsl app factories, and parses the
resulting GROMACS log for throughput (ns/day).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from mdfactory.orchestration.config import ExecutorConfig

# ---------------------------------------------------------------------------
# GROMACS md.log performance parser
# ---------------------------------------------------------------------------

#: Regex for the GROMACS performance line: ``Performance:   <ns/day>   <hour/ns>``
_PERF_RE = re.compile(r"^\s*Performance:\s+([\d.]+)\s+([\d.]+)\s*$", re.MULTILINE)


def parse_mdlog_performance(log_path: Path) -> float | None:
    """Extract ns/day from a GROMACS log file's performance summary.

    Reads the last 4 KB of the log (the performance block is always at the
    end) and returns the ns/day value.  If the log contains multiple
    performance blocks (e.g. from a restart with ``-append``), the last one
    is used.

    Parameters
    ----------
    log_path : Path
        Path to a GROMACS ``.log`` file (e.g. ``prod.log``, ``min.log``).

    Returns
    -------
    float or None
        Throughput in ns/day, or ``None`` if the performance block is
        missing (truncated run, crashed before completion).

    """
    if not log_path.exists():
        return None

    # Read only the tail — performance block is always at the end
    size = log_path.stat().st_size
    read_bytes = min(size, 4096)
    with open(log_path, "r") as fh:
        if size > read_bytes:
            fh.seek(size - read_bytes)
        tail = fh.read()

    matches = list(_PERF_RE.finditer(tail))
    if not matches:
        return None

    # Last match wins (restart with -append produces multiple blocks)
    ns_per_day = float(matches[-1].group(1))
    return ns_per_day


# ---------------------------------------------------------------------------
# Config and result models
# ---------------------------------------------------------------------------


class TrialResult(BaseModel, frozen=True):
    """Result of a single benchmark trial."""

    cpu_count: int
    gpu_replicas: int
    ns_per_day: float | None
    wall_seconds: float | None
    error: str | None = None

    @property
    def efficiency(self) -> float | None:
        """Throughput per core-hour (ns/day per core)."""
        if self.ns_per_day is None or self.cpu_count == 0:
            return None
        return self.ns_per_day / self.cpu_count


class BenchmarkResult(BaseModel, frozen=True):
    """Aggregated results of a benchmark sweep."""

    system_path: str
    trials: list[TrialResult]
    optimum: TrialResult | None = None
    selection_criterion: str = "ns_per_day"

    def save(self, path: Path) -> None:
        """Write results as JSON sidecar."""
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResult":
        """Load results from a JSON sidecar."""
        return cls.model_validate_json(path.read_text())


class BenchmarkConfig(BaseModel, frozen=True):
    """Configuration for a scalability benchmark sweep.

    Parameters
    ----------
    cpu_counts : list[int]
        Core counts to sweep (e.g. ``[1, 2, 4, 8, 16]``).
    gpu_replicas : list[int]
        GPU sharing replicas to sweep (e.g. ``[1, 2, 4]``).
        An empty list skips the GPU sweep dimension.
    duration_ps : float
        Benchmark trial duration in picoseconds.
    selection : str
        Optimum selection: ``"ns_per_day"`` for raw throughput or
        ``"efficiency"`` for throughput per core-hour.
    mdp_overrides : dict[str, str]
        Additional MDP parameter overrides applied to the benchmark MDP
        (e.g. ``{"nstxout-compressed": "0"}`` to disable trajectory output).

    """

    cpu_counts: list[int] = [1, 2, 4, 8]
    gpu_replicas: list[int] = []
    duration_ps: float = 100.0
    selection: Literal["ns_per_day", "efficiency"] = "ns_per_day"
    mdp_overrides: dict[str, str] = {}

    @field_validator("cpu_counts")
    @classmethod
    def _cpu_counts_nonempty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("cpu_counts must not be empty")
        return v

    @field_validator("duration_ps")
    @classmethod
    def _duration_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duration_ps must be positive")
        return v


# ---------------------------------------------------------------------------
# Short benchmark MDP generator
# ---------------------------------------------------------------------------


def _generate_benchmark_mdp(
    source_mdp: Path,
    output_mdp: Path,
    duration_ps: float,
    dt: float | None = None,
    extra_overrides: dict[str, str] | None = None,
) -> None:
    """Create a short-duration MDP for benchmarking from an existing MDP.

    Reads ``source_mdp``, overrides ``nsteps`` to achieve ``duration_ps``,
    reduces output frequency to minimize I/O overhead, and writes the result.

    Parameters
    ----------
    source_mdp : Path
        Source production MDP file.
    output_mdp : Path
        Output benchmark MDP path.
    duration_ps : float
        Target simulation duration in picoseconds.
    dt : float, optional
        Timestep override in ps.  If ``None``, reads ``dt`` from the source MDP.
    extra_overrides : dict, optional
        Additional key-value overrides (e.g. ``{"nstxout-compressed": "0"}``).

    """
    from mdfactory.orchestration.mdp import (
        get_mdp_value,
        modify_mdp_value,
        parse_mdp,
        write_mdp,
    )

    parsed = parse_mdp(source_mdp)

    # Determine timestep
    if dt is None:
        dt_str = get_mdp_value(parsed, "dt")
        dt = float(dt_str) if dt_str else 0.002  # GROMACS default

    nsteps = int(duration_ps / dt)
    parsed = modify_mdp_value(parsed, "nsteps", str(nsteps))

    # Reduce output frequency to minimize I/O overhead during benchmark
    # Use large intervals — we only care about performance, not trajectory
    io_interval = str(max(nsteps, 1000))
    for key in ("nstxout", "nstvout", "nstfout", "nstxout_compressed", "nstenergy", "nstlog"):
        if get_mdp_value(parsed, key) is not None:
            parsed = modify_mdp_value(parsed, key, io_interval)

    # Apply any user-specified overrides
    if extra_overrides:
        for key, value in extra_overrides.items():
            norm_key = key.lower().replace("-", "_")
            parsed = modify_mdp_value(parsed, norm_key, value)

    write_mdp(parsed, output_mdp)
    logger.debug(f"Benchmark MDP: {output_mdp} (nsteps={nsteps}, dt={dt})")


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


def _select_optimum(
    trials: list[TrialResult],
    criterion: str,
) -> TrialResult | None:
    """Pick the best trial based on the selection criterion."""
    successful = [t for t in trials if t.ns_per_day is not None]
    if not successful:
        return None

    if criterion == "efficiency":
        return max(successful, key=lambda t: t.efficiency or 0.0)
    # Default: raw throughput
    return max(successful, key=lambda t: t.ns_per_day or 0.0)


def _build_sweep_configs(
    base_config: "ExecutorConfig",
    benchmark_config: BenchmarkConfig,
) -> list[dict]:
    """Generate sweep point descriptors from base config and benchmark params.

    Each descriptor contains the fields needed to create a per-trial
    executor config variant and to record the trial result.

    Returns
    -------
    list[dict]
        Each dict has ``cpu_count``, ``gpu_replicas``, and ``config_overrides``.

    """
    points = []

    for cpu_count in benchmark_config.cpu_counts:
        if benchmark_config.gpu_replicas:
            for gpu_rep in benchmark_config.gpu_replicas:
                points.append(
                    {
                        "cpu_count": cpu_count,
                        "gpu_replicas": gpu_rep,
                        "config_overrides": {
                            "cpus_per_node": cpu_count,
                            "max_workers_per_node": gpu_rep,
                        },
                    }
                )
        else:
            points.append(
                {
                    "cpu_count": cpu_count,
                    "gpu_replicas": 0,
                    "config_overrides": {
                        "cpus_per_node": cpu_count,
                    },
                }
            )

    return points


def run_benchmark_sweep(
    system_path: Path,
    base_config: "ExecutorConfig",
    benchmark_config: BenchmarkConfig | None = None,
    *,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Run a scalability benchmark sweep for a simulation system.

    Generates one executor config variant per sweep point, runs a short
    mdrun trial for each, parses the GROMACS log for ns/day, and selects
    the optimum configuration.

    Parameters
    ----------
    system_path : Path
        Path to a prepared simulation directory (must contain topology,
        structure, and MDP files).
    base_config : ExecutorConfig
        Base executor configuration.  Sweep points are derived by
        overriding ``cpus_per_node`` and ``max_workers_per_node``.
    benchmark_config : BenchmarkConfig, optional
        Sweep parameters.  Defaults to ``BenchmarkConfig()`` with
        CPU counts ``[1, 2, 4, 8]``.
    dry_run : bool
        If ``True``, log what would be run without submitting work.

    Returns
    -------
    BenchmarkResult
        All trial results with the selected optimum.

    """
    if benchmark_config is None:
        benchmark_config = BenchmarkConfig()

    system_path = Path(system_path)
    sweep_points = _build_sweep_configs(base_config, benchmark_config)

    logger.info(f"Benchmark sweep: {len(sweep_points)} point(s) for {system_path.name}")

    # Dry-run: show plan without loading Parsl
    if dry_run:
        trials = []
        for point in sweep_points:
            desc = (
                f"  cpus={point['cpu_count']}, "
                f"gpu_replicas={point['gpu_replicas']}, "
                f"duration={benchmark_config.duration_ps} ps"
            )
            logger.info(f"[dry-run] Trial: {desc}")
            trials.append(
                TrialResult(
                    cpu_count=point["cpu_count"],
                    gpu_replicas=point["gpu_replicas"],
                    ns_per_day=None,
                    wall_seconds=None,
                )
            )
        logger.info(f"[dry-run] {len(trials)} trial(s) would be run")
        return BenchmarkResult(
            system_path=str(system_path),
            trials=trials,
            selection_criterion=benchmark_config.selection,
        )

    # Prepare benchmark MDP from the production MDP
    from mdfactory.orchestration.stages import STAGE_BY_NAME

    prod_spec = STAGE_BY_NAME["Production"]
    source_mdp = system_path / prod_spec.mdp_file
    if not source_mdp.exists():
        raise FileNotFoundError(
            f"Production MDP not found: {source_mdp}. "
            "Benchmark requires a prepared simulation directory."
        )

    bench_mdp = system_path / "benchmark.mdp"
    _generate_benchmark_mdp(
        source_mdp,
        bench_mdp,
        duration_ps=benchmark_config.duration_ps,
        extra_overrides=benchmark_config.mdp_overrides,
    )

    # Run each sweep point as a sequential Parsl session
    import time

    from mdfactory.orchestration.apps import get_grompp_app, get_mdrun_app
    from mdfactory.orchestration.session import parsl_session
    from mdfactory.orchestration.stages import _extract_resource_hints
    from mdfactory.orchestration.trajectory import find_structure_file

    structure = find_structure_file(system_path)
    if not structure:
        raise FileNotFoundError(f"No structure file found in {system_path}")

    trials: list[TrialResult] = []

    for point in sweep_points:
        cpu_count = point["cpu_count"]
        gpu_reps = point["gpu_replicas"]
        overrides = point["config_overrides"]

        # Create a config variant for this sweep point
        trial_config = base_config.model_copy(update=overrides)

        logger.info(f"Trial: cpus={cpu_count}, gpu_replicas={gpu_reps}")

        trial_dir = system_path / f".benchmark/cpus{cpu_count}_gpu{gpu_reps}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Symlink inputs into trial directory
        for src_file in [bench_mdp, structure, system_path / "topology.top"]:
            dst = trial_dir / src_file.name
            if not dst.exists():
                dst.symlink_to(src_file.resolve())

        # Rename benchmark.mdp symlink to md.mdp for the Production stage
        bench_link = trial_dir / "benchmark.mdp"
        md_link = trial_dir / prod_spec.mdp_file
        if bench_link.exists() and not md_link.exists():
            bench_link.rename(md_link)

        deffnm = "bench"
        log_file = trial_dir / f"{deffnm}.log"

        start_time = time.monotonic()

        try:
            hints = _extract_resource_hints(trial_config)

            with parsl_session(trial_config):
                grompp_app = get_grompp_app()
                mdrun_app = get_mdrun_app()

                # grompp
                grompp_future = grompp_app(
                    work_dir=str(trial_dir),
                    mdp_file=prod_spec.mdp_file,
                    structure_file=structure.name,
                    topology_file="topology.top",
                    tpr_output=f"{deffnm}.tpr",
                    maxwarn=prod_spec.maxwarn,
                )
                grompp_future.result()

                # mdrun
                mdrun_future = mdrun_app(
                    work_dir=str(trial_dir),
                    deffnm=deffnm,
                    ntasks=hints.ntasks,
                    disable_gpu=hints.disable_gpu,
                    gmx_binary=hints.gmx_binary,
                )
                mdrun_future.result()

            wall_seconds = time.monotonic() - start_time
            ns_per_day = parse_mdlog_performance(log_file)

            trials.append(
                TrialResult(
                    cpu_count=cpu_count,
                    gpu_replicas=gpu_reps,
                    ns_per_day=ns_per_day,
                    wall_seconds=wall_seconds,
                )
            )
            logger.info(f"  Result: {ns_per_day or 'N/A'} ns/day, {wall_seconds:.1f}s wall")

        except Exception as exc:
            wall_seconds = time.monotonic() - start_time
            logger.warning(f"  Trial failed: {exc}")
            trials.append(
                TrialResult(
                    cpu_count=cpu_count,
                    gpu_replicas=gpu_reps,
                    ns_per_day=None,
                    wall_seconds=wall_seconds,
                    error=str(exc),
                )
            )

    optimum = _select_optimum(trials, benchmark_config.selection)
    result = BenchmarkResult(
        system_path=str(system_path),
        trials=trials,
        optimum=optimum,
        selection_criterion=benchmark_config.selection,
    )

    # Persist results
    result_path = system_path / "benchmark_result.json"
    result.save(result_path)
    logger.info(f"Benchmark results saved to {result_path}")

    if optimum:
        logger.info(
            f"Optimum: cpus={optimum.cpu_count}, "
            f"gpu_replicas={optimum.gpu_replicas}, "
            f"{optimum.ns_per_day:.3f} ns/day"
        )

    return result
