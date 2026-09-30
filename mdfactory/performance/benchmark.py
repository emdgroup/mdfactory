# ABOUTME: Scalability benchmark sweep for optimal resource discovery
# ABOUTME: Runs short GROMACS trials across CPU/GPU configs, parses ns/day, selects optimum
"""Scalability benchmark sweep.

Discovers optimal resource configuration for a given system size by running
short production trials across a sweep of executor configs.  The whole study
runs inside a single :func:`~mdfactory.orchestration.session.parsl_session`;
each sweep point submits a short mdrun (a single simulation, scaled across
increasing GPU counts via ``srun``) via the standard Parsl app factories and
reads throughput (ns/day) from the worker-written performance sidecar.
"""

from __future__ import annotations

import fcntl
import os
import re
import socket
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger
from pydantic import BaseModel, field_validator

from mdfactory.orchestration.apps import MDRUN_PERF_MARKER

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


#: Regex for the marker ``MDFACTORY_NS_PER_DAY=<value>`` echoed by mdrun.
_PERF_MARKER_RE = re.compile(re.escape(MDRUN_PERF_MARKER) + r"([\d.]+)")


def parse_performance_marker(stdout: str) -> float | None:
    """Extract ns/day from the mdrun app's stdout marker.

    The mdrun bash script greps the performance value out of its own log and
    echoes ``MDFACTORY_NS_PER_DAY=<value>`` on stdout.  Because that script
    runs on the node that wrote the log, this avoids the driver-side shared
    filesystem staleness that can otherwise return a *previous* run's value
    (GROMACS renames old outputs to ``#<name>.<n>#``, but a stale filesystem
    view can still resolve ``bench.log`` to the pre-rename inode).

    Parameters
    ----------
    stdout : str
        Standard output returned by the mdrun Parsl app.

    Returns
    -------
    float or None
        Throughput in ns/day, or ``None`` when the marker is absent or its
        value is empty.

    """
    if not stdout:
        return None
    matches = list(_PERF_MARKER_RE.finditer(stdout))
    if not matches:
        return None
    return float(matches[-1].group(1))


def read_performance_file(perf_path: Path) -> float | None:
    """Read ns/day from the mdrun performance sidecar file.

    The mdrun bash script writes the performance marker to a caller-named
    file (see ``performance_file``).  Reading that file here is safe as long
    as the name is unique per trial: a path the driver has never opened cannot
    be served from a stale filesystem cache entry.

    Parameters
    ----------
    perf_path : Path
        Sidecar file written by the mdrun script.

    Returns
    -------
    float or None
        Throughput in ns/day, or ``None`` when the file is missing or holds
        no marker value.

    """
    try:
        text = perf_path.read_text()
    except FileNotFoundError:
        return None
    return parse_performance_marker(text)


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
        """Throughput per core (ns/day / cpu_count)."""
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
        GPU counts to sweep (e.g. ``[1, 2, 4]``): a point with ``R`` runs ONE
        simulation across ``R`` GPUs (one MPI rank each, ``srun -n R``, no
        MPS).  Cores are tied 1:1 to GPUs — each point uses ``R`` cores, one
        per rank, 1 thread each (pure MPI, no OpenMP hybrid), so threads/rank
        is constant across the sweep.  ``cpu_counts`` is ignored for a GPU
        sweep.  The recorded ``ns_per_day`` is that simulation's throughput.
        An empty list runs a CPU-only sweep.
    duration_ps : float
        Benchmark trial duration in picoseconds.
    selection : str
        Optimum selection: ``"ns_per_day"`` for raw throughput or
        ``"efficiency"`` for throughput per core.
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
        if any(c < 1 for c in v):
            # A 0/negative core count would make ntasks floor to the
            # auto-detect sentinel, mislabeling the trial (finding 1).
            raise ValueError("cpu_counts values must be >= 1")
        return v

    @field_validator("gpu_replicas")
    @classmethod
    def _gpu_replicas_positive(cls, v: list[int]) -> list[int]:
        if any(g < 1 for g in v):
            raise ValueError("gpu_replicas values must be >= 1")
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

    # Reduce output frequency to minimize I/O overhead during benchmark.
    # Skip keys already set to "0" (intentionally disabled outputs).
    io_interval = str(max(nsteps, 1000))
    for key in ("nstxout", "nstvout", "nstfout", "nstxout_compressed", "nstenergy", "nstlog"):
        current = get_mdp_value(parsed, key)
        if current is not None and current != "0":
            parsed = modify_mdp_value(parsed, key, io_interval)

    # Apply any user-specified overrides
    if extra_overrides:
        for key, value in extra_overrides.items():
            norm_key = key.lower().replace("-", "_")
            if get_mdp_value(parsed, norm_key) is None:
                # modify_mdp_value only replaces existing lines; an override
                # naming a key absent from the source MDP would be silently
                # dropped (finding 4).  Append it so the override actually
                # takes effect, matching the documented behavior.
                parsed.append((f"{key} = {value}", norm_key, value))
                logger.debug(f"Benchmark MDP override appended: {key} = {value}")
            else:
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

    A GPU sweep ties cores 1:1 to GPUs — each point runs one MPI rank per GPU
    with a single thread (pure MPI, no OpenMP hybrid), so threads/rank is
    constant across the sweep; ``cpu_counts`` applies only to a CPU-only sweep.

    Each descriptor contains the fields needed to create a per-trial
    executor config variant and to record the trial result.

    Returns
    -------
    list[dict]
        Each dict has ``cpu_count``, ``gpu_replicas``, and ``config_overrides``.

    """
    points = []

    if benchmark_config.gpu_replicas:
        # Cores = GPUs: one core and one MPI rank per GPU, 1 thread each.
        # extract_resource_hints then yields ntasks = gpu_count // gpu_count = 1
        # (pure MPI), so threads/rank never varies across the sweep.
        for gpu_count in benchmark_config.gpu_replicas:
            points.append(
                {
                    "cpu_count": gpu_count,  # cores = GPUs (1 core per GPU)
                    "gpu_replicas": gpu_count,
                    "config_overrides": {
                        "cpus_per_node": gpu_count,
                        "max_workers_per_node": gpu_count,
                    },
                }
            )
    else:
        for cpu_count in benchmark_config.cpu_counts:
            points.append(
                {
                    "cpu_count": cpu_count,
                    "gpu_replicas": 0,
                    "config_overrides": {
                        "cpus_per_node": cpu_count,
                        # Normalize to one worker so extract_resource_hints
                        # yields ntasks == cpu_count regardless of the base
                        # config's max_workers_per_node.
                        "max_workers_per_node": 1,
                    },
                }
            )

    return points


@contextmanager
def _sweep_lock(system_path: Path):
    """Serialize benchmark sweeps for one system via an exclusive file lock.

    Two concurrent sweeps would share the same ``.benchmark/cpus*`` trial
    directories and race on ``bench.log``/``bench.tpr``, yielding meaningless
    results.  An ``flock`` on ``.benchmark/.lock`` makes a second sweep fail
    fast, and the lock is released automatically if the holder dies.

    Parameters
    ----------
    system_path : Path
        Prepared simulation directory.

    Yields
    ------
    None
        While the lock is held.

    Raises
    ------
    RuntimeError
        If another sweep currently holds the lock.

    """
    lock_dir = system_path / ".benchmark"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".lock"
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"Another benchmark sweep is already running for {system_path} "
            f"(lock held on {lock_path}). Wait for it to finish."
        ) from exc
    try:
        handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _scale_gres(gres: str, count: int) -> str:
    """Return ``gres`` with its trailing device count set to ``count``.

    Handles the common SLURM gres spellings — ``gpu``, ``gpu:1``,
    ``gpu:l40s``, ``gpu:l40s:1`` — by replacing an existing trailing numeric
    count or appending one.  Used so a GPU sweep can request ``N`` GPUs on the
    single sweep allocation (one simulation spanning ``N`` GPUs).

    Parameters
    ----------
    gres : str
        Base gres string from the executor config (e.g. ``"gpu:l40s:1"``).
    count : int
        Desired device count.

    Returns
    -------
    str
        gres string with the device count set to ``count``.

    """
    parts = gres.split(":")
    if parts[-1].isdigit():
        parts[-1] = str(count)
    else:
        parts.append(str(count))
    return ":".join(parts)


def run_benchmark_sweep(
    system_path: Path,
    base_config: "ExecutorConfig",
    benchmark_config: BenchmarkConfig | None = None,
    *,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Run a scalability benchmark sweep for a simulation system.

    Opens **one** executor allocation sized to the largest requested CPU
    count, then runs a short mdrun trial for every sweep point inside it,
    varying only the mdrun thread count.  Keeping the whole study on a single
    allocation/node removes job-placement and queueing variance from the
    scaling curve.  A GPU point with ``R`` runs one simulation across ``R``
    GPUs (``gres`` scales to the max; ``srun -n R`` gives each rank one GPU).
    Reads ns/day from each trial's performance sidecar and selects the optimum.

    Parameters
    ----------
    system_path : Path
        Path to a prepared simulation directory (must contain topology,
        structure, and MDP files).
    base_config : ExecutorConfig
        Base executor configuration.  The allocation is derived from it by
        overriding ``cpus_per_node`` with the maximum sweep value, forcing
        ``max_workers_per_node=1``, and — for a GPU sweep — scaling ``gres``
        to the largest GPU count.  When the config supports it (SLURM), the
        sweep allocation is marked ``exclusive`` so neighbour jobs cannot skew
        the measurement.
    benchmark_config : BenchmarkConfig, optional
        Sweep parameters.  Defaults to ``BenchmarkConfig()`` with
        CPU counts ``[1, 2, 4, 8]``.
    dry_run : bool
        If ``True``, log what would be run without submitting work.

    Returns
    -------
    BenchmarkResult
        All trial results with the selected optimum.  For a GPU point with
        ``R`` GPUs, ``TrialResult.ns_per_day`` is that single simulation's
        throughput (one run spanning ``R`` ranks).

    Raises
    ------
    ValueError
        If the executor targets more than one node (trials are single-rank).

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

    # A GPU replica sweep must actually have a GPU, or every trial would run
    # CPU-only while labeled as a GPU point and still feed _select_optimum
    # (finding 5).  Validated here (not before dry-run) so a cluster-free
    # preview still works — dry-run must run without a GPU/SLURM config.
    if benchmark_config.gpu_replicas:
        from mdfactory.orchestration.stages import extract_resource_hints

        if extract_resource_hints(base_config).disable_gpu:
            gres = getattr(base_config, "gres", None)
            raise ValueError(
                f"gpu_replicas={benchmark_config.gpu_replicas} requested but the "
                f"executor config has no GPU resource (gres={gres!r}); trials would "
                "run CPU-only while labeled as GPU points. Add a gres field "
                "(e.g. 'gpu:l40s:1') to the SLURM config."
            )

    # The whole study runs inside ONE allocation sized to the largest requested
    # point, so trial results differ only by mdrun thread count — not by job
    # placement or queueing.  Validate it before doing any work.
    max_cpus = max(point["cpu_count"] for point in sweep_points)
    max_replicas = max((point["gpu_replicas"] for point in sweep_points), default=0)
    allocation_update = {
        "cpus_per_node": max_cpus,
        # One Parsl worker runs the (possibly multi-rank) simulation at a time;
        # points run sequentially, and a multi-GPU point's ranks are spawned by
        # srun inside that worker's task.
        "max_workers_per_node": 1,
        # Hard-guarantee a SINGLE SLURM allocation for the whole sweep (like
        # exclusive/gres, overridden only on this copy so `simulate` keeps the
        # yaml's multi-block parallelism).  The sequential design already keeps
        # Parsl to one block; pinning max_blocks=1 makes "one allocation" a
        # guarantee rather than an emergent property, so the sweep can never
        # span multiple jobs and mix placement/queueing across nodes.
        "max_blocks": 1,
    }
    # A benchmark must measure the allocation, not our neighbours.  On a
    # shared node other jobs steal cores, which swung the 8-core point ~37%
    # between two runs of the same sweep.  Opt the *sweep* into an exclusive
    # whole-node allocation (``#SBATCH --exclusive``); production runs keep
    # the default shared behaviour.  Only SLURM configs carry the field.
    if "exclusive" in type(base_config).model_fields:
        allocation_update["exclusive"] = True
    # A GPU sweep requests the largest GPU count so the single simulation can
    # span up to N GPUs.  available_accelerators=0 keeps per-worker pinning off
    # so the one worker sees all N GPUs and `srun --gpus-per-task=1` hands one
    # to each rank.  CPU-only sweeps leave gres/available_accelerators alone.
    if max_replicas > 0 and getattr(base_config, "gres", None):
        allocation_update["gres"] = _scale_gres(base_config.gres, max_replicas)
        allocation_update["available_accelerators"] = 0
    allocation_config = base_config.model_copy(update=allocation_update)
    if getattr(allocation_config, "nodes", 1) > 1:
        raise ValueError(
            "Benchmark trials run as a single MPI rank; use a single-node "
            f"allocation (got nodes={allocation_config.nodes})."
        )

    logger.info(
        f"Single allocation: cpus={max_cpus}, "
        f"max_workers_per_node={allocation_config.max_workers_per_node}"
        + (f", gres={allocation_config.gres}" if getattr(allocation_config, "gres", None) else "")
        + (
            f", exclusive={allocation_config.exclusive}"
            if hasattr(allocation_config, "exclusive")
            else ""
        )
    )

    # Run every sweep point inside the single allocation computed above.
    import time

    from mdfactory.orchestration.apps import get_grompp_app, get_mdrun_app
    from mdfactory.orchestration.session import parsl_session
    from mdfactory.orchestration.stages import STAGE_BY_NAME, extract_resource_hints
    from mdfactory.orchestration.trajectory import find_structure_file

    trials: list[TrialResult] = []

    # The lock is taken before touching any shared state: the benchmark MDP
    # generated below is exposed to every trial as ``md.mdp``, so a second
    # sweep regenerating it would silently mutate this sweep's trials.  The
    # MDP write must therefore happen while the lock is held.
    with _sweep_lock(system_path):
        # Prepare benchmark MDP from the production MDP
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

        structure = find_structure_file(system_path)
        if not structure:
            raise FileNotFoundError(f"No structure file found in {system_path}")

        with parsl_session(allocation_config):
            grompp_app = get_grompp_app()
            mdrun_app = get_mdrun_app()

            for point in sweep_points:
                cpu_count = point["cpu_count"]
                gpu_reps = point["gpu_replicas"]

                # Derive mdrun thread hints from the per-point overrides; the
                # allocation itself stays at the maximum requested size.
                trial_config = base_config.model_copy(update=point["config_overrides"])
                hints = extract_resource_hints(trial_config)

                # A GPU point with R GPUs runs ONE simulation across R MPI
                # ranks (srun -n R, one GPU each); CPU-only runs one rank.
                # hints.ntasks is threads per rank = cores // ranks
                # (cores tied 1:1 to GPUs → 1 thread per rank, pure MPI).
                gpus = max(1, gpu_reps)

                logger.info(
                    f"Trial: cpus={cpu_count}, gpu_replicas={gpu_reps} "
                    f"({gpus} GPU(s), single simulation)"
                )

                base_deffnm = "bench"
                trial_dir = system_path / f".benchmark/cpus{cpu_count}_gpu{gpu_reps}"

                start_time = time.monotonic()

                try:
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    # Symlink all top-level inputs into the trial directory
                    # (replacing stale symlinks).  GROMACS resolves ``#include``
                    # paths relative to the topology file, so every companion
                    # ``.itp`` file and force-field directory must be reachable
                    # from the trial dir too — not just ``topology.top`` itself.
                    for src_file in system_path.iterdir():
                        if src_file.name == ".benchmark":
                            continue
                        dst = trial_dir / src_file.name
                        if dst.is_symlink():
                            dst.unlink()
                        if not dst.exists():
                            dst.symlink_to(
                                src_file.resolve(), target_is_directory=src_file.is_dir()
                            )

                    # Point the Production stage's expected MDP name at the benchmark MDP
                    bench_link = trial_dir / "benchmark.mdp"
                    md_link = trial_dir / prod_spec.mdp_file
                    if bench_link.exists() or bench_link.is_symlink():
                        if md_link.is_symlink():
                            md_link.unlink()
                        if not md_link.exists():
                            bench_link.rename(md_link)

                    # grompp — once per point; every rank shares the TPR.
                    grompp_future = grompp_app(
                        work_dir=str(trial_dir),
                        mdp_file=prod_spec.mdp_file,
                        gro_file=structure.name,
                        top_file="topology.top",
                        tpr_file=f"{base_deffnm}.tpr",
                        maxwarn=prod_spec.maxwarn,
                        inputs=[],
                    )
                    grompp_future.result()

                    # One mdrun task: for R>1 the script launches
                    # `srun -n R --gpus-per-task=1`, so the single simulation
                    # spans R GPUs (one rank each).  The sidecar name is unique
                    # so the driver never reads a stale filesystem cache entry.
                    perf_name = f"{base_deffnm}.{uuid.uuid4().hex}.perf"
                    perf_path = trial_dir / perf_name
                    mdrun_future = mdrun_app(
                        work_dir=str(trial_dir),
                        deffnm=base_deffnm,
                        ntasks=hints.ntasks,
                        gpus=gpus,
                        disable_gpu=hints.disable_gpu,
                        gmx_binary=hints.gmx_binary,
                        performance_file=perf_name,
                        inputs=[grompp_future],
                    )
                    # The mdrun script extracts ns/day on the node that wrote
                    # the log and writes it to the perf sidecar.  Parsing the
                    # log from the driver is unsafe: the shared filesystem can
                    # serve a stale pre-backup inode, which previously
                    # reported a previous trial's throughput.
                    mdrun_future.result()

                    wall_seconds = time.monotonic() - start_time
                    ns_per_day = read_performance_file(perf_path)

                    error = None
                    if ns_per_day is None:
                        error = f"no performance data in {perf_path}"
                        logger.warning(f"  {error}")

                    trials.append(
                        TrialResult(
                            cpu_count=cpu_count,
                            gpu_replicas=gpu_reps,
                            ns_per_day=ns_per_day,
                            wall_seconds=wall_seconds,
                            error=error,
                        )
                    )
                    logger.info(f"  Result: {ns_per_day or 'N/A'} ns/day, {wall_seconds:.1f}s wall")

                except Exception as exc:
                    wall_seconds = time.monotonic() - start_time
                    logger.opt(exception=True).warning(f"  Trial failed: {exc}")
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
