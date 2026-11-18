# ABOUTME: CPU scalability benchmark sweep for the 75% parallel-efficiency rule
# ABOUTME: Runs short GROMACS trials per core count and picks the highest justified count
"""CPU scalability benchmark.

Finds the highest core count still justified by the 75% parallel-efficiency
rule of thumb.  Short production trials run inside **one** exclusive
allocation, varying only the mdrun thread count, so job placement and queueing
do not enter the scaling curve.  A point is justified while its parallel
efficiency — speedup divided by core count, both relative to the smallest
swept count — stays at or above ``efficiency_threshold``.

GPU scaling is out of scope: every trial is CPU-only.
"""

from __future__ import annotations

import fcntl
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, field_validator

from mdfactory.orchestration.apps import MDRUN_PERF_MARKER

if TYPE_CHECKING:
    from mdfactory.orchestration.config import ExecutorConfig
    from mdfactory.orchestration.stages import StageSpec

#: Core counts swept by default: a 1-core reference plus doublings.
DEFAULT_CPU_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

#: Efficiency floor of the rule of thumb: 75% of linear speedup.
DEFAULT_EFFICIENCY = 0.75

#: Trial directory (relative to the system dir) and its sweep lock.
TRIAL_ROOT = ".benchmark"
RESULT_FILE = "benchmark_result.json"
BENCHMARK_MDP = "benchmark.mdp"

#: Regex for the marker ``MDFACTORY_NS_PER_DAY=<value>`` written by mdrun.
_PERF_MARKER_RE = re.compile(re.escape(MDRUN_PERF_MARKER) + r"([\d.]+)")


def read_performance_file(perf_path: Path) -> float | None:
    """Read ns/day from the sidecar the mdrun script writes on the compute node.

    The script greps the ``Performance:`` line out of its own log and writes
    ``MDFACTORY_NS_PER_DAY=<value>`` to a caller-named file.  A driver-side log
    read can be served from a stale pre-backup inode and report a *previous*
    trial's value; a path the driver has never opened cannot.

    Parameters
    ----------
    perf_path : Path
        Sidecar file written by the mdrun script.

    Returns
    -------
    float or None
        Throughput in ns/day, or ``None`` when the file or value is missing.

    """
    try:
        text = perf_path.read_text()
    except FileNotFoundError:
        return None
    values = _PERF_MARKER_RE.findall(text)
    return float(values[-1]) if values else None


class TrialResult(BaseModel, frozen=True):
    """One short mdrun at a fixed core count."""

    cpu_count: int
    ns_per_day: float | None
    wall_seconds: float | None
    error: str | None = None


class BenchmarkConfig(BaseModel, frozen=True):
    """Sweep parameters for a CPU scalability benchmark.

    Parameters
    ----------
    cpu_counts : list[int]
        Thread counts to sweep.  The smallest successful one is the efficiency
        baseline, so keep ``1`` in the list for the standard rule of thumb.
    duration_ps : float
        Duration of each trial in picoseconds.
    efficiency_threshold : float
        Fraction of linear speedup (0, 1] a core count must reach to count as
        justified.  Defaults to the 75% rule of thumb.
    mdp_overrides : dict[str, str]
        Extra MDP overrides applied on top of the short-trial MDP.

    """

    cpu_counts: list[int] = list(DEFAULT_CPU_COUNTS)
    duration_ps: float = 100.0
    efficiency_threshold: float = DEFAULT_EFFICIENCY
    mdp_overrides: dict[str, str] = {}

    @field_validator("cpu_counts")
    @classmethod
    def _positive_counts(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("cpu_counts must not be empty")
        if any(c < 1 for c in v):
            # 0 would fall back to node-wide auto-detection under a false label.
            raise ValueError("cpu_counts values must be >= 1")
        return v

    @field_validator("duration_ps")
    @classmethod
    def _positive_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duration_ps must be positive")
        return v

    @field_validator("efficiency_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("efficiency_threshold must be in (0, 1]")
        return v


class BenchmarkResult(BaseModel, frozen=True):
    """Sweep results and the highest core count they justify."""

    system_path: str
    trials: list[TrialResult]
    efficiency_threshold: float = DEFAULT_EFFICIENCY

    def efficiency(self, trial: TrialResult) -> float | None:
        """Parallel efficiency of ``trial`` relative to the smallest swept count.

        ``1.0`` is perfect linear scaling; ``0.75`` means the trial reaches 75%
        of the throughput a perfectly scaling machine would deliver per core.

        """
        base = self.baseline
        if base is None or not base.ns_per_day or trial.ns_per_day is None:
            return None
        speedup = trial.ns_per_day / base.ns_per_day
        return speedup * base.cpu_count / trial.cpu_count

    @property
    def baseline(self) -> TrialResult | None:
        """Smallest successful trial — the reference for parallel efficiency."""
        successful = [t for t in self.trials if t.ns_per_day is not None]
        return min(successful, key=lambda t: t.cpu_count, default=None)

    @property
    def max_justified(self) -> TrialResult | None:
        """Highest core count whose efficiency still meets the threshold."""
        justified = [
            t
            for t in self.trials
            if (eff := self.efficiency(t)) is not None and eff >= self.efficiency_threshold
        ]
        return max(justified, key=lambda t: t.cpu_count, default=None)

    def save(self, path: Path) -> None:
        """Write the result as a JSON sidecar."""
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResult":
        """Load a result from a JSON sidecar."""
        return cls.model_validate_json(path.read_text())


def _generate_benchmark_mdp(
    source_mdp: Path,
    output_mdp: Path,
    duration_ps: float,
    dt: float | None = None,
    extra_overrides: dict[str, str] | None = None,
) -> None:
    """Write a short-trial MDP derived from the production MDP.

    Overrides ``nsteps`` for ``duration_ps`` and stretches output intervals to
    the trial length so I/O does not distort the throughput measurement.

    Parameters
    ----------
    source_mdp : Path
        Production MDP to derive from.
    output_mdp : Path
        Benchmark MDP to write.
    duration_ps : float
        Target trial duration in picoseconds.
    dt : float, optional
        Timestep override in ps.  Reads ``dt`` from the source MDP when None.
    extra_overrides : dict, optional
        Additional key-value overrides applied last.

    """
    from mdfactory.orchestration.mdp import (
        get_mdp_value,
        modify_mdp_value,
        parse_mdp,
        write_mdp,
    )

    parsed = parse_mdp(source_mdp)

    if dt is None:
        dt_str = get_mdp_value(parsed, "dt")
        dt = float(dt_str) if dt_str else 0.002  # GROMACS default

    nsteps = int(duration_ps / dt)
    parsed = modify_mdp_value(parsed, "nsteps", str(nsteps))

    # Skip keys already "0": those outputs are intentionally disabled.
    io_interval = str(max(nsteps, 1000))
    for key in ("nstxout", "nstvout", "nstfout", "nstxout_compressed", "nstenergy", "nstlog"):
        current = get_mdp_value(parsed, key)
        if current is not None and current != "0":
            parsed = modify_mdp_value(parsed, key, io_interval)

    for key, value in (extra_overrides or {}).items():
        norm_key = key.lower().replace("-", "_")
        if get_mdp_value(parsed, norm_key) is None:
            # modify_mdp_value only replaces existing lines, so append instead
            # of silently dropping an override naming an absent key.
            parsed.append((f"{key} = {value}", norm_key, value))
        else:
            parsed = modify_mdp_value(parsed, norm_key, value)

    write_mdp(parsed, output_mdp)
    logger.debug(f"Benchmark MDP: {output_mdp} (nsteps={nsteps}, dt={dt})")


@contextmanager
def _sweep_lock(system_path: Path):
    """Serialize sweeps for one system via an exclusive file lock.

    Two concurrent sweeps would share the ``.benchmark/cpus*`` directories and
    the shared benchmark MDP.  The lock is released automatically if the
    holder dies.

    Raises
    ------
    RuntimeError
        If another sweep currently holds the lock.

    """
    lock_dir = system_path / TRIAL_ROOT
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


def _stage_trial_dir(system_path: Path, trial_dir: Path, mdp_file: str) -> None:
    """Link every system input into ``trial_dir``, with the short MDP as ``mdp_file``.

    GROMACS resolves ``#include`` relative to the topology file, so companion
    ``.itp`` files and force-field directories must be reachable from the trial
    directory too — not just ``topology.top``.  Stale symlinks from a previous
    sweep are replaced.

    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    for src in system_path.iterdir():
        if src.name == TRIAL_ROOT:
            continue
        dst = trial_dir / src.name
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())

    bench_link = trial_dir / BENCHMARK_MDP
    md_link = trial_dir / mdp_file
    if bench_link.exists() or bench_link.is_symlink():
        if md_link.is_symlink():
            md_link.unlink()
        if not md_link.exists():
            bench_link.rename(md_link)


def _run_trial(
    system_path: Path,
    cpu_count: int,
    spec: "StageSpec",
    structure: Path,
    grompp_app,
    mdrun_app,
    gmx_binary: str,
) -> TrialResult:
    """Run one short CPU-only mdrun at ``cpu_count`` threads and record ns/day."""
    trial_dir = system_path / TRIAL_ROOT / f"cpus{cpu_count}"
    # Unique per trial: a path the driver has never opened cannot be served
    # from a stale filesystem cache entry.
    perf_name = f"bench.{uuid.uuid4().hex}.perf"
    start = time.monotonic()
    ns_per_day: float | None = None
    error: str | None = None

    try:
        _stage_trial_dir(system_path, trial_dir, spec.mdp_file)
        grompp_future = grompp_app(
            work_dir=str(trial_dir),
            mdp_file=spec.mdp_file,
            gro_file=structure.name,
            top_file="topology.top",
            tpr_file="bench.tpr",
            maxwarn=spec.maxwarn,
            inputs=[],
        )
        grompp_future.result()
        mdrun_app(
            work_dir=str(trial_dir),
            deffnm="bench",
            ntasks=cpu_count,
            disable_gpu=True,
            gmx_binary=gmx_binary,
            performance_file=perf_name,
            inputs=[grompp_future],
        ).result()
        ns_per_day = read_performance_file(trial_dir / perf_name)
        if ns_per_day is None:
            error = f"no performance data in {perf_name}"
    except Exception as exc:
        logger.opt(exception=True).warning(f"  Trial cpus={cpu_count} failed: {exc}")
        error = str(exc)

    wall_seconds = time.monotonic() - start
    logger.info(
        f"  cpus={cpu_count}: {ns_per_day if ns_per_day else 'N/A'} ns/day, "
        f"{wall_seconds:.1f}s wall" + (f" ({error})" if error else "")
    )
    return TrialResult(
        cpu_count=cpu_count,
        ns_per_day=ns_per_day,
        wall_seconds=wall_seconds,
        error=error,
    )


def run_benchmark_sweep(
    system_path: Path,
    base_config: "ExecutorConfig",
    benchmark_config: BenchmarkConfig | None = None,
    *,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Sweep core counts for one system and report the highest justified count.

    Opens **one** allocation sized to the largest swept core count and forces it
    exclusive, so neighbours cannot skew the measurement, and to a single Parsl
    block and worker, so trials run sequentially on the same node.  Each trial
    is a short CPU-only mdrun whose only variable is the thread count.

    Parameters
    ----------
    system_path : Path
        Prepared simulation directory (topology, structure, and MDP files).
    base_config : ExecutorConfig
        Base executor config.  Only ``cpus_per_node``, ``max_workers_per_node``,
        ``max_blocks``, and (for SLURM) ``exclusive`` are overridden, on a copy.
    benchmark_config : BenchmarkConfig, optional
        Sweep parameters.  Defaults to :class:`BenchmarkConfig`.
    dry_run : bool
        Log the plan without submitting anything.

    Returns
    -------
    BenchmarkResult
        All trial results; ``max_justified`` holds the answer.

    Raises
    ------
    ValueError
        If the allocation would span more than one node.
    FileNotFoundError
        If the production MDP or the structure file is missing.

    """
    config = benchmark_config or BenchmarkConfig()
    system_path = Path(system_path)
    counts = sorted(config.cpu_counts)

    if dry_run:
        logger.info(f"[dry-run] CPU sweep for {system_path.name}: {counts}")
        return BenchmarkResult(
            system_path=str(system_path),
            trials=[TrialResult(cpu_count=c, ns_per_day=None, wall_seconds=None) for c in counts],
            efficiency_threshold=config.efficiency_threshold,
        )

    allocation_update = {
        "cpus_per_node": max(counts),
        # Trials are sequential: one worker runs one mdrun at a time, and one
        # Parsl block guarantees the whole sweep stays on a single allocation.
        "max_workers_per_node": 1,
        "max_blocks": 1,
    }
    if "exclusive" in type(base_config).model_fields:
        # A shared node lets neighbours steal cores, which swung one 8-core
        # point ~37% between identical runs.
        allocation_update["exclusive"] = True
    allocation_config = base_config.model_copy(update=allocation_update)
    if getattr(allocation_config, "nodes", 1) > 1:
        raise ValueError(
            "A benchmark trial is one mdrun on one node; use a single-node "
            f"allocation (got nodes={allocation_config.nodes})."
        )

    from mdfactory.orchestration.apps import get_grompp_app, get_mdrun_app
    from mdfactory.orchestration.session import parsl_session
    from mdfactory.orchestration.stages import STAGE_BY_NAME
    from mdfactory.orchestration.trajectory import find_structure_file

    logger.info(
        f"CPU sweep for {system_path.name}: {counts} "
        f"(allocation cpus={allocation_config.cpus_per_node}, efficiency >= "
        f"{config.efficiency_threshold:.0%})"
    )

    # The lock comes before the shared MDP write: every trial reads
    # ``benchmark.mdp``, so a losing sweep regenerating it would mutate the
    # winner's trials.
    with _sweep_lock(system_path):
        spec = STAGE_BY_NAME["Production"]
        source_mdp = system_path / spec.mdp_file
        if not source_mdp.exists():
            raise FileNotFoundError(
                f"Production MDP not found: {source_mdp}. "
                "Benchmark requires a prepared simulation directory."
            )
        _generate_benchmark_mdp(
            source_mdp,
            system_path / BENCHMARK_MDP,
            duration_ps=config.duration_ps,
            extra_overrides=config.mdp_overrides,
        )

        structure = find_structure_file(system_path)
        if not structure:
            raise FileNotFoundError(f"No structure file found in {system_path}")

        gmx_binary = getattr(allocation_config, "gmx_binary", "auto")
        with parsl_session(allocation_config):
            grompp_app = get_grompp_app()
            mdrun_app = get_mdrun_app()
            trials = [
                _run_trial(system_path, n, spec, structure, grompp_app, mdrun_app, gmx_binary)
                for n in counts
            ]

    result = BenchmarkResult(
        system_path=str(system_path),
        trials=trials,
        efficiency_threshold=config.efficiency_threshold,
    )
    result.save(system_path / RESULT_FILE)
    logger.info(f"Benchmark results saved to {system_path / RESULT_FILE}")
    return result
