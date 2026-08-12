# ABOUTME: Thread-safe progress tracking for per-stage simulation monitoring
# ABOUTME: Provides StageProgressTracker and Rich progress display for EM/NVT/NPT/Production stages
"""Per-stage progress tracking for simulation orchestration.

Provides :class:`StageProgressTracker`, a thread-safe tracker that worker
threads report into, and :func:`display_stage_progress`, a Rich-based
display that polls the tracker and renders one progress bar per simulation
stage (EM, NVT, NPT, Production).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class SimState(Enum):
    """State of a single simulation in a single stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


_TERMINAL = frozenset({SimState.SUCCEEDED, SimState.FAILED, SimState.SKIPPED})


@dataclass
class StageProgressTracker:
    """Thread-safe tracker for per-stage, per-simulation progress.

    Worker threads call :meth:`mark_running`, :meth:`mark_succeeded`,
    :meth:`mark_failed`, :meth:`mark_skipped`.  The main thread calls
    :meth:`snapshot` for counts and :meth:`all_done` for termination.

    Parameters
    ----------
    stages : list[str]
        Ordered stage names (e.g. ``["EM", "NVT", "NPT", "Production"]``).
    sim_hashes : list[str]
        Simulation directory hashes, one per simulation.

    """

    stages: list[str]
    sim_hashes: list[str]

    _state: dict[str, dict[str, SimState]] = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _results: dict[str, dict] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._state = {
            stage: {h: SimState.PENDING for h in self.sim_hashes} for stage in self.stages
        }
        self._lock = threading.Lock()
        self._results = {}

    def mark_running(self, stage: str, sim_hash: str) -> None:
        with self._lock:
            self._state[stage][sim_hash] = SimState.RUNNING

    def mark_succeeded(self, stage: str, sim_hash: str) -> None:
        with self._lock:
            self._state[stage][sim_hash] = SimState.SUCCEEDED

    def mark_failed(self, stage: str, sim_hash: str) -> None:
        with self._lock:
            self._state[stage][sim_hash] = SimState.FAILED

    def mark_skipped(self, stage: str, sim_hash: str) -> None:
        with self._lock:
            self._state[stage][sim_hash] = SimState.SKIPPED

    def store_result(self, sim_hash: str, result: dict) -> None:
        """Store the final result dict for a simulation."""
        with self._lock:
            self._results[sim_hash] = result

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return per-stage counts.

        Returns
        -------
        dict[str, dict[str, int]]
            ``{stage: {succeeded: N, failed: N, running: N, pending: N, skipped: N}}``.
        """
        with self._lock:
            out: dict[str, dict[str, int]] = {}
            for stage in self.stages:
                counts = {s.value: 0 for s in SimState}
                for h in self.sim_hashes:
                    counts[self._state[stage][h].value] += 1
                out[stage] = counts
            return out

    def all_done(self) -> bool:
        """Return ``True`` when every simulation's last stage is terminal."""
        with self._lock:
            last_stage = self.stages[-1]
            return all(self._state[last_stage][h] in _TERMINAL for h in self.sim_hashes)

    def collect_results(self) -> list[dict]:
        """Return stored result dicts in ``sim_hashes`` order."""
        with self._lock:
            return [
                self._results.get(h, {"hash": h, "status": "unknown"}) for h in self.sim_hashes
            ]


def display_stage_progress(
    tracker: StageProgressTracker,
    *,
    poll_interval: float = 2.0,
) -> None:
    """Poll the tracker and render Rich progress bars until all simulations finish.

    Runs on the main thread.  Blocks until :meth:`StageProgressTracker.all_done`
    returns ``True``.

    Parameters
    ----------
    tracker : StageProgressTracker
        Shared tracker updated by worker threads.
    poll_interval : float
        Seconds between display refreshes.

    """
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
    from rich.text import Text

    from .build import _get_block_status

    console = Console()
    total = len(tracker.sim_hashes)

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[succeeded]} ✓[/]"),
        TextColumn("[red]{task.fields[failed]} ✗[/]"),
        TextColumn("[yellow]{task.fields[running]} ●[/]"),
        console=console,
        transient=False,
    )

    task_ids = {}
    max_len = max(len(s) for s in tracker.stages)
    for stage in tracker.stages:
        tid = progress.add_task(
            f"⚒ {stage:<{max_len}}",
            total=total,
            succeeded=0,
            failed=0,
            running=0,
        )
        task_ids[stage] = tid

    def _render():
        snap = tracker.snapshot()
        for stage in tracker.stages:
            counts = snap[stage]
            done = counts["succeeded"] + counts["failed"] + counts["skipped"]
            progress.update(
                task_ids[stage],
                completed=done,
                succeeded=counts["succeeded"],
                failed=counts["failed"],
                running=counts["running"],
            )
        parts: list = [progress]
        block_info = _get_block_status()
        if block_info:
            parts.append(Text.from_markup(f"  ▸ SLURM: {block_info}"))
        return Group(*parts)

    try:
        with Live(_render(), console=console, refresh_per_second=2) as live:
            while not tracker.all_done():
                time.sleep(poll_interval)
                live.update(_render())
            live.update(_render())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted — cancelling SLURM jobs...[/]")
        raise
