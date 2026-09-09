# ABOUTME: Thread-safe progress tracking and shared Rich progress display loop
# ABOUTME: Provides StageProgressTracker, run_progress_loop, and display_stage_progress
"""Progress tracking and display for orchestration workflows."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
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
            return [self._results.get(h, {"hash": h, "status": "unknown"}) for h in self.sim_hashes]


def _get_block_status() -> str:
    """Query the active Parsl DFK for SLURM executor block statuses.

    Returns a Rich-markup string summarising block counts, or ``""``
    when no executor exposes status.
    """
    try:
        import parsl  # type: ignore[import-not-found]

        dfk = parsl.dfk()
        counts: dict[str, int] = {}
        for executor in dfk.executors.values():
            if not hasattr(executor, "status"):
                continue
            for _block_id, job_status in executor.status().items():
                state = str(job_status.state.name).lower()
                counts[state] = counts.get(state, 0) + 1
        if not counts:
            return ""
        parts = []
        if counts.get("running", 0):
            parts.append(f"[green]{counts['running']} running[/]")
        if counts.get("pending", 0):
            parts.append(f"[yellow]{counts['pending']} pending[/]")
        if counts.get("completed", 0):
            parts.append(f"[dim]{counts['completed']} completed[/]")
        if counts.get("failed", 0):
            parts.append(f"[red]{counts['failed']} failed[/]")
        for state, count in counts.items():
            if state not in ("running", "pending", "completed", "failed"):
                parts.append(f"[dim]{count} {state}[/]")
        return " · ".join(parts)
    except Exception:
        return ""


def _make_progress():
    """Create a Rich Progress bar with the standard orchestration column layout."""
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    return Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[succeeded]} ✓[/]"),
        TextColumn("[red]{task.fields[failed]} ✗[/]"),
        TextColumn("[yellow]{task.fields[running]} ●[/]"),
        console=Console(),
        transient=False,
    )


def run_progress_loop(
    progress,
    *,
    update: Callable[[], bool],
    render_extras: Callable[[], list] = lambda: [],
    poll_interval: float = 2.0,
) -> None:
    """Run a Live poll loop around a Progress bar until *update* returns ``True``.

    Handles the SLURM block-status line and ``KeyboardInterrupt`` uniformly.
    """
    from rich.console import Group
    from rich.live import Live
    from rich.text import Text

    console = progress.console

    def _render():
        parts: list = [progress]
        block_info = _get_block_status()
        if block_info:
            parts.append(Text.from_markup(f"  ▸ SLURM: {block_info}"))
        parts.extend(render_extras())
        return Group(*parts)

    try:
        with Live(_render(), console=console, refresh_per_second=2) as live:
            while True:
                done = update()
                live.update(_render())
                if done:
                    break
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted[/]")
        raise


def display_stage_progress(
    tracker: StageProgressTracker,
    *,
    poll_interval: float = 2.0,
) -> None:
    """Poll *tracker* and render Rich progress bars until all simulations finish."""
    total, progress = len(tracker.sim_hashes), _make_progress()
    max_len = max(len(s) for s in tracker.stages)
    task_ids = {
        stage: progress.add_task(
            f"⚒ {stage:<{max_len}}",
            total=total,
            succeeded=0,
            failed=0,
            running=0,
        )
        for stage in tracker.stages
    }

    def _update():
        snap = tracker.snapshot()
        for stage, tid in task_ids.items():
            c = snap[stage]
            progress.update(
                tid,
                completed=c["succeeded"] + c["failed"] + c["skipped"],
                succeeded=c["succeeded"],
                failed=c["failed"],
                running=c["running"],
            )
        return tracker.all_done()

    run_progress_loop(progress, update=_update, poll_interval=poll_interval)
