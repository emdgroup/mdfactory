# ABOUTME: Build orchestration dispatch via Parsl
# ABOUTME: Submits parallel builds and provides dry-run preview
"""Build orchestration via Parsl."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from mdfactory.models.input import BuildInput

from .apps import get_build_app
from .session import (
    _get_slurm_job_ids,
    _scancel_jobs,
    _shutdown_parsl,
    parsl_session,
)

if TYPE_CHECKING:
    from .config import ExecutorConfig

# Re-exported for backward compatibility — the canonical home is ``session``.
__all__ = [
    "build_systems",
    "parsl_session",
    "_shutdown_parsl",
    "_get_slurm_job_ids",
    "_scancel_jobs",
]


def build_systems(
    build_inputs: list,
    config: "ExecutorConfig",
    *,
    output_dir: Path | None = None,
    wait: bool = True,
    dry_run: bool = False,
) -> list:
    """Submit parallel builds via Parsl (or preview with dry_run).

    Parameters
    ----------
    build_inputs : list
        List of BuildInput models or dicts.
    config : ExecutorConfig
        Executor configuration for Parsl.
    output_dir : Path, optional
        Base output directory for builds. Defaults to current directory.
    wait : bool, optional
        Whether to wait for all futures to complete. Default True.
        If False, returns raw AppFutures and the caller is responsible
        for calling ``parsl.clear()`` after all futures complete.
    dry_run : bool, optional
        If True, log what would be built and return descriptions without
        loading Parsl or submitting any work. Default False.

    Returns
    -------
    list
        If dry_run=True, list of description dicts.
        If wait=True, list of result dicts.
        If wait=False, list of AppFutures.

    """
    output_dir = Path(output_dir) if output_dir else Path.cwd()

    # Resolve all inputs upfront
    resolved: list[tuple[BuildInput, dict]] = []
    for inp in build_inputs:
        if isinstance(inp, BuildInput):
            model = inp
            input_dict = inp.model_dump()
        elif isinstance(inp, dict):
            model = BuildInput(**inp)
            input_dict = inp.copy()
        else:
            raise TypeError(f"Expected BuildInput or dict, got {type(inp)}")
        input_dict["_build_dir"] = str(output_dir / model.hash)
        resolved.append((model, input_dict))

    # Dry-run: log plan and return without loading Parsl
    if dry_run:
        descriptions = []
        for model, _ in resolved:
            desc = {
                "hash": model.hash,
                "simulation_type": model.simulation_type,
                "parametrization": model.parametrization,
                "engine": model.engine,
                "output_directory": str(output_dir / model.hash),
            }
            descriptions.append(desc)
            logger.info(
                f"[dry-run] {model.hash} | {model.simulation_type} | "
                f"{model.parametrization} | {model.engine} -> {output_dir / model.hash}"
            )
        logger.info(f"[dry-run] {len(descriptions)} system(s) would be built")
        logger.info(f"[dry-run] Provider: {config.provider}")
        return descriptions

    # parsl_session owns the full DFK lifecycle: guard, load, and shutdown
    # (including scancel of lingering SLURM jobs) on exit.
    with parsl_session(config) as session:
        build_app = get_build_app()

        # Submit all builds
        futures = []
        input_hashes = []
        for model, input_dict in resolved:
            input_hashes.append(model.hash)
            futures.append(build_app(input_dict))

        logger.info(f"Submitted {len(futures)} build(s) to Parsl")

        if not wait:
            logger.warning("Returning raw futures — caller must call parsl.clear() when done.")
            session.detach()
            return futures

        # Poll futures with live status reporting
        return _wait_with_progress(futures, hashes=input_hashes, label="Parsl Builds")


def _describe_failure(exc: BaseException) -> tuple[str, str]:
    """Extract ``(failure_type, error_detail)`` from a future's exception.

    Parsl surfaces worker errors differently across versions: modern Parsl
    re-raises the *original* exception (via ``RemoteExceptionWrapper.reraise()``),
    while older versions wrapped it and exposed the underlying error on
    ``.e_value``. Unwrap defensively so callers always see the *underlying*
    error type — letting future retry logic distinguish, e.g., a GROMACS crash
    (``CalledProcessError``) from an infrastructure failure (OOM / preemption).

    Parameters
    ----------
    exc : BaseException
        The exception raised by ``future.result()``.

    Returns
    -------
    tuple[str, str]
        ``(failure_type, error_detail)`` — the underlying exception's class
        name and its string representation.

    """
    underlying = getattr(exc, "e_value", None) or exc
    return type(underlying).__name__, str(underlying)


def _collect_results(results: list, hashes: list[str]) -> list[dict]:
    """Return all captured results, raising if any slot is still uncaptured.

    The polling loop only exits once every future has been recorded, so a
    ``None`` slot here signals an internal bug rather than a normal outcome.
    Failing explicitly is safer than silently returning a shorter list than
    the caller submitted (which would corrupt any ``len()``-based bookkeeping).

    Parameters
    ----------
    results : list
        Per-future result slots; ``None`` marks an uncaptured future.
    hashes : list[str]
        Display hashes aligned with ``results``, used for the error message.

    Returns
    -------
    list[dict]
        The complete list of result dicts.

    Raises
    ------
    RuntimeError
        If one or more result slots were never captured.

    """
    missing = [hashes[i] for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} build result(s) were never captured "
            f"({', '.join(h[:12] for h in missing)}); the polling loop exited "
            "prematurely. This is an internal error."
        )
    return list(results)


def _wait_with_progress(
    futures: list,
    *,
    hashes: list[str] | None = None,
    label: str = "Parsl Builds",
    poll_interval: float = 2.0,
) -> list[dict]:
    """Wait for futures with a live terminal progress display.

    Shows a progress bar with summary counts and a scrolling activity log
    of recent completions/failures. Scales to any number of builds.

    Parameters
    ----------
    futures : list
        List of Parsl AppFutures.
    hashes : list[str], optional
        Known hashes for each build (displayed in activity log).
    label : str
        Heading shown next to the progress bar. Defaults to ``"Parsl Builds"``;
        pass e.g. ``"Simulations"`` to reuse this display for other workflows.
    poll_interval : float
        Seconds between status polls.

    Returns
    -------
    list[dict]
        Result dicts for each future.

    """
    import time

    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
    from rich.text import Text

    total = len(futures)
    results: list[dict | None] = [None] * total
    display_hashes = hashes or [f"build-{i}" for i in range(total)]
    console = Console()

    # Track counts
    succeeded = 0
    failed = 0
    # Activity log (last N events)
    max_activity = 12
    activity: list[Text] = []

    # Progress bar
    progress = Progress(
        TextColumn(f"⚒ [bold]{label}[/]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[succeeded]} ✓[/]"),
        TextColumn("[red]{task.fields[failed]} ✗[/]"),
        TextColumn("[yellow]{task.fields[running]} ●[/]"),
        console=console,
        transient=False,
    )
    task_id = progress.add_task("builds", total=total, succeeded=0, failed=0, running=0)

    def _get_block_status() -> str:
        """Query Parsl for SLURM block (job) statuses."""
        try:
            import parsl  # type: ignore[import-not-found]

            dfk = parsl.dfk()
            counts: dict[str, int] = {}
            for executor in dfk.executors.values():
                if not hasattr(executor, "status"):
                    continue
                block_statuses = executor.status()
                for _block_id, job_status in block_statuses.items():
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
            # Catch-all for other states
            for state, count in counts.items():
                if state not in ("running", "pending", "completed", "failed"):
                    parts.append(f"[dim]{count} {state}[/]")
            return " · ".join(parts)
        except Exception:
            return ""

    def _render():
        done_count = succeeded + failed
        running = sum(1 for i, f in enumerate(futures) if results[i] is None and _is_running(f))
        progress.update(
            task_id,
            completed=done_count,
            succeeded=succeeded,
            failed=failed,
            running=running,
        )
        # Combine progress bar + SLURM status + activity log
        parts = [progress]
        block_info = _get_block_status()
        if block_info:
            parts.append(Text.from_markup(f"  ▸ SLURM: {block_info}"))
        if activity:
            parts.append(Text(""))  # blank line
            for line in activity[-max_activity:]:
                parts.append(line)
        return Group(*parts)

    def _is_running(future) -> bool:
        try:
            status = future.task_status()
            return status in ("launched", "running", "running_ended")
        except Exception:
            return False

    try:
        with Live(_render(), console=console, refresh_per_second=2) as live:
            while True:
                done_count = 0
                for i, future in enumerate(futures):
                    if results[i] is not None:
                        done_count += 1
                        continue
                    if future.done():
                        try:
                            result = future.result()
                            results[i] = result
                            succeeded += 1
                            line = Text()
                            line.append("  ✓ ", style="bold green")
                            line.append(display_hashes[i][:12], style="cyan")
                            line.append("  done", style="dim")
                            activity.append(line)
                        except Exception as exc:
                            failure_type, error_detail = _describe_failure(exc)
                            results[i] = {
                                "hash": display_hashes[i],
                                "status": "failed",
                                "error": error_detail,
                                "failure_type": failure_type,
                                "error_detail": error_detail,
                            }
                            failed += 1
                            line = Text()
                            line.append("  ✗ ", style="bold red")
                            line.append(display_hashes[i][:12], style="cyan")
                            line.append(f"  {error_detail}", style="red")
                            activity.append(line)
                        done_count += 1

                live.update(_render())

                if done_count == total:
                    break

                time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted — cancelling SLURM jobs...[/]")
        # Don't call _shutdown_parsl() here — build_systems' finally block owns cleanup
        raise

    return _collect_results(results, display_hashes)
