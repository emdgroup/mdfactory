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


from .errors import _describe_failure  # noqa: E402


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


from .progress import _get_block_status  # noqa: E402


def _is_running(future) -> bool:
    """Return ``True`` when a Parsl future is actively executing.

    Parameters
    ----------
    future : AppFuture
        Parsl future to inspect.

    Returns
    -------
    bool

    """
    try:
        return future.task_status() in ("launched", "running", "running_ended")
    except Exception:
        return False


def _poll_future(future, display_hash: str, result_transform) -> "tuple[dict, object]":
    """Collect the result (or error dict) from a completed future.

    Separates per-future result extraction from the polling loop, making
    both independently testable.

    Parameters
    ----------
    future : AppFuture
        A *completed* Parsl future (caller must check ``future.done()``).
    display_hash : str
        Short identifier used in the activity-log line.
    result_transform : callable or None
        ``(raw_result, display_hash) -> dict`` applied on success.

    Returns
    -------
    result : dict
        Result dict (success or failure).
    line : rich.text.Text
        Activity-log line to append to the live display.

    """
    from rich.text import Text

    try:
        raw = future.result()
        result = result_transform(raw, display_hash) if result_transform is not None else raw
        line = Text()
        line.append("  ✓ ", style="bold green")
        line.append(display_hash[:12], style="cyan")
        line.append("  done", style="dim")
        return result, line
    except Exception as exc:
        failure_type, error_detail = _describe_failure(exc)
        result = {
            "hash": display_hash,
            "status": "failed",
            "error": error_detail,
            "failure_type": failure_type,
            "error_detail": error_detail,
        }
        line = Text()
        line.append("  ✗ ", style="bold red")
        line.append(display_hash[:12], style="cyan")
        line.append(f"  {error_detail}", style="red")
        return result, line


def _wait_with_progress(
    futures: list,
    *,
    hashes: list[str] | None = None,
    label: str = "Parsl Builds",
    result_transform=None,
    poll_interval: float = 2.0,
) -> list[dict]:
    """Wait for futures with a live terminal progress display.

    Shows a progress bar with summary counts and a scrolling activity log
    of recent completions/failures. Scales to any number of builds.

    Per-future result handling is delegated to :func:`_poll_future`; SLURM
    block status querying to :func:`_get_block_status`.

    Parameters
    ----------
    futures : list
        List of Parsl AppFutures.
    hashes : list[str], optional
        Known hashes for each build (displayed in activity log).
    label : str
        Heading shown next to the progress bar.
    result_transform : callable or None, optional
        ``(raw_result, display_hash) -> dict`` applied to each successful
        future's return value before it is stored.
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

    succeeded = 0
    failed = 0
    max_activity = 12
    activity: list[Text] = []

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
        parts = [progress]
        block_info = _get_block_status()
        if block_info:
            parts.append(Text.from_markup(f"  ▸ SLURM: {block_info}"))
        if activity:
            parts.append(Text(""))
            for line in activity[-max_activity:]:
                parts.append(line)
        return Group(*parts)

    try:
        with Live(_render(), console=console, refresh_per_second=2) as live:
            while True:
                done_count = 0
                for i, future in enumerate(futures):
                    if results[i] is not None:
                        done_count += 1
                        continue
                    if future.done():
                        result, line = _poll_future(future, display_hashes[i], result_transform)
                        results[i] = result
                        activity.append(line)
                        if result.get("status") == "failed":
                            failed += 1
                        else:
                            succeeded += 1
                        done_count += 1

                live.update(_render())

                if done_count == total:
                    break

                time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted[/]")
        raise

    return _collect_results(results, display_hashes)
