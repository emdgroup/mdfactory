# ABOUTME: Build orchestration dispatch via Parsl
# ABOUTME: Submits parallel builds and provides dry-run preview
"""Build orchestration via Parsl."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from mdfactory.models.input import BuildInput

from .apps import get_build_app

if TYPE_CHECKING:
    from .config import ExecutorConfig


def build_systems(
    build_inputs: list,
    config: "ExecutorConfig",
    *,
    output_dir: Path | None = None,
    wait: bool = True,
) -> list:
    """Submit parallel builds via Parsl.

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

    Returns
    -------
    list
        If wait=True, list of result dicts. If wait=False, list of AppFutures.

    """
    try:
        import parsl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "parsl is required for build orchestration. "
            "Install with `pip install 'mdfactory[parsl]'`."
        ) from exc

    output_dir = Path(output_dir) if output_dir else Path.cwd()

    # Build parsl config and load
    parsl_config = config.to_parsl_config()
    parsl.load(parsl_config)

    cleanup_needed = True
    try:
        build_app = get_build_app()

        # Prepare inputs and submit
        futures = []
        input_hashes = []
        for inp in build_inputs:
            if isinstance(inp, BuildInput):
                input_dict = inp.model_dump()
                input_dict["_build_dir"] = str(output_dir / inp.hash)
                input_hashes.append(inp.hash)
            elif isinstance(inp, dict):
                model = BuildInput(**inp)
                input_dict = inp.copy()
                input_dict["_build_dir"] = str(output_dir / model.hash)
                input_hashes.append(model.hash)
            else:
                raise TypeError(f"Expected BuildInput or dict, got {type(inp)}")
            futures.append(build_app(input_dict))

        logger.info(f"Submitted {len(futures)} build(s) to Parsl")

        if not wait:
            logger.warning("Returning raw futures — caller must call parsl.clear() when done.")
            cleanup_needed = False
            return futures

        # Poll futures with live status reporting
        results = _wait_with_progress(futures, hashes=input_hashes)
        return results
    finally:
        if cleanup_needed:
            _shutdown_parsl()


def _shutdown_parsl():
    """Shut down Parsl and cancel any remaining SLURM jobs."""
    try:
        import parsl  # type: ignore[import-not-found]
    except ImportError:
        return

    job_ids = []
    try:
        dfk = parsl.dfk()
        job_ids = _get_slurm_job_ids(dfk)
    except Exception as exc:
        logger.debug(f"Could not query Parsl DFK for SLURM jobs: {exc}")

    try:
        parsl.clear()
    except Exception as exc:
        logger.warning(f"Parsl shutdown failed — DFK may still be loaded: {exc}")

    # Always attempt scancel if we found job IDs
    if job_ids:
        _scancel_jobs(job_ids)


def _get_slurm_job_ids(dfk) -> list[str]:
    """Extract active SLURM job IDs from the Parsl DataFlowKernel."""
    job_ids = []
    try:
        for executor in dfk.executors.values():
            if hasattr(executor, "provider") and hasattr(executor.provider, "resources"):
                for _block_id, resource in executor.provider.resources.items():
                    job_id = resource.get("remote_job_id") or resource.get("job_id")
                    if job_id:
                        job_ids.append(str(job_id))
    except Exception as exc:
        logger.debug(f"Could not enumerate SLURM jobs for explicit scancel: {exc}")
    return job_ids


def _scancel_jobs(job_ids: list[str]):
    """Run scancel on the given SLURM job IDs."""
    if not job_ids:
        return
    try:
        cmd = ["scancel"] + job_ids
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            logger.info(f"Cancelled SLURM jobs: {', '.join(job_ids)}")
        else:
            logger.debug(f"scancel stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        logger.debug("scancel not found (not on SLURM cluster)")
    except Exception as exc:
        logger.debug(f"scancel failed: {exc}")


def _wait_with_progress(
    futures: list,
    *,
    hashes: list[str] | None = None,
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
        TextColumn("🔨 [bold]Parsl Builds[/]"),
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
            parts.append(Text.from_markup(f"  🖥  SLURM: {block_info}"))
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
                            error_msg = str(exc)
                            results[i] = {
                                "hash": display_hashes[i],
                                "status": "failed",
                                "error": error_msg,
                            }
                            failed += 1
                            line = Text()
                            line.append("  ✗ ", style="bold red")
                            line.append(display_hashes[i][:12], style="cyan")
                            line.append(f"  {error_msg}", style="red")
                            activity.append(line)
                        done_count += 1

                live.update(_render())

                if done_count == total:
                    break

                time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted — cancelling SLURM jobs...[/]")
        _shutdown_parsl()
        raise

    return [r for r in results if r is not None]


def build_systems_dry_run(
    build_inputs: list,
    config: "ExecutorConfig",
    *,
    output_dir: Path | None = None,
) -> list[dict]:
    """Print what would be built without submitting to Parsl.

    Parameters
    ----------
    build_inputs : list
        List of BuildInput models or dicts.
    config : ExecutorConfig
        Executor configuration (used for display, Parsl not loaded).
    output_dir : Path, optional
        Base output directory for builds. Defaults to current directory.

    Returns
    -------
    list[dict]
        List of dicts describing each planned build.

    """
    output_dir = Path(output_dir) if output_dir else Path.cwd()

    descriptions = []
    for inp in build_inputs:
        if isinstance(inp, BuildInput):
            model = inp
        elif isinstance(inp, dict):
            model = BuildInput(**inp)
        else:
            raise TypeError(f"Expected BuildInput or dict, got {type(inp)}")

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
