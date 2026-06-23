# ABOUTME: Generic Parsl session lifecycle management (guard, load, shutdown)
# ABOUTME: Reusable context manager shared by build/simulation/benchmark orchestration
"""Parsl session management.

Provides :func:`parsl_session`, a context manager that owns the full Parsl
``DataFlowKernel`` lifecycle — guarding against an already-active DFK, loading
the executor config, and guaranteeing shutdown (including ``scancel`` of any
lingering SLURM jobs) on exit. Build, simulation, and benchmark orchestration
all share this single implementation rather than duplicating the guard / load /
cleanup boilerplate.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from .config import _import_parsl

if TYPE_CHECKING:
    from .config import ExecutorConfig


@dataclass
class ParslSession:
    """Handle yielded by :func:`parsl_session`.

    Parameters
    ----------
    parsl : module
        The imported ``parsl`` module, for submitting apps or inspecting the DFK.
    detached : bool
        When ``True``, the context manager will **not** shut down the DFK on
        exit; the caller takes ownership of cleanup. Set via :meth:`detach`.

    """

    parsl: object
    detached: bool = False

    def detach(self) -> None:
        """Transfer DFK ownership to the caller (skip shutdown on exit).

        Used when returning raw futures (e.g. ``wait=False``): the caller is
        then responsible for calling ``parsl.clear()`` once all futures
        complete.
        """
        self.detached = True


@contextmanager
def parsl_session(config: "ExecutorConfig") -> Iterator[ParslSession]:
    """Manage a Parsl ``DataFlowKernel`` lifecycle for an orchestration run.

    Guards against an already-active DFK, loads ``config``'s Parsl config,
    yields a :class:`ParslSession`, and guarantees shutdown on exit unless the
    session was detached via :meth:`ParslSession.detach`.

    Examples
    --------
    >>> with parsl_session(config) as session:  # doctest: +SKIP
    ...     futures = [app(x) for x in inputs]
    ...     results = wait(futures)

    Parameters
    ----------
    config : ExecutorConfig
        Executor configuration whose ``to_parsl_config()`` drives the DFK.

    Yields
    ------
    ParslSession
        Session handle exposing the ``parsl`` module and a ``detach()`` escape
        hatch.

    Raises
    ------
    RuntimeError
        If a Parsl ``DataFlowKernel`` is already active.

    """
    parsl = _import_parsl()
    _guard_no_active_dfk(parsl)
    parsl.load(config.to_parsl_config())
    session = ParslSession(parsl=parsl)
    try:
        yield session
    finally:
        if not session.detached:
            _shutdown_parsl()


def _guard_no_active_dfk(parsl) -> None:
    """Raise if a Parsl ``DataFlowKernel`` is already loaded.

    Parameters
    ----------
    parsl : module
        The imported ``parsl`` module.

    Raises
    ------
    RuntimeError
        If a DFK is already active.

    """
    try:
        parsl.dfk()
    except Exception:
        return  # No active DFK, good
    raise RuntimeError(
        "A Parsl DataFlowKernel is already active. "
        "Call parsl.clear() before starting a new session."
    )


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
    except subprocess.TimeoutExpired:
        logger.warning(
            f"scancel timed out after 10 s — SLURM jobs {', '.join(job_ids)} may still "
            f"be running. Run `scancel {' '.join(job_ids)}` manually."
        )
    except Exception as exc:
        logger.debug(f"scancel failed: {exc}")
