# ABOUTME: GROMACS error classification for rescue retry decisions
# ABOUTME: Distinguishes physics failures (rescuable) from infrastructure failures
"""GROMACS error classification for adaptive rescue retry.

Classifies simulation failures as physics-driven (rescuable via parameter
adjustment) or infrastructure-driven (handled by Parsl's built-in retry).
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from loguru import logger

#: Regex patterns matching GROMACS physics-failure messages in stderr/log.
#: Each pattern is compiled once at import time for performance.
_PHYSICS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"LINCS WARNING",
        r"the coordinate constraints could not be satisfied",
        r"Particle .+ moved more than",
        r"The domain decomposition cell size",
        r"Fatal error.*blowing up",
        r"step \d+.*: Water molecule starting at",
        r"Too many LINCS warnings",
        r"can not continue",
        r"force .+ is not finite",
        r"A]?norm(al termination|al end| of program)",
        r"There are \d+ perturbed non-bonded pair interactions",
        r"atoms .+ are involved in more than \d+ constraints",
    ]
]


class FailureType(Enum):
    """Classification of a simulation stage failure.

    Attributes
    ----------
    PHYSICS
        Failure due to simulation physics (overlapping atoms, box explosion,
        constraint failures). Rescuable via MDP parameter adjustment.
    INFRASTRUCTURE
        Failure due to infrastructure (OOM, timeout, SLURM preemption,
        filesystem errors). Handled by Parsl's built-in retry.
    UNKNOWN
        Failure whose cause could not be determined. Treated as
        non-rescuable (same as INFRASTRUCTURE).

    """

    PHYSICS = "physics"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


def classify_failure(
    exc: BaseException,
    sim_dir: Path | None = None,
    stage: str | None = None,
) -> FailureType:
    """Classify a GROMACS simulation failure.

    Inspects the exception message and, optionally, the GROMACS log file
    in the simulation directory to determine whether the failure is due
    to physics (rescuable) or infrastructure.

    Parameters
    ----------
    exc : BaseException
        The exception from a failed Parsl future.
    sim_dir : Path, optional
        Simulation directory to check for GROMACS log files.
    stage : str, optional
        Stage name (used to locate the correct log file).

    Returns
    -------
    FailureType
        The classified failure type.

    """
    # Unwrap Parsl's exception wrapping (legacy and modern)
    underlying = getattr(exc, "e_value", None) or exc
    error_text = str(underlying)

    # Check the exception message first
    if _matches_physics_patterns(error_text):
        return FailureType.PHYSICS

    # Check GROMACS log file if available
    if sim_dir is not None and stage is not None:
        log_text = _read_stage_log(sim_dir, stage)
        if log_text and _matches_physics_patterns(log_text):
            return FailureType.PHYSICS

    # Check for known infrastructure patterns
    infra_keywords = [
        "oom",
        "out of memory",
        "killed",
        "timeout",
        "time limit",
        "cancelled",
        "preempted",
        "no space left",
        "disk quota",
        "connection reset",
    ]
    error_lower = error_text.lower()
    for keyword in infra_keywords:
        if keyword in error_lower:
            return FailureType.INFRASTRUCTURE

    logger.debug(f"Could not classify failure: {error_text[:200]}")
    return FailureType.UNKNOWN


def _matches_physics_patterns(text: str) -> bool:
    """Return True if any physics-failure pattern matches the text.

    Parameters
    ----------
    text : str
        Error message or log content to search.

    Returns
    -------
    bool

    """
    for pattern in _PHYSICS_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _read_stage_log(sim_dir: Path, stage: str) -> str:
    """Read the GROMACS log file for a stage, if it exists.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    stage : str
        Stage name.

    Returns
    -------
    str
        Log file content, or empty string if not found.

    """
    from .stages import STAGE_BY_NAME

    spec = STAGE_BY_NAME.get(stage)
    if spec is None:
        return ""

    log_file = sim_dir / f"{spec.deffnm}.log"
    if not log_file.exists():
        return ""

    try:
        # Read only the last 10KB to avoid huge logs
        size = log_file.stat().st_size
        with open(log_file) as f:
            if size > 10240:
                f.seek(size - 10240)
                f.readline()  # Skip partial line
            return f.read()
    except Exception as e:
        logger.debug(f"Could not read log file {log_file}: {e}")
        return ""
