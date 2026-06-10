# ABOUTME: General-purpose utility functions for mdfactory
# ABOUTME: Provides working directory management, YAML loading, and file locking
# ABOUTME: Includes subprocess execution wrapper with graceful failure modes
"""General-purpose utility functions for mdfactory."""

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import yaml


def run_command(
    cmd: list[str],
    *,
    timeout: int | None = None,
    capture_output: bool = True,
    check: bool = False,
    text: bool = True,
    graceful: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess | str | None:
    """Run a shell command with flexible error handling.

    Provides a unified interface for subprocess execution with common
    patterns: graceful failure for optional commands, strict checking
    for critical operations, and timeout support.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to execute.
    timeout : int or None
        Timeout in seconds. None for no timeout (default).
    capture_output : bool
        Capture stdout/stderr. Default True.
    check : bool
        Raise CalledProcessError on non-zero exit. Default False.
        Ignored when graceful=True.
    text : bool
        Return output as text (str) rather than bytes. Default True.
    graceful : bool
        Return None on any failure (timeout, non-zero exit, OSError)
        instead of raising. Default False. When True, overrides check=True.
    **kwargs
        Additional arguments passed to subprocess.run (e.g., cwd, env,
        stdout, stderr, stdin).

    Returns
    -------
    subprocess.CompletedProcess, str, or None
        - If graceful=True: Returns stdout string on success, None on any failure.
        - If capture_output=True and not graceful: Returns stripped stdout string.
        - Otherwise: Returns CompletedProcess object.

    Raises
    ------
    subprocess.CalledProcessError
        If check=True and command exits with non-zero status.
    subprocess.TimeoutExpired
        If timeout is exceeded and graceful=False.
    FileNotFoundError
        If command binary not found and graceful=False.
    OSError
        On other OS-level failures and graceful=False.

    Examples
    --------
    >>> # Graceful failure for optional commands
    >>> output = run_command(["sinfo", "--version"], timeout=30, graceful=True)
    >>> if output is None:
    ...     print("SLURM not available")

    >>> # Strict checking for critical operations
    >>> run_command(["vmd", "-e", "script.tcl"], check=True)

    >>> # Manual error handling
    >>> result = run_command(["fdt", "config"], capture_output=False)
    >>> if result.returncode != 0:
    ...     print("Validation failed")
    """
    if graceful:
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=text,
                timeout=timeout,
                check=False,  # Handle manually in graceful mode
                **kwargs,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip() if capture_output and text else result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
    else:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
            **kwargs,
        )
        # If caller wants just stdout (most common case)
        if capture_output and text and not check:
            return result.stdout.strip()
        return result


@contextlib.contextmanager
def working_directory(path, create=False, cleanup=False, exists_ok=True):
    """Change working directory and return to previous on exit.

    Parameters
    ----------
    path : str or Path
        Target directory to change into.
    create : bool, optional
        Create the directory if it does not exist. Default is False.
    cleanup : bool, optional
        Remove the directory before creating it. Implies ``create=True``.
        Default is False.
    exists_ok : bool, optional
        If False, raise ``FileExistsError`` when the directory already exists.
        Default is True.

    Yields
    ------
    Path
        Resolved path of the working directory.

    Raises
    ------
    FileExistsError
        If the directory exists and ``exists_ok`` is False.

    """
    path = Path(path).resolve()
    if path.is_dir() and not exists_ok:
        raise FileExistsError("Path already exists.")
    if cleanup:
        create = True
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev_cwd)


@contextlib.contextmanager
def temporary_working_directory(prefix="temp_"):
    """Create a temporary directory and change into it.

    Parameters
    ----------
    prefix : str, optional
        Prefix for the temporary directory name. Default is ``"temp_"``.

    Yields
    ------
    Path
        Resolved path of the temporary directory.

    """
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        with working_directory(temp_dir) as tmp:
            yield tmp


def load_yaml_file(yaml_file_path: str) -> Dict[str, Any]:
    """Load data from a YAML file.

    Parameters
    ----------
    yaml_file_path : str
        The path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        The loaded data from the YAML file.

    """
    with open(yaml_file_path, "r") as file:
        return yaml.safe_load(file)


@contextlib.contextmanager
def lock_local_folder(folder, retries: int = 120, wait: float = 2.0, message: str = ""):
    """Acquire a file-based lock on a folder, blocking until available.

    Create a ``<folder>.lock`` sentinel file and yield. If the lock file
    already exists, retry up to *retries* times with *wait* seconds between
    attempts. The lock file is removed on exit.

    Parameters
    ----------
    folder : str or Path
        Path to the folder to lock.
    retries : int, optional
        Maximum number of acquisition attempts. Default is 120.
    wait : float, optional
        Seconds to sleep between retries. Default is 2.0.
    message : str, optional
        Informational message (currently unused, reserved for logging).

    Yields
    ------
    None

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within the retry limit.

    """
    lockfile = Path(f"{folder}.lock")
    was_locked = False
    created_lockfile = False
    other_exception = None
    for _ in range(retries):
        try:
            lockfile.touch(exist_ok=False)
            created_lockfile = True
            was_locked = False
            # print(f"Acquired lock for folder {folder}.", message)
            yield
        except FileExistsError as e:
            if created_lockfile:
                other_exception = e
            # print(f"Folder {folder} is already locked.", message, e)
            was_locked = True
            time.sleep(wait)
        except Exception as e:
            other_exception = e
        finally:
            if other_exception:
                lockfile.unlink()
                raise other_exception
            if was_locked:
                # print(f"Folder {folder} was locked.", message)
                continue
            lockfile.unlink()
            return
    raise TimeoutError(f"Could not acquire lock for folder {folder} after {retries} retries.")
