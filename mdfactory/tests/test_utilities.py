# ABOUTME: Tests for general utility functions including YAML file loading
# ABOUTME: and file locking behavior under concurrent access.
"""Tests for general utility functions including YAML file loading."""

import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mdfactory.utils.utilities import (
    load_yaml_file,
    run_command,
)

from .utils import ProcessWithException


def test_load_yaml_file(tmp_path):
    """Test loading data from a YAML file."""
    test_data = {
        "name": "Test Config",
        "version": 1.0,
        "settings": {"enabled": True, "timeout": 30},
    }

    # Create a test YAML file
    test_yaml_path = tmp_path / "test_config.yaml"
    with open(test_yaml_path, "w") as file:
        yaml.dump(test_data, file)

    # Load the file using our function
    loaded_data = load_yaml_file(test_yaml_path)

    # Verify the data matches
    assert loaded_data == test_data


def test_load_yaml_file_nonexistent(tmp_path):
    """Test loading a YAML file that doesn't exist raises an error."""
    nonexistent_file = tmp_path / "doesnt_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml_file(nonexistent_file)


def test_lock_folder(tmp_path):
    import threading

    from mdfactory.utils.utilities import lock_local_folder

    lf = tmp_path / "lockfolder"

    def do_stuff(i):
        print(f"Thread {i} starting...")
        with lock_local_folder(lf, message=f"from thread {i}"):
            print("lolol")
            time.sleep(0.1 * i)
        print(f"Thread {i} done.")

    threads = []
    for i in range(5):
        thread = threading.Thread(target=do_stuff, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def test_working_directory_with_exception(tmp_path):
    from mdfactory.utils.utilities import temporary_working_directory, working_directory

    d = tmp_path / "subdir"
    d.mkdir()

    with pytest.raises(Exception, match="Some error"):
        with working_directory(d):
            raise Exception("Some error occurred")

    existed = tmp_path / "existed"
    existed.mkdir()
    with pytest.raises(FileExistsError):
        with working_directory(existed, exists_ok=False):
            pass

    with working_directory(existed, exists_ok=True, cleanup=True):
        pass

    tmp = None
    with temporary_working_directory() as temp_dir:
        assert temp_dir.exists()
        tmp = temp_dir
    assert not tmp.exists()


def test_lock_folder_processes(tmp_path):
    from mdfactory.utils.utilities import lock_local_folder

    lockpath = (tmp_path / "testlock").resolve()
    lockfile = Path(f"{lockpath}.lock")

    lockfile.touch()

    with pytest.raises(TimeoutError):
        with lock_local_folder(lockpath, retries=5, wait=0.1):
            print("Locked folder:", lockpath)

    lockfile.unlink()

    with pytest.raises(RuntimeError, match="custom exception"):
        with lock_local_folder(lockpath, retries=1, wait=0.1):
            raise RuntimeError("custom exception")

    assert not lockfile.is_file()

    def lock_stuff(i):
        with lock_local_folder(lockpath, retries=2, wait=0.1):
            print("Locked folder:", lockpath)

    processes = []
    n_procs = 16
    for i in range(n_procs):
        proc = ProcessWithException(target=lock_stuff, args=(i,))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    for proc in processes:
        if proc.exception:
            raise Exception(f"{proc.exception[1]}") from proc.exception[0]
    assert not lockfile.is_file()


# ---------------------------------------------------------------------------
# Tests: run_command
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Test run_command with various failure modes and options."""

    def test_graceful_returns_none_on_timeout(self):
        """Test that graceful=True returns None on timeout."""
        with patch(
            "mdfactory.utils.utilities.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["test_cmd"], timeout=30),
        ):
            result = run_command(["test_cmd"], timeout=30, graceful=True)
        assert result is None

    def test_graceful_returns_none_on_file_not_found(self):
        """Test that graceful=True returns None when binary not found."""
        with patch(
            "mdfactory.utils.utilities.subprocess.run",
            side_effect=FileNotFoundError("No such file"),
        ):
            result = run_command(["nonexistent_binary"], graceful=True)
        assert result is None

    def test_graceful_returns_none_on_nonzero_exit(self):
        """Test that graceful=True returns None on non-zero exit."""
        with patch(
            "mdfactory.utils.utilities.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["test"], returncode=1, stdout="", stderr="error"
            ),
        ):
            result = run_command(["test", "--flag"], graceful=True)
        assert result is None

    def test_graceful_returns_stdout_on_success(self):
        """Test that graceful=True returns stdout on success."""
        with patch(
            "mdfactory.utils.utilities.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["echo"], returncode=0, stdout="hello\n", stderr=""
            ),
        ):
            result = run_command(["echo", "hello"], graceful=True)
        assert result == "hello"

    def test_check_raises_on_nonzero_exit(self):
        """Test that check=True raises on non-zero exit."""
        with patch(
            "mdfactory.utils.utilities.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                returncode=1, cmd=["false"], output="", stderr="error"
            ),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                run_command(["false"], check=True)

    def test_returns_completed_process_when_no_capture(self):
        """Test that result is CompletedProcess when not capturing output."""
        with patch(
            "mdfactory.utils.utilities.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["echo"], returncode=0, stdout=None, stderr=None
            ),
        ) as mock_run:
            result = run_command(["echo", "test"], capture_output=False)
            assert isinstance(result, subprocess.CompletedProcess)
            assert result.returncode == 0
            mock_run.assert_called_once()
