# ABOUTME: Regression tests for wheel packaging of run-schedule resources
# ABOUTME: Covers Git-checkout and metadata-free source build contexts
"""Validate that wheel artifacts contain the packaged GROMACS templates."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
RUN_SCHEDULE_PREFIX = "mdfactory/run_schedules/gromacs/"
EXPECTED_RUN_SCHEDULES = {
    f"{RUN_SCHEDULE_PREFIX}{system_type}/{stage}.mdp"
    for system_type in ("bilayer", "mixedbox")
    for stage in ("em", "md", "npt", "nvt")
}


def _copy_tracked_source(destination: Path) -> bytes:
    """Copy the working tree's tracked files and return their pathspecs."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout

    for relative_path in tracked.rstrip(b"\0").split(b"\0"):
        source = REPOSITORY_ROOT / os.fsdecode(relative_path)
        target = destination / source.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return tracked


def _prepare_source(destination: Path, *, with_vcs_metadata: bool) -> None:
    """Create a build source with optional Git index metadata."""
    destination.mkdir()
    tracked = _copy_tracked_source(destination)
    if not with_vcs_metadata:
        return

    subprocess.run(["git", "init", "--quiet"], cwd=destination, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "--force",
            "--pathspec-from-file=-",
            "--pathspec-file-nul",
        ],
        cwd=destination,
        input=tracked,
        check=True,
    )


def _build_wheel(source: Path) -> Path:
    """Build and return the sole wheel produced from a source tree."""
    hatch = shutil.which("hatch")
    if hatch is None:
        pytest.fail("hatch executable is required for packaging tests")

    subprocess.run([hatch, "build", "-t", "wheel"], cwd=source, check=True)
    wheels = list((source / "dist").glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _assert_run_schedule_members(wheel: Path) -> None:
    """Assert that a wheel contains each expected template exactly once."""
    with ZipFile(wheel) as archive:
        members = [item.filename for item in archive.infolist()]

    run_schedule_members = [name for name in members if name.startswith(RUN_SCHEDULE_PREFIX)]
    assert set(run_schedule_members) == EXPECTED_RUN_SCHEDULES
    assert all(count == 1 for count in Counter(run_schedule_members).values())


def _assert_runtime_resolution(wheel: Path, destination: Path) -> None:
    """Resolve packaged templates while importing from extracted wheel contents."""
    with ZipFile(wheel) as archive:
        archive.extractall(destination)

    runtime_directory = destination.parent / "runtime"
    runtime_directory.mkdir()
    script = """
import sys
from pathlib import Path

package_root = Path(sys.argv[1]).resolve()
from mdfactory.run_schedules import RunScheduleManager
import mdfactory

assert Path(mdfactory.__file__).resolve().is_relative_to(package_root)
manager = RunScheduleManager()
for system_type in ("bilayer", "mixedbox"):
    paths = manager.get_all_run_file_paths("gromacs", system_type)
    assert set(paths) == {"em.mdp", "md.mdp", "npt.mdp", "nvt.mdp"}
    assert all(path.is_file() for path in paths.values())
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(destination)
    subprocess.run(
        [sys.executable, "-c", script, str(destination)],
        cwd=runtime_directory,
        env=environment,
        check=True,
    )


@pytest.mark.parametrize(
    "with_vcs_metadata",
    [True, False],
    ids=["git-checkout", "exported-source"],
)
def test_build_wheel_contains_run_schedule_templates(tmp_path, with_vcs_metadata):
    """Build wheels that contain usable, non-duplicated MDP templates."""
    source = tmp_path / "source"
    _prepare_source(source, with_vcs_metadata=with_vcs_metadata)

    wheel = _build_wheel(source)
    _assert_run_schedule_members(wheel)
    _assert_runtime_resolution(wheel, tmp_path / "installed")


def test_packaged_mdp_templates_are_only_ignore_exception():
    """Keep generated MDP files ignored while allowing package templates."""
    packaged = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "mdfactory/run_schedules/gromacs/bilayer/em.mdp",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    generated = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "simulation/em.mdp"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert packaged.returncode == 1
    assert generated.returncode == 0
