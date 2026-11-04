# ABOUTME: Tests for the simulate-time GROMACS version check in generated grompp scripts
# ABOUTME: Covers requirement detection from topology.top, run_stage wiring, and the bash check
"""Tests for the simulate-time GROMACS version check in generated grompp scripts."""

import shutil
import subprocess
import textwrap
from unittest.mock import MagicMock

import pytest

from mdfactory.orchestration.apps import _build_grompp_script
from mdfactory.orchestration.stages import STAGE_BY_NAME, min_gmx_version_for_sim_dir, run_stage

AMBER_TOP = '#include "amber14sb.ff/forcefield.itp"\n#include "amber14sb.ff/tip3p.itp"\n'
CHARMM_TOP = '#include "charmm36-feb2026_cgenff-5.0.ff/forcefield.itp"\n'


def _script(work_dir, min_gmx_version=""):
    return _build_grompp_script(
        "em.mdp",
        "system.pdb",
        "topology.top",
        "min.tpr",
        str(work_dir),
        min_gmx_version=min_gmx_version,
    )


class TestRequirementDetection:
    @pytest.mark.parametrize(("top", "expected"), [(AMBER_TOP, "2026"), (CHARMM_TOP, "")])
    def test_from_topology(self, tmp_path, top, expected):
        (tmp_path / "topology.top").write_text(top)
        assert min_gmx_version_for_sim_dir(tmp_path) == expected

    def test_missing_topology_has_no_requirement(self, tmp_path):
        assert min_gmx_version_for_sim_dir(tmp_path) == ""

    @pytest.mark.parametrize(("top", "expected"), [(AMBER_TOP, "2026"), (CHARMM_TOP, None)])
    def test_run_stage_passes_requirement_only_when_needed(self, tmp_path, top, expected):
        (tmp_path / "topology.top").write_text(top)
        grompp_app = MagicMock(return_value=MagicMock())
        run_stage(STAGE_BY_NAME["EM"], tmp_path, None, grompp_app, MagicMock())
        assert grompp_app.call_args.kwargs.get("min_gmx_version") == expected


class TestGromppScript:
    def test_no_check_without_minimum(self, tmp_path):
        script = _script(tmp_path)
        assert "GMX_VERSION" not in script
        assert script == _build_grompp_script(
            "em.mdp", "system.pdb", "topology.top", "min.tpr", str(tmp_path)
        )

    def test_check_runs_after_detection_and_before_grompp(self, tmp_path):
        script = _script(tmp_path, "2026")
        assert "GROMACS >= 2026" in script
        detect = script.index("GMX_BIN=gmx")
        check = script.index("GMX_VERSION=")
        assert detect < check < script.index("$GMX_BIN grompp")

    @pytest.mark.parametrize("bad", ["abc", "2026.x", "-1"])
    def test_rejects_invalid_minimum(self, tmp_path, bad):
        with pytest.raises(ValueError, match="Invalid minimum GROMACS version"):
            _script(tmp_path, bad)


def _fake_gmx(bin_dir, name="gmx"):
    """Write a fake gmx that reports $FAKE_VERSION and records grompp calls."""
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / name
    fake.write_text(
        textwrap.dedent("""\
        #!/bin/bash
        if [ "$1" = "--version" ]; then
            echo "                 :-) GROMACS - gmx, $FAKE_VERSION (-:"
            echo
            echo "GROMACS version:    $FAKE_VERSION"
            echo "Precision:          mixed"
            exit 0
        fi
        echo "$@" >> "$FAKE_LOG"
        while [ $# -gt 0 ]; do
            if [ "$1" = "-o" ]; then touch "$2"; fi
            shift
        done
        """)
    )
    fake.chmod(0o755)
    return fake


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
class TestVersionCheckExecution:
    @pytest.mark.parametrize(
        ("version", "passes"),
        [
            ("2026.3-conda_forge", True),
            ("2026.0-dev-20250101-abcdef", True),
            ("2024.3", False),
            ("2025.4", False),
            ("garbage", False),
        ],
    )
    def test_fake_gmx(self, tmp_path, version, passes):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        bin_dir = tmp_path / "bin"
        _fake_gmx(bin_dir)
        log = tmp_path / "grompp.log"
        result = subprocess.run(
            ["bash", "-c", _script(sim_dir, "2026")],
            capture_output=True,
            text=True,
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "FAKE_VERSION": version,
                "FAKE_LOG": str(log),
            },
            timeout=30,
            check=False,
        )
        if passes:
            assert result.returncode == 0, result.stderr
            assert (sim_dir / "min.tpr").is_file()
            assert "grompp" in log.read_text()
        else:
            assert result.returncode != 0
            assert "requires GROMACS >= 2026" in result.stderr
            assert str(bin_dir / "gmx") in result.stderr
            # grompp (and therefore any mdrun) never ran.
            assert not log.exists()
            assert not (sim_dir / "min.tpr").exists()
