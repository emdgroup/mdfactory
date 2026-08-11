# ABOUTME: Integration tests for rescue retry with real GROMACS execution.
# ABOUTME: Validates that oversized dt crashes NVT at tier 0 and succeeds after rescue.
"""Integration tests: rescue retry with real GROMACS.

These tests require ``gmx`` on PATH and are marked ``slow`` — they are
excluded from the default test suite and must be run explicitly::

    pixi run pytest -m slow mdfactory/tests/test_rescue_integration.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if GROMACS is not available
# ---------------------------------------------------------------------------

GMX = shutil.which("gmx") or shutil.which("gmx_mpi")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(GMX is None, reason="GROMACS (gmx) not found on PATH"),
]

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rescue_test"

# Force fields to try, in preference order.  Each entry is
# (forcefield_dir, water_include) — we pick the first one that exists
# in the GROMACS data directory.
_FF_CANDIDATES = [
    ("oplsaa.ff", "oplsaa.ff/spce.itp"),
    ("amber99sb-ildn.ff", "amber99sb-ildn.ff/spce.itp"),
    ("gromos54a7.ff", "gromos54a7.ff/spc.itp"),
    ("charmm27.ff", "charmm27.ff/spc.itp"),
]


def _find_gmx_topdir() -> Path | None:
    """Discover the GROMACS topology data directory.

    Returns
    -------
    Path or None
        Path to the ``top`` directory (e.g. ``/usr/share/gromacs/top``),
        or None if it cannot be determined.
    """
    # GMXLIB takes precedence
    import os

    gmxlib = os.environ.get("GMXLIB")
    if gmxlib:
        p = Path(gmxlib)
        if p.is_dir():
            return p

    # Parse 'gmx --version' for the data prefix
    try:
        result = subprocess.run(
            [GMX, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "Data prefix" in line:
                prefix = line.split(":", 1)[1].strip()
                topdir = Path(prefix) / "share" / "gromacs" / "top"
                if topdir.is_dir():
                    return topdir
    except Exception:
        pass

    return None


def _detect_forcefield() -> tuple[str, str]:
    """Return (forcefield_itp, water_itp) for the first available FF.

    Raises
    ------
    pytest.skip
        If no supported force field is found.
    """
    topdir = _find_gmx_topdir()
    if topdir is None:
        pytest.skip("Cannot locate GROMACS topology data directory")

    for ff_dir, water_itp in _FF_CANDIDATES:
        if (topdir / ff_dir).is_dir():
            ff_itp = f"{ff_dir}/forcefield.itp"
            return ff_itp, water_itp

    available = [d.name for d in topdir.iterdir() if d.suffix == ".ff"]
    pytest.skip(f"No supported force field found in {topdir}. Available: {available}")


def _write_topology(dest: Path, ff_itp: str, water_itp: str) -> None:
    """Write a minimal SPC water topology file."""
    dest.write_text(
        textwrap.dedent(f"""\
        ; Auto-generated topology for rescue integration test
        #include "{ff_itp}"
        #include "{water_itp}"

        [ system ]
        Rescue integration test

        [ molecules ]
        SOL     3
        """)
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sim_dir(tmp_path: Path) -> Path:
    """Copy fixture files to a temp directory with a working topology.

    Copies ``system.gro``, ``em.mdp``, and ``nvt.mdp`` from the static
    fixtures, then generates ``topology.top`` using whichever force field
    is available in the local GROMACS installation.
    """
    # Copy static fixtures
    for name in ("system.gro", "em.mdp", "nvt.mdp"):
        shutil.copy(FIXTURE_DIR / name, tmp_path / name)

    # Generate topology with detected force field
    ff_itp, water_itp = _detect_forcefield()
    _write_topology(tmp_path / "topology.top", ff_itp, water_itp)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRescueIntegrationNVT:
    """Verify NVT rescue with real GROMACS execution."""

    def _run_gmx(
        self,
        args: list[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess:
        """Run a gmx command and return the result."""
        return subprocess.run(
            [GMX, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def _run_grompp(
        self, sim_dir: Path, mdp: str = "nvt.mdp", tpr: str = "nvt.tpr"
    ) -> subprocess.CompletedProcess:
        """Run grompp and return the result."""
        return self._run_gmx(
            [
                "grompp",
                "-f",
                mdp,
                "-c",
                "system.gro",
                "-p",
                "topology.top",
                "-o",
                tpr,
                "-maxwarn",
                "10",
            ],
            cwd=sim_dir,
        )

    def _run_mdrun(self, sim_dir: Path, deffnm: str = "nvt") -> subprocess.CompletedProcess:
        """Run mdrun and return the result."""
        return self._run_gmx(
            ["mdrun", "-deffnm", deffnm, "-ntomp", "1"],
            cwd=sim_dir,
        )

    def _run_em(self, sim_dir: Path) -> subprocess.CompletedProcess:
        """Run energy minimisation to prepare a valid starting structure."""
        grompp = self._run_grompp(sim_dir, mdp="em.mdp", tpr="em.tpr")
        assert grompp.returncode == 0, f"EM grompp failed: {grompp.stderr}"
        mdrun = self._run_gmx(
            ["mdrun", "-deffnm", "em", "-ntomp", "1"],
            cwd=sim_dir,
        )
        assert mdrun.returncode == 0, f"EM mdrun failed: {mdrun.stderr}"
        # Use EM output as starting structure for NVT
        shutil.copy(sim_dir / "em.gro", sim_dir / "system.gro")
        return mdrun

    def test_grompp_succeeds(self, sim_dir: Path):
        """Sanity check: grompp accepts the fixture files for NVT."""
        self._run_em(sim_dir)
        result = self._run_grompp(sim_dir)
        assert result.returncode == 0, (
            f"grompp failed — the fixture files may be invalid.\nstderr: {result.stderr}"
        )
        assert (sim_dir / "nvt.tpr").exists()

    def test_nvt_crashes_at_tier_0(self, sim_dir: Path):
        """The oversized dt (0.1 ps) causes a LINCS/constraint failure."""
        self._run_em(sim_dir)
        grompp = self._run_grompp(sim_dir)
        assert grompp.returncode == 0, f"grompp failed: {grompp.stderr}"

        mdrun = self._run_mdrun(sim_dir)
        assert mdrun.returncode != 0, (
            "mdrun succeeded with dt=0.1 — the fixture is not "
            "triggering a crash. Increase dt or adjust the system."
        )

        # Confirm it's a physics failure, not a missing-file error
        combined = mdrun.stdout + mdrun.stderr
        log_file = sim_dir / "nvt.log"
        if log_file.exists():
            combined += log_file.read_text()

        physics_markers = [
            "LINCS",
            "SETTLE",
            "moved more than",
            "blowing up",
            "coordinate constraints",
            "Abnorm",
            "not finite",
            "can not continue",
            "too large",
            "Bond/angle/dihedral",
        ]
        assert any(m in combined for m in physics_markers), (
            f"mdrun failed but not with a recognisable physics error.\n"
            f"Output (last 500 chars): ...{combined[-500:]}"
        )

    def test_apply_rescue_tier_produces_valid_mdp(self, sim_dir: Path):
        """apply_rescue_tier writes a rescue MDP with halved dt."""
        from mdfactory.orchestration.mdp import (
            apply_rescue_tier,
            get_mdp_value,
            parse_mdp,
        )

        rescue_path = apply_rescue_tier(sim_dir, "NVT", tier=1)

        assert rescue_path.exists()
        assert rescue_path.name == "nvt_rescue_t1.mdp"

        parsed = parse_mdp(rescue_path)
        dt = float(get_mdp_value(parsed, "dt"))
        nsteps = int(get_mdp_value(parsed, "nsteps"))

        assert dt == pytest.approx(0.05)  # 0.1 / 2^1
        assert nsteps == 200  # 100 * 2^1

    def test_nvt_succeeds_with_rescue_tier(self, sim_dir: Path):
        """NVT runs when rescue reduces dt sufficiently.

        Tries rescue tiers 1 through 6 until one succeeds. At least one
        tier must complete without error for the test to pass — this
        validates the core premise that halving the timestep eventually
        stabilises the dynamics.
        """
        from mdfactory.orchestration.mdp import apply_rescue_tier

        self._run_em(sim_dir)

        max_tier = 6
        for tier in range(1, max_tier + 1):
            rescue_mdp = apply_rescue_tier(sim_dir, "NVT", tier=tier)

            # Clean previous run outputs
            for f in sim_dir.glob("nvt.*"):
                if f.suffix != ".mdp":
                    f.unlink()

            grompp = self._run_grompp(sim_dir, mdp=rescue_mdp.name)
            assert grompp.returncode == 0, f"grompp failed at tier {tier}: {grompp.stderr}"

            mdrun = self._run_mdrun(sim_dir)
            if mdrun.returncode == 0:
                return

        pytest.fail(
            f"NVT did not succeed at any rescue tier (1–{max_tier}). "
            f"Last dt = {0.1 / 2**max_tier}. "
            f"Consider adjusting the fixture."
        )

    def test_classify_failure_detects_physics(self, sim_dir: Path):
        """classify_failure correctly identifies the crash as PHYSICS."""
        from mdfactory.orchestration.errors import FailureType, classify_failure

        self._run_em(sim_dir)
        grompp = self._run_grompp(sim_dir)
        if grompp.returncode != 0:
            pytest.skip(f"grompp failed: {grompp.stderr}")

        mdrun = self._run_mdrun(sim_dir)
        if mdrun.returncode == 0:
            pytest.skip("mdrun did not crash — cannot test classification")

        error_text = mdrun.stderr + mdrun.stdout
        log_file = sim_dir / "nvt.log"
        if log_file.exists():
            error_text += log_file.read_text()

        exc = RuntimeError(error_text)

        result = classify_failure(exc, sim_dir=sim_dir, stage="NVT")
        assert result == FailureType.PHYSICS, (
            f"Expected PHYSICS, got {result}. Error text (last 300 chars): ...{error_text[-300:]}"
        )
