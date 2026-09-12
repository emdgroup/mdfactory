# ABOUTME: Tests for analysis utility functions including simulation discovery,
# ABOUTME: species/parameter flattening, per-frame analysis, and trajectory windowing.
"""Tests for analysis utility functions including simulation discovery,."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from mdfactory.analysis.bilayer.utils import run_per_frame_analysis, trajectory_window
from mdfactory.analysis.utils import (
    STATUS_ORDER,
    discover_simulations,
    extract_all_species,
    flatten_species_composition,
    flatten_system_parameters,
)
from mdfactory.models.species import SolutionSpecies


@pytest.fixture
def mock_build_input():
    """Create a mock BuildInput object."""
    mock = Mock()
    mock.hash = "test_hash_123"
    return mock


@pytest.fixture
def temp_simulation_dir(tmp_path):
    """Create a temporary simulation directory structure."""
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "prod.xtc").touch()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "config.yaml").write_text("test: data")
    return tmp_path


def test_discover_simulations_basic(temp_simulation_dir, mock_build_input):
    """Test basic discovery of simulations."""
    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        result = discover_simulations(temp_simulation_dir)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["hash", "path", "simulation", "status"]
        assert result.iloc[0]["hash"] == "test_hash_123"
        # Verify Simulation instance is present
        from mdfactory.analysis.simulation import Simulation

        assert isinstance(result.iloc[0]["simulation"], Simulation)


def test_discover_simulations_no_directories(tmp_path):
    """Test with no valid directories."""
    result = discover_simulations(tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_discover_simulations_missing_files(tmp_path):
    """Test directory missing required files."""
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "prod.xtc").touch()
    # Missing system.pdb

    result = discover_simulations(tmp_path)
    assert len(result) == 0


def test_discover_simulations_no_yaml(tmp_path):
    """Test directory with no YAML files."""
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "prod.xtc").touch()
    (sim_dir / "system.pdb").touch()

    result = discover_simulations(tmp_path)
    assert len(result) == 0


def test_discover_simulations_multiple_valid_yaml(temp_simulation_dir, mock_build_input):
    """Test error when multiple valid YAML files exist."""
    sim_dir = temp_simulation_dir / "sim1"
    (sim_dir / "config2.yaml").write_text("test: data2")

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        with pytest.raises(ValueError, match="Multiple valid YAML files found"):
            discover_simulations(temp_simulation_dir)


def test_discover_simulations_invalid_yaml(temp_simulation_dir):
    """Test handling of invalid YAML files."""
    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", side_effect=Exception("Invalid")),
    ):
        result = discover_simulations(temp_simulation_dir)
        assert len(result) == 0


def test_discover_simulations_custom_filenames(tmp_path, mock_build_input):
    """Test with custom trajectory and structure filenames."""
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "custom.xtc").touch()
    (sim_dir / "custom.pdb").touch()
    (sim_dir / "config.yaml").write_text("test: data")

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        # Note: status is determined by standard file names (prod.xtc/prod.gro),
        # not custom file names. Use min_status="build" to find this simulation.
        result = discover_simulations(
            tmp_path, trajectory_file="custom.xtc", structure_file="custom.pdb", min_status="build"
        )
        assert len(result) == 1


def test_discover_simulations_string_path(temp_simulation_dir, mock_build_input):
    """Test with string path instead of Path object."""
    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        result = discover_simulations(str(temp_simulation_dir))
        assert len(result) == 1


def test_discover_simulations_multiple_dirs(tmp_path, mock_build_input):
    """Test discovery across multiple simulation directories."""
    for i in range(3):
        sim_dir = tmp_path / f"sim{i}"
        sim_dir.mkdir()
        (sim_dir / "prod.xtc").touch()
        (sim_dir / "system.pdb").touch()
        (sim_dir / "config.yaml").write_text(f"test: data{i}")

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        result = discover_simulations(tmp_path)
        assert len(result) == 3


def test_flatten_species_composition() -> None:
    """Test flatten_species_composition returns expected keys."""
    species = [
        SimpleNamespace(resname="LIP", count=10, fraction=0.5),
        SimpleNamespace(resname="WAT", count=10, fraction=0.5),
    ]
    build_input = SimpleNamespace(
        simulation_type="bilayer",
        system=SimpleNamespace(total_count=20, species=species),
    )

    result = flatten_species_composition(build_input, prefix="sys_")

    assert result["simulation_type"] == "bilayer"
    assert result["total_count"] == 20
    assert result["sys_LIP_count"] == 10
    assert result["sys_LIP_fraction"] == 0.5
    assert result["sys_WAT_count"] == 10
    assert result["sys_WAT_fraction"] == 0.5


def test_extract_all_species_concentration_based_total_is_none() -> None:
    """A concentration-based species has no count, so the total is None, not a crash."""
    species = [
        SolutionSpecies(smiles="O", resname="SOL", concentration=55.0),
        SolutionSpecies(smiles="CCO", resname="ETH", count=200),
    ]
    build_input = SimpleNamespace(
        simulation_type="protein_mixedbox",
        system=SimpleNamespace(species=species),
    )

    result = extract_all_species(build_input)

    assert result["SOL_count"] is None
    assert result["ETH_count"] == 200
    assert result["total_species_count"] == 2
    assert result["total_molecule_count"] is None


def test_extract_all_species_count_based_total_sums() -> None:
    """When every species has a count, the total is their sum."""
    species = [
        SolutionSpecies(smiles="O", resname="SOL", count=10000),
        SolutionSpecies(smiles="CCO", resname="ETH", count=200),
    ]
    build_input = SimpleNamespace(
        simulation_type="protein_mixedbox",
        system=SimpleNamespace(species=species),
    )

    result = extract_all_species(build_input)

    assert result["total_molecule_count"] == 10200


def test_flatten_system_parameters() -> None:
    """Test flatten_system_parameters merges system_specific metadata."""
    build_input = SimpleNamespace(
        simulation_type="mixedbox",
        engine="gromacs",
        parametrization="cgenff",
        metadata={"system_specific": {"z_padding": 10.0, "target_density": 1.0}},
    )

    result = flatten_system_parameters(build_input)

    assert result["simulation_type"] == "mixedbox"
    assert result["engine"] == "gromacs"
    assert result["parametrization"] == "cgenff"
    assert result["z_padding"] == 10.0
    assert result["target_density"] == 1.0


# GROUP: min_status parameter tests


def test_discover_simulations_invalid_min_status(tmp_path):
    """Test ValueError raised for invalid min_status."""
    with pytest.raises(ValueError, match="Invalid min_status 'invalid'"):
        discover_simulations(tmp_path, min_status="invalid")


def test_discover_simulations_status_order_constant():
    """Test STATUS_ORDER contains expected values in correct order."""
    assert STATUS_ORDER == ["build", "equilibrated", "production", "completed"]


def test_discover_simulations_min_status_default(temp_simulation_dir, mock_build_input):
    """Test that min_status defaults to 'production' (backward compatibility)."""
    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        # Default call should find production-status simulation
        result = discover_simulations(temp_simulation_dir)
        assert len(result) == 1
        assert result.iloc[0]["status"] == "production"


def test_discover_simulations_min_status_build(tmp_path, mock_build_input):
    """Test min_status='build' includes simulations without trajectory."""
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "config.yaml").write_text("test: data")
    # No prod.xtc - this is a build-only simulation

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        result = discover_simulations(tmp_path, min_status="build")
        assert len(result) == 1
        assert result.iloc[0]["status"] == "build"


def test_discover_simulations_min_status_completed(tmp_path, mock_build_input):
    """Test min_status='completed' excludes non-completed simulations."""
    # Create a production-status simulation (has prod.xtc but no prod.gro)
    sim_dir = tmp_path / "sim_prod"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    (sim_dir / "config.yaml").write_text("test: data")

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        # Should NOT find production simulation when requiring completed
        result = discover_simulations(tmp_path, min_status="completed")
        assert len(result) == 0


def test_discover_simulations_status_column_values(tmp_path, mock_build_input):
    """Test status column returns correct status values based on files."""
    # Create a completed simulation (has prod.gro)
    sim_dir = tmp_path / "sim_completed"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    (sim_dir / "prod.gro").touch()
    (sim_dir / "config.yaml").write_text("test: data")

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        result = discover_simulations(tmp_path, min_status="completed")
        assert len(result) == 1
        assert result.iloc[0]["status"] == "completed"


def test_discover_simulations_filters_by_status_threshold(tmp_path, mock_build_input):
    """Test that simulations below min_status threshold are excluded."""
    # Create build-only simulation
    build_dir = tmp_path / "sim_build"
    build_dir.mkdir()
    (build_dir / "system.pdb").touch()
    (build_dir / "config.yaml").write_text("test: data")

    # Create production simulation
    prod_dir = tmp_path / "sim_prod"
    prod_dir.mkdir()
    (prod_dir / "system.pdb").touch()
    (prod_dir / "prod.xtc").touch()
    (prod_dir / "config2.yaml").write_text("test: data2")

    with (
        patch("mdfactory.analysis.utils.load_yaml_file", return_value={}),
        patch("mdfactory.analysis.utils.BuildInput", return_value=mock_build_input),
    ):
        # min_status="production" should only find prod_dir
        result = discover_simulations(tmp_path, min_status="production")
        assert len(result) == 1
        assert result.iloc[0]["status"] == "production"

        # min_status="build" should find both
        result_all = discover_simulations(tmp_path, min_status="build")
        assert len(result_all) == 2
        statuses = set(result_all["status"].values)
        assert statuses == {"build", "production"}


# --- trajectory_window tests ---


class _FakeTrajectory:
    """Minimal trajectory stub that supports len() and .dt."""

    def __init__(self, n_frames: int, dt_ps: float):
        self.dt = dt_ps
        self._n_frames = n_frames

    def __len__(self) -> int:
        return self._n_frames


def _fake_universe(n_frames: int, dt_ps: float = 10.0):
    """Build a minimal object that satisfies trajectory_window's interface."""
    return SimpleNamespace(trajectory=_FakeTrajectory(n_frames, dt_ps))


def test_trajectory_window_defaults() -> None:
    """No start_ns or last_ns returns the full trajectory."""
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u)
    assert start == 0
    assert stop == 100
    assert step == 1


def test_trajectory_window_start_ns() -> None:
    """start_ns crops the beginning of the trajectory."""
    # 100 frames @ 10 ps/frame = 1 ns total; start at 0.5 ns = 500 ps = frame 50
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, start_ns=0.5)
    assert start == 50
    assert stop == 100
    assert step == 1


def test_trajectory_window_last_ns() -> None:
    """last_ns selects the trailing portion of the trajectory."""
    # 100 frames @ 10 ps/frame = 1 ns; last 0.3 ns = 300 ps = 30 frames
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, last_ns=0.3)
    assert start == 70
    assert stop == 100
    assert step == 1


def test_trajectory_window_last_ns_overrides_start_ns() -> None:
    """last_ns takes precedence when both are provided."""
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, start_ns=0.1, last_ns=0.3)
    assert start == 70  # last_ns wins


def test_trajectory_window_stride() -> None:
    """stride is passed through as the step value."""
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, start_ns=0.5, stride=5)
    assert start == 50
    assert step == 5


def test_trajectory_window_start_ns_beyond_trajectory() -> None:
    """start_ns past the end clamps to total_frames."""
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, start_ns=999.0)
    assert start == 100
    assert stop == 100


def test_trajectory_window_end_ns() -> None:
    """end_ns caps the stop frame."""
    # 100 frames @ 10 ps/frame = 1 ns total; end at 0.5 ns = 500 ps = frame 50 (+1 inclusive)
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, end_ns=0.5)
    assert start == 0
    assert stop == 51
    assert step == 1


def test_trajectory_window_end_ns_none() -> None:
    """end_ns=None returns full trajectory (default)."""
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, end_ns=None)
    assert start == 0
    assert stop == 100


def test_trajectory_window_start_ns_and_end_ns() -> None:
    """start_ns and end_ns together define a sub-window."""
    # 100 frames @ 10 ps; start=0.2ns (frame 20), end=0.7ns (frame 70+1)
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, start_ns=0.2, end_ns=0.7)
    assert start == 20
    assert stop == 71
    assert step == 1


def test_trajectory_window_end_ns_beyond_trajectory() -> None:
    """end_ns past the end clamps to total_frames."""
    u = _fake_universe(100, dt_ps=10.0)
    start, stop, step = trajectory_window(u, end_ns=999.0)
    assert start == 0
    assert stop == 100


def test_run_per_frame_analysis_preserves_ragged_results() -> None:
    """Ragged per-frame outputs should be returned as a Python list."""
    mda = pytest.importorskip("MDAnalysis")

    u = mda.Universe.empty(1, trajectory=True)
    coordinates = np.zeros((3, 1, 3), dtype=float)
    u.load_new(coordinates, dt=1.0)

    def _frame_fn(atomgroup):
        frame = int(atomgroup.universe.trajectory.ts.frame)
        return list(range(frame + 1))

    timeseries = run_per_frame_analysis(
        _frame_fn,
        u.trajectory,
        u.atoms,
        backend="serial",
        n_workers=1,
    )

    assert timeseries == [[0], [0, 1], [0, 1, 2]]
