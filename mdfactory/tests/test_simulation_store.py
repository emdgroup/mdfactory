# ABOUTME: Tests for the SimulationStore class covering simulation querying,
# ABOUTME: filtering, iteration, and database-backed simulation access.
"""Tests for SimulationStore class."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from mdfactory.analysis.artifacts import ARTIFACT_REGISTRY
from mdfactory.analysis.store import SimulationStore
from mdfactory.models.input import BuildInput


@pytest.fixture
def mock_build_input():
    """Create mock BuildInput."""
    mock = Mock(spec=BuildInput)
    mock.hash = "ABC123"
    mock.simulation_type = "bilayer"
    mock.engine = "gromacs"
    mock.parametrization = "cgenff"
    mock.tags = None
    mock.system = Mock()
    mock.system.total_count = 600
    return mock


@pytest.fixture
def mock_discovery_df(tmp_path, mock_build_input):
    """Create mock discovery DataFrame with new schema."""
    from mdfactory.analysis.simulation import Simulation

    sim1 = tmp_path / "sim1"
    sim1.mkdir()
    sim2 = tmp_path / "sim2"
    sim2.mkdir()

    # need to create structure/trajectory files for Simulation init
    (sim1 / "system.pdb").touch()
    (sim1 / "prod.xtc").touch()
    (sim2 / "system.pdb").touch()
    (sim2 / "prod.xtc").touch()

    # Create Simulation instances
    simulation1 = Simulation(sim1, build_input=mock_build_input)
    simulation2 = Simulation(sim2, build_input=mock_build_input)

    return pd.DataFrame(
        {
            "hash": ["HASH1", "HASH2"],
            "path": [sim1, sim2],
            "simulation": [simulation1, simulation2],
        }
    )


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for analysis data."""
    return pd.DataFrame(
        {
            "time": [0, 1, 2],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_init_single_root(tmp_path):
    """Test SimulationStore initialization with single root."""
    store = SimulationStore(tmp_path)

    assert len(store.roots) == 1
    assert store.roots[0] == Path(tmp_path)


def test_init_multiple_roots(tmp_path):
    """Test SimulationStore initialization with multiple roots."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"

    store = SimulationStore([root1, root2])

    assert len(store.roots) == 2
    assert store.roots[0] == Path(root1)
    assert store.roots[1] == Path(root2)


def test_init_string_path(tmp_path):
    """Test SimulationStore accepts string path."""
    store = SimulationStore(str(tmp_path))

    assert len(store.roots) == 1
    assert isinstance(store.roots[0], Path)


def test_init_custom_filenames(tmp_path):
    """Test SimulationStore with custom trajectory/structure filenames."""
    store = SimulationStore(
        tmp_path,
        trajectory_file="custom.xtc",
        structure_file="custom.pdb",
    )

    assert store.trajectory_file == "custom.xtc"
    assert store.structure_file == "custom.pdb"


@patch("mdfactory.analysis.store.discover_simulations")
def test_discover(mock_discover, tmp_path, mock_discovery_df):
    """Test discover() calls discover_simulations and caches result."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    result = store.discover()

    # Check discover_simulations was called
    mock_discover.assert_called_once()

    # Check result is cached
    assert store._discovery_df is not None
    pd.testing.assert_frame_equal(result, mock_discovery_df)


@patch("mdfactory.analysis.store.discover_simulations")
def test_discover_caching(mock_discover, tmp_path, mock_discovery_df):
    """Test discover() uses cached result on second call."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)

    # First call
    result1 = store.discover()
    # Second call
    result2 = store.discover()

    # Should only call discover_simulations once
    assert mock_discover.call_count == 1

    # Results should be same
    pd.testing.assert_frame_equal(result1, result2)


@patch("mdfactory.analysis.store.discover_simulations")
def test_discover_refresh(mock_discover, tmp_path, mock_discovery_df):
    """Test discover(refresh=True) clears cache and re-discovers."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)

    # First call
    store.discover()
    # Note: Simulation cache is populated automatically in discover()
    # so it should have 2 entries
    assert len(store._simulations) == 2

    # Refresh
    store.discover(refresh=True)

    # Should call discover_simulations twice
    assert mock_discover.call_count == 2

    # Simulation cache should be repopulated after refresh
    assert len(store._simulations) == 2


@patch("mdfactory.analysis.store.discover_simulations")
def test_discover_multiple_roots(mock_discover, tmp_path, mock_discovery_df):
    """Test discover() with multiple roots concatenates results."""
    root1 = tmp_path / "root1"
    root1.mkdir()
    root2 = tmp_path / "root2"
    root2.mkdir()

    # Mock different results for each root
    df1 = mock_discovery_df.iloc[[0]]
    df2 = mock_discovery_df.iloc[[1]]

    mock_discover.side_effect = [df1, df2]

    store = SimulationStore([root1, root2])
    result = store.discover()

    # Should call discover_simulations twice
    assert mock_discover.call_count == 2

    # Result should contain both
    assert len(result) == 2


@patch("mdfactory.analysis.store.discover_simulations")
def test_discover_nonexistent_root_warns(mock_discover, tmp_path):
    """Test discover() warns for nonexistent root."""
    nonexistent = tmp_path / "nonexistent"

    store = SimulationStore(nonexistent)
    result = store.discover()

    # Should not call discover_simulations
    mock_discover.assert_not_called()

    # Should return empty DataFrame
    assert len(result) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_get_simulation(mock_discover, tmp_path, mock_discovery_df):
    """Test get_simulation() returns Simulation instance."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()

    sim_hash = mock_discovery_df.iloc[0]["hash"]
    sim_path = mock_discovery_df.iloc[0]["path"]
    sim = store.get_simulation(sim_hash)

    assert sim.path == sim_path.resolve()


@patch("mdfactory.analysis.store.discover_simulations")
def test_get_simulation_caching(mock_discover, tmp_path, mock_discovery_df):
    """Test get_simulation() caches Simulation instances."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()

    sim_hash = mock_discovery_df.iloc[0]["hash"]

    # Get twice
    sim1 = store.get_simulation(sim_hash)
    sim2 = store.get_simulation(sim_hash)

    # Should be same instance
    assert sim1 is sim2


@patch("mdfactory.analysis.store.discover_simulations")
def test_get_simulation_invalid_hash_raises(mock_discover, tmp_path, mock_discovery_df):
    """Test get_simulation() with invalid hash raises ValueError."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()

    with pytest.raises(ValueError, match="not found in discovered"):
        store.get_simulation("NONEXISTENT_HASH")


@patch("mdfactory.analysis.store.discover_simulations")
def test_get_simulation_auto_discovers(mock_discover, tmp_path, mock_discovery_df):
    """Test get_simulation() automatically runs discover if needed."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    # Don't call discover() explicitly

    sim_hash = mock_discovery_df.iloc[0]["hash"]
    sim_path = mock_discovery_df.iloc[0]["path"]
    sim = store.get_simulation(sim_hash)

    # Should have called discover automatically
    mock_discover.assert_called_once()
    assert sim.path == sim_path.resolve()


@patch("mdfactory.analysis.store.discover_simulations")
def test_list_simulations(mock_discover, tmp_path, mock_discovery_df):
    """Test list_simulations() returns sorted hashes."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    result = store.list_simulations()

    assert len(result) == 2
    assert all(isinstance(h, str) for h in result)
    # Check sorted
    assert result == sorted(result)


@patch("mdfactory.analysis.store.discover_simulations")
def test_list_simulations_empty(mock_discover, tmp_path):
    """Test list_simulations() with no simulations."""
    mock_discover.return_value = pd.DataFrame(columns=["hash", "path", "simulation"])

    store = SimulationStore(tmp_path)
    result = store.list_simulations()

    assert result == []


@patch("mdfactory.analysis.store.discover_simulations")
def test_build_metadata_table(mock_discover, tmp_path, mock_discovery_df):
    """Test build_metadata_table() with flatten function."""
    mock_discover.return_value = mock_discovery_df

    def flatten_fn(build_input):
        return {
            "simulation_type": build_input.simulation_type,
            "total_count": build_input.system.total_count,
        }

    store = SimulationStore(tmp_path)
    result = store.build_metadata_table(flatten_fn)

    assert len(result) == 2
    assert "path" in result.columns
    assert "hash" in result.columns
    assert "simulation_type" in result.columns
    assert "total_count" in result.columns
    assert all(result["total_count"] == 600)


@patch("mdfactory.analysis.store.discover_simulations")
def test_build_metadata_table_empty(mock_discover, tmp_path):
    """Test build_metadata_table() with no simulations."""
    mock_discover.return_value = pd.DataFrame(columns=["hash", "path", "simulation"])

    def flatten_fn(build_input):
        return {}

    store = SimulationStore(tmp_path)
    result = store.build_metadata_table(flatten_fn)

    assert len(result) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_build_metadata_table_flatten_error(mock_discover, tmp_path, mock_discovery_df):
    """Test build_metadata_table() raises if flatten function fails."""
    mock_discover.return_value = mock_discovery_df

    def flatten_fn(build_input):
        raise ValueError("Flatten failed")

    store = SimulationStore(tmp_path)

    with pytest.raises(ValueError, match="Flatten function failed"):
        store.build_metadata_table(flatten_fn)


@patch("mdfactory.analysis.store.discover_simulations")
def test_load_analysis_with_metadata(mock_discover, tmp_path, mock_discovery_df, sample_dataframe):
    """Test load_analysis_with_metadata() joins analysis with metadata."""
    mock_discover.return_value = mock_discovery_df

    def flatten_fn(build_input):
        return {"simulation_type": build_input.simulation_type}

    store = SimulationStore(tmp_path)
    store.discover()

    # Create simulations with analysis data
    for _, row in mock_discovery_df.iterrows():
        sim_hash = row["hash"]
        sim_path = row["path"]
        (sim_path / "system.pdb").touch()
        (sim_path / "prod.xtc").touch()

        sim = store.get_simulation(sim_hash)
        sim.save_analysis("test_analysis", sample_dataframe)

    # Load with metadata
    result = store.load_analysis_with_metadata("test_analysis", flatten_fn)

    # Should have data from both simulations
    assert len(result) == 6  # 3 rows * 2 simulations

    # Should have analysis columns
    assert "time" in result.columns
    assert "value" in result.columns

    # Should have metadata columns
    assert "hash" in result.columns
    assert "simulation_type" in result.columns


@patch("mdfactory.analysis.store.discover_simulations")
def test_load_analysis_with_metadata_missing_ok_true(
    mock_discover, tmp_path, mock_discovery_df, sample_dataframe
):
    """Test load_analysis_with_metadata() with missing_ok=True skips missing."""
    mock_discover.return_value = mock_discovery_df

    def flatten_fn(build_input):
        return {}

    store = SimulationStore(tmp_path)
    store.discover()

    # Create only one simulation with analysis
    sim_hash = mock_discovery_df.iloc[0]["hash"]
    sim_path = mock_discovery_df.iloc[0]["path"]
    (sim_path / "system.pdb").touch()
    (sim_path / "prod.xtc").touch()

    sim = store.get_simulation(sim_hash)
    sim.save_analysis("test_analysis", sample_dataframe)

    # Load with missing_ok=True
    result = store.load_analysis_with_metadata("test_analysis", flatten_fn, missing_ok=True)

    # Should only have data from one simulation
    assert len(result) == 3


@patch("mdfactory.analysis.store.discover_simulations")
def test_load_analysis_with_metadata_missing_ok_false(mock_discover, tmp_path, mock_discovery_df):
    """Test load_analysis_with_metadata() with missing_ok=False raises."""
    mock_discover.return_value = mock_discovery_df

    def flatten_fn(build_input):
        return {}

    store = SimulationStore(tmp_path)
    store.discover()

    # Don't create any analysis files

    # Load with missing_ok=False
    with pytest.raises(FileNotFoundError, match="not found in simulation"):
        store.load_analysis_with_metadata("test_analysis", flatten_fn, missing_ok=False)


@patch("mdfactory.analysis.store.discover_simulations")
def test_load_analysis_with_metadata_no_simulations(mock_discover, tmp_path):
    """Test load_analysis_with_metadata() with no simulations."""
    mock_discover.return_value = pd.DataFrame(columns=["hash", "path", "simulation"])

    def flatten_fn(build_input):
        return {}

    store = SimulationStore(tmp_path)
    result = store.load_analysis_with_metadata("test_analysis", flatten_fn)

    assert len(result) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_load_analysis_with_metadata_no_analyses(mock_discover, tmp_path, mock_discovery_df):
    """Test load_analysis_with_metadata() when no simulations have analysis."""
    mock_discover.return_value = mock_discovery_df

    def flatten_fn(build_input):
        return {}

    store = SimulationStore(tmp_path)
    result = store.load_analysis_with_metadata("nonexistent", flatten_fn, missing_ok=True)

    assert len(result) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_list_artifacts_status_empty(mock_discover, tmp_path, mock_discovery_df):
    """Test list_artifacts_status returns rows for available artifacts."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    result = store.list_artifacts_status()

    assert len(result) == len(mock_discovery_df) * len(ARTIFACT_REGISTRY["bilayer"])
    assert "artifact_name" in result.columns
    assert "status" in result.columns


@patch("mdfactory.analysis.store.discover_simulations")
def test_list_analyses_status_completed(mock_discover, tmp_path, mock_discovery_df):
    """Test list_analyses_status marks completed analyses."""
    mock_discover.return_value = mock_discovery_df

    for sim in mock_discovery_df["simulation"]:
        sim.list_analyses = Mock(return_value=["area_per_lipid"])

    store = SimulationStore(tmp_path)
    result = store.list_analyses_status()

    completed_rows = result[result["analysis_name"] == "area_per_lipid"]
    assert (completed_rows["status"] == "completed").all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_list_analyses_status_filtered_empty(mock_discover, tmp_path, mock_discovery_df):
    """Test list_analyses_status returns empty when filtered out."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    result = store.list_analyses_status(simulation_type="unknown")

    assert result.empty
    assert list(result.columns) == ["hash", "simulation_type", "analysis_name", "status"]


@patch("mdfactory.analysis.store.discover_simulations")
def test_run_artifacts_batch(mock_discover, tmp_path, mock_discovery_df):
    """Test run_artifacts_batch executes artifact producers."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    with patch(
        "mdfactory.analysis.simulation.Simulation.run_artifact",
        return_value=[tmp_path / "artifact.pdb"],
    ):
        summary = store.run_artifacts_batch(
            artifact_names=["last_frame_pdb"],
        )

    assert len(summary) == len(mock_discovery_df)
    assert (summary["status"] == "success").all()
    assert (summary["files"] == 1).all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_run_analyses_batch_success(mock_discover, tmp_path, mock_discovery_df, sample_dataframe):
    """Test run_analyses_batch executes analysis functions."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    with patch(
        "mdfactory.analysis.simulation.Simulation.run_analysis",
        return_value=sample_dataframe,
    ):
        summary = store.run_analyses_batch(analysis_names=["area_per_lipid"])

    assert len(summary) == len(mock_discovery_df)
    assert (summary["status"] == "success").all()
    assert (summary["rows"] == len(sample_dataframe)).all()

    store.remove_all_analyses()
    stat = store.list_analyses_status()
    assert (stat["status"] == "not yet run").all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_remove_all_analyses_summary(mock_discover, tmp_path, mock_discovery_df):
    """Test remove_all_analyses returns summary rows."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    summary = store.remove_all_analyses()

    assert len(summary) == len(mock_discovery_df)
    assert (summary["status"] == "success").all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_run_analyses_batch_failure(mock_discover, tmp_path, mock_discovery_df):
    """Test run_analyses_batch reports failures."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    with patch(
        "mdfactory.analysis.simulation.Simulation.run_analysis",
        side_effect=RuntimeError("boom"),
    ):
        summary = store.run_analyses_batch(analysis_names=["area_per_lipid"])

    assert (summary["status"] == "failed").all()
    assert summary["error"].str.contains("boom").all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_run_analyses_batch_unknown_name(mock_discover, tmp_path, mock_discovery_df):
    """Test run_analyses_batch skips unknown analyses."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    summary = store.run_analyses_batch(analysis_names=["missing_analysis"])

    assert summary.empty
    assert list(summary.columns) == [
        "hash",
        "simulation_type",
        "analysis_name",
        "status",
        "error",
        "rows",
        "duration_seconds",
    ]


@patch("mdfactory.analysis.store.discover_simulations")
def test_run_artifacts_batch_skip_existing(mock_discover, tmp_path, mock_discovery_df):
    """Test run_artifacts_batch skips existing artifacts."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    with patch(
        "mdfactory.analysis.simulation.Simulation.run_artifact",
        return_value=[tmp_path / "artifact.pdb"],
    ):
        store.run_artifacts_batch(artifact_names=["last_frame_pdb"])

    for sim in mock_discovery_df["simulation"]:
        sim.list_artifacts = Mock(return_value=["last_frame_pdb"])

    summary = store.run_artifacts_batch(artifact_names=["last_frame_pdb"])

    assert summary.empty
    assert list(summary.columns) == [
        "hash",
        "simulation_type",
        "artifact_name",
        "status",
        "error",
        "files",
        "duration_seconds",
    ]


@patch("mdfactory.analysis.store.discover_simulations")
def test_remove_all_artifacts(mock_discover, tmp_path, mock_discovery_df):
    """Test remove_all_artifacts clears artifacts across simulations."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    with patch(
        "mdfactory.analysis.simulation.Simulation.run_artifact",
        return_value=[tmp_path / "artifact.pdb"],
    ):
        store.run_artifacts_batch(artifact_names=["last_frame_pdb"])

    summary = store.remove_all_artifacts()

    assert len(summary) == len(mock_discovery_df)
    assert (summary["status"] == "success").all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_remove_all_artifacts_filtered(mock_discover, tmp_path, mock_discovery_df):
    """Test remove_all_artifacts respects simulation_type filter."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    with patch(
        "mdfactory.analysis.simulation.Simulation.run_artifact",
        return_value=[tmp_path / "artifact.pdb"],
    ):
        store.run_artifacts_batch(artifact_names=["last_frame_pdb"])

    filtered = store.remove_all_artifacts(simulation_type="mixedbox")
    assert filtered.empty

    summary = store.remove_all_artifacts(simulation_type="bilayer")
    assert len(summary) == len(mock_discovery_df)
    assert (summary["status"] == "success").all()


@patch("mdfactory.analysis.store.discover_simulations")
def test_integration_full_workflow(mock_discover, tmp_path, mock_discovery_df, sample_dataframe):
    """Test full workflow: discover, save analyses, aggregate."""
    mock_discover.return_value = mock_discovery_df

    # Initialize store
    store = SimulationStore(tmp_path)

    # Discover simulations
    discovered = store.discover()
    assert len(discovered) == 2

    # Create simulation files and save analyses
    for sim_hash in store.list_simulations():
        sim = store.get_simulation(sim_hash)
        # Create required files
        (sim.path / "system.pdb").touch()
        (sim.path / "prod.xtc").touch()

        # Save different data per simulation
        df = sample_dataframe.copy()
        df["value"] = df["value"] * (1 if sim.path.name == "sim1" else 2)
        sim.save_analysis("test_analysis", df)

    # Build metadata table
    def flatten_fn(build_input):
        return {
            "sim_type": build_input.simulation_type,
            "count": build_input.system.total_count,
        }

    metadata_df = store.build_metadata_table(flatten_fn)
    assert len(metadata_df) == 2
    assert "sim_type" in metadata_df.columns

    # Load analysis with metadata
    combined_df = store.load_analysis_with_metadata("test_analysis", flatten_fn)

    # Should have data from both simulations
    assert len(combined_df) == 6

    # Should have both analysis and metadata columns
    assert "time" in combined_df.columns
    assert "value" in combined_df.columns
    assert "hash" in combined_df.columns
    assert "sim_type" in combined_df.columns
    assert "count" in combined_df.columns

    # Verify data from different simulations
    hash1 = mock_discovery_df.iloc[0]["hash"]
    hash2 = mock_discovery_df.iloc[1]["hash"]
    sim1_data = combined_df[combined_df["hash"] == hash1]
    sim2_data = combined_df[combined_df["hash"] == hash2]
    assert len(sim1_data) == 3
    assert len(sim2_data) == 3


@patch("mdfactory.analysis.store.discover_simulations")
def test_ensure_discovered_auto_runs(mock_discover, tmp_path, mock_discovery_df):
    """Test _ensure_discovered() automatically runs discover()."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)

    # Call method that uses _ensure_discovered without calling discover first
    result = store.list_simulations()

    # Should have auto-discovered
    mock_discover.assert_called_once()
    assert len(result) == 2


@patch("mdfactory.analysis.store.discover_simulations")
def test_min_status_passthrough(mock_discover, tmp_path, mock_discovery_df):
    """Test that SimulationStore passes min_status to discover_simulations."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path, min_status="build")
    store.discover()

    # Check min_status was passed through
    call_kwargs = mock_discover.call_args[1]
    assert call_kwargs.get("min_status") == "build"


@patch("mdfactory.analysis.store.discover_simulations")
def test_min_status_default(mock_discover, tmp_path, mock_discovery_df):
    """Test that SimulationStore defaults to production min_status."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()

    call_kwargs = mock_discover.call_args[1]
    assert call_kwargs.get("min_status") == "production"


@patch("mdfactory.analysis.store.discover_simulations")
def test_metadata_table_includes_tags(mock_discover, tmp_path):
    """Test that build_metadata_table includes tag columns."""
    from mdfactory.analysis.simulation import Simulation

    # Create mock with tags
    mock_bi = Mock(spec=BuildInput)
    mock_bi.hash = "HASH1"
    mock_bi.simulation_type = "mixedbox"
    mock_bi.tags = {"project": "test", "batch": "001"}
    mock_bi.system = Mock()
    mock_bi.system.total_count = 1000

    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    sim = Simulation(sim_dir, build_input=mock_bi)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["HASH1"],
            "path": [sim_dir],
            "simulation": [sim],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    def simple_flatten(bi):
        return {"sim_type": bi.simulation_type}

    table = store.build_metadata_table(simple_flatten)

    assert "tag_project" in table.columns
    assert "tag_batch" in table.columns
    assert table.iloc[0]["tag_project"] == "test"
    assert table.iloc[0]["tag_batch"] == "001"


@patch("mdfactory.analysis.store.discover_simulations")
def test_metadata_table_no_tags(mock_discover, tmp_path):
    """Test that build_metadata_table works when tags are None."""
    from mdfactory.analysis.simulation import Simulation

    mock_bi = Mock(spec=BuildInput)
    mock_bi.hash = "HASH1"
    mock_bi.simulation_type = "mixedbox"
    mock_bi.tags = None
    mock_bi.system = Mock()
    mock_bi.system.total_count = 1000

    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    sim = Simulation(sim_dir, build_input=mock_bi)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["HASH1"],
            "path": [sim_dir],
            "simulation": [sim],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    def simple_flatten(bi):
        return {"sim_type": bi.simulation_type}

    table = store.build_metadata_table(simple_flatten)

    assert "hash" in table.columns
    assert "sim_type" in table.columns
    # No tag_ columns should be present
    tag_cols = [c for c in table.columns if c.startswith("tag_")]
    assert len(tag_cols) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_no_filters(mock_discover, tmp_path, mock_discovery_df):
    """Test search with no filters returns all simulations."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()
    results = store.search()

    assert len(results) == 2
    assert "hash" in results.columns
    assert "path" in results.columns
    assert "simulation_type" in results.columns
    assert "status" in results.columns
    assert "tags" in results.columns


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_by_simulation_type(mock_discover, tmp_path, mock_discovery_df):
    """Test search filters by simulation_type."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()

    # mock_build_input has simulation_type="bilayer"
    results = store.search(simulation_type="bilayer")
    assert len(results) == 2

    results = store.search(simulation_type="mixedbox")
    assert len(results) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_by_hash_prefix(mock_discover, tmp_path, mock_discovery_df):
    """Test search filters by hash prefix."""
    mock_discover.return_value = mock_discovery_df

    store = SimulationStore(tmp_path)
    store.discover()

    results = store.search(hash_prefix="HASH1")
    assert len(results) == 1
    assert results.iloc[0]["hash"] == "HASH1"

    results = store.search(hash_prefix="HASH")
    assert len(results) == 2

    results = store.search(hash_prefix="NONEXISTENT")
    assert len(results) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_by_tags(mock_discover, tmp_path):
    """Test search filters by tags."""
    from mdfactory.analysis.simulation import Simulation

    # Create two sims with different tags
    mock_bi1 = Mock(spec=BuildInput)
    mock_bi1.hash = "HASH1"
    mock_bi1.simulation_type = "mixedbox"
    mock_bi1.tags = {"project": "alpha", "batch": "001"}
    mock_bi1.system = Mock()
    mock_bi1.system.species = []

    mock_bi2 = Mock(spec=BuildInput)
    mock_bi2.hash = "HASH2"
    mock_bi2.simulation_type = "mixedbox"
    mock_bi2.tags = {"project": "beta"}
    mock_bi2.system = Mock()
    mock_bi2.system.species = []

    sim1_dir = tmp_path / "sim1"
    sim1_dir.mkdir()
    (sim1_dir / "system.pdb").touch()
    (sim1_dir / "prod.xtc").touch()
    sim2_dir = tmp_path / "sim2"
    sim2_dir.mkdir()
    (sim2_dir / "system.pdb").touch()
    (sim2_dir / "prod.xtc").touch()

    sim1 = Simulation(sim1_dir, build_input=mock_bi1)
    sim2 = Simulation(sim2_dir, build_input=mock_bi2)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["HASH1", "HASH2"],
            "path": [sim1_dir, sim2_dir],
            "simulation": [sim1, sim2],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    results = store.search(tags={"project": "alpha"})
    assert len(results) == 1
    assert results.iloc[0]["hash"] == "HASH1"

    results = store.search(tags={"project": "beta"})
    assert len(results) == 1
    assert results.iloc[0]["hash"] == "HASH2"

    # Multiple tag filter (AND)
    results = store.search(tags={"project": "alpha", "batch": "001"})
    assert len(results) == 1

    results = store.search(tags={"project": "alpha", "batch": "999"})
    assert len(results) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_no_tags_skipped(mock_discover, tmp_path):
    """Test that simulations without tags are excluded when filtering by tags."""
    from mdfactory.analysis.simulation import Simulation

    mock_bi = Mock(spec=BuildInput)
    mock_bi.hash = "HASH1"
    mock_bi.simulation_type = "mixedbox"
    mock_bi.tags = None
    mock_bi.system = Mock()
    mock_bi.system.species = []

    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    sim = Simulation(sim_dir, build_input=mock_bi)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["HASH1"],
            "path": [sim_dir],
            "simulation": [sim],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    results = store.search(tags={"project": "alpha"})
    assert len(results) == 0


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_combined_filters(mock_discover, tmp_path):
    """Test search with multiple filters applied together."""
    from mdfactory.analysis.simulation import Simulation

    mock_bi1 = Mock(spec=BuildInput)
    mock_bi1.hash = "HASH1"
    mock_bi1.simulation_type = "bilayer"
    mock_bi1.tags = {"project": "alpha"}
    mock_bi1.system = Mock()
    mock_bi1.system.species = []

    mock_bi2 = Mock(spec=BuildInput)
    mock_bi2.hash = "HASH2"
    mock_bi2.simulation_type = "mixedbox"
    mock_bi2.tags = {"project": "alpha"}
    mock_bi2.system = Mock()
    mock_bi2.system.species = []

    sim1_dir = tmp_path / "sim1"
    sim1_dir.mkdir()
    (sim1_dir / "system.pdb").touch()
    (sim1_dir / "prod.xtc").touch()
    sim2_dir = tmp_path / "sim2"
    sim2_dir.mkdir()
    (sim2_dir / "system.pdb").touch()
    (sim2_dir / "prod.xtc").touch()

    sim1 = Simulation(sim1_dir, build_input=mock_bi1)
    sim2 = Simulation(sim2_dir, build_input=mock_bi2)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["HASH1", "HASH2"],
            "path": [sim1_dir, sim2_dir],
            "simulation": [sim1, sim2],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    # Both match project=alpha
    results = store.search(tags={"project": "alpha"})
    assert len(results) == 2

    # Only HASH1 is bilayer
    results = store.search(tags={"project": "alpha"}, simulation_type="bilayer")
    assert len(results) == 1
    assert results.iloc[0]["hash"] == "HASH1"


@patch("mdfactory.analysis.store.discover_simulations")
def test_search_empty_store(mock_discover, tmp_path):
    """Test search on empty store returns empty DataFrame with correct columns."""
    mock_discover.return_value = pd.DataFrame(columns=["hash", "path", "simulation"])

    store = SimulationStore(tmp_path)
    store.discover()

    results = store.search()
    assert len(results) == 0
    assert list(results.columns) == ["hash", "path", "simulation_type", "status", "tags"]
