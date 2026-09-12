# ABOUTME: Tests for the Simulation class covering artifact resolution, metadata
# ABOUTME: loading, trajectory access, and analysis result retrieval.
"""Tests for Simulation class."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from mdfactory.analysis.artifacts import ARTIFACT_REGISTRY
from mdfactory.analysis.simulation import ANALYSIS_REGISTRY, Simulation
from mdfactory.models.input import BuildInput


@pytest.fixture
def temp_sim_dir(tmp_path):
    """Create temporary simulation directory with required files."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    return sim_dir


@pytest.fixture
def mock_build_input_bilayer():
    """Create mock BuildInput for bilayer simulation."""
    mock = Mock(spec=BuildInput)
    mock.hash = "ABC123"
    mock.simulation_type = "bilayer"
    mock.engine = "gromacs"
    mock.parametrization = "cgenff"

    # Mock system with species
    mock.system = Mock()
    mock.system.total_count = 600
    mock.system.z_padding = 20.0
    mock.system.monolayer = False
    mock.system.ionization = Mock()
    mock.system.ionization.model_dump.return_value = {
        "neutralize": True,
        "concentration": 0.15,
    }

    # Mock species
    species1 = Mock()
    species1.resname = "ILN"
    species1.count = 150
    species1.fraction = 0.25

    species2 = Mock()
    species2.resname = "CHL"
    species2.count = 240
    species2.fraction = 0.4

    mock.system.species = [species1, species2]

    mock.model_dump_json.return_value = '{"simulation_type": "bilayer"}'

    # Mock metadata property
    mock.metadata = {
        "hash": "ABC123",
        "simulation_type": "bilayer",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "total_count": 600,
        "species_composition": [
            {"resname": "ILN", "count": 150, "fraction": 0.25},
            {"resname": "CHL", "count": 240, "fraction": 0.4},
        ],
        "system_specific": {
            "z_padding": 20.0,
            "monolayer": False,
            "ionization": {"neutralize": True, "concentration": 0.15},
        },
        "build_input_json": '{"simulation_type": "bilayer"}',
    }

    return mock


@pytest.fixture
def mock_build_input_mixedbox():
    """Create mock BuildInput for mixedbox simulation."""
    mock = Mock(spec=BuildInput)
    mock.hash = "DEF456"
    mock.simulation_type = "mixedbox"
    mock.engine = "gromacs"
    mock.parametrization = "cgenff"

    # Mock system
    mock.system = Mock()
    mock.system.total_count = 1000
    mock.system.target_density = 1.0
    mock.system.ionization = Mock()
    mock.system.ionization.model_dump.return_value = {
        "neutralize": True,
        "concentration": 0.15,
    }

    # Mock species
    species1 = Mock()
    species1.resname = "MOL1"
    species1.count = 500
    species1.fraction = 0.5

    mock.system.species = [species1]

    mock.model_dump_json.return_value = '{"simulation_type": "mixedbox"}'

    # Mock metadata property
    mock.metadata = {
        "hash": "DEF456",
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "total_count": 1000,
        "species_composition": [
            {"resname": "MOL1", "count": 500, "fraction": 0.5},
        ],
        "system_specific": {
            "target_density": 1.0,
            "ionization": {"neutralize": True, "concentration": 0.15},
        },
        "build_input_json": '{"simulation_type": "mixedbox"}',
    }

    return mock


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for analysis data."""
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_init_with_build_input(temp_sim_dir, mock_build_input_bilayer):
    """Test Simulation initialization with BuildInput provided."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    assert sim.path == temp_sim_dir.resolve()
    assert sim._build_input == mock_build_input_bilayer
    assert sim._registry is None


def test_init_without_build_input(temp_sim_dir):
    """Test Simulation initialization without BuildInput."""
    sim = Simulation(temp_sim_dir)

    assert sim.path == temp_sim_dir.resolve()
    assert sim._build_input is None


def test_init_not_directory_raises(tmp_path):
    """Test initialization with non-directory raises error."""
    file_path = tmp_path / "file.txt"
    file_path.touch()

    with pytest.raises(NotADirectoryError):
        Simulation(file_path)


def test_init_nonexistent_raises(tmp_path):
    """Test initialization with nonexistent path raises error."""
    with pytest.raises(NotADirectoryError):
        Simulation(tmp_path / "nonexistent")


def test_build_input_property_returns_existing(temp_sim_dir, mock_build_input_bilayer):
    """Test build_input property returns existing instance."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    assert sim.build_input == mock_build_input_bilayer


@patch("mdfactory.analysis.simulation.Simulation.discover_build_input")
def test_build_input_property_loads_if_none(mock_discover, temp_sim_dir, mock_build_input_bilayer):
    """Test build_input property loads from YAML if not provided."""
    mock_discover.return_value = mock_build_input_bilayer
    sim = Simulation(temp_sim_dir)

    result = sim.build_input

    assert result == mock_build_input_bilayer
    mock_discover.assert_called_once_with(temp_sim_dir.resolve())


def test_metadata_property_bilayer(temp_sim_dir, mock_build_input_bilayer):
    """Test metadata property generates correct structure for bilayer."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    metadata = sim.metadata

    assert metadata["hash"] == "ABC123"
    assert metadata["simulation_type"] == "bilayer"
    assert metadata["engine"] == "gromacs"
    assert metadata["parametrization"] == "cgenff"
    assert metadata["total_count"] == 600
    assert len(metadata["species_composition"]) == 2
    assert metadata["species_composition"][0]["resname"] == "ILN"
    assert metadata["species_composition"][0]["count"] == 150
    assert metadata["species_composition"][0]["fraction"] == 0.25
    assert metadata["system_specific"]["z_padding"] == 20.0
    assert metadata["system_specific"]["monolayer"] is False
    assert "build_input_json" in metadata


def test_metadata_property_mixedbox(temp_sim_dir, mock_build_input_mixedbox):
    """Test metadata property generates correct structure for mixedbox."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_mixedbox)

    metadata = sim.metadata

    assert metadata["simulation_type"] == "mixedbox"
    assert metadata["total_count"] == 1000
    assert metadata["system_specific"]["target_density"] == 1.0


def test_analysis_dir_property(temp_sim_dir, mock_build_input_bilayer):
    """Test analysis_dir property returns correct path."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    assert sim.analysis_dir == temp_sim_dir / ".analysis"


def test_registry_property_lazy_loads(temp_sim_dir, mock_build_input_bilayer):
    """Test registry property lazy-loads AnalysisRegistry."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    assert sim._registry is None

    registry = sim.registry

    assert sim._registry is not None
    assert registry == sim._registry


def test_registry_property_returns_same_instance(temp_sim_dir, mock_build_input_bilayer):
    """Test registry property returns same instance on multiple calls."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    registry1 = sim.registry
    registry2 = sim.registry

    assert registry1 is registry2


def test_save_metadata(temp_sim_dir, mock_build_input_bilayer):
    """Test saving metadata to disk."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    sim.save_metadata()

    metadata_path = temp_sim_dir / "metadata.json"
    assert metadata_path.exists()

    with open(metadata_path, "r") as f:
        loaded = json.load(f)

    assert loaded["hash"] == "ABC123"
    assert loaded["simulation_type"] == "bilayer"


def test_load_metadata(temp_sim_dir, mock_build_input_bilayer):
    """Test loading metadata from disk."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    # Save first
    sim.save_metadata()

    # Load back
    loaded = sim.load_metadata()

    assert loaded["hash"] == "ABC123"
    assert loaded["total_count"] == 600


def test_load_metadata_missing_raises(temp_sim_dir, mock_build_input_bilayer):
    """Test loading nonexistent metadata raises FileNotFoundError."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    with pytest.raises(FileNotFoundError):
        sim.load_metadata()


def test_save_load_metadata_roundtrip(temp_sim_dir, mock_build_input_bilayer):
    """Test save/load metadata round-trip preserves data."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    original = sim.metadata
    sim.save_metadata()
    loaded = sim.load_metadata()

    assert loaded == original


def test_save_analysis(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test saving analysis creates parquet and updates registry."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    sim.save_analysis("test_analysis", sample_dataframe)
    sim.save_analysis("test_analysis_2", sample_dataframe)

    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)
    # sim.registry.load()

    # Check parquet file exists
    parquet_path = temp_sim_dir / ".analysis" / "test_analysis.parquet"
    assert parquet_path.exists()

    # Check registry updated
    assert "test_analysis" in sim.registry.list_analyses()
    assert "test_analysis_2" in sim.registry.list_analyses()

    # Delete analysis for cleanup
    sim.remove_analysis("test_analysis")
    sim.registry.load()
    assert "test_analysis" not in sim.registry.list_analyses()
    assert not parquet_path.exists()

    sim.remove_all_analyses()
    assert sim.registry.list_analyses() == []


def test_save_analysis_creates_directory(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test save_analysis creates .analysis directory if needed."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    assert not sim.analysis_dir.exists()

    sim.save_analysis("test_analysis", sample_dataframe)

    assert sim.analysis_dir.exists()


def test_save_analysis_with_extras(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test save_analysis stores extra metadata."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    sim.save_analysis("test_analysis", sample_dataframe, custom_key="custom_value")

    entry = sim.registry.get_entry("test_analysis")
    assert entry["extras"]["custom_key"] == "custom_value"


def test_load_analysis(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test loading analysis from parquet."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    # Save first
    sim.save_analysis("test_analysis", sample_dataframe)

    # Load back
    loaded = sim.load_analysis("test_analysis")

    pd.testing.assert_frame_equal(loaded, sample_dataframe)


def test_load_analysis_missing_raises(temp_sim_dir, mock_build_input_bilayer):
    """Test loading nonexistent analysis raises FileNotFoundError."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    with pytest.raises(FileNotFoundError, match="not found"):
        sim.load_analysis("nonexistent")


def test_save_load_analysis_roundtrip(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test save/load analysis round-trip preserves data."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    sim.save_analysis("test_analysis", sample_dataframe)
    loaded = sim.load_analysis("test_analysis")

    pd.testing.assert_frame_equal(loaded, sample_dataframe)


def test_list_analyses_empty(temp_sim_dir, mock_build_input_bilayer):
    """Test list_analyses returns empty list when no analyses exist."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    assert sim.list_analyses() == []


def test_list_analyses(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test list_analyses returns analysis names."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    sim.save_analysis("analysis1", sample_dataframe)
    sim.save_analysis("analysis2", sample_dataframe)

    analyses = sim.list_analyses()

    assert sorted(analyses) == ["analysis1", "analysis2"]


def test_run_analysis_not_registered_raises(temp_sim_dir, mock_build_input_bilayer):
    """Test run_analysis with unregistered analysis raises ValueError."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    with pytest.raises(ValueError, match="not registered"):
        sim.run_analysis("nonexistent_analysis")


def test_run_analysis_generates_data(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test run_analysis executes a registered analysis."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    def mock_analysis(simulation, **kwargs):
        return sample_dataframe

    original_func = ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"]
    try:
        ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"] = mock_analysis

        result = sim.run_analysis("area_per_lipid")

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)
        assert "area_per_lipid" in sim.list_analyses()
    finally:
        ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"] = original_func


def test_run_analysis_executes_function(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test run_analysis executes and saves result."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    # Replace stub with custom function
    def mock_analysis(simulation, **kwargs):
        return sample_dataframe

    original_func = ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"]
    try:
        ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"] = mock_analysis

        result = sim.run_analysis("area_per_lipid")

        # Check result returned
        pd.testing.assert_frame_equal(result, sample_dataframe)

        # Check result saved
        assert "area_per_lipid" in sim.list_analyses()
    finally:
        # Restore original
        ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"] = original_func


def test_run_analysis_ignores_unsupported_kwargs(
    temp_sim_dir, mock_build_input_bilayer, sample_dataframe
):
    """run_analysis should drop kwargs not accepted by the analysis function."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)
    captured: dict[str, float | None] = {"start_ns": None}

    def mock_analysis(simulation, start_ns=None):
        captured["start_ns"] = start_ns
        return sample_dataframe

    original_func = ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"]
    try:
        ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"] = mock_analysis

        result = sim.run_analysis(
            "area_per_lipid",
            start_ns=50.0,
            unsupported_flag=True,
        )
        pd.testing.assert_frame_equal(result, sample_dataframe)
        assert captured["start_ns"] == 50.0

        entry = sim.registry.get_entry("area_per_lipid")
        assert entry["extras"] == {"start_ns": 50.0}
    finally:
        ANALYSIS_REGISTRY["bilayer"]["area_per_lipid"] = original_func


def test_check_integrity_valid(temp_sim_dir, mock_build_input_bilayer):
    """Test check_integrity returns valid for complete simulation."""
    # Create YAML file
    (temp_sim_dir / "test.yaml").write_text("simulation_type: bilayer\nsystem: {}")

    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)
    sim.save_metadata()

    with patch.object(Simulation, "discover_build_input", return_value=mock_build_input_bilayer):
        result = sim.check_integrity()

    assert result["valid"] is True
    assert result["missing_metadata"] is False
    assert result["missing_build_input"] is False


def test_check_integrity_missing_metadata(temp_sim_dir, mock_build_input_bilayer):
    """Test check_integrity detects missing metadata."""
    # Create YAML file
    (temp_sim_dir / "test.yaml").write_text("simulation_type: bilayer\nsystem: {}")

    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    with patch.object(Simulation, "discover_build_input", return_value=mock_build_input_bilayer):
        result = sim.check_integrity()

    assert result["valid"] is False
    assert result["missing_metadata"] is True


def test_check_integrity_missing_yaml(temp_sim_dir, mock_build_input_bilayer):
    """Test check_integrity detects missing BuildInput YAML."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)
    sim.save_metadata()

    result = sim.check_integrity()

    assert result["valid"] is False
    assert result["missing_build_input"] is True


def test_check_integrity_registry_issues(temp_sim_dir, mock_build_input_bilayer, sample_dataframe):
    """Test check_integrity detects registry issues."""
    (temp_sim_dir / "test.yaml").write_text("simulation_type: bilayer\nsystem: {}")

    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)
    sim.save_metadata()

    # Save analysis but delete parquet file
    sim.save_analysis("test_analysis", sample_dataframe)
    (sim.analysis_dir / "test_analysis.parquet").unlink()

    with patch.object(Simulation, "discover_build_input", return_value=mock_build_input_bilayer):
        result = sim.check_integrity()

    assert result["valid"] is False
    assert result["registry_issues"]["valid"] is False


def test_save_load_artifact_last_frame_pdb(temp_sim_dir, mock_build_input_bilayer):
    """Test saving and loading last-frame PDB artifact."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    mock_atoms = Mock()
    mock_atoms.write.side_effect = lambda path: Path(path).touch()
    mock_universe = Mock()
    mock_universe.trajectory = [None]
    mock_universe.atoms = mock_atoms
    with patch.object(Simulation, "universe", new=mock_universe):
        artifact_paths = sim.run_artifact("last_frame_pdb", filename="last_frame.pdb")

    assert len(artifact_paths) == 1
    assert "last_frame_pdb" in sim.list_artifacts()


def test_artifact_integrity_checksum_mismatch(temp_sim_dir, mock_build_input_bilayer):
    """Test registry integrity detects artifact checksum mismatch."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    mock_atoms = Mock()
    mock_atoms.write.side_effect = lambda path: Path(path).touch()
    mock_universe = Mock()
    mock_universe.trajectory = [None]
    mock_universe.atoms = mock_atoms
    with patch.object(Simulation, "universe", new=mock_universe):
        artifact_paths = sim.run_artifact("last_frame_pdb", filename="last_frame.pdb")

    artifact_paths[0].write_text("changed")

    result = sim.registry.check_integrity()

    assert result["valid"] is False
    assert result["artifact_checksum_mismatches"]


def test_artifact_integrity_missing_file(temp_sim_dir, mock_build_input_bilayer):
    """Test registry integrity detects missing artifact files."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    mock_atoms = Mock()
    mock_atoms.write.side_effect = lambda path: Path(path).touch()
    mock_universe = Mock()
    mock_universe.trajectory = [None]
    mock_universe.atoms = mock_atoms
    with patch.object(Simulation, "universe", new=mock_universe):
        artifact_paths = sim.run_artifact("last_frame_pdb", filename="last_frame.pdb")

    artifact_paths[0].unlink()

    result = sim.registry.check_integrity()

    assert result["valid"] is False
    assert result["artifact_missing_files"]


def test_remove_artifact(temp_sim_dir, mock_build_input_bilayer):
    """Test removing artifact files and registry entries."""
    sim = Simulation(temp_sim_dir, build_input=mock_build_input_bilayer)

    mock_atoms = Mock()
    mock_atoms.write.side_effect = lambda path: Path(path).touch()
    mock_universe = Mock()
    mock_universe.trajectory = [None]
    mock_universe.atoms = mock_atoms
    with patch.object(Simulation, "universe", new=mock_universe):
        artifact_paths = sim.run_artifact("last_frame_pdb", filename="last_frame.pdb")

    sim.remove_artifact("last_frame_pdb")

    assert "last_frame_pdb" not in sim.list_artifacts()
    assert not artifact_paths[0].exists()
    assert not (sim.artifact_dir / "last_frame_pdb").exists()


@patch("mdfactory.analysis.simulation.load_yaml_file")
def test_discover_build_input(mock_load_yaml, tmp_path, mock_build_input_bilayer):
    """Test discover_build_input static method."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "test.yaml").touch()

    mock_load_yaml.return_value = {"simulation_type": "bilayer", "system": {}}

    with patch("mdfactory.analysis.simulation.BuildInput", return_value=mock_build_input_bilayer):
        result = Simulation.discover_build_input(sim_dir)

    assert result == mock_build_input_bilayer


def test_discover_build_input_no_yaml(tmp_path):
    """Test discover_build_input raises if no YAML found."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()

    with pytest.raises(ValueError, match="No YAML files found"):
        Simulation.discover_build_input(sim_dir)


@patch("mdfactory.analysis.simulation.load_yaml_file")
def test_discover_build_input_no_valid_yaml(mock_load_yaml, tmp_path):
    """Test discover_build_input raises if no valid BuildInput."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "invalid.yaml").touch()

    mock_load_yaml.return_value = {"invalid": "data"}

    with patch("mdfactory.analysis.simulation.BuildInput", side_effect=ValueError("Invalid")):
        with pytest.raises(ValueError, match="No valid BuildInput"):
            Simulation.discover_build_input(sim_dir)


@patch("mdfactory.analysis.simulation.load_yaml_file")
def test_discover_build_input_multiple_yaml(mock_load_yaml, tmp_path, mock_build_input_bilayer):
    """Test discover_build_input raises if multiple valid YAMLs."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "test1.yaml").touch()
    (sim_dir / "test2.yaml").touch()

    mock_load_yaml.return_value = {"simulation_type": "bilayer", "system": {}}

    with patch("mdfactory.analysis.simulation.BuildInput", return_value=mock_build_input_bilayer):
        with pytest.raises(ValueError, match="Multiple valid"):
            Simulation.discover_build_input(sim_dir)


def test_discover_build_input_nonexistent(tmp_path):
    """Test discover_build_input raises for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        Simulation.discover_build_input(tmp_path / "nonexistent")


def test_simulation_with_string_path(tmp_path, mock_build_input_bilayer):
    """Test Simulation accepts string path."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()

    sim = Simulation(str(sim_dir), build_input=mock_build_input_bilayer)

    assert isinstance(sim.path, Path)
    assert sim.path == sim_dir.resolve()


def test_analysis_registry_has_bilayer_analyses():
    """Test ANALYSIS_REGISTRY includes expected bilayer analyses."""
    assert "bilayer" in ANALYSIS_REGISTRY
    assert "area_per_lipid" in ANALYSIS_REGISTRY["bilayer"]
    assert "density_distribution" in ANALYSIS_REGISTRY["bilayer"]
    assert "bilayer_thickness_map" in ANALYSIS_REGISTRY["bilayer"]
    assert "tail_order_parameter" in ANALYSIS_REGISTRY["bilayer"]


def test_analysis_registry_has_mixedbox_analyses():
    """Test ANALYSIS_REGISTRY includes expected mixedbox analyses."""
    assert "mixedbox" in ANALYSIS_REGISTRY
    assert "system_chemistry" in ANALYSIS_REGISTRY["mixedbox"]


def test_artifact_registry_has_defaults():
    """Test ARTIFACT_REGISTRY includes expected artifact entries."""
    assert "bilayer" in ARTIFACT_REGISTRY
    assert "mixedbox" in ARTIFACT_REGISTRY
    assert "protein_mixedbox" in ARTIFACT_REGISTRY
    assert "last_frame_pdb" in ARTIFACT_REGISTRY["bilayer"]
    assert "last_frame_pdb" in ARTIFACT_REGISTRY["mixedbox"]
    assert "last_frame_pdb" in ARTIFACT_REGISTRY["protein_mixedbox"]
