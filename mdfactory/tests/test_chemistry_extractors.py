# ABOUTME: Tests for chemistry extraction utilities (extract_all_species,
# ABOUTME: get_chemistry_extractor) and SimulationStore convenience methods.
"""Tests for chemistry extraction utilities."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from mdfactory.analysis.utils import extract_all_species, get_chemistry_extractor
from mdfactory.models.input import BuildInput


def _make_species(resname, count, fraction, smiles=None):
    """Create a mock species object."""
    sp = Mock()
    sp.resname = resname
    sp.count = count
    sp.fraction = fraction
    sp.smiles = smiles
    return sp


def _make_build_input(species, simulation_type="bilayer"):
    """Create a mock BuildInput with the given species."""
    bi = Mock(spec=BuildInput)
    bi.simulation_type = simulation_type
    bi.system = Mock()
    bi.system.species = species
    bi.system.total_count = sum(sp.count for sp in species)
    bi.hash = "TESTHASH"
    bi.tags = None
    return bi


# ---------------------------------------------------------------------------
# extract_all_species tests
# ---------------------------------------------------------------------------


class TestExtractAllSpecies:
    """Tests for extract_all_species."""

    def test_basic_extraction(self):
        """Test extracting species with SMILES."""
        species = [
            _make_species("HL", 100, 0.5, smiles="CCCCCC"),
            _make_species("CHL", 100, 0.5, smiles="C1CCCCC1"),
        ]
        bi = _make_build_input(species)

        result = extract_all_species(bi)

        assert result["HL_count"] == 100
        assert result["HL_fraction"] == 0.5
        assert result["HL_smiles"] == "CCCCCC"
        assert result["CHL_count"] == 100
        assert result["CHL_fraction"] == 0.5
        assert result["CHL_smiles"] == "C1CCCCC1"
        assert result["total_species_count"] == 2
        assert result["total_molecule_count"] == 200

    def test_no_species(self):
        """Test extraction when no species are present."""
        bi = _make_build_input([])

        result = extract_all_species(bi)

        assert result["total_species_count"] == 0
        assert result["total_molecule_count"] == 0

    def test_species_without_smiles(self):
        """Test extraction for species lacking a smiles attribute."""
        sp = Mock()
        sp.resname = "WAT"
        sp.count = 5000
        sp.fraction = 1.0
        # Deliberately delete smiles to test getattr fallback
        del sp.smiles

        bi = _make_build_input([sp])

        result = extract_all_species(bi)

        assert result["WAT_count"] == 5000
        assert result["WAT_fraction"] == 1.0
        assert result["WAT_smiles"] is None
        assert result["total_species_count"] == 1
        assert result["total_molecule_count"] == 5000

    def test_multiple_species_with_smiles(self):
        """Test extraction with multiple species all having SMILES."""
        species = [
            _make_species("ILN", 50, 0.25, smiles="CC(=O)O"),
            _make_species("ILP", 50, 0.25, smiles="CC(=O)[O-]"),
            _make_species("HL", 60, 0.30, smiles="CCCCCC"),
            _make_species("CHL", 40, 0.20, smiles="OC1CCC2C1CCC1C3CCC(O)CC3CCC12"),
        ]
        bi = _make_build_input(species)

        result = extract_all_species(bi)

        assert result["total_species_count"] == 4
        assert result["total_molecule_count"] == 200
        assert result["ILN_smiles"] == "CC(=O)O"
        assert result["ILP_smiles"] == "CC(=O)[O-]"


# ---------------------------------------------------------------------------
# get_chemistry_extractor tests
# ---------------------------------------------------------------------------


class TestGetChemistryExtractor:
    """Tests for get_chemistry_extractor."""

    def test_mode_all(self):
        """Test mode='all' returns extract_all_species."""
        extractor = get_chemistry_extractor(mode="all")
        assert extractor is extract_all_species

    def test_mode_lnp(self):
        """Test mode='lnp' returns a working LNP extractor."""
        from mdfactory.analysis.utils import extract_lnp_chemistry

        extractor = get_chemistry_extractor(mode="lnp")
        assert extractor is extract_lnp_chemistry

    def test_mode_custom(self):
        """Test mode='custom' creates an extractor from species_groups."""
        groups = {"lipid": ["HL"], "sterol": ["CHL"]}
        extractor = get_chemistry_extractor(mode="custom", species_groups=groups)

        species = [
            _make_species("HL", 80, 0.4, smiles="CCCC"),
            _make_species("CHL", 120, 0.6, smiles="C1CCCCC1"),
        ]
        bi = _make_build_input(species)

        result = extractor(bi)

        assert result["lipid_count"] == 80
        assert result["lipid_fraction"] == 0.4
        assert result["lipid_smiles"] == "CCCC"
        assert result["sterol_count"] == 120
        assert result["sterol_fraction"] == 0.6
        assert result["sterol_smiles"] == "C1CCCCC1"

    def test_mode_custom_without_groups_raises(self):
        """Test mode='custom' without species_groups raises ValueError."""
        with pytest.raises(ValueError, match="species_groups required"):
            get_chemistry_extractor(mode="custom")

    def test_mode_invalid_raises(self):
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            get_chemistry_extractor(mode="bogus")


# ---------------------------------------------------------------------------
# SimulationStore.build_all_species_table convenience method
# ---------------------------------------------------------------------------


@patch("mdfactory.analysis.store.discover_simulations")
def test_build_all_species_table(mock_discover, tmp_path):
    """Test SimulationStore.build_all_species_table integration."""
    from mdfactory.analysis.simulation import Simulation
    from mdfactory.analysis.store import SimulationStore

    species = [
        _make_species("HL", 100, 0.5, smiles="CCCC"),
        _make_species("CHL", 100, 0.5, smiles="C1CCCCC1"),
    ]
    mock_bi = _make_build_input(species, simulation_type="bilayer")

    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    sim = Simulation(sim_dir, build_input=mock_bi)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["TESTHASH"],
            "path": [sim_dir],
            "simulation": [sim],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    table = store.build_all_species_table()

    assert len(table) == 1
    assert "HL_count" in table.columns
    assert "CHL_count" in table.columns
    assert "HL_smiles" in table.columns
    assert "total_species_count" in table.columns
    assert "total_molecule_count" in table.columns
    assert table.iloc[0]["HL_count"] == 100
    assert table.iloc[0]["CHL_smiles"] == "C1CCCCC1"


@patch("mdfactory.analysis.store.discover_simulations")
def test_build_chemistry_table_modes(mock_discover, tmp_path):
    """Test SimulationStore.build_chemistry_table with different modes."""
    from mdfactory.analysis.simulation import Simulation
    from mdfactory.analysis.store import SimulationStore

    species = [
        _make_species("HL", 60, 0.3, smiles="CCCC"),
        _make_species("CHL", 40, 0.2, smiles="OC1CC1"),
        _make_species("ILN", 50, 0.25, smiles="CC=O"),
        _make_species("ILP", 50, 0.25, smiles="CC[O-]"),
    ]
    mock_bi = _make_build_input(species, simulation_type="bilayer")

    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()
    (sim_dir / "system.pdb").touch()
    (sim_dir / "prod.xtc").touch()
    sim = Simulation(sim_dir, build_input=mock_bi)

    mock_discover.return_value = pd.DataFrame(
        {
            "hash": ["TESTHASH"],
            "path": [sim_dir],
            "simulation": [sim],
        }
    )

    store = SimulationStore(tmp_path)
    store.discover()

    # mode="all" — every species gets its own columns
    table_all = store.build_chemistry_table(mode="all")
    assert "HL_count" in table_all.columns
    assert "ILN_count" in table_all.columns
    assert "total_species_count" in table_all.columns

    # mode="lnp" — grouped columns
    table_lnp = store.build_chemistry_table(mode="lnp")
    assert "HL_count" in table_lnp.columns
    assert "CHL_count" in table_lnp.columns
    assert "IL_count" in table_lnp.columns  # merged ILN+ILP
    assert table_lnp.iloc[0]["IL_count"] == 100

    # mode="custom"
    table_custom = store.build_chemistry_table(
        mode="custom",
        species_groups={"lipids": ["HL", "CHL"]},
    )
    assert "lipids_count" in table_custom.columns
    assert table_custom.iloc[0]["lipids_count"] == 100  # 60 + 40
