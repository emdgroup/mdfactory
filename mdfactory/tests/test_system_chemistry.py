# ABOUTME: Tests for the system_chemistry analysis function.
# ABOUTME: Verifies long-format species extraction and ANALYSIS_REGISTRY registration.

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from mdfactory.analysis.artifacts import ARTIFACT_REGISTRY
from mdfactory.analysis.simulation import ANALYSIS_REGISTRY
from mdfactory.analysis.utils import system_chemistry


@pytest.fixture
def mock_simulation_bilayer():
    """Create a mock Simulation with bilayer BuildInput containing 3 species."""
    species = [
        SimpleNamespace(resname="ILN", smiles="CC(=O)OC", count=150, fraction=0.25),
        SimpleNamespace(
            resname="CHL", smiles="OC1CCC2C1CCC1C3CCCCC3CCC12", count=240, fraction=0.4
        ),
        SimpleNamespace(resname="HL", smiles="CCCCCCCCCCCCCCCC(=O)OCC", count=210, fraction=0.35),
    ]
    build_input = SimpleNamespace(
        hash="ABC123",
        simulation_type="bilayer",
        system=SimpleNamespace(species=species),
    )
    return SimpleNamespace(build_input=build_input)


@pytest.fixture
def mock_simulation_mixedbox():
    """Create a mock Simulation with mixedbox BuildInput containing 2 species."""
    species = [
        SimpleNamespace(resname="butane", smiles="CCCC", count=2000, fraction=0.1),
        SimpleNamespace(resname="water", smiles="O", count=18000, fraction=0.9),
    ]
    build_input = SimpleNamespace(
        hash="DEF456",
        simulation_type="mixedbox",
        system=SimpleNamespace(species=species),
    )
    return SimpleNamespace(build_input=build_input)


@pytest.fixture
def mock_simulation_no_smiles():
    """Create a mock Simulation with base Species (no smiles attribute)."""
    species = [
        SimpleNamespace(resname="UNK", count=100, fraction=1.0),
    ]
    build_input = SimpleNamespace(
        hash="GHI789",
        simulation_type="mixedbox",
        system=SimpleNamespace(species=species),
    )
    return SimpleNamespace(build_input=build_input)


def test_system_chemistry_returns_long_format(mock_simulation_bilayer):
    """system_chemistry returns one row per species."""
    df = system_chemistry(mock_simulation_bilayer)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_system_chemistry_columns(mock_simulation_bilayer):
    """Output has exactly the expected columns."""
    df = system_chemistry(mock_simulation_bilayer)

    expected_columns = {"resname", "smiles", "count", "fraction", "simulation_type"}
    assert set(df.columns) == expected_columns


def test_system_chemistry_values(mock_simulation_bilayer):
    """Spot-check extracted values."""
    df = system_chemistry(mock_simulation_bilayer)

    iln_row = df[df["resname"] == "ILN"].iloc[0]
    assert iln_row["smiles"] == "CC(=O)OC"
    assert iln_row["count"] == 150
    assert iln_row["fraction"] == 0.25
    assert iln_row["simulation_type"] == "bilayer"


def test_system_chemistry_mixedbox(mock_simulation_mixedbox):
    """Works for mixedbox simulation type."""
    df = system_chemistry(mock_simulation_mixedbox)

    assert len(df) == 2
    assert set(df["simulation_type"]) == {"mixedbox"}
    assert set(df["resname"]) == {"butane", "water"}


def test_system_chemistry_species_without_smiles(mock_simulation_no_smiles):
    """Species without smiles attribute produces None."""
    df = system_chemistry(mock_simulation_no_smiles)

    assert len(df) == 1
    assert df.iloc[0]["smiles"] is None
    assert df.iloc[0]["resname"] == "UNK"


def test_system_chemistry_fractions(mock_simulation_bilayer):
    """Fractions are preserved correctly from species."""
    df = system_chemistry(mock_simulation_bilayer)

    assert pytest.approx(df["fraction"].sum()) == 1.0


def test_system_chemistry_registered_for_all_types():
    """system_chemistry is registered for every simulation type in ANALYSIS_REGISTRY."""
    for sim_type in ANALYSIS_REGISTRY:
        assert "system_chemistry" in ANALYSIS_REGISTRY[sim_type], (
            f"system_chemistry not registered for '{sim_type}'"
        )
        assert ANALYSIS_REGISTRY[sim_type]["system_chemistry"] is system_chemistry


def test_proteinbox_registered_in_analysis_registry():
    """proteinbox is a valid analysis target with system_chemistry."""
    assert "proteinbox" in ANALYSIS_REGISTRY
    assert ANALYSIS_REGISTRY["proteinbox"]["system_chemistry"] is system_chemistry


def test_proteinbox_registered_in_artifact_registry():
    """proteinbox can produce the last-frame PDB artifact."""
    assert "proteinbox" in ARTIFACT_REGISTRY
    assert "last_frame_pdb" in ARTIFACT_REGISTRY["proteinbox"]


def test_system_chemistry_absorbs_kwargs(mock_simulation_bilayer):
    """Extra kwargs are absorbed without error (analysis dispatch may pass them)."""
    df = system_chemistry(mock_simulation_bilayer, backend="local", n_workers=4)

    assert len(df) == 3


def _protein_mixedbox_simulation(tmp_path):
    """Build a protein_mixedbox mock Simulation with one concentration-based species."""
    species = [
        SimpleNamespace(resname="SOL", smiles="O", count=None, fraction=None, concentration=55.0)
    ]
    protein = SimpleNamespace(resname="LYZ", count=1, fraction=1.0)
    build_input = SimpleNamespace(
        simulation_type="protein_mixedbox",
        system=SimpleNamespace(species=species, protein=protein),
    )
    report = {
        "species": [
            {
                "resname": "SOL",
                "smiles": "O",
                "resolved_count": 1000,
                "final_count": 998,
                "achieved_concentration_molar": 54.8,
            }
        ]
    }
    (tmp_path / "build_metadata.json").write_text(json.dumps(report))
    return SimpleNamespace(build_input=build_input, path=tmp_path)


def test_system_chemistry_reads_protein_mixedbox_build_metadata(tmp_path):
    """Resolved counts and achieved concentrations come from the build report."""
    simulation = _protein_mixedbox_simulation(tmp_path)

    df = system_chemistry(simulation)
    row = df[df["resname"] == "SOL"].iloc[0]
    assert row["concentration"] == 55.0
    assert row["resolved_count"] == 1000
    assert row["final_count"] == 998
    assert row["achieved_concentration"] == 54.8


def test_system_chemistry_protein_mixedbox_includes_protein_row(tmp_path):
    """The protein is listed as its own species row and consumes no achieved index."""
    simulation = _protein_mixedbox_simulation(tmp_path)

    df = system_chemistry(simulation)
    assert list(df["resname"]) == ["LYZ", "SOL"]

    protein_row = df.iloc[0]
    assert protein_row["resname"] == "LYZ"
    assert protein_row["count"] == 1
    assert protein_row["fraction"] == 1.0
    assert protein_row["smiles"] is None
    # The protein has no build-resolved solution counts; pandas represents the
    # missing numeric values as NaN once the column holds the solution row's floats.
    assert pd.isna(protein_row["concentration"])
    assert pd.isna(protein_row["resolved_count"])
    assert pd.isna(protein_row["final_count"])
    assert pd.isna(protein_row["achieved_concentration"])

    # The solution row still reads its own build_metadata entry (alignment intact).
    sol_row = df[df["resname"] == "SOL"].iloc[0]
    assert sol_row["resolved_count"] == 1000
