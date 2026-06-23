# ABOUTME: Tests for the CLI search command
# ABOUTME: Validates search flag parsing and output formatting
"""Tests for CLI search command."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mdfactory import cli


def test_search_no_results(tmp_path, capsys):
    """Test search command with no results prints message."""
    with patch.object(cli, "SimulationStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.search.return_value = pd.DataFrame(
            columns=["hash", "path", "simulation_type", "status", "tags"]
        )

        cli.search_simulations(tmp_path)

        captured = capsys.readouterr()
        assert "No simulations found" in captured.out


def test_search_with_results(tmp_path, capsys):
    """Test search command prints results table."""
    with patch.object(cli, "SimulationStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.search.return_value = pd.DataFrame(
            {
                "hash": ["ABC123DEF456"],
                "path": [Path("/tmp/sim1")],
                "simulation_type": ["mixedbox"],
                "status": ["production"],
                "tags": [{"project": "test"}],
            }
        )

        cli.search_simulations(tmp_path)

        captured = capsys.readouterr()
        assert "ABC123DEF456"[:12] in captured.out
        assert "mixedbox" in captured.out
        assert "project=test" in captured.out


def test_search_tag_parsing(tmp_path, capsys):
    """Test that --tag key=value is parsed correctly."""
    with patch.object(cli, "SimulationStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store.search.return_value = pd.DataFrame(
            columns=["hash", "path", "simulation_type", "status", "tags"]
        )

        cli.search_simulations(
            tmp_path,
            tag=["project=alpha", "batch=001"],
        )

        mock_store.search.assert_called_once_with(
            simulation_type=None,
            status=None,
            hash_prefix=None,
            tags={"project": "alpha", "batch": "001"},
            smiles=None,
        )


def test_search_invalid_tag_format(tmp_path):
    """Test that invalid tag format raises SystemExit."""
    with patch.object(cli, "SimulationStore"):
        with pytest.raises(SystemExit):
            cli.search_simulations(tmp_path, tag=["invalid_no_equals"])
