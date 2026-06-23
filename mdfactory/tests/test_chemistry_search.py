# ABOUTME: Tests for SMILES substructure matching utility
# ABOUTME: Validates chemical substructure search for simulation filtering
"""Tests for SMILES substructure matching."""

import pytest

from mdfactory.utils.chemistry_utilities import smiles_substructure_match


def test_substructure_match_basic():
    """Test basic substructure matching."""
    # Ethanol is substructure of propanol
    assert smiles_substructure_match("CCO", "CCCO")
    # Methanol in ethanol
    assert smiles_substructure_match("CO", "CCO")
    # Exact match
    assert smiles_substructure_match("CCO", "CCO")


def test_substructure_no_match():
    """Test non-matching substructures."""
    # Benzene not in propane
    assert not smiles_substructure_match("c1ccccc1", "CCC")
    # Nitrogen not in ethanol
    assert not smiles_substructure_match("N", "CCO")


def test_substructure_match_ring():
    """Test ring substructure matching."""
    # Benzene in toluene
    assert smiles_substructure_match("c1ccccc1", "Cc1ccccc1")


def test_substructure_invalid_smiles():
    """Test that invalid SMILES raises ValueError."""
    with pytest.raises(ValueError, match="Invalid query SMILES"):
        smiles_substructure_match("INVALID_SMILES_XYZ", "CCO")

    with pytest.raises(ValueError, match="Invalid target SMILES"):
        smiles_substructure_match("CCO", "INVALID_SMILES_XYZ")
