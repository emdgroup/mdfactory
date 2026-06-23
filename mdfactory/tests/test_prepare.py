# ABOUTME: Tests for input preparation logic that converts CSV rows and dictionaries
# ABOUTME: into BuildInput models with nested species configurations.
"""Tests for input preparation logic that converts CSV rows and dictionaries."""

from copy import deepcopy

import pandas as pd
import pytest

from mdfactory.models.input import BuildInput
from mdfactory.prepare import df_to_build_input_models, dict_to_nested_dict_with_species_prefix


def test_row_to_nested_dict():
    row = {
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": 1000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.2,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.8,
    }
    ret = dict_to_nested_dict_with_species_prefix(row)
    print(ret)
    assert ret == {
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system": {
            "total_count": 1000,
            "species": [
                {"resname": "ABC", "smiles": "CCC", "fraction": 0.2},
                {"resname": "DEF", "smiles": "CCO", "fraction": 0.8},
            ],
        },
    }
    inp = BuildInput(**ret)
    print(inp)

    row2 = deepcopy(row)
    row2["system.species"] = "12"
    with pytest.raises(ValueError):
        ret = dict_to_nested_dict_with_species_prefix(row2)


def test_df_to_models():
    row1 = {
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": 1000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.2,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.8,
    }
    row2 = {
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": 1000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.2,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.8,
    }
    df_nan = pd.DataFrame(data=[row1, row2])
    print(df_nan)

    with pytest.raises(ValueError, match="NaN"):
        df_to_build_input_models(df_nan)

    df_dup = pd.DataFrame(data=[row1, row1])
    with pytest.raises(ValueError, match="duplicate"):
        df_to_build_input_models(df_dup)

    row3 = {
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": 2000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.3,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.7,
    }
    df = pd.DataFrame(data=[row1, row3])
    df_to_build_input_models(df)
    # df.to_csv("test.csv")

    row4 = {
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": -2000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.3,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.7,
    }
    df = pd.DataFrame(data=[row1, row4, row3])
    models, errors = df_to_build_input_models(df)
    assert 1 in errors

    row5 = {
        "simulation_type": "chickenburger",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": 2000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.3,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.7,
    }
    df = pd.DataFrame(data=[row1, row4, row3, row5])
    models, errors = df_to_build_input_models(df)
    print(errors)
    assert len(models) == 2
    assert 1 in errors
    assert 3 in errors
    # df.to_csv("test_errors.csv")


def test_tags_from_csv_dot_notation():
    """Test that tags.key columns in CSV produce BuildInput with tags dict."""
    row = {
        "simulation_type": "mixedbox",
        "engine": "gromacs",
        "parametrization": "cgenff",
        "system.total_count": 1000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.5,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.5,
        "tags.formulation_id": "F42",
        "tags.project": "lnp_screen",
    }
    nested = dict_to_nested_dict_with_species_prefix(row)
    assert nested["tags"] == {"formulation_id": "F42", "project": "lnp_screen"}

    inp = BuildInput(**nested)
    assert inp.tags == {"formulation_id": "F42", "project": "lnp_screen"}


def test_tags_from_csv_df():
    """Test tags columns flow through df_to_build_input_models."""
    rows = [
        {
            "simulation_type": "mixedbox",
            "engine": "gromacs",
            "parametrization": "cgenff",
            "system.total_count": 1000,
            "system.species.ABC.smiles": "CCC",
            "system.species.ABC.fraction": 0.5,
            "system.species.DEF.smiles": "CCO",
            "system.species.DEF.fraction": 0.5,
            "tags.project": "test",
        },
        {
            "simulation_type": "mixedbox",
            "engine": "gromacs",
            "parametrization": "cgenff",
            "system.total_count": 1500,
            "system.species.ABC.smiles": "CCC",
            "system.species.ABC.fraction": 0.5,
            "system.species.DEF.smiles": "CCO",
            "system.species.DEF.fraction": 0.5,
            "tags.project": "test",
        },
    ]
    df = pd.DataFrame(rows)
    models, errors = df_to_build_input_models(df)
    assert not errors
    assert len(models) == 2
    assert models[0].tags == {"project": "test"}
    assert models[1].tags == {"project": "test"}
