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


def _proteinbox_row(tmp_path, **overrides):
    pdb = tmp_path / "prot.pdb"
    pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
    row = {
        "simulation_type": "proteinbox",
        "engine": "gromacs",
        "parametrization": "pdb2gmx",
        "parametrization_config.type": "pdb2gmx",
        "parametrization_config.forcefield": "charmm36m",
        "parametrization_config.water_model": "tip3p",
        "system.protein.resname": "LYZ",
        "system.protein.pdb_path": str(pdb),
        "system.padding": 12.0,
    }
    row.update(overrides)
    return row


def test_proteinbox_row_parses_delimited_and_dotted_fields(tmp_path):
    row = _proteinbox_row(
        tmp_path,
        **{
            "system.protein.chains": "H;L;Y",
            "system.protein.disulfide_bonds": "CYS6-CYS127;CYS30-CYS115",
            "system.protein.protonation_states.HIS15": "HIE",
        },
    )
    nested = dict_to_nested_dict_with_species_prefix(row)
    protein = nested["system"]["protein"]
    assert protein["chains"] == ["H", "L", "Y"]
    assert protein["disulfide_bonds"] == [("CYS6", "CYS127"), ("CYS30", "CYS115")]
    assert protein["protonation_states"] == {"HIS15": "HIE"}

    inp = BuildInput(**nested)
    assert inp.system.protein.chains == ["H", "L", "Y"]
    assert inp.system.protein.disulfide_bonds == [("CYS6", "CYS127"), ("CYS30", "CYS115")]
    assert inp.system.protein.protonation_states == {"HIS15": "HIE"}


def test_proteinbox_single_chain_no_optional_fields(tmp_path):
    row = _proteinbox_row(tmp_path, **{"system.protein.chains": "A"})
    inp = BuildInput(**dict_to_nested_dict_with_species_prefix(row))
    assert inp.system.protein.chains == ["A"]
    assert inp.system.protein.disulfide_bonds == []
    assert inp.system.protein.protonation_states == {}


def test_proteinbox_invalid_disulfide_cell_raises(tmp_path):
    row = _proteinbox_row(tmp_path, **{"system.protein.disulfide_bonds": "CYS6_CYS127"})
    with pytest.raises(ValueError, match="Invalid disulfide bond"):
        dict_to_nested_dict_with_species_prefix(row)


def test_df_to_models_allows_blank_optional_protein_fields(tmp_path):
    # Two proteins in one CSV: one with disulfides+protonation, one multi-chain with
    # neither. The blank optional cells become NaN but must not fail the NaN guard.
    with_extras = _proteinbox_row(
        tmp_path,
        **{
            "system.protein.chains": "A",
            "system.protein.disulfide_bonds": "CYS6-CYS127",
            "system.protein.protonation_states.HIS15": "HIE",
        },
    )
    multichain = _proteinbox_row(
        tmp_path,
        **{"system.protein.resname": "INS", "system.protein.chains": "A;B"},
    )
    df = pd.DataFrame(data=[with_extras, multichain])
    models, errors = df_to_build_input_models(df)
    assert errors == {}
    assert len(models) == 2
    assert models[0].system.protein.disulfide_bonds == [("CYS6", "CYS127")]
    assert models[1].system.protein.chains == ["A", "B"]
    assert models[1].system.protein.disulfide_bonds == []
    assert models[1].system.protein.protonation_states == {}


def test_df_to_models_still_flags_missing_required_field(tmp_path):
    good = _proteinbox_row(tmp_path, **{"system.protein.chains": "A"})
    bad = _proteinbox_row(tmp_path, **{"system.protein.chains": "A"})
    del bad["simulation_type"]
    df = pd.DataFrame(data=[good, bad])
    with pytest.raises(ValueError, match="NaN"):
        df_to_build_input_models(df)
