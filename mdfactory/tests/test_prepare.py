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
        "parametrization": "charmm",
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
        "parametrization": "charmm",
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
        "parametrization": "charmm",
        "system.total_count": 1000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.2,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.8,
    }
    row2 = {
        "engine": "gromacs",
        "parametrization": "charmm",
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
        "parametrization": "charmm",
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
        "parametrization": "charmm",
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
        "parametrization": "charmm",
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


def _proteinbox_row(pdb, **overrides):
    row = {
        "simulation_type": "proteinbox",
        "engine": "gromacs",
        "parametrization.family": "charmm",
        "parametrization.protein": "charmm36m",
        "parametrization.water_model": "tip3p",
        "system.protein.resname": "LYZ",
        "system.protein.pdb_path": str(pdb),
        "system.padding": 12.0,
    }
    row.update(overrides)
    return row


def test_proteinbox_row_parses_delimited_and_dotted_fields(protein_pdb):
    row = _proteinbox_row(
        protein_pdb,
        **{
            "system.protein.chains": "H;L;Y",
            "system.protein.disulfide_bonds": "CYS6-CYS127;CYS30-CYS115",
            "system.protein.protonation_states.HIS15": "HIE",
        },
    )
    inp = BuildInput(**dict_to_nested_dict_with_species_prefix(row))
    assert inp.system.protein.chains == ["H", "L", "Y"]
    assert inp.system.protein.disulfide_bonds == [("CYS6", "CYS127"), ("CYS30", "CYS115")]
    assert inp.system.protein.protonation_states == {"HIS15": "HIE"}


def test_proteinbox_invalid_disulfide_cell_raises(protein_pdb):
    row = _proteinbox_row(protein_pdb, **{"system.protein.disulfide_bonds": "CYS6_CYS127"})
    with pytest.raises(ValueError, match="Invalid disulfide bond"):
        dict_to_nested_dict_with_species_prefix(row)


def test_df_to_models_allows_blank_optional_protein_fields(protein_pdb):
    # Two proteins in one CSV: one single-chain with disulfides+protonation, one
    # multi-chain with neither. The blank optional cells become NaN but must not
    # fail the NaN guard.
    with_extras = _proteinbox_row(
        protein_pdb,
        **{
            "system.protein.chains": "A",
            "system.protein.disulfide_bonds": "CYS6-CYS127",
            "system.protein.protonation_states.HIS15": "HIE",
        },
    )
    multichain = _proteinbox_row(
        protein_pdb,
        **{"system.protein.resname": "INS", "system.protein.chains": "A;B"},
    )
    df = pd.DataFrame(data=[with_extras, multichain])
    models, errors = df_to_build_input_models(df)
    assert errors == {}
    assert len(models) == 2
    assert models[0].system.protein.chains == ["A"]
    assert models[0].system.protein.disulfide_bonds == [("CYS6", "CYS127")]
    assert models[1].system.protein.chains == ["A", "B"]
    assert models[1].system.protein.disulfide_bonds == []
    assert models[1].system.protein.protonation_states == {}


def test_df_to_models_still_flags_missing_required_field(protein_pdb):
    good = _proteinbox_row(protein_pdb, **{"system.protein.chains": "A"})
    bad = dict(good)
    del bad["simulation_type"]
    df = pd.DataFrame(data=[good, bad])
    with pytest.raises(ValueError, match="NaN"):
        df_to_build_input_models(df)


def test_proteinbox_row_with_dotted_parametrization_and_prep_options(protein_pdb):
    row = _proteinbox_row(
        protein_pdb,
        **{"system.protein.merge_all": True, "system.protein.ignore_hydrogens": False},
    )
    inp = BuildInput(**dict_to_nested_dict_with_species_prefix(row))
    assert inp.parametrization.family == "charmm"
    assert (inp.parametrization.protein, inp.parametrization.water_model) == (
        "charmm36m",
        "tip3p",
    )
    assert inp.system.protein.merge_all is True
    assert inp.system.protein.ignore_hydrogens is False


def _mixedbox_row(**overrides):
    row = {
        "simulation_type": "mixedbox",
        "system.total_count": 1000,
        "system.species.ABC.smiles": "CCC",
        "system.species.ABC.fraction": 0.2,
        "system.species.DEF.smiles": "CCO",
        "system.species.DEF.fraction": 0.8,
    }
    row.update(overrides)
    return row


def test_mixedbox_rows_with_shorthand_and_dotted_parametrization():
    shorthand = _mixedbox_row(parametrization="openff")
    dotted = _mixedbox_row(
        **{
            "parametrization.family": "openff",
            "parametrization.small_molecules.charge_method": "am1bcc",
            "parametrization.water_model": "tip3p",
        }
    )
    short_inp = BuildInput(**dict_to_nested_dict_with_species_prefix(shorthand))
    dotted_inp = BuildInput(**dict_to_nested_dict_with_species_prefix(dotted))

    assert short_inp.parametrization.small_molecules.charge_method != "am1bcc"
    assert dotted_inp.parametrization.small_molecules.charge_method == "am1bcc"
    assert dotted_inp.parametrization.water_model == "tip3p"
    assert short_inp.hash != dotted_inp.hash
    # The shorthand and the fully spelled family defaults are the same build.
    spelled = _mixedbox_row(
        **{
            "parametrization.family": "openff",
            "parametrization.small_molecules.forcefield": "openff-2.2.0.offxml",
            "parametrization.water_model": "opc3",
        }
    )
    assert BuildInput(**dict_to_nested_dict_with_species_prefix(spelled)).hash == short_inp.hash
