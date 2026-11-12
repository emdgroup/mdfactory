# ABOUTME: Tests for the per-component force-field family specs on BuildInput
# ABOUTME: Covers shorthand, per-type defaults, validation rules, persistence, and examples
"""Tests for the per-component force-field family specs on BuildInput."""

import json

import pytest
import yaml
from pydantic import ValidationError

from mdfactory.models.input import BuildInput, type_mapping
from mdfactory.models.parametrization import (
    ALL_SIMULATION_TYPES,
    DEFAULT_NAGL_MODEL,
    DEFAULT_OPENFF_FORCEFIELD,
    PROTEIN_SIMULATION_TYPES,
    SMALL_MOLECULE_SIMULATION_TYPES,
    AmberParametrization,
    CgenffSmallMolecules,
    CharmmParametrization,
    OpenFFParametrization,
    OpenFFSmallMolecules,
)
from mdfactory.models.simulation_settings import DEFAULT_PRODUCTION_LENGTH_NS
from mdfactory.prepare import load_build_input_csv, resolve_protein_pdb_path
from mdfactory.run_schedules.gromacs import COUPLING

from .conftest import EXAMPLES_DIR, PROTEIN_TYPES, protein_input_data

SMALL_MOLECULE_TYPES = ["mixedbox", "bilayer", "lnp"]

VALID_COMBINATIONS = [
    ("charmm", "mixedbox"),
    ("charmm", "bilayer"),
    ("charmm", "lnp"),
    ("charmm", "proteinbox"),
    ("charmm", "protein_mixedbox"),
    ("amber", "proteinbox"),
    ("amber", "protein_mixedbox"),
    ("openff", "mixedbox"),
    ("openff", "bilayer"),
    ("openff", "lnp"),
]

# Every component spelled out with its family default, per simulation type.
SPELLED_DEFAULTS = {
    ("charmm", "small"): {
        "family": "charmm",
        "small_molecules": {"forcefield": "cgenff"},
        "water_model": "tip3p",
    },
    ("charmm", "proteinbox"): {"family": "charmm", "protein": "charmm36m", "water_model": "tip3p"},
    ("charmm", "protein_mixedbox"): {
        "family": "charmm",
        "protein": "charmm36m",
        "small_molecules": {"forcefield": "cgenff"},
        "water_model": "tip3p",
    },
    ("amber", "proteinbox"): {"family": "amber", "protein": "amber14sb", "water_model": "tip3p"},
    ("amber", "protein_mixedbox"): {
        "family": "amber",
        "protein": "amber14sb",
        "small_molecules": {
            "forcefield": DEFAULT_OPENFF_FORCEFIELD,
            "charge_method": DEFAULT_NAGL_MODEL,
        },
        "water_model": "tip3p",
    },
    ("openff", "small"): {
        "family": "openff",
        "small_molecules": {
            "forcefield": DEFAULT_OPENFF_FORCEFIELD,
            "charge_method": DEFAULT_NAGL_MODEL,
        },
        "water_model": "opc3",
    },
}


def _small_molecule_system(simulation_type):
    """Return a valid system mapping, taken from the shipped examples."""
    example = {
        "mixedbox": "mixedbox/water_ethanol_cgenff.yaml",
        "bilayer": "bilayer/popc_cgenff.yaml",
        "lnp": "lnp/lnp_alc0315.yaml",
    }[simulation_type]
    return yaml.safe_load((EXAMPLES_DIR / example).read_text())["system"]


def _data(simulation_type, parametrization, pdb):
    if simulation_type in PROTEIN_TYPES:
        return protein_input_data(simulation_type, pdb, parametrization=parametrization)
    return {
        "simulation_type": simulation_type,
        "parametrization": parametrization,
        "system": _small_molecule_system(simulation_type),
    }


def _build(simulation_type, parametrization, pdb):
    return BuildInput(**_data(simulation_type, parametrization, pdb))


@pytest.mark.parametrize(("family", "simulation_type"), VALID_COMBINATIONS)
class TestFamilyResolution:
    def test_shorthand_equals_spelled_defaults(self, family, simulation_type, protein_pdb):
        key = (family, "small" if simulation_type in SMALL_MOLECULE_TYPES else simulation_type)
        shorthand = _build(simulation_type, family, protein_pdb)
        explicit = _build(simulation_type, {"family": family}, protein_pdb)
        spelled = _build(simulation_type, SPELLED_DEFAULTS[key], protein_pdb)

        assert shorthand.parametrization == explicit.parametrization == spelled.parametrization
        assert shorthand.hash == explicit.hash == spelled.hash

    def test_components_that_do_not_apply_are_none(self, family, simulation_type, protein_pdb):
        spec = _build(simulation_type, family, protein_pdb).parametrization
        components = spec.components()
        assert (spec.protein is not None) == (simulation_type in PROTEIN_TYPES)
        assert (spec.small_molecules is not None) == (simulation_type != "proteinbox")
        assert spec.water_model is not None
        assert components["water"] == spec.water_model
        assert components["ions"] is not None
        assert spec.summary().startswith(f"{family} (")

    def test_json_round_trip(self, family, simulation_type, protein_pdb):
        inp = _build(simulation_type, family, protein_pdb)
        again = BuildInput(**json.loads(inp.model_dump_json()))
        assert again == inp
        assert again.hash == inp.hash
        assert BuildInput.from_data_row(inp.to_data_row()).hash == inp.hash

    def test_persistence_uses_family_name(self, family, simulation_type, protein_pdb):
        inp = _build(simulation_type, family, protein_pdb)
        assert inp.to_data_row()["parametrization"] == family
        if simulation_type != "lnp":  # BuildInput.metadata does not support LNP systems yet
            assert inp.metadata["parametrization"] == family
            components = inp.parametrization.components()
            assert inp.metadata["parametrization_components"] == components
        # The full component spec is kept in the stored input JSON.
        stored = json.loads(inp.to_data_row()["input_data"])["parametrization"]
        assert stored["family"] == family


@pytest.mark.parametrize(
    ("family", "scheme"), [("charmm", "charmm"), ("amber", "sage"), ("openff", "sage")]
)
def test_nonbonded_scheme(family, scheme):
    spec = {"charmm": CharmmParametrization, "amber": AmberParametrization}.get(
        family, OpenFFParametrization
    )()
    assert spec.nonbonded_scheme == scheme


@pytest.mark.parametrize(
    ("simulation_type", "spec", "expected"),
    [
        (
            "mixedbox",
            "charmm",
            {"protein": None, "small_molecules": "cgenff", "water": "tip3p", "ions": "charmm36"},
        ),
        (
            "proteinbox",
            "charmm",
            {
                "protein": "charmm36m",
                "small_molecules": None,
                "water": "tip3p",
                "ions": "charmm36m",
            },
        ),
        (
            "protein_mixedbox",
            "amber",
            {
                "protein": "amber14sb",
                "small_molecules": f"{DEFAULT_OPENFF_FORCEFIELD} (charges: {DEFAULT_NAGL_MODEL})",
                "water": "tip3p",
                "ions": "amber14sb",
            },
        ),
        (
            "bilayer",
            {"family": "openff", "small_molecules": {"charge_method": "nagl"}},
            {
                "protein": None,
                "small_molecules": f"{DEFAULT_OPENFF_FORCEFIELD} (charges: {DEFAULT_NAGL_MODEL})",
                "water": "opc3",
                "ions": DEFAULT_OPENFF_FORCEFIELD,
            },
        ),
    ],
)
def test_component_descriptions(simulation_type, spec, expected, protein_pdb):
    assert _build(simulation_type, spec, protein_pdb).parametrization.components() == expected


def test_overrides_are_kept(protein_pdb):
    spec = _build(
        "lnp",
        {
            "family": "openff",
            "small_molecules": {"forcefield": "openff-2.1.0.offxml", "charge_method": "am1bcc"},
            "water_model": "tip3p",
        },
        protein_pdb,
    ).parametrization
    assert spec.small_molecules == OpenFFSmallMolecules(
        forcefield="openff-2.1.0.offxml", charge_method="am1bcc"
    )
    assert spec.water_model == "tip3p"


def test_charmm_proteinbox_accepts_other_charmm_forcefields_and_waters(protein_pdb):
    spec = _build(
        "proteinbox",
        {"family": "charmm", "protein": "charmm27", "water_model": "spce"},
        protein_pdb,
    ).parametrization
    assert (spec.protein, spec.water_model) == ("charmm27", "spce")


def test_resolved_spec_keeps_cgenff_small_molecules(protein_pdb):
    spec = _build("mixedbox", "charmm", protein_pdb).parametrization
    assert isinstance(spec.small_molecules, CgenffSmallMolecules)


@pytest.mark.parametrize(
    ("simulation_type", "family", "attribute", "value"),
    [
        ("mixedbox", "charmm", "water_model", "spce"),
        *[(t, f, "protein", "other") for t in PROTEIN_TYPES for f in ("charmm", "amber")],
    ],
)
def test_resolved_spec_is_frozen(simulation_type, family, attribute, value, protein_pdb):
    spec = _build(simulation_type, family, protein_pdb).parametrization
    with pytest.raises(ValidationError):
        setattr(spec, attribute, value)


# Unsupported protein force-field and water choices, rejected for both protein types.
PROTEIN_CHOICE_REJECTIONS = [
    (sim_type, {"family": family, **overrides}, match)
    for sim_type in PROTEIN_TYPES
    for family, overrides, match in [
        ("charmm", {"protein": "amber99sb-ildn"}, "CHARMM"),
        ("charmm", {"water_model": "tip4p"}, "3-site"),
        ("charmm", {"protein": "charmm36m-ljpme"}, "LJ-PME"),
        ("amber", {"protein": "amber19sb"}, "Only 'amber14sb'"),
        ("amber", {"protein": "charmm36m"}, "Only 'amber14sb'"),
        ("amber", {"water_model": "opc"}, "Only 'tip3p'"),
        ("amber", {"water_model": "spce"}, "Only 'tip3p'"),
    ]
] + [
    ("protein_mixedbox", {"family": "charmm", "protein": "charmm27"}, "only the registered"),
    ("protein_mixedbox", {"family": "charmm", "water_model": "spce"}, "only 'tip3p'"),
]


@pytest.mark.parametrize(
    ("simulation_type", "parametrization", "match"),
    [
        # Family not allowed for the simulation type.
        ("proteinbox", "openff", "family 'openff' is not valid for simulation type 'proteinbox'"),
        ("protein_mixedbox", "openff", "Allowed families: \\['amber', 'charmm'\\]"),
        ("mixedbox", "amber", "family 'amber' is not valid for simulation type 'mixedbox'"),
        ("lnp", "amber", "Allowed families: \\['charmm', 'openff'\\]"),
        ("bilayer", "amber", "family 'amber' is not valid for simulation type 'bilayer'"),
        # Components that do not apply to the simulation type.
        ("mixedbox", {"family": "charmm", "protein": "charmm36m"}, "protein does not apply"),
        (
            "proteinbox",
            {"family": "charmm", "small_molecules": {"forcefield": "cgenff"}},
            "small_molecules does not apply",
        ),
        ("bilayer", {"family": "openff", "protein": "x"}, "protein does not apply"),
        # CHARMM restrictions.
        ("proteinbox", {"family": "charmm", "protein": "amber14sb"}, "only CHARMM force fields"),
        ("proteinbox", {"family": "charmm", "water_model": "tip4p"}, "3-site"),
        ("mixedbox", {"family": "charmm", "water_model": "spce"}, "TIP3P water of its"),
        ("protein_mixedbox", {"family": "charmm", "protein": "charmm27"}, "'charmm36m' bundle"),
        ("protein_mixedbox", {"family": "charmm", "water_model": "spc"}, "only 'tip3p'"),
        (
            "mixedbox",
            {"family": "charmm", "small_molecules": {"forcefield": "openff-2.2.0.offxml"}},
            "Input should be 'cgenff'",
        ),
        # Amber restrictions.
        ("proteinbox", {"family": "amber", "protein": "amber19sb"}, "Only 'amber14sb'"),
        ("protein_mixedbox", {"family": "amber", "water_model": "opc"}, "Only 'tip3p'"),
        # OpenFF component values.
        ("mixedbox", {"family": "openff", "water_model": "opc3.offxml"}, "use 'opc3' instead"),
        (
            "mixedbox",
            {"family": "openff", "small_molecules": {"forcefield": "openff-2.2.0"}},
            "must be an .offxml file name",
        ),
        *PROTEIN_CHOICE_REJECTIONS,
    ],
)
def test_rejections(simulation_type, parametrization, match, protein_pdb):
    with pytest.raises(ValidationError, match=match):
        _build(simulation_type, parametrization, protein_pdb)


@pytest.mark.parametrize("simulation_type", ["mixedbox", "proteinbox"])
def test_parametrization_is_required(simulation_type, protein_pdb):
    data = _data(simulation_type, "charmm", protein_pdb)
    del data["parametrization"]
    with pytest.raises(ValidationError, match="parametrization\n  Field required"):
        BuildInput(**data)


@pytest.mark.parametrize("old_value", ["cgenff", "smirnoff", "pdb2gmx"])
def test_old_parametrization_values_are_rejected(old_value, protein_pdb):
    with pytest.raises(ValidationError, match=f"Input tag '{old_value}' found using 'family'"):
        _build("mixedbox", old_value, protein_pdb)


@pytest.mark.parametrize(
    ("parametrization", "extra_top_level", "match"),
    [
        ("charmm", {"parametrization_config": {"type": "cgenff"}}, "parametrization_config"),
        ({"family": "charmm", "type": "charmm"}, {}, "charmm.type"),
        ({"family": "charmm", "ignore_hydrogens": True}, {}, "charmm.ignore_hydrogens"),
        ({"family": "amber", "merge_all": True}, {}, "amber.merge_all"),
        ({"family": "amber", "sage_forcefield": "x.offxml"}, {}, "amber.sage_forcefield"),
    ],
)
def test_unknown_keys_are_rejected(parametrization, extra_top_level, match, protein_pdb):
    data = _data("proteinbox", parametrization, protein_pdb)
    data.update(extra_top_level)
    with pytest.raises(ValidationError, match=f"{match}\n  Extra inputs are not permitted"):
        BuildInput(**data)


def test_missing_family_is_rejected(protein_pdb):
    with pytest.raises(ValidationError, match="Unable to extract tag using discriminator 'family'"):
        _build("mixedbox", {"water_model": "tip3p"}, protein_pdb)


# =============================================================================
# Examples: every example YAML and CSV must parse into BuildInput.
# =============================================================================

EXAMPLE_YAMLS = sorted(EXAMPLES_DIR.rglob("*.yaml"))
EXAMPLE_CSVS = sorted(EXAMPLES_DIR.rglob("*.csv"))
BUILD_EXAMPLE_YAMLS = [p for p in EXAMPLE_YAMLS if "simulation_type" in p.read_text()]


def test_examples_are_discovered():
    assert len(BUILD_EXAMPLE_YAMLS) >= 12
    assert len(EXAMPLE_CSVS) >= 4


@pytest.mark.parametrize(
    "path", BUILD_EXAMPLE_YAMLS, ids=[str(p.relative_to(EXAMPLES_DIR)) for p in BUILD_EXAMPLE_YAMLS]
)
def test_example_yaml_parses(path):
    data = yaml.safe_load(path.read_text())
    resolve_protein_pdb_path(data, path.parent)
    inp = BuildInput(**data)
    assert inp.parametrization.family in {"charmm", "amber", "openff"}


@pytest.mark.parametrize(
    "path", EXAMPLE_CSVS, ids=[str(p.relative_to(EXAMPLES_DIR)) for p in EXAMPLE_CSVS]
)
def test_example_csv_parses(path):
    df, models, errors = load_build_input_csv(path)
    assert errors == {}
    assert len(models) == len(df) > 0
    assert {m.parametrization.family for m in models} <= {"charmm", "amber", "openff"}


def test_simulation_type_tables_cover_the_literal():
    """Per-type tables are keyed by the one SimulationType definition."""
    assert set(type_mapping) == ALL_SIMULATION_TYPES
    assert set(DEFAULT_PRODUCTION_LENGTH_NS) == ALL_SIMULATION_TYPES
    assert set(COUPLING) == ALL_SIMULATION_TYPES
    assert PROTEIN_SIMULATION_TYPES <= ALL_SIMULATION_TYPES
    assert SMALL_MOLECULE_SIMULATION_TYPES <= ALL_SIMULATION_TYPES
    assert CharmmParametrization.simulation_types == ALL_SIMULATION_TYPES
    assert OpenFFParametrization.simulation_types.isdisjoint(PROTEIN_SIMULATION_TYPES)
