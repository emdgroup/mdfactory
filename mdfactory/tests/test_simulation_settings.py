# ABOUTME: Tests for the SimulationSettings contract, its identity rule and its supply paths
# ABOUTME: Covers defaults, validation, hashing, YAML/CSV/file/object inputs, precedence, duplicates
"""Tests for :mod:`mdfactory.models.simulation_settings` and its BuildInput integration."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from mdfactory.models.input import BuildInput, check_unique_hashes, write_build_record
from mdfactory.models.simulation_settings import (
    DEFAULT_PRODUCTION_LENGTH_NS,
    SimulationSettings,
    load_simulation_settings,
    merge_simulation_settings,
)
from mdfactory.prepare import (
    apply_simulation_settings_file,
    df_to_build_input_models,
    load_build_input_csv,
)

WATER = {
    "simulation_type": "mixedbox",
    "parametrization": "charmm",
    "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
}

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

CSV_HEADER = "simulation_type,parametrization,system.species.SOL.smiles,system.species.SOL.count"


def _water(**settings) -> BuildInput:
    return BuildInput(**WATER, simulation_settings=settings)


# --- defaults and resolution -------------------------------------------------------


@pytest.mark.parametrize("simulation_type", sorted(DEFAULT_PRODUCTION_LENGTH_NS))
def test_resolve_fills_per_type_production_length(simulation_type):
    resolved = SimulationSettings().resolve(simulation_type)
    expected = 1000.0 if simulation_type == "bilayer" else 100.0
    assert resolved.production.length == expected
    assert resolved.temperature == 300.0 and resolved.pressure == 1.0
    assert resolved.timestep == 2.0


def test_resolve_keeps_explicit_production_length():
    resolved = SimulationSettings(production={"length": 5.0}).resolve("bilayer")
    assert resolved.production.length == 5.0


def test_input_model_resolves_settings():
    assert _water().simulation_settings.production.length == 100.0
    assert BuildInput(**WATER).simulation_settings == _water().simulation_settings


# --- identity rule -------------------------------------------------------------------


def test_identity_fields_are_pinned():
    """A new settings field must be classified; only the state point is identity."""
    assert SimulationSettings.IDENTITY_FIELDS == {"temperature", "pressure"}
    assert set(SimulationSettings.model_fields) == {
        "temperature",
        "pressure",
        "timestep",
        "minimization",
        "nvt",
        "npt",
        "production",
        "gromacs_overrides",
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"production": {"length": 500.0}},
        {"production": {"output_interval": 20.0}},
        {"timestep": 1.0},
        {"nvt": {"length": 200.0, "output_interval": 2.0}},
        {"npt": {"length": 50.0}},
        {"minimization": {"max_steps": 1000, "force_tolerance": 500.0}},
        {"gromacs_overrides": {"npt": {"pcoupl": "C-rescale"}}},
    ],
)
def test_non_identity_settings_keep_hash(settings):
    assert _water(**settings).hash == _water().hash


@pytest.mark.parametrize("settings", [{"temperature": 310.0}, {"pressure": 2.0}])
def test_state_point_changes_hash(settings):
    assert _water(**settings).hash != _water().hash


def test_spelled_out_defaults_give_same_hash():
    spelled = SimulationSettings().resolve("mixedbox").model_dump(mode="json")
    assert BuildInput(**WATER).hash == _water(**spelled).hash
    # Integer and float spellings of the state point are the same identity.
    assert _water(temperature=300).hash == _water(temperature=300.0).hash


def test_metadata_and_record_carry_settings(tmp_path):
    model = _water(temperature=310.0, production={"length": 50.0})
    assert model.metadata["temperature"] == 310.0
    assert model.metadata["pressure"] == 1.0
    record = write_build_record(model, tmp_path)
    assert record.name == f"{model.hash}.yaml"
    data = yaml.safe_load(record.read_text())
    assert data["simulation_settings"]["production"] == {"length": 50.0, "output_interval": 10.0}
    reloaded = BuildInput(**data)
    assert reloaded.hash == model.hash
    assert reloaded.simulation_settings == model.simulation_settings


def test_record_round_trip_keeps_hash_of_dict_fields(tmp_path):
    """Dict fields (LNP input_fractions) keep their order, so the record keeps the hash."""
    data = yaml.safe_load((EXAMPLES / "lnp" / "lnp_alc0315_smirnoff.yaml").read_text())
    model = BuildInput(**data)
    fractions = model.system.shell.input_fractions
    assert list(fractions) != sorted(fractions)  # the case that a sorted dump breaks
    record = write_build_record(model, tmp_path)
    assert BuildInput(**yaml.safe_load(record.read_text())).hash == model.hash


# --- validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"temperature": 0.0}, "greater than 0"),
        ({"pressure": -1.0}, "greater than 0"),
        ({"timestep": 0.0}, "greater than 0"),
        ({"timestep": 4.0}, "less than or equal to 2"),
        ({"minimization": {"max_steps": 0}}, "greater than 0"),
        ({"nvt": {"length": 100.001}}, "integer multiple of the timestep"),
        ({"timestep": 1.5, "npt": {"output_interval": 1.0}}, "integer multiple"),
        ({"nvt": {"length": 1.0, "output_interval": 2.0}}, "must not exceed"),
        ({"production": {"length": 1.0, "output_interval": 30.0}}, "multiple of production"),
        ({"gromacs_overrides": {"equilibration": {"tau-p": 1.0}}}, "Extra inputs"),
        ({"unknown": 1}, "Extra inputs"),
    ],
)
def test_invalid_settings_rejected(settings, match):
    with pytest.raises(ValidationError, match=match):
        SimulationSettings(**settings)


def test_per_type_default_is_validated_on_resolve():
    """An interval that does not divide the type's default length fails at resolve time."""
    settings = SimulationSettings(production={"output_interval": 30.0})
    with pytest.raises(ValidationError, match="multiple of production"):
        settings.resolve("mixedbox")


@pytest.mark.parametrize("key", ["ref_t", "REF-T", "nsteps", "dt", "nstxtcout", "DispCorr"])
def test_override_of_typed_key_rejected(key):
    with pytest.raises(ValidationError, match="cannot be overridden"):
        SimulationSettings(gromacs_overrides={"production": {key: 1}})


@pytest.mark.parametrize("key", ["tc_grps", "rvdw-switch", "vdw_modifier", "compressibility"])
def test_override_of_derived_key_rejected(key):
    with pytest.raises(ValidationError, match="cannot be overridden"):
        SimulationSettings(gromacs_overrides={"nvt": {key: "x"}})


def test_override_spellings_of_one_key_rejected():
    with pytest.raises(ValidationError, match="name the same mdp key"):
        SimulationSettings(gromacs_overrides={"npt": {"tau-p": 1.0, "tau_p": 2.0}})


@pytest.mark.parametrize(
    "overrides", [{"bad key": 1}, {"a=b": 1}, {"pcoupl": "no ; comment"}, {"x": "a\nb"}]
)
def test_malformed_override_rejected(overrides):
    with pytest.raises(ValidationError):
        SimulationSettings(gromacs_overrides={"npt": overrides})


def test_load_settings_file(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text("temperature: 310\nproduction: {length: 10}\n")
    assert load_simulation_settings(path) == {"temperature": 310, "production": {"length": 10}}
    path.write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        load_simulation_settings(path)
    path.write_text("timestep: 3\n")
    with pytest.raises(ValidationError):
        load_simulation_settings(path)


# --- supply paths and precedence -------------------------------------------------------

SETTINGS = {
    "temperature": 310.0,
    "production": {"length": 50.0},
    "gromacs_overrides": {"npt": {"pcoupl": "C-rescale"}},
}


def test_all_supply_paths_resolve_identically(tmp_path):
    """Inline YAML, dotted CSV columns, a protocol file and a Python object agree."""
    inline = BuildInput(**{**WATER, "simulation_settings": SETTINGS})

    csv = tmp_path / "rows.csv"
    csv.write_text(
        f"{CSV_HEADER},simulation_settings.temperature,simulation_settings.production.length,"
        "simulation_settings.gromacs_overrides.npt.pcoupl\n"
        "mixedbox,charmm,O,100,310,50,C-rescale\n"
    )
    _, (from_csv,), errors = load_build_input_csv(csv)
    assert errors == {}

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump(SETTINGS))
    plain = tmp_path / "plain.csv"
    plain.write_text(f"{CSV_HEADER}\nmixedbox,charmm,O,100\n")
    _, (from_file,), _ = load_build_input_csv(plain, load_simulation_settings(protocol))

    obj = SimulationSettings(
        temperature=310.0,
        production={"length": 50.0},
        gromacs_overrides={"npt": {"pcoupl": "C-rescale"}},
    )
    from_object = BuildInput(**apply_simulation_settings_file(dict(WATER), obj))

    for model in (from_csv, from_file, from_object):
        assert model.simulation_settings == inline.simulation_settings
        assert model.hash == inline.hash


def test_precedence_row_over_file_over_defaults():
    file_settings = {
        "temperature": 310.0,
        "production": {"length": 50.0},
        "gromacs_overrides": {"npt": {"tau-p": 3.0, "pcoupl": "C-rescale"}},
    }
    row = {**WATER, "simulation_settings": {"temperature": 320.0, "npt": {"length": 20.0}}}
    row["simulation_settings"]["gromacs_overrides"] = {"npt": {"tau_p": 4.0}}
    model = BuildInput(**apply_simulation_settings_file(row, file_settings))
    settings = model.simulation_settings
    assert settings.temperature == 320.0  # row
    assert settings.production.length == 50.0  # file
    assert settings.production.output_interval == 10.0  # default
    assert settings.npt.length == 20.0 and settings.npt.output_interval == 1.0
    # The row's spelling of tau-p replaces the file's; other file overrides stay.
    assert settings.gromacs_overrides.npt == {"pcoupl": "C-rescale", "tau_p": 4.0}


def test_merge_with_object_uses_only_explicit_fields():
    merged = merge_simulation_settings({"temperature": 310.0}, SimulationSettings(pressure=2.0))
    assert merged == {"temperature": 310.0, "pressure": 2.0}


def test_csv_settings_cells_may_be_blank():
    df = pd.DataFrame(
        {
            "simulation_type": ["mixedbox", "mixedbox"],
            "parametrization": ["charmm", "charmm"],
            "system.species.SOL.smiles": ["O", "O"],
            "system.species.SOL.count": [100, 100],
            "simulation_settings.temperature": [310.0, None],
        }
    )
    models, errors = df_to_build_input_models(df, {"temperature": 290.0})
    assert errors == {}
    assert [m.simulation_settings.temperature for m in models] == [310.0, 290.0]


def test_csv_integer_override_with_blank_cells_stays_integer(tmp_path):
    # A blank cell makes pandas read the column as float; grompp rejects "100.0"
    # for integer parameters, so the CSV path must resolve like the YAML path.
    column = "simulation_settings.gromacs_overrides.npt.nstcalcenergy"
    csv = tmp_path / "rows.csv"
    csv.write_text(
        f"{CSV_HEADER},simulation_settings.temperature,{column}\n"
        "mixedbox,charmm,O,100,300,100\nmixedbox,charmm,O,100,310,\n"
    )
    _, models, errors = load_build_input_csv(csv)
    assert errors == {}
    from_yaml = _water(gromacs_overrides={"npt": {"nstcalcenergy": 100}})
    assert models[0].simulation_settings == from_yaml.simulation_settings
    assert models[1].simulation_settings.gromacs_overrides.npt == {}

    from mdfactory.run_schedules.gromacs import format_mdp_value, render_run_schedule

    assert format_mdp_value(render_run_schedule(models[0])["npt"]["nstcalcenergy"]) == "100"


# --- duplicates ------------------------------------------------------------------------


def test_duplicate_hash_with_different_settings_rejected():
    models = [_water(production={"length": 50.0}), _water(production={"length": 60.0})]
    with pytest.raises(ValueError, match="differ only in simulation settings"):
        check_unique_hashes(models, ["row 0", "row 1"])


def test_identical_inputs_rejected():
    with pytest.raises(ValueError, match="identical"):
        check_unique_hashes([_water(), _water(production={"length": 100.0})])


def test_temperature_series_accepted():
    check_unique_hashes([_water(temperature=300.0), _water(temperature=310.0)])


def test_cli_csv_duplicates_exit_before_any_system_is_made(tmp_path):
    from mdfactory.cli import _build_from_csv

    csv = tmp_path / "rows.csv"
    csv.write_text(
        f"{CSV_HEADER},simulation_settings.production.length\n"
        "mixedbox,charmm,O,100,50\nmixedbox,charmm,O,100,60\n"
    )
    with patch("mdfactory.cli.run_build_from_dict") as run, pytest.raises(SystemExit):
        _build_from_csv(csv, tmp_path / "out")
    run.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_orchestration_duplicates_raise_before_submission(tmp_path):
    from mdfactory.orchestration import build as orchestration_build
    from mdfactory.orchestration.config import ExecutorConfig

    inputs = [
        {**WATER, "simulation_settings": {"production": {"length": 50.0}}},
        {**WATER, "simulation_settings": {"production": {"length": 60.0}}},
    ]
    with (
        patch.object(orchestration_build, "get_build_app") as app,
        pytest.raises(ValueError, match="Duplicate"),
    ):
        orchestration_build.build_systems(inputs, ExecutorConfig(), output_dir=tmp_path)
    app.assert_not_called()


# --- CLI and orchestration supply ------------------------------------------------------


@pytest.fixture()
def protocol_file(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text("temperature: 310\nproduction: {length: 20}\n")
    return path


def _captured_models(command, *args, **kwargs):
    with patch("mdfactory.orchestration.build_systems", return_value=[]) as systems:
        command(*args, **kwargs)
    return systems.call_args[0][0]


def test_cli_csv_dry_run_applies_protocol_file(tmp_path, protocol_file):
    from mdfactory.cli import build_system

    csv = tmp_path / "rows.csv"
    csv.write_text(f"{CSV_HEADER},simulation_settings.temperature\nmixedbox,charmm,O,100,320\n")
    (model,) = _captured_models(
        build_system, csv, tmp_path / "out", dry_run=True, simulation_settings=protocol_file
    )
    assert model.simulation_settings.temperature == 320.0  # the row wins
    assert model.simulation_settings.production.length == 20.0  # from the file


def test_cli_yaml_dry_run_applies_protocol_file(tmp_path, protocol_file):
    from mdfactory.cli import build_system

    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(WATER))
    (model,) = _captured_models(
        build_system, path, tmp_path / "out", dry_run=True, simulation_settings=protocol_file
    )
    assert model.simulation_settings.temperature == 310.0
    assert model.simulation_settings.production.length == 20.0


def test_cli_summary_yaml_rejects_protocol_file(tmp_path, protocol_file):
    from mdfactory.cli import build_system

    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump({"system_directory": [str(tmp_path)], "hash": ["X"]}))
    with pytest.raises(SystemExit):
        build_system(summary, tmp_path / "out", simulation_settings=protocol_file)


def test_cli_invalid_protocol_file_exits(tmp_path):
    from mdfactory.cli import build_system

    bad = tmp_path / "bad.yaml"
    bad.write_text("timestep: 5\n")
    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(WATER))
    with pytest.raises(SystemExit):
        build_system(path, tmp_path / "out", dry_run=True, simulation_settings=bad)


def test_prepare_command_records_protocol_settings(tmp_path, protocol_file):
    from mdfactory.cli import prepare_build

    csv = tmp_path / "rows.csv"
    csv.write_text(f"{CSV_HEADER}\nmixedbox,charmm,O,100\n")
    with patch("mdfactory.cli.check_run_exists", return_value=False):
        prepare_build(csv, tmp_path / "out", simulation_settings=protocol_file)
    (record,) = (tmp_path / "out").glob("*/*.yaml")
    model = BuildInput(**yaml.safe_load(record.read_text()))
    assert record.stem == model.hash
    assert model.simulation_settings.temperature == 310.0
    assert model.simulation_settings.production.length == 20.0


def test_dry_run_line_shows_state_point(tmp_path):
    from loguru import logger

    from mdfactory.orchestration.build import build_systems
    from mdfactory.orchestration.config import ExecutorConfig

    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        (desc,) = build_systems(
            [WATER],
            ExecutorConfig(),
            output_dir=tmp_path,
            dry_run=True,
            simulation_settings={"temperature": 310.0, "pressure": 2.0},
        )
    finally:
        logger.remove(sink)
    assert desc["temperature"] == 310.0 and desc["pressure"] == 2.0
    line = next(message for message in messages if f"[dry-run] {desc['hash']}" in message)
    assert "| 310 K, 2 bar |" in line


def test_orchestration_rejects_settings_with_resolved_objects(tmp_path):
    from mdfactory.orchestration.build import build_systems
    from mdfactory.orchestration.config import ExecutorConfig

    with pytest.raises(ValueError, match="already resolved"):
        build_systems(
            [BuildInput(**WATER)],
            ExecutorConfig(),
            output_dir=tmp_path,
            dry_run=True,
            simulation_settings={"temperature": 310.0},
        )


def test_orchestration_sends_resolved_settings_to_worker(tmp_path):
    """The worker dict carries the merged, resolved settings and rebuilds the same hash."""
    from mdfactory.orchestration import build as orchestration_build
    from mdfactory.orchestration.config import ExecutorConfig

    sent = []
    with (
        patch.object(orchestration_build, "parsl_session"),
        patch.object(orchestration_build, "get_build_app", return_value=sent.append),
        patch.object(orchestration_build, "_wait_with_progress", return_value=[]),
    ):
        orchestration_build.build_systems(
            [WATER], ExecutorConfig(), output_dir=tmp_path, simulation_settings={"pressure": 2.0}
        )
    (payload,) = sent
    model = BuildInput(**{k: v for k, v in payload.items() if not k.startswith("_")})
    assert model.simulation_settings.pressure == 2.0
    assert payload["simulation_settings"]["production"]["length"] == 100.0
    assert payload["_build_dir"] == str(tmp_path / model.hash)
