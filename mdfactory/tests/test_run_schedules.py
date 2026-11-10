# ABOUTME: Tests for the generated GROMACS run schedules and the re-render of existing directories
# ABOUTME: Checks files, orchestration contract, settings propagation, nonbonded table, invariants
"""Tests for :mod:`mdfactory.run_schedules`."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mdfactory.models.input import BuildInput, write_build_record
from mdfactory.models.parametrization import (
    DEFAULT_OPENFF_FORCEFIELD,
    AmberParametrization,
    CharmmParametrization,
    OpenFFParametrization,
)
from mdfactory.models.simulation_settings import STAGES, SimulationSettings
from mdfactory.orchestration.mdp import apply_rescue_tier, get_mdp_value, parse_mdp
from mdfactory.orchestration.simulate import _missing_build_files
from mdfactory.orchestration.stages import STAGE_REGISTRY
from mdfactory.orchestration.trajectory import _extract_expected_frames_from_mdp
from mdfactory.run_schedules import (
    STAGE_FILES,
    apply_simulation_settings,
    render_run_schedule,
    write_mdp_files,
)

# Every simulation type with each family that supports it (both nonbonded schemes).
COMBINATIONS = [
    ("mixedbox", CharmmParametrization),
    ("mixedbox", OpenFFParametrization),
    ("bilayer", CharmmParametrization),
    ("bilayer", OpenFFParametrization),
    ("lnp", CharmmParametrization),
    ("lnp", OpenFFParametrization),
    ("proteinbox", CharmmParametrization),
    ("proteinbox", AmberParametrization),
    ("protein_mixedbox", CharmmParametrization),
    ("protein_mixedbox", AmberParametrization),
]
PROTEIN_TYPES = ("proteinbox", "protein_mixedbox")


def _inp(simulation_type, family=CharmmParametrization, **settings):
    return SimpleNamespace(
        simulation_type=simulation_type,
        parametrization=family().resolve(simulation_type),
        simulation_settings=SimulationSettings(**settings).resolve(simulation_type),
    )


def _read(path: Path) -> dict[str, str]:
    """Return the file's ``normalized key -> value`` pairs, rejecting duplicate keys."""
    pairs = [(key, value) for _, key, value in parse_mdp(path) if key is not None]
    keys = [key.replace("_", "") for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate key in {path.name}"
    return dict(pairs)


def _written(tmp_path, simulation_type, family=CharmmParametrization, **settings):
    files = write_mdp_files(_inp(simulation_type, family, **settings), tmp_path)
    return {stage: _read(path) for stage, path in files.items()}


def _tokens(value: str) -> list:
    """Split a value into tokens, numbers as floats and words lowercased."""
    out = []
    for token in value.split():
        try:
            out.append(float(token))
        except ValueError:
            out.append(token.lower())
    return out


# --- files ------------------------------------------------------------------------------


def test_stage_files_match_orchestration_stages():
    assert [STAGE_FILES[stage] for stage in STAGES] == [s.mdp_file for s in STAGE_REGISTRY]


@pytest.mark.parametrize(("simulation_type", "family"), COMBINATIONS)
def test_writes_four_parseable_files_with_header(tmp_path, simulation_type, family):
    (tmp_path / "em.mdp").write_text("stale = 1\n")
    files = write_mdp_files(_inp(simulation_type, family), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(s.mdp_file for s in STAGE_REGISTRY)
    scheme = family().nonbonded_scheme
    for stage, path in files.items():
        text = path.read_text()
        header = [line for line in text.splitlines() if line.startswith(";")]
        assert "mdfactory" in header[0] and stage in header[0]
        assert simulation_type in header[1] and scheme in header[1]
        assert "300 K, 1 bar" in header[2]
        assert "stale" not in text
        values = _read(path)
        assert "title" not in values and "ns_type" not in values


# --- orchestration contract (unchanged consumers) ---------------------------------------


def test_rescue_tier_on_generated_files(tmp_path):
    write_mdp_files(_inp("mixedbox"), tmp_path)
    em = parse_mdp(apply_rescue_tier(tmp_path, "EM", 1))
    assert get_mdp_value(em, "emstep") == "0.005"
    assert get_mdp_value(em, "nsteps") == "100000"
    nvt = parse_mdp(apply_rescue_tier(tmp_path, "NVT", 2))
    assert get_mdp_value(nvt, "dt") == "0.0005"
    assert get_mdp_value(nvt, "nsteps") == "200000"


@pytest.mark.parametrize(
    ("settings", "frames"),
    [({}, 10_000), ({"production": {"length": 2.0, "output_interval": 4.0}}, 500)],
)
def test_expected_frames_follow_settings(tmp_path, settings, frames):
    write_mdp_files(_inp("mixedbox", **settings), tmp_path)
    assert _extract_expected_frames_from_mdp(tmp_path, "Production") == frames


def test_required_files_present(tmp_path):
    write_mdp_files(_inp("bilayer"), tmp_path)
    (tmp_path / "system.pdb").touch()
    (tmp_path / "topology.top").touch()
    assert _missing_build_files(tmp_path, [s.name for s in STAGE_REGISTRY]) == []


# --- settings propagation ---------------------------------------------------------------


@pytest.mark.parametrize("simulation_type", ["mixedbox", "bilayer", "proteinbox"])
def test_state_point_reaches_every_coupling_key(tmp_path, simulation_type):
    stages = _written(tmp_path, simulation_type, temperature=310.0, pressure=2.5)
    for stage in ("nvt", "npt", "production"):
        assert set(_tokens(stages[stage]["ref_t"])) == {310.0}
    assert _tokens(stages["nvt"]["gen_temp"]) == [310.0]
    n_pressure = 2 if simulation_type == "bilayer" else 1
    for stage in ("npt", "production"):
        assert _tokens(stages[stage]["ref_p"]) == [2.5] * n_pressure
        assert len(_tokens(stages[stage]["compressibility"])) == n_pressure


def test_lengths_and_intervals_become_exact_step_counts(tmp_path):
    stages = _written(
        tmp_path,
        "mixedbox",
        timestep=1.0,
        minimization={"max_steps": 1234, "force_tolerance": 500.0},
        nvt={"length": 50.0, "output_interval": 0.5},
        npt={"length": 20.0, "output_interval": 2.0},
        production={"length": 3.0, "output_interval": 6.0},
    )
    assert stages["minimization"]["nsteps"] == "1234"
    assert float(stages["minimization"]["emtol"]) == 500.0
    assert stages["nvt"]["dt"] == "0.001"
    assert (stages["nvt"]["nsteps"], stages["nvt"]["nstxout"]) == ("50000", "500")
    assert (stages["npt"]["nsteps"], stages["npt"]["nstenergy"]) == ("20000", "2000")
    md = stages["production"]
    assert (md["nsteps"], md["nstxout_compressed"], md["nstlog"]) == ("3000000", "6000", "6000")


def test_overrides_are_applied_last(tmp_path):
    stages = _written(
        tmp_path,
        "bilayer",
        gromacs_overrides={
            "npt": {"pcoupl": "C-rescale", "tau_p": 1.5},
            "production": {"nstcalcenergy": 50, "gen_vel": False},
        },
    )
    assert stages["npt"]["pcoupl"] == "C-rescale"
    assert stages["npt"]["tau_p"] == "1.5"  # replaces the generated tau-p, no duplicate
    assert stages["production"]["nstcalcenergy"] == "50"
    assert stages["production"]["gen_vel"] == "no"
    assert stages["nvt"]["pcoupl"] == "no"


# --- nonbonded table and protocol invariants --------------------------------------------

NONBONDED = {
    "charmm": {
        "coulombtype": "pme",
        "rcoulomb": "1.2",
        "vdwtype": "cutoff",
        "vdw_modifier": "force-switch",
        "rvdw": "1.2",
        "rvdw_switch": "1.0",
    },
    "sage": {
        "coulombtype": "pme",
        "rcoulomb": "0.9",
        "vdwtype": "cutoff",
        "vdw_modifier": "potential-switch",
        "rvdw": "0.9",
        "rvdw_switch": "0.8",
    },
}
DISPCORR = {
    ("charmm", "mixedbox"): "enerpres",
    ("charmm", "lnp"): "enerpres",
    ("charmm", "bilayer"): "no",
    ("charmm", "proteinbox"): "no",
    ("charmm", "protein_mixedbox"): "no",
}


@pytest.mark.parametrize(("simulation_type", "family"), COMBINATIONS)
def test_nonbonded_block_follows_family_in_all_stages(tmp_path, simulation_type, family):
    scheme = family().nonbonded_scheme
    dispcorr = DISPCORR.get((scheme, simulation_type), "enerpres")
    expected = {**NONBONDED[scheme], "dispcorr": dispcorr}
    for stage, values in _written(tmp_path, simulation_type, family).items():
        assert {key: values[key].lower() for key in expected} == expected, stage


@pytest.mark.parametrize(("simulation_type", "family"), COMBINATIONS)
def test_protocol_invariants(tmp_path, simulation_type, family):
    stages = _written(tmp_path, simulation_type, family)
    protein = simulation_type in PROTEIN_TYPES
    groups = stages["nvt"]["tc_grps"].split()
    assert groups == (["Protein", "Non-Protein"] if protein else ["System"])
    for stage in ("nvt", "npt", "production"):
        assert stages[stage]["tc_grps"].split() == groups
        assert len(stages[stage]["tau_t"].split()) == len(groups)
        assert len(stages[stage]["ref_t"].split()) == len(groups)
    assert [stage for stage in STAGES if stages[stage].get("define") == "-DPOSRES"] == [
        "nvt",
        "npt",
    ]
    coupling = "semi-isotropic" if simulation_type == "bilayer" else "isotropic"
    assert stages["npt"]["pcoupltype"] == stages["production"]["pcoupltype"] == coupling
    assert stages["npt"]["pcoupl"] == "Berendsen"
    assert stages["production"]["pcoupl"] == ("Parrinello-Rahman" if protein else "Berendsen")
    assert stages["nvt"]["gen_vel"] == "yes" and stages["npt"]["gen_vel"] == "no"
    assert stages["production"]["nstxout"] == stages["production"]["nstvout"] == "0"


@pytest.mark.parametrize("family", [CharmmParametrization, OpenFFParametrization])
def test_lnp_runs_the_mixedbox_protocol(family):
    lnp, mixedbox = _inp("lnp", family), _inp("mixedbox", family)
    assert render_run_schedule(lnp) == render_run_schedule(mixedbox)


# The protein schedules the issue names, from the static files they replace
# (obsolete title/ns_type dropped; the nonbonded block is checked above).
PROTEIN_DYNAMICS = {
    "integrator": "md",
    "dt": "0.002",
    "continuation": "yes",
    "constraint_algorithm": "lincs",
    "constraints": "h-bonds",
    "lincs_iter": "1",
    "lincs_order": "4",
    "cutoff_scheme": "Verlet",
    "nstlist": "10",
    "pbc": "xyz",
    "pme_order": "4",
    "fourierspacing": "0.16",
    "tcoupl": "V-rescale",
    "tc_grps": "Protein Non-Protein",
    "tau_t": "0.1 0.1",
    "ref_t": "300 300",
    "gen_vel": "no",
}
PROTEIN_EQUILIBRATION = {
    "define": "-DPOSRES",
    "nsteps": "50000",
    "nstxout": "500",
    "nstvout": "500",
    "nstenergy": "500",
    "nstlog": "500",
}
PROTEIN_PRESSURE = {"pcoupltype": "isotropic", "ref_p": "1.0", "compressibility": "4.5e-5"}
PROTEIN_SCHEDULE = {
    "minimization": {
        "integrator": "steep",
        "emtol": "1000.0",
        "emstep": "0.01",
        "nsteps": "50000",
        "nstlist": "1",
        "cutoff_scheme": "Verlet",
        "pbc": "xyz",
    },
    "nvt": {
        **PROTEIN_DYNAMICS,
        **PROTEIN_EQUILIBRATION,
        "continuation": "no",
        "pcoupl": "no",
        "gen_vel": "yes",
        "gen_temp": "300",
        "gen_seed": "-1",
    },
    "npt": {
        **PROTEIN_DYNAMICS,
        **PROTEIN_EQUILIBRATION,
        **PROTEIN_PRESSURE,
        "pcoupl": "Berendsen",
        "tau_p": "2.0",
    },
    "production": {
        **PROTEIN_DYNAMICS,
        **PROTEIN_PRESSURE,
        "nsteps": "50000000",
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstenergy": "5000",
        "nstlog": "5000",
        "nstxout_compressed": "5000",
        "compressed_x_grps": "System",
        "pcoupl": "Parrinello-Rahman",
        "tau_p": "5.0",
        "refcoord_scaling": "com",
    },
}
# Keys the generator adds at their GROMACS default.
WRITTEN_AT_DEFAULT = {"nvt": {"nstfout": "0"}, "npt": {"nstfout": "0"}}


@pytest.mark.parametrize("simulation_type", PROTEIN_TYPES)
@pytest.mark.parametrize("family", [CharmmParametrization, AmberParametrization])
def test_protein_schedules_reproduce_the_static_files(tmp_path, simulation_type, family):
    nonbonded = {"coulombtype", "rcoulomb", "vdwtype", "vdw_modifier", "rvdw", "rvdw_switch"}
    for stage, values in _written(tmp_path, simulation_type, family).items():
        generated = {k: _tokens(v) for k, v in values.items() if k not in nonbonded | {"dispcorr"}}
        expected = {**PROTEIN_SCHEDULE[stage], **WRITTEN_AT_DEFAULT.get(stage, {})}
        assert generated == {k: _tokens(v) for k, v in expected.items()}, stage


def test_sage_scheme_matches_the_openff_force_fields(tmp_path):
    """The sage block equals the vdW handler of the default Sage and water force fields."""
    toolkit = pytest.importorskip("openff.toolkit")
    from openff.units import unit

    values = _written(tmp_path, "mixedbox", OpenFFParametrization)["production"]
    water = f"{OpenFFParametrization().resolve('mixedbox').water_model}.offxml"
    for name in (DEFAULT_OPENFF_FORCEFIELD, water):
        handler = toolkit.ForceField(name).get_parameter_handler("vdW")
        cutoff = handler.cutoff.m_as(unit.nanometer)
        switch = cutoff - handler.switch_width.m_as(unit.nanometer)
        assert float(values["rvdw"]) == pytest.approx(cutoff), name
        assert float(values["rcoulomb"]) == pytest.approx(cutoff), name
        assert float(values["rvdw_switch"]) == pytest.approx(switch), name


# --- re-render existing directories -----------------------------------------------------

WATER = {
    "simulation_type": "mixedbox",
    "parametrization": "charmm",
    "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
}


@pytest.fixture()
def prepared(tmp_path):
    """Return a prepared directory: build record and mdp files, no simulation outputs."""
    model = BuildInput(**WATER)
    directory = tmp_path / model.hash
    directory.mkdir()
    write_build_record(model, directory)
    write_mdp_files(model, directory)
    return directory


def _record(directory: Path) -> BuildInput:
    return BuildInput(**yaml.safe_load((directory / f"{directory.name}.yaml").read_text()))


def test_rerender_rewrites_stages_and_keeps_hash(prepared):
    (update,) = apply_simulation_settings([prepared], {"production": {"length": 200.0}})
    assert update.status == "updated"
    assert update.changed_stages == ["production"]
    assert _read(prepared / "md.mdp")["nsteps"] == "100000000"
    record = _record(prepared)
    assert record.hash == prepared.name
    assert record.simulation_settings.production.length == 200.0


def test_rerender_refuses_changed_stage_with_outputs(prepared):
    before = (prepared / "md.mdp").read_text()
    (prepared / "prod.tpr").touch()
    (update,) = apply_simulation_settings([prepared], {"production": {"length": 200.0}})
    assert update.status == "refused" and "prod.tpr" in update.reason
    assert (prepared / "md.mdp").read_text() == before
    assert _record(prepared).simulation_settings.production.length == 100.0
    (prepared / "prod.tpr").unlink()
    (update,) = apply_simulation_settings([prepared], {"production": {"length": 200.0}})
    assert update.status == "updated"


def test_rerender_leaves_unchanged_stages_with_outputs(prepared):
    (prepared / "em.mdp").write_text("; edited by hand\n")
    (prepared / "min.gro").touch()
    (update,) = apply_simulation_settings([prepared], {"production": {"length": 200.0}})
    assert update.status == "updated"
    assert prepared / "em.mdp" not in update.written
    assert (prepared / "em.mdp").read_text() == "; edited by hand\n"


@pytest.mark.parametrize("settings", [{"temperature": 310.0}, {"pressure": 2.0}])
def test_rerender_refuses_state_point_change(prepared, settings):
    (update,) = apply_simulation_settings([prepared], settings)
    assert update.status == "refused" and "identity" in update.reason


def test_rerender_dry_run_writes_nothing(prepared):
    files = {p: p.read_text() for p in prepared.iterdir()}
    (update,) = apply_simulation_settings([prepared], {"nvt": {"length": 50.0}}, dry_run=True)
    assert update.status == "updated" and update.changed_stages == ["nvt"]
    assert prepared / "nvt.mdp" in update.written
    assert {p: p.read_text() for p in prepared.iterdir()} == files


def test_rerender_unchanged_and_missing_record(prepared, tmp_path):
    (update,) = apply_simulation_settings([prepared], {"temperature": 300.0})
    assert update.status == "unchanged"
    empty = tmp_path / "empty"
    empty.mkdir()
    (update,) = apply_simulation_settings([empty], {})
    assert update.status == "refused" and "build record" in update.reason


def test_rerender_reads_settings_file(prepared, tmp_path):
    path = tmp_path / "longer.yaml"
    path.write_text("production: {length: 300}\n")
    (update,) = apply_simulation_settings([prepared], path)
    assert update.status == "updated"
    assert _record(prepared).simulation_settings.production.length == 300.0


def test_apply_settings_command(prepared, tmp_path):
    from mdfactory.cli import apply_settings

    (prepared / "system.pdb").touch()
    path = tmp_path / "longer.yaml"
    path.write_text("production: {length: 300}\n")
    apply_settings(path, prepared.parent)
    assert _read(prepared / "md.mdp")["nsteps"] == "150000000"
    path.write_text("temperature: 310\n")
    with pytest.raises(SystemExit):
        apply_settings(path, prepared.parent)
