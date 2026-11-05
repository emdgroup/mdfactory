# ABOUTME: Tests for force field parametrization of molecular species: CGenFF and
# ABOUTME: OpenFF parameter generation, cache keys/locking, atom types, and topology files.
"""Tests for force field parametrization of molecular species."""

import contextlib
import json
import os
import shutil
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mdfactory import parametrize as parametrize_module
from mdfactory.models.parametrization import (
    DEFAULT_NAGL_MODEL,
    DEFAULT_OPENFF_FORCEFIELD,
    CharmmParametrization,
    GromacsSingleMoleculeParameterSet,
    OpenFFParametrization,
    OpenFFSmallMolecules,
)
from mdfactory.models.species import SingleMoleculeSpecies
from mdfactory.parametrize import (
    OPENFF_CACHE_COMPLETE_MARKER,
    generate_gromacs_topology,
    openff_ion_cache_entry,
    openff_small_molecule_cache_entry,
    openff_water_cache_entry,
    parametrize_cgenff_gromacs,
    parametrize_openff_gromacs,
)
from mdfactory.settings import settings as config
from mdfactory.utils.data_manager import DataManager, SQLiteDataSource
from mdfactory.utils.utilities import working_directory

from .utils import ProcessWithException

CHARMM_MIXEDBOX = CharmmParametrization().resolve("mixedbox")
OPENFF_MIXEDBOX = OpenFFParametrization().resolve("mixedbox")


def is_openff_available():
    """Check if OpenFF toolkit and interchange are available."""
    try:
        from openff.interchange import Interchange  # noqa: F401, PLC0415
        from openff.toolkit import ForceField  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def is_cgenff_available():
    silcsbiodir = os.environ.get("SILCSBIODIR")
    if not silcsbiodir:
        return False

    # Construct the command
    script_path = Path(silcsbiodir) / "cgenff" / "cgenff_batch.sh"
    if not script_path.exists():
        print(f"CGenFF script not found at {script_path}")
        return False

    return True


@pytest.mark.skipif(not is_cgenff_available(), reason="CGenFF not available")
def test_parametrization_simple(tmp_path, mocker):
    mock_data_source = SQLiteDataSource(
        db_path=str(tmp_path / "molecule_database.db"), table_name="molecules"
    )

    patcher = patch.object(DataManager, "_initialize_data_source", return_value=mock_data_source)
    patcher.start()

    config.parameter_store = tmp_path / "parameters"
    path = config.parameter_store / "gromacs" / "cgenff"

    parametrize = partial(parametrize_cgenff_gromacs, spec=CHARMM_MIXEDBOX)

    specs = [
        SingleMoleculeSpecies(smiles="CCC", count=1, resname="PRO"),
        SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH"),
        SingleMoleculeSpecies(smiles="CCOC", count=1, resname="BLA"),
        SingleMoleculeSpecies(smiles="O", count=1, resname="SOL"),
        SingleMoleculeSpecies(smiles="[Na+]", count=1, resname="NA"),
        SingleMoleculeSpecies(smiles="[Cl-]", count=1, resname="CL"),
    ]

    parameters = [parametrize(spec) for spec in specs]
    # 2nd time should fetch from db
    parameters = [parametrize(spec) for spec in specs]
    del parameters

    # mock the query to return empty dataframe to simulate missing entries
    patcher2 = patch.object(DataManager, "query_data", return_value=pd.DataFrame())
    patcher2.start()
    parameters = [parametrize(spec) for spec in specs]
    patcher2.stop()

    files = list(path.glob("*"))
    assert len(files) == 3  # water and ions should not be saved
    for file in files:
        assert file.is_dir()

    df = mock_data_source.load_data()
    print(df)

    patcher.stop()

    u = mocker.MagicMock()
    u.residues.resnames = 5 * ["PRO"] + 3 * ["ETH"] + 2 * ["BLA"] + 10 * ["SOL"] + ["NA"] + ["CL"]

    with working_directory(tmp_path):
        generate_gromacs_topology(u, specs, parameters, "test_system")
        top_path = tmp_path / "topology.top"
        assert top_path.is_file()

        with pytest.raises(ValueError, match="Must have parameters for each species"):
            generate_gromacs_topology(u, specs[:-1], parameters, "test_system")


@pytest.mark.skipif(not is_cgenff_available(), reason="CGenFF not available")
def test_parametrization_simultaneously(tmp_path):
    config.parameter_store = tmp_path / "parameters"
    path = config.parameter_store / "gromacs" / "cgenff"

    parametrize = partial(parametrize_cgenff_gromacs, spec=CHARMM_MIXEDBOX)

    specs = [
        SingleMoleculeSpecies(smiles="CCC", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="CCO", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="CCOC", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="CCCCCCCCCCCCCOCCCCCCCCCOCCCCCCOCCC", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="OCCCCCCCCCCCCCOCCCCCCCCCOCCCCCCOCCC", count=1, resname="XYZ"),
        SingleMoleculeSpecies(
            smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
            count=1,
            resname="XYZ",
        ),
        SingleMoleculeSpecies(
            smiles=r"CCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
            count=1,
            resname="XYZ",
        ),
        SingleMoleculeSpecies(
            smiles=r"CCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCC",
            count=1,
            resname="XYZ",
        ),
    ]
    n_specs = len(specs)

    mock_data_source = SQLiteDataSource(
        db_path=str(tmp_path / "molecule_database.db"), table_name="molecules"
    )

    patcher = patch.object(DataManager, "_initialize_data_source", return_value=mock_data_source)
    patcher.start()
    dm = DataManager("RUN_DATABASE")
    df = dm.load_data()
    assert df.empty

    def parametrize_many(i, specs_list):
        print(f"Thread {i} starting...")
        parameters = [parametrize(spec) for spec in specs_list]
        del parameters
        print(f"Thread {i} done:")

    processes = []
    n_procs = 8
    import random

    random.seed(420)
    for i in range(n_procs):
        specs_shuffled = random.sample(specs, len(specs))
        # specs_shuffled = specs
        proc = ProcessWithException(target=parametrize_many, args=(i, specs_shuffled))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    for proc in processes:
        if proc.exception:
            raise Exception(f"{proc.exception[1]}") from proc.exception[0]

    # NOTE: db is disabled for now
    # df = dm.load_data()
    # print(df)
    # assert len(df) == len(specs)

    files = list(path.glob("*"))
    assert len(files) == n_specs
    for file in files:
        assert file.is_dir()

    patcher.stop()


@pytest.mark.skipif(not is_cgenff_available(), reason="CGenFF not available")
def test_parallel_cgenff_run(tmp_path):
    from mdfactory.utils.topology_utilities import (
        run_cgenff_to_gmx,
        write_mol2_from_rdkit,
    )
    from mdfactory.utils.utilities import working_directory

    def run_cgenff(species):
        workdir = tmp_path / species.hash
        with working_directory(workdir, exists_ok=False, create=True) as tmpdir:
            mol2_file = tmpdir / "XXX.mol2"
            write_mol2_from_rdkit(species.rdkit_molecule, mol2_file, title="XXX")

            if not mol2_file.is_file():
                raise FileNotFoundError(f"Could not create mol2 file at {mol2_file}.")

            print("Running cgenff...")
            run_cgenff_to_gmx(mol2_file)
        shutil.rmtree(workdir, ignore_errors=True)

    specs = [
        SingleMoleculeSpecies(smiles="CCC", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="CCO", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="CCOC", count=1, resname="XYZ"),
        SingleMoleculeSpecies(smiles="CCOC", count=1, resname="XYZ"),
    ]

    # since working directories are process-specific, use multiprocessing instead
    import multiprocessing

    n_procs = 2
    processes = []
    for i in range(n_procs):
        process = multiprocessing.Process(target=run_cgenff, args=(specs[i],))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


# =============================================================================
# CGenFF cache directory and locking (CGenFF itself is faked)
# =============================================================================


def _fake_cgenff(monkeypatch, calls):
    """Replace the CGenFF executable with a stub writing a minimal moleculetype file."""

    def fake_run(mol2_file):
        calls.append(Path.cwd())
        out = Path("1_cgenff_batch") / "gromacs_output"
        out.mkdir(parents=True)
        (out / "XXX.itp").write_text("; fake\n")
        return 0, "", ""

    monkeypatch.setattr(parametrize_module, "run_cgenff_to_gmx", fake_run)
    monkeypatch.setattr(
        parametrize_module,
        "extract_reusable_parts_from_cgenff_gmx_top",
        lambda itp, replace_name: ("[ moleculetype ]\n", "[ atomtypes ]\n", []),
    )


def _record_locks(monkeypatch):
    """Record every path parametrize.py locks, keeping the real locking behavior."""
    locked = []
    real_lock = parametrize_module.lock_local_folder

    @contextlib.contextmanager
    def recording_lock(folder, *args, **kwargs):
        locked.append(Path(folder))
        with real_lock(folder, *args, **kwargs):
            yield

    monkeypatch.setattr(parametrize_module, "lock_local_folder", recording_lock)
    return locked


def test_cgenff_locks_written_directory_and_regenerates_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "parameter_store", tmp_path / "parameters")
    calls = []
    _fake_cgenff(monkeypatch, calls)
    locked = _record_locks(monkeypatch)
    spec = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH")
    # A failed earlier attempt left a directory without outputs.
    stale = tmp_path / "parameters" / "gromacs" / "cgenff" / spec.hash
    stale.mkdir(parents=True)
    (stale / "mol2gmx_1.log").write_text("LICENSE_ERROR\n")

    param = parametrize_cgenff_gromacs(spec, spec=CHARMM_MIXEDBOX)

    assert locked == [param.itp.parent] == [stale.resolve()]
    assert not (stale / "mol2gmx_1.log").exists()
    assert (param.component, param.forcefield, param.topology_merge) == (
        "small_molecule",
        "cgenff",
        "cgenff",
    )
    assert param.charge_method is None
    # A complete entry is reused without running CGenFF again.
    assert parametrize_cgenff_gromacs(spec, spec=CHARMM_MIXEDBOX) == param
    assert len(calls) == 1


def test_cgenff_native_water_and_ions_record_forcefield(tmp_path, monkeypatch):
    ff_dir = tmp_path / "charmm36.ff"
    ff_dir.mkdir()
    for name in ("forcefield.itp", "tip3p.itp", "ions.itp"):
        (ff_dir / name).write_text(f"; {name}\n")
    monkeypatch.setattr(config, "parameter_store", tmp_path / "parameters")
    native = parametrize_module.NativeForceField("charmm36m", ff_dir, "tip3p")

    water, na = (
        parametrize_cgenff_gromacs(
            SingleMoleculeSpecies(smiles=smiles, count=1, resname=resname),
            spec=CHARMM_MIXEDBOX,
            native_forcefield=native,
        )
        for smiles, resname in (("O", "SOL"), ("[Na+]", "NA"))
    )

    assert (water.itp, water.component, water.forcefield, water.water_model) == (
        ff_dir / "tip3p.itp",
        "water",
        "charmm36m",
        "tip3p",
    )
    assert (na.itp, na.moleculetype, na.component, na.water_model) == (
        ff_dir / "ions.itp",
        "NA",
        "ion",
        None,
    )
    assert {water.topology_merge, na.topology_merge} == {"native"}
    assert not (tmp_path / "parameters").exists()


# =============================================================================
# OpenFF (SMIRNOFF) cache keys (no parametrization needed)
# =============================================================================


@pytest.fixture
def parameter_store(tmp_path, monkeypatch):
    store = tmp_path / "parameters"
    monkeypatch.setattr(config, "parameter_store", store)
    return store


def test_openff_small_molecule_cache_key(parameter_store):
    spec = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH")
    default = openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules())
    alias = openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules(charge_method="nagl"))
    am1bcc = openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules(charge_method="am1bcc"))
    older_ff = openff_small_molecule_cache_entry(
        spec, OpenFFSmallMolecules(forcefield="openff-2.1.0.offxml")
    )

    # "nagl" resolves to the default NAGL model, so both share one entry.
    assert alias.workdir == default.workdir
    assert len({default.workdir, am1bcc.workdir, older_ff.workdir}) == 3
    assert default.key["charge_method"] == DEFAULT_NAGL_MODEL
    # Readable layout: <root>/small_molecules/<ff stem>_<digest>/<species hash>.
    root = (parameter_store / "gromacs" / "openff" / "small_molecules").resolve()
    assert default.workdir.parent.parent == root
    assert default.workdir.name == spec.hash
    assert default.workdir.parent.name.startswith("openff-2.2.0_")
    assert older_ff.workdir.parent.name.startswith("openff-2.1.0_")


def test_openff_water_and_ion_cache_keys(parameter_store):
    water = SingleMoleculeSpecies(smiles="O", count=1, resname="SOL")
    na = SingleMoleculeSpecies(smiles="[Na+]", count=1, resname="NA")
    cl = SingleMoleculeSpecies(smiles="[Cl-]", count=1, resname="CL")

    tip3p = openff_water_cache_entry(water, "tip3p")
    opc3 = openff_water_cache_entry(water, "opc3")
    assert tip3p.workdir != opc3.workdir
    assert tip3p.workdir.name.startswith("tip3p_") and opc3.workdir.name.startswith("opc3_")

    na_default = openff_ion_cache_entry(na, DEFAULT_OPENFF_FORCEFIELD)
    na_older = openff_ion_cache_entry(na, "openff-2.1.0.offxml")
    cl_default = openff_ion_cache_entry(cl, DEFAULT_OPENFF_FORCEFIELD)
    assert na_default.workdir != na_older.workdir
    # Each ion has its own (separately locked) directory under the force-field key.
    assert na_default.workdir.parent == cl_default.workdir.parent
    assert (na_default.workdir.name, cl_default.workdir.name) == ("NA", "CL")


def test_openff_atomtype_prefix_is_short_and_deterministic(parameter_store):
    spec = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH")
    other = SingleMoleculeSpecies(smiles="CCCO", count=1, resname="ETH")
    entry = openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules())
    prefix = entry.atomtype_prefix

    assert prefix == openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules()).atomtype_prefix
    assert (
        prefix != openff_small_molecule_cache_entry(other, OpenFFSmallMolecules()).atomtype_prefix
    )
    assert prefix[0].isalpha() and prefix.isalnum() and prefix.isascii()
    # Leaves room for "_<index>" within 16 characters for molecules up to 99999 atoms.
    assert len(prefix) + len("_99999") <= 16
    assert "ETH" not in prefix


# =============================================================================
# OpenFF (SMIRNOFF) parametrization
# =============================================================================


def _atomtype_names(params_itp: Path) -> list[str]:
    names, in_atomtypes = [], False
    for raw in params_itp.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            in_atomtypes = line.strip("[] ").lower() == "atomtypes"
        elif in_atomtypes and line:
            names.append(line.split()[0])
    return names


def _itp_atom_types(itp: Path) -> list[str]:
    types, in_atoms = [], False
    for raw in itp.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            in_atoms = line.strip("[] ").lower() == "atoms"
        elif in_atoms and line:
            types.append(line.split()[1])
    return types


def _sections(path: Path) -> set[str]:
    return {
        line.strip().strip("[] ").lower()
        for line in path.read_text().splitlines()
        if line.strip().startswith("[")
    }


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_openff_small_molecule_parameter_set(parameter_store):
    spec = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH")
    param = parametrize_openff_gromacs(spec, spec=OPENFF_MIXEDBOX)

    assert param.moleculetype == spec.hash
    assert (param.component, param.forcefield, param.charge_method) == (
        "small_molecule",
        DEFAULT_OPENFF_FORCEFIELD,
        DEFAULT_NAGL_MODEL,
    )
    assert param.topology_merge == "smirnoff"
    assert param.itp.is_file() and param.parameter_itp.is_file()
    assert param.forcefield_itp is None
    assert (
        param.itp.parent == openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules()).workdir
    )

    # Atom types carry the set's prefix (not the resname) and match between files.
    prefix = openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules()).atomtype_prefix
    types = _atomtype_names(param.parameter_itp)
    assert types == [f"{prefix}_{i}" for i in range(len(types))]
    assert set(_itp_atom_types(param.itp)) <= set(types)
    assert not any("ETH" in name for name in types + _itp_atom_types(param.itp))
    # Interchange inlines bonded parameters, so no other section references a type.
    atomtypes_file = param.itp.parent / f"{spec.hash}_atomtypes.itp"
    assert _sections(atomtypes_file) == {"atomtypes"}
    assert not {s for s in _sections(param.itp) if s.endswith("types")}

    provenance = json.loads((param.itp.parent / "provenance.json").read_text())
    assert provenance["key"] == {
        "format": parametrize_module.OPENFF_CACHE_FORMAT_VERSION,
        "component": "small_molecule",
        "forcefield": DEFAULT_OPENFF_FORCEFIELD,
        "charge_method": DEFAULT_NAGL_MODEL,
    }
    assert provenance["smiles"] == spec.smiles
    assert provenance["atomtype_prefix"] == prefix
    assert set(provenance["versions"]) == {"openff-toolkit", "openff-interchange"}
    assert (param.itp.parent / OPENFF_CACHE_COMPLETE_MARKER).is_file()


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_openff_cache_reuse_and_nagl_alias(parameter_store):
    spec = SingleMoleculeSpecies(smiles="CCCC", count=1, resname="BUT")
    first = parametrize_openff_gromacs(spec, spec=OPENFF_MIXEDBOX)
    mtime = first.itp.stat().st_mtime_ns

    alias_spec = OpenFFParametrization(small_molecules={"charge_method": "nagl"}).resolve(
        "mixedbox"
    )
    second = parametrize_openff_gromacs(spec, spec=alias_spec)

    assert second == first
    assert second.itp.stat().st_mtime_ns == mtime


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_openff_incomplete_entry_is_regenerated(parameter_store):
    spec = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH")
    entry = openff_small_molecule_cache_entry(spec, OpenFFSmallMolecules())
    # An interrupted run: files exist, but the completion marker was never written.
    entry.workdir.mkdir(parents=True)
    entry.itp.write_text("; truncated\n")
    (entry.workdir / "leftover.txt").write_text("junk\n")

    param = parametrize_openff_gromacs(spec, spec=OPENFF_MIXEDBOX)

    assert entry.is_complete()
    assert not (entry.workdir / "leftover.txt").exists()
    assert "; truncated" not in param.itp.read_text()

    # Removing an artifact of a complete entry also forces regeneration.
    entry.params.unlink()
    parametrize_openff_gromacs(spec, spec=OPENFF_MIXEDBOX)
    assert entry.params.is_file()


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
@pytest.mark.parametrize("resname", ["SOL", "WAT"])
def test_openff_water(parameter_store, resname):
    water = SingleMoleculeSpecies(smiles="O", count=1, resname=resname)
    opc3 = parametrize_openff_gromacs(water, spec=OPENFF_MIXEDBOX)
    tip3p_spec = OpenFFParametrization(water_model="tip3p").resolve("mixedbox")
    tip3p = parametrize_openff_gromacs(water, spec=tip3p_spec)

    for param, model in ((opc3, "opc3"), (tip3p, "tip3p")):
        assert param.moleculetype == "SOL"
        assert (param.component, param.forcefield, param.water_model) == (
            "water",
            f"{model}.offxml",
            model,
        )
        assert param.parameter_itp.is_file()
        assert param.itp.parent == openff_water_cache_entry(water, model).workdir
    assert opc3.itp.parent != tip3p.itp.parent
    assert _atomtype_names(opc3.parameter_itp) != _atomtype_names(tip3p.parameter_itp)


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_openff_ions(parameter_store):
    na = parametrize_openff_gromacs(
        SingleMoleculeSpecies(smiles="[Na+]", count=1, resname="NA"), spec=OPENFF_MIXEDBOX
    )
    cl = parametrize_openff_gromacs(
        SingleMoleculeSpecies(smiles="[Cl-]", count=1, resname="CL"), spec=OPENFF_MIXEDBOX
    )

    assert (na.moleculetype, cl.moleculetype) == ("NA", "CL")
    for param in (na, cl):
        assert (param.component, param.forcefield, param.topology_merge) == (
            "ion",
            DEFAULT_OPENFF_FORCEFIELD,
            "smirnoff",
        )
        assert param.itp.is_file()
    assert set(_atomtype_names(na.parameter_itp)).isdisjoint(_atomtype_names(cl.parameter_itp))


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_openff_locks_the_directory_it_writes(parameter_store, monkeypatch):
    locked = _record_locks(monkeypatch)
    species = [
        SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH"),
        SingleMoleculeSpecies(smiles="O", count=1, resname="SOL"),
        SingleMoleculeSpecies(smiles="[Na+]", count=1, resname="NA"),
    ]

    params = [parametrize_openff_gromacs(spec, spec=OPENFF_MIXEDBOX) for spec in species]

    assert locked == [param.itp.parent for param in params]
    assert not list(parameter_store.rglob("*.lock"))


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_openff_same_resname_molecules_combine_in_mixedbox_topology(
    parameter_store, tmp_path, mocker
):
    """Two different molecules with the same resname no longer share atom-type names."""
    first = SingleMoleculeSpecies(smiles="CCO", count=2, resname="LIG")
    second = SingleMoleculeSpecies(smiles="CCCO", count=3, resname="LIG")
    water = SingleMoleculeSpecies(smiles="O", count=4, resname="SOL")
    params = [
        parametrize_openff_gromacs(first, spec=OPENFF_MIXEDBOX),
        parametrize_openff_gromacs(second, spec=OPENFF_MIXEDBOX),
        parametrize_openff_gromacs(water, spec=OPENFF_MIXEDBOX),
    ]
    first_types = _atomtype_names(params[0].parameter_itp)
    second_types = _atomtype_names(params[1].parameter_itp)
    assert set(first_types).isdisjoint(second_types)

    # In the box the two molecules need distinct resnames; their parameter sets
    # (cached by SMILES) still carry the shared LIG resname they were made with.
    in_box = [
        SingleMoleculeSpecies(smiles="CCO", count=2, resname="LG1"),
        SingleMoleculeSpecies(smiles="CCCO", count=3, resname="LG2"),
        water,
    ]
    u = mocker.MagicMock()
    u.residues.resnames = ["LG1"] * 2 + ["LG2"] * 3 + ["SOL"] * 4
    build = tmp_path / "build"
    build.mkdir()
    with working_directory(build):
        generate_gromacs_topology(u, in_box, params, "mixedbox")
        merged = _atomtype_names(build / "extra_params.itp")
        molecules = (build / "topology.top").read_text().split("[ molecules ]", 1)[1]
    assert len(merged) == len(set(merged))
    assert set(first_types) | set(second_types) <= set(merged)
    assert f"{first.hash} 2" in molecules and f"{second.hash} 3" in molecules


def _fake_smirnoff_set(tmp_path, name, epsilon):
    folder = tmp_path / name
    folder.mkdir()
    itp = folder / f"{name}.itp"
    itp.write_text(f"[ moleculetype ]\n{name} 3\n\n[ atoms ]\n1 T_0 1 LIG C 1 0.0 12.011\n")
    params = folder / f"{name}_params.itp"
    params.write_text(
        f"[ defaults ]\n1 2 yes 0.5 0.833333\n\n[ atomtypes ]\nT_0 6 12.011 0.0 A 0.34 {epsilon}\n"
    )
    return GromacsSingleMoleculeParameterSet(
        moleculetype=name,
        smiles="C",
        component="small_molecule",
        forcefield=DEFAULT_OPENFF_FORCEFIELD,
        charge_method=DEFAULT_NAGL_MODEL,
        topology_merge="smirnoff",
        itp=itp,
        parameter_itp=params,
        forcefield_itp=None,
    )


def test_conflicting_smirnoff_atomtypes_raise_in_mixedbox_topology(tmp_path, mocker):
    first = _fake_smirnoff_set(tmp_path, "AAA", "0.45")
    second = _fake_smirnoff_set(tmp_path, "BBB", "0.99")
    species = [
        SingleMoleculeSpecies(smiles="C", count=1, resname="AAA"),
        SingleMoleculeSpecies(smiles="CC", count=1, resname="BBB"),
    ]
    u = mocker.MagicMock()
    u.residues.resnames = ["AAA", "BBB"]
    with working_directory(tmp_path):
        with pytest.raises(ValueError, match="'T_0' is defined with different parameters"):
            generate_gromacs_topology(u, species, [first, second], "mixedbox")
