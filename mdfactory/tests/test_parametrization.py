# ABOUTME: Tests for force field parametrization of molecular species, including
# ABOUTME: CGenFF parameter generation and topology file handling.
"""Tests for force field parametrization of molecular species, including."""

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mdfactory.models.parametrization import SmirnoffConfig
from mdfactory.models.species import SingleMoleculeSpecies
from mdfactory.parametrize import (
    generate_gromacs_topology,
    parametrize_cgenff_gromacs,
    parametrize_smirnoff_gromacs,
    retrieve_or_deposit_parameters,
)
from mdfactory.settings import settings as config
from mdfactory.utils.data_manager import DataManager, SQLiteDataSource
from mdfactory.utils.utilities import working_directory

from .utils import ProcessWithException


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

    dec = retrieve_or_deposit_parameters("gromacs", "cgenff")
    parametrize = dec(parametrize_cgenff_gromacs)

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

    dec = retrieve_or_deposit_parameters("gromacs", "cgenff")
    parametrize = dec(parametrize_cgenff_gromacs)

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
# SMIRNOFF Parametrization Tests
# =============================================================================


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_parametrization_simple(tmp_path):
    """Test basic SMIRNOFF parametrization."""
    config.parameter_store = tmp_path / "parameters"

    spec = SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH")
    param = parametrize_smirnoff_gromacs(spec)

    assert param.itp.is_file()
    assert param.parametrization == "smirnoff"
    assert param.parameter_itp is not None
    assert param.parameter_itp.is_file()
    assert param.forcefield_itp is None


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_parametrization_water(tmp_path):
    """Test SMIRNOFF parametrization of water."""
    config.parameter_store = tmp_path / "parameters"

    spec = SingleMoleculeSpecies(smiles="O", count=1, resname="SOL")
    param = parametrize_smirnoff_gromacs(spec)

    assert param.itp.is_file()
    assert param.moleculetype == "SOL"
    assert param.parametrization == "smirnoff"


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_parametrization_water_wat_resname(tmp_path):
    """Test SMIRNOFF parametrization of water with non-SOL resname (regression).

    Verifies that water parametrization with resname="WAT" still produces
    consistent SOL-based atom types and ITP files. Before the fix, this
    caused a KeyError due to mismatched atom type prefixes (WAT_0 vs SOL_0).
    """
    config.parameter_store = tmp_path / "parameters"

    spec = SingleMoleculeSpecies(smiles="O", count=1, resname="WAT")
    param = parametrize_smirnoff_gromacs(spec)

    assert param.itp.is_file()
    # Molecule type should always be SOL regardless of input resname
    assert param.moleculetype == "SOL"
    assert param.parametrization == "smirnoff"
    # Verify the parameter ITP was generated correctly
    assert param.parameter_itp is not None
    assert param.parameter_itp.is_file()


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_parametrization_ions(tmp_path):
    """Test SMIRNOFF parametrization of ions."""
    config.parameter_store = tmp_path / "parameters"

    na_spec = SingleMoleculeSpecies(smiles="[Na+]", count=1, resname="NA")
    cl_spec = SingleMoleculeSpecies(smiles="[Cl-]", count=1, resname="CL")

    na_param = parametrize_smirnoff_gromacs(na_spec)
    cl_param = parametrize_smirnoff_gromacs(cl_spec)

    assert na_param.itp.is_file()
    assert na_param.moleculetype == "NA"
    assert cl_param.itp.is_file()
    assert cl_param.moleculetype == "CL"


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_with_config(tmp_path):
    """Test SMIRNOFF with custom config."""
    config.parameter_store = tmp_path / "parameters"

    smirnoff_config = SmirnoffConfig(
        forcefield="openff-2.1.0.offxml",
        charge_method="openff-gnn-am1bcc-0.1.0-rc.3.pt",  # NAGL model
    )
    spec = SingleMoleculeSpecies(smiles="CCC", count=1, resname="PRO")
    param = parametrize_smirnoff_gromacs(spec, smirnoff_config=smirnoff_config)

    assert param.itp.is_file()
    assert param.parametrization == "smirnoff"

    # Verify the cache path includes the forcefield version
    expected_ff_hash = "openff_2.1.0"
    assert expected_ff_hash in str(param.itp)


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_caching(tmp_path):
    """Test that SMIRNOFF parameters are cached correctly."""
    config.parameter_store = tmp_path / "parameters"

    spec = SingleMoleculeSpecies(smiles="CCCC", count=1, resname="BUT")

    # First call generates parameters
    param1 = parametrize_smirnoff_gromacs(spec)
    assert param1.itp.is_file()
    mtime1 = param1.itp.stat().st_mtime

    # Second call should use cache (same file, same mtime)
    param2 = parametrize_smirnoff_gromacs(spec)
    assert param2.itp == param1.itp
    mtime2 = param2.itp.stat().st_mtime
    assert mtime1 == mtime2  # File was not regenerated


@pytest.mark.skipif(not is_openff_available(), reason="OpenFF not available")
def test_smirnoff_multiple_species(tmp_path, mocker):
    """Test SMIRNOFF parametrization of multiple species and topology generation."""
    config.parameter_store = tmp_path / "parameters"

    specs = [
        SingleMoleculeSpecies(smiles="CCC", count=1, resname="PRO"),
        SingleMoleculeSpecies(smiles="CCO", count=1, resname="ETH"),
        SingleMoleculeSpecies(smiles="O", count=1, resname="SOL"),
        SingleMoleculeSpecies(smiles="[Na+]", count=1, resname="NA"),
        SingleMoleculeSpecies(smiles="[Cl-]", count=1, resname="CL"),
    ]

    parameters = [parametrize_smirnoff_gromacs(spec) for spec in specs]

    # Verify all parameters were generated
    assert len(parameters) == len(specs)
    for param in parameters:
        assert param.itp.is_file()
        assert param.parametrization == "smirnoff"
