# ABOUTME: Force field parametrization for molecular species
# ABOUTME: Handles CGenFF and SMIRNOFF parametrization with GROMACS topology generation
"""Force field parametrization for molecular species."""

import shutil
from functools import wraps
from pathlib import Path
from typing import Optional

from .models.parametrization import CgenffConfig, GromacsSingleMoleculeParameterSet, SmirnoffConfig
from .models.species import SingleMoleculeSpecies
from .settings import settings
from .utils.topology_utilities import (
    count_contiguous_strings,
    extract_reusable_parts_from_cgenff_gmx_top,
    merge_extra_parameter_itps,
    run_cgenff_to_gmx,
    write_mol2_from_rdkit,
)
from .utils.utilities import lock_local_folder, working_directory

parameter_set_types = {
    "gromacs": GromacsSingleMoleculeParameterSet,
}


def _replace_moleculetype(itp_path: Path, new_name: str) -> None:
    with open(itp_path, "r") as fb:
        lines = fb.readlines()

    in_moleculetype = False
    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("[") and "moleculetype" in stripped.lower():
            in_moleculetype = True
            continue
        if in_moleculetype:
            if not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if parts and parts[0] == new_name:
                return
            parts[0] = new_name
            lines[i] = line.replace(stripped, " ".join(parts))
            updated = True
            break

    if updated:
        with open(itp_path, "w") as fb:
            fb.writelines(lines)


def _extract_defaults_block(top_path: Path) -> str:
    defaults_lines = []
    in_defaults = False
    with open(top_path, "r") as fb:
        for line in fb:
            stripped = line.strip()
            if stripped.startswith("#include") and in_defaults:
                break
            if stripped.lower().startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1].strip().lower()
                if section == "defaults":
                    in_defaults = True
                    defaults_lines.append(line.rstrip("\n"))
                    continue
                if in_defaults:
                    break
            if in_defaults:
                defaults_lines.append(line.rstrip("\n"))
    return "\n".join(defaults_lines).strip()


def _build_smirnoff_parameter_itp(top_path: Path, atomtypes_path: Path, out_path: Path) -> None:
    defaults_block = _extract_defaults_block(top_path)
    with open(out_path, "w") as fb:
        if defaults_block:
            fb.write(defaults_block)
            fb.write("\n\n")
        if atomtypes_path.is_file():
            with open(atomtypes_path, "r") as atom_fb:
                fb.write(atom_fb.read())


def retrieve_or_deposit_parameters(engine, parametrization):
    """Return a decorator that caches parametrization results in the local parameter store.

    Parameters
    ----------
    engine : str
        Simulation engine (e.g., "gromacs")
    parametrization : str
        Force field scheme (e.g., "cgenff", "smirnoff")

    """

    def inner(function):
        @wraps(function)
        def wrapper(species: SingleMoleculeSpecies):
            pardir = settings.parameter_store / engine / parametrization
            pardir.mkdir(parents=True, exist_ok=True)
            workdir = pardir / species.hash
            with lock_local_folder(workdir):
                params = function(species)
                return params

        return wrapper

    return inner


def parametrize_cgenff_gromacs(species: SingleMoleculeSpecies) -> GromacsSingleMoleculeParameterSet:
    """Parametrize a species using CGenFF for GROMACS.

    Parameters
    ----------
    species : SingleMoleculeSpecies
        Molecule to parametrize

    Returns
    -------
    GromacsSingleMoleculeParameterSet
        Parameter set with paths to ITP and forcefield files

    """
    # local folder for parameter storage
    pardir = settings.parameter_store / "gromacs" / "cgenff"
    charmm36_dir = settings.cgenff_dir / "data" / "gromacs" / "charmm36.ff"
    forcefield_itp = charmm36_dir / "forcefield.itp"
    cgenff_config = CgenffConfig()

    pardir.mkdir(parents=True, exist_ok=True)

    # check for water/ions and return standard parameters
    if species.is_water:
        return GromacsSingleMoleculeParameterSet(
            moleculetype="SOL",
            smiles=species.smiles,
            parametrization="cgenff",
            parametrization_config=cgenff_config,
            itp=charmm36_dir / "tip3p.itp",
            parameter_itp=None,
            forcefield_itp=forcefield_itp,
        )
    elif species.is_ion:
        typemap = {
            "[Na+]": "NA",
            "[Cl-]": "CL",
        }
        return GromacsSingleMoleculeParameterSet(
            moleculetype=typemap[species.smiles],
            smiles=species.smiles,
            parametrization="cgenff",
            parametrization_config=cgenff_config,
            itp=charmm36_dir / "ions.itp",
            parameter_itp=None,
            forcefield_itp=forcefield_itp,
        )

    # parametrize molecule with cgenff
    workdir = pardir / species.hash
    itp_path = workdir / f"{species.hash}.itp"
    prm_path = workdir / f"{species.hash}_extra_params.itp"
    itp_path = itp_path.resolve()
    prm_path = prm_path.resolve()

    if itp_path.is_file() and prm_path.is_file():
        prm = GromacsSingleMoleculeParameterSet(
            moleculetype=species.hash,
            smiles=species.smiles,
            parametrization="cgenff",
            parametrization_config=cgenff_config,
            itp=itp_path,
            parameter_itp=prm_path,
            forcefield_itp=forcefield_itp,
        )
        return prm

    workdir.mkdir(parents=True, exist_ok=False)
    with working_directory(workdir, exists_ok=True) as tmpdir:
        mol2_file = tmpdir / "XXX.mol2"
        write_mol2_from_rdkit(species.rdkit_molecule, mol2_file, title="XXX")

        return_code, stdout, stderr = run_cgenff_to_gmx(mol2_file)
        if return_code != 0:
            raise RuntimeError(
                f"CGenFF parametrization failed for '{mol2_file}' "
                f"(return code {return_code}).\nstdout: {stdout}\nstderr: {stderr}"
            )

        itp = tmpdir / "1_cgenff_batch" / "gromacs_output" / "XXX.itp"
        if not itp.is_file():
            raise FileNotFoundError(
                f"CGenFF did not produce expected output at '{itp}'. "
                f"stdout: {stdout}\nstderr: {stderr}"
            )
        itp_str, prm_str, _ = extract_reusable_parts_from_cgenff_gmx_top(
            itp, replace_name=species.hash
        )
        with open(itp_path, "w") as fb:
            fb.write(itp_str)
        with open(prm_path, "w") as fb:
            fb.write(prm_str)

    prm = GromacsSingleMoleculeParameterSet(
        moleculetype=species.hash,
        smiles=species.smiles,
        parametrization="cgenff",
        parametrization_config=cgenff_config,
        itp=itp_path,
        parameter_itp=prm_path,
        forcefield_itp=forcefield_itp,
    )
    return prm


def parametrize_smirnoff_gromacs(
    species: SingleMoleculeSpecies,
    smirnoff_config: Optional[SmirnoffConfig] = None,
) -> GromacsSingleMoleculeParameterSet:
    """Parametrize a molecule using OpenFF SMIRNOFF force field.

    Uses OpenFF Interchange to generate GROMACS ITP files.

    Parameters
    ----------
    species : SingleMoleculeSpecies
        The molecule species to parametrize.
    smirnoff_config : SmirnoffConfig, optional
        Configuration for SMIRNOFF parametrization. If None, uses defaults.

    Returns
    -------
    GromacsSingleMoleculeParameterSet
        The parameter set containing paths to generated ITP files.

    """
    # Lazy imports to avoid loading OpenFF when not needed
    import numpy as np  # noqa: PLC0415
    from openff.interchange import Interchange  # noqa: PLC0415
    from openff.toolkit import ForceField, Topology  # noqa: PLC0415
    from openff.units import unit  # noqa: PLC0415

    # Use provided config or defaults
    if smirnoff_config is None:
        smirnoff_config = SmirnoffConfig()

    ff_name = smirnoff_config.forcefield
    water_model = smirnoff_config.water_model
    charge_method = smirnoff_config.charge_method
    if charge_method.lower() == "nagl":
        charge_method = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

    # Cache path includes forcefield version for separation
    # Structure: parameter_store/gromacs/smirnoff/{ff_hash}/{mol_hash}/
    ff_hash = ff_name.replace(".offxml", "").replace("-", "_")
    pardir = settings.parameter_store / "gromacs" / "smirnoff" / ff_hash
    pardir.mkdir(parents=True, exist_ok=True)

    # Handle water
    if species.is_water:
        workdir = pardir / "water"
        itp_path = workdir / "SOL.itp"
        atomtypes_path = workdir / "SOL_atomtypes.itp"
        params_itp_path = workdir / "SOL_params.itp"
        top_path = workdir / "SOL.top"

        if not itp_path.is_file():
            workdir.mkdir(parents=True, exist_ok=True)
            molecule = species.openff_molecule
            # Force molecule name to "SOL" for consistent Interchange output,
            # regardless of the user-provided resname (e.g. "WAT").
            # This ensures atom types are always SOL_0, SOL_1, ... and the
            # generated ITP is always SOL_SOL.itp.
            molecule.name = "SOL"
            topology = Topology.from_molecules([molecule])
            forcefield = ForceField(water_model)
            interchange = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
            interchange.box = np.eye(3) * 3.0 * unit.nanometer

            with working_directory(workdir, exists_ok=True):
                interchange.to_top("SOL.top", monolithic=False)
                # Interchange creates SOL_SOL.itp, rename to SOL.itp
                generated_itp = workdir / "SOL_SOL.itp"
                if generated_itp.is_file():
                    generated_itp.rename(itp_path)
            # Remove stale params if atomtypes were regenerated
            if params_itp_path.is_file():
                params_itp_path.unlink()
        if top_path.is_file() and atomtypes_path.is_file() and not params_itp_path.is_file():
            _build_smirnoff_parameter_itp(top_path, atomtypes_path, params_itp_path)

        return GromacsSingleMoleculeParameterSet(
            moleculetype="SOL",
            smiles=species.smiles,
            parametrization="smirnoff",
            parametrization_config=smirnoff_config,
            itp=itp_path.resolve(),
            parameter_itp=params_itp_path.resolve() if params_itp_path.is_file() else None,
            forcefield_itp=None,
        )

    # Handle ions
    if species.is_ion:
        typemap = {"[Na+]": "NA", "[Cl-]": "CL"}
        ion_type = typemap[species.smiles]
        workdir = pardir / "ions"
        itp_path = workdir / f"{ion_type}.itp"
        atomtypes_path = workdir / f"{ion_type}_atomtypes.itp"
        params_itp_path = workdir / f"{ion_type}_params.itp"
        top_path = workdir / f"{ion_type}.top"

        if not itp_path.is_file():
            workdir.mkdir(parents=True, exist_ok=True)
            molecule = species.openff_molecule
            molecule.assign_partial_charges("formal_charge")
            topology = Topology.from_molecules([molecule])
            forcefield = ForceField(ff_name)
            interchange = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
            interchange.box = np.eye(3) * 3.0 * unit.nanometer

            with working_directory(workdir, exists_ok=True):
                interchange.to_top(f"{ion_type}.top", monolithic=False)
                # Interchange creates {ion_type}_{ion_type}.itp, rename to {ion_type}.itp
                generated_itp = workdir / f"{ion_type}_{ion_type}.itp"
                if generated_itp.is_file():
                    generated_itp.rename(itp_path)
        if top_path.is_file() and atomtypes_path.is_file() and not params_itp_path.is_file():
            _build_smirnoff_parameter_itp(top_path, atomtypes_path, params_itp_path)

        return GromacsSingleMoleculeParameterSet(
            moleculetype=ion_type,
            smiles=species.smiles,
            parametrization="smirnoff",
            parametrization_config=smirnoff_config,
            itp=itp_path.resolve(),
            parameter_itp=params_itp_path.resolve() if params_itp_path.is_file() else None,
            forcefield_itp=None,
        )

    # Regular molecules
    workdir = pardir / species.hash
    itp_path = workdir / f"{species.hash}.itp"
    atomtypes_path = workdir / f"{species.hash}_atomtypes.itp"
    params_itp_path = workdir / f"{species.hash}_params.itp"
    top_path = workdir / f"{species.hash}.top"

    # Check cache
    if itp_path.is_file():
        _replace_moleculetype(itp_path, species.hash)
        if top_path.is_file() and atomtypes_path.is_file():
            _build_smirnoff_parameter_itp(top_path, atomtypes_path, params_itp_path)
        return GromacsSingleMoleculeParameterSet(
            moleculetype=species.hash,
            smiles=species.smiles,
            parametrization="smirnoff",
            parametrization_config=smirnoff_config,
            itp=itp_path.resolve(),
            parameter_itp=params_itp_path.resolve() if params_itp_path.is_file() else None,
            forcefield_itp=None,
        )

    # Generate parameters
    workdir.mkdir(parents=True, exist_ok=False)

    with working_directory(workdir, exists_ok=True):
        molecule = species.openff_molecule

        # Generate conformer if needed (required for charge assignment)
        if molecule.n_conformers == 0:
            raise ValueError(
                "Molecule has no conformers. Generate at least one conformer "
                "before calling parametrization so partial charges can be assigned."
            )

        # Assign partial charges using configured method
        # 'nagl' is fast (neural network), 'am1bcc' is slower but standard
        molecule.assign_partial_charges(charge_method)

        topology = Topology.from_molecules([molecule])
        forcefield = ForceField(ff_name)
        interchange = Interchange.from_smirnoff(
            force_field=forcefield,
            topology=topology,
            # need to specify charge_from_molecules to avoid OpenFF trying to be
            # smart and reassigning charges with a different method
            charge_from_molecules=[molecule],
        )
        interchange.box = np.eye(3) * 3.0 * unit.nanometer

        # Export to GROMACS format
        interchange.to_top(f"{species.hash}.top", monolithic=False)

        # Rename output ITP to expected name
        # Interchange creates {top_basename}_{molecule.name}.itp
        generated_itp = workdir / f"{species.hash}_{species.resname.upper()}.itp"
        if generated_itp.is_file() and generated_itp != itp_path:
            generated_itp.rename(itp_path)
        if itp_path.is_file():
            _replace_moleculetype(itp_path, species.hash)
        if top_path.is_file() and atomtypes_path.is_file():
            _build_smirnoff_parameter_itp(top_path, atomtypes_path, params_itp_path)

    return GromacsSingleMoleculeParameterSet(
        moleculetype=species.hash,
        smiles=species.smiles,
        parametrization="smirnoff",
        parametrization_config=smirnoff_config,
        itp=itp_path.resolve(),
        parameter_itp=params_itp_path.resolve() if params_itp_path.is_file() else None,
        forcefield_itp=None,
    )


def generate_gromacs_topology(u, species, parameters, system_name) -> str:
    """Write a GROMACS topology file from species and their parameters.

    Parameters
    ----------
    u : mda.Universe
        Universe with the system structure
    species : list
        List of species objects
    parameters : list[GromacsSingleMoleculeParameterSet]
        Matching parameter sets for each species
    system_name : str
        Name for the [ system ] section

    """
    if len(species) != len(parameters):
        raise ValueError("Must have parameters for each species.")

    cgenff_top_template = """{ff_includes}
{prm_includes}
{itp_includes}

[ system ]
{system_name}

[ molecules ]
{mol_table}"""
    residues = count_contiguous_strings(u.residues.resnames)

    # NOTE: maybe copy itp files and change "resname" to what was specified in the input?
    # might be more convenient for analysis later?
    ff_files = set(x.forcefield_itp for x in parameters if x.forcefield_itp is not None)

    ff_local = []
    for ff in ff_files:
        shutil.copytree(ff.parent, ff.parent.name, dirs_exist_ok=True)
        ff_local.append(Path(ff.parent.name) / ff.name)

    prm_files = set(x.parameter_itp for x in parameters if x.parameter_itp is not None)
    itp_files = set(x.itp for x in parameters)

    # check for duplicate lines in parameter files and merge them
    prm_merged = merge_extra_parameter_itps(prm_files)
    with open("extra_params.itp", "w") as fb:
        fb.write(f"; merged from {prm_files}\n")
        fb.write(prm_merged)
    prm_local = [Path("extra_params.itp")]
    # for prm in prm_files:
    #     shutil.copy(prm, prm.name)
    #     prm_local.append(Path(prm.name))

    itp_local = []
    for itp in itp_files:
        shutil.copy(itp, itp.name)
        itp_local.append(Path(itp.name))

    moleculetype_for_resname = {
        spec.resname: par.moleculetype for spec, par in zip(species, parameters)
    }
    smiles_for_resname = {spec.resname: spec.smiles for spec in species}

    def join_includes(fl):
        return "\n".join(f'#include "{f}"' for f in fl)

    ff_includes = join_includes(ff_local)
    prm_includes = join_includes(prm_local)
    itp_includes = join_includes(itp_local)
    mol_table = "\n".join(
        f"{moleculetype_for_resname[res]} {count}  ; {res}, {smiles_for_resname[res]}"
        for res, count in residues
    )
    top_str = cgenff_top_template.format(
        ff_includes=ff_includes,
        prm_includes=prm_includes,
        itp_includes=itp_includes,
        system_name=system_name,
        mol_table=mol_table,
    )

    with open("topology.top", "w") as fb:
        fb.write(top_str)


DISPATCH_ENGINE_PARAMETRIZE = {
    "gromacs": {
        "cgenff": parametrize_cgenff_gromacs,
        "smirnoff": parametrize_smirnoff_gromacs,
    }
}
DISPATCH_TOPOLOGY_BUILD = {"gromacs": generate_gromacs_topology}
