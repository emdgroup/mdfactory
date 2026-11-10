# ABOUTME: Force field parametrization for molecular species
# ABOUTME: Handles CGenFF and OpenFF (SMIRNOFF) parametrization with GROMACS topology generation
"""Force field parametrization for molecular species."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable

from .models.parametrization import (
    GromacsSingleMoleculeParameterSet,
    OpenFFSmallMolecules,
)
from .models.species import SingleMoleculeSpecies
from .settings import settings
from .utils.topology_utilities import (
    check_atomtype_collisions,
    check_defaults_compatible,
    collect_forcefield_atomtype_definitions,
    count_contiguous_strings,
    extract_reusable_parts_from_cgenff_gmx_top,
    merge_extra_parameter_itps,
    read_itp_atoms,
    rename_atomtypes,
    run_cgenff_to_gmx,
    unique_atom_names,
    validate_cgenff_atomtype_compatibility,
    write_itp_with_atom_names,
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


NATIVE_ION_MOLECULETYPES = {"[Na+]": "NA", "[Cl-]": "CL"}


@dataclass(frozen=True)
class NativeForceField:
    """A GROMACS force-field directory that provides native water and ion topologies.

    Parameters
    ----------
    name : str
        Force-field name recorded on the parameter sets (e.g. ``charmm36m``).
    directory : Path
        The ``.ff`` directory holding ``forcefield.itp`` and the water/ion ITPs.
    water_model : str
        Water model name selecting the native water (and, for Amber, ion) files.

    """

    name: str
    directory: Path
    water_model: str


def _default_cgenff_native_forcefield(water_model: str) -> NativeForceField:
    """Return the ``charmm36.ff`` shipped with the CGenFF installation."""
    return NativeForceField(
        name="charmm36",
        directory=settings.cgenff_dir / "data" / "gromacs" / "charmm36.ff",
        water_model=water_model,
    )


def parametrize_cgenff_gromacs(
    species: SingleMoleculeSpecies, spec, native_forcefield: NativeForceField | None = None
) -> GromacsSingleMoleculeParameterSet:
    """Parametrize a species using CGenFF for GROMACS.

    Water and Na+/Cl- are taken from a CHARMM force-field directory
    (``<water_model>.itp`` and ``ions.itp``). Other molecules are parametrized
    with CGenFF and cached in ``parameter_store/gromacs/cgenff/<species hash>/``,
    which is locked while it is checked and written.

    Parameters
    ----------
    species : SingleMoleculeSpecies
        Molecule to parametrize
    spec : CharmmParametrization
        Resolved charmm family spec of the build.
    native_forcefield : NativeForceField or None, optional
        CHARMM directory providing native water/ion topologies (e.g. the bundled
        charmm36m of a protein build). The ``charmm36.ff`` of the CGenFF
        installation is used when omitted.

    Returns
    -------
    GromacsSingleMoleculeParameterSet
        Parameter set with paths to ITP and forcefield files

    """
    native = native_forcefield or _default_cgenff_native_forcefield(spec.water_model)
    forcefield_itp = native.directory / "forcefield.itp"

    # check for water/ions and return standard parameters
    if species.is_water:
        return GromacsSingleMoleculeParameterSet(
            moleculetype="SOL",
            smiles=species.smiles,
            component="water",
            forcefield=native.name,
            water_model=native.water_model,
            topology_merge="native",
            itp=native.directory / f"{native.water_model}.itp",
            parameter_itp=None,
            forcefield_itp=forcefield_itp,
        )
    if species.is_ion:
        return GromacsSingleMoleculeParameterSet(
            moleculetype=NATIVE_ION_MOLECULETYPES[species.smiles],
            smiles=species.smiles,
            component="ion",
            forcefield=native.name,
            topology_merge="native",
            itp=native.directory / "ions.itp",
            parameter_itp=None,
            forcefield_itp=forcefield_itp,
        )

    # parametrize molecule with cgenff
    pardir = settings.parameter_store / "gromacs" / "cgenff"
    pardir.mkdir(parents=True, exist_ok=True)
    workdir = (pardir / species.hash).resolve()
    itp_path = workdir / f"{species.hash}.itp"
    prm_path = workdir / f"{species.hash}_extra_params.itp"

    def parameter_set():
        return GromacsSingleMoleculeParameterSet(
            moleculetype=species.hash,
            smiles=species.smiles,
            component="small_molecule",
            forcefield="cgenff",
            topology_merge="cgenff",
            itp=itp_path,
            parameter_itp=prm_path,
            forcefield_itp=forcefield_itp,
        )

    with lock_local_folder(workdir):
        if itp_path.is_file() and prm_path.is_file():
            return parameter_set()

        # A directory without both outputs is a failed or interrupted attempt.
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
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

    return parameter_set()


# Bump when the layout or content of OpenFF cache entries changes, so entries
# written by an older mdfactory are never reused.
OPENFF_CACHE_FORMAT_VERSION = 1
OPENFF_CACHE_COMPLETE_MARKER = ".complete"
OPENFF_CACHE_PROVENANCE = "provenance.json"


def _key_digest(key: dict, length: int = 10) -> str:
    """Return a short, stable hex digest of a JSON-serializable cache key."""
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode("UTF-8")).hexdigest()[:length]


def _readable_stem(name: str) -> str:
    """Return a directory-safe stem of a force-field or water-model name."""
    stem = name.removesuffix(".offxml")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)


@dataclass(frozen=True)
class OpenFFCacheEntry:
    """Location and identity of one OpenFF parameter set in the parameter store.

    Parameters
    ----------
    workdir : Path
        Directory holding the entry; it is also the locked path.
    name : str
        File stem of the generated files (species hash, ``SOL`` or the ion name).
    key : dict
        Inputs that determine the parameters (force field, charge method or
        water model, and the cache format version).
    species_hash : str
        Hash of the parametrized species.

    """

    workdir: Path
    name: str
    key: dict
    species_hash: str

    @property
    def itp(self) -> Path:
        return self.workdir / f"{self.name}.itp"

    @property
    def atomtypes(self) -> Path:
        return self.workdir / f"{self.name}_atomtypes.itp"

    @property
    def params(self) -> Path:
        return self.workdir / f"{self.name}_params.itp"

    @property
    def top(self) -> Path:
        return self.workdir / f"{self.name}.top"

    @property
    def provenance(self) -> Path:
        return self.workdir / OPENFF_CACHE_PROVENANCE

    @property
    def marker(self) -> Path:
        return self.workdir / OPENFF_CACHE_COMPLETE_MARKER

    @property
    def atomtype_prefix(self) -> str:
        """Return the atom-type prefix: a letter plus nine hex digits of the set identity."""
        identity = {"species": self.species_hash, "key": self.key}
        return "T" + _key_digest(identity, length=9).upper()

    def is_complete(self) -> bool:
        """Return True if every artifact exists and the completion marker was written."""
        artifacts = (self.itp, self.atomtypes, self.params, self.top, self.provenance)
        return self.marker.is_file() and all(path.is_file() for path in artifacts)


def _openff_root() -> Path:
    return settings.parameter_store / "gromacs" / "openff"


def openff_small_molecule_cache_entry(
    species: SingleMoleculeSpecies, small_molecules: OpenFFSmallMolecules
) -> OpenFFCacheEntry:
    """Return the cache entry of an OpenFF small molecule.

    Keyed on the force field, the resolved charge method (``nagl`` and the
    default NAGL model share an entry) and the cache format version.
    """
    key = {
        "format": OPENFF_CACHE_FORMAT_VERSION,
        "component": "small_molecule",
        "forcefield": small_molecules.forcefield,
        "charge_method": small_molecules.resolved_charge_method,
    }
    keydir = f"{_readable_stem(small_molecules.forcefield)}_{_key_digest(key)}"
    workdir = _openff_root() / "small_molecules" / keydir / species.hash
    return OpenFFCacheEntry(workdir.resolve(), species.hash, key, species.hash)


def openff_water_cache_entry(species: SingleMoleculeSpecies, water_model: str) -> OpenFFCacheEntry:
    """Return the cache entry of OpenFF water, keyed on the water model."""
    key = {"format": OPENFF_CACHE_FORMAT_VERSION, "component": "water", "water_model": water_model}
    workdir = _openff_root() / "water" / f"{_readable_stem(water_model)}_{_key_digest(key)}"
    return OpenFFCacheEntry(workdir.resolve(), "SOL", key, species.hash)


def openff_ion_cache_entry(species: SingleMoleculeSpecies, forcefield: str) -> OpenFFCacheEntry:
    """Return the cache entry of an OpenFF ion, keyed on the force field."""
    ion = NATIVE_ION_MOLECULETYPES[species.smiles]
    key = {"format": OPENFF_CACHE_FORMAT_VERSION, "component": "ion", "forcefield": forcefield}
    keydir = f"{_readable_stem(forcefield)}_{_key_digest(key)}"
    workdir = _openff_root() / "ions" / keydir / ion
    return OpenFFCacheEntry(workdir.resolve(), ion, key, species.hash)


def _openff_versions() -> dict[str, str]:
    """Return the installed OpenFF toolkit and interchange versions (provenance only)."""
    import openff.interchange  # noqa: PLC0415
    import openff.toolkit  # noqa: PLC0415

    return {
        "openff-toolkit": openff.toolkit.__version__,
        "openff-interchange": openff.interchange.__version__,
    }


def _ensure_openff_entry(
    entry: OpenFFCacheEntry,
    species: SingleMoleculeSpecies,
    moleculetype: str,
    build_interchange: Callable[[], tuple[object, str]],
) -> None:
    """Generate *entry* under its lock unless a complete entry already exists.

    An incomplete directory (no completion marker, or a missing artifact) is
    removed and regenerated. Atom types are renamed to the entry's
    deterministic prefix, so sets of different molecules never share names.

    Parameters
    ----------
    entry : OpenFFCacheEntry
        Cache entry to check or write.
    species : SingleMoleculeSpecies
        Parametrized species (recorded in the provenance file).
    moleculetype : str
        Moleculetype name written into the ITP.
    build_interchange : callable
        Returns ``(interchange, molecule_name)``; Interchange names the
        molecule ITP ``<entry.name>_<molecule_name>.itp``.

    """
    entry.workdir.parent.mkdir(parents=True, exist_ok=True)
    with lock_local_folder(entry.workdir):
        if entry.is_complete():
            return
        if entry.workdir.exists():
            shutil.rmtree(entry.workdir)
        entry.workdir.mkdir()

        interchange, molecule_name = build_interchange()
        with working_directory(entry.workdir, exists_ok=True):
            interchange.to_top(entry.top.name, monolithic=False)
        generated_itp = entry.workdir / f"{entry.name}_{molecule_name}.itp"
        if not generated_itp.is_file() or not entry.atomtypes.is_file():
            raise RuntimeError(
                f"OpenFF Interchange did not write the expected '{generated_itp.name}' and "
                f"'{entry.atomtypes.name}' in {entry.workdir}."
            )
        generated_itp.rename(entry.itp)
        _replace_moleculetype(entry.itp, moleculetype)
        rename_atomtypes(entry.itp, entry.atomtypes, entry.atomtype_prefix)
        _build_smirnoff_parameter_itp(entry.top, entry.atomtypes, entry.params)

        provenance = {
            "key": entry.key,
            "smiles": species.smiles,
            "species_hash": species.hash,
            "moleculetype": moleculetype,
            "atomtype_prefix": entry.atomtype_prefix,
            "versions": _openff_versions(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        entry.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
        # Written last: only a fully written entry is ever reused.
        entry.marker.touch()


def _native_openff_parameter_set(
    species: SingleMoleculeSpecies, native: NativeForceField
) -> GromacsSingleMoleculeParameterSet:
    """Return water or ion parameters taken from a GROMACS-native force-field directory."""
    if species.is_water:
        moleculetype = "SOL"
        component = "water"
        itp = native.directory / f"{native.water_model}.itp"
    else:
        moleculetype = NATIVE_ION_MOLECULETYPES[species.smiles]
        component = "ion"
        itp = native.directory / f"ions_{native.water_model}.itp"
    return GromacsSingleMoleculeParameterSet(
        moleculetype=moleculetype,
        smiles=species.smiles,
        component=component,
        forcefield=native.name,
        water_model=native.water_model,
        topology_merge="native",
        itp=itp,
        parameter_itp=None,
        forcefield_itp=native.directory / "forcefield.itp",
    )


def parametrize_openff_gromacs(
    species: SingleMoleculeSpecies, spec, native_forcefield: NativeForceField | None = None
) -> GromacsSingleMoleculeParameterSet:
    """Parametrize a molecule with an OpenFF (SMIRNOFF) force field for GROMACS.

    Uses OpenFF Interchange to generate GROMACS ITP files, cached under
    ``parameter_store/gromacs/openff/``: small molecules per force field and
    resolved charge method, water per water model (``<water_model>.offxml``),
    and ions per force field. Each entry directory is locked while it is
    checked and written.

    Parameters
    ----------
    species : SingleMoleculeSpecies
        The molecule species to parametrize.
    spec : AmberParametrization or OpenFFParametrization
        Resolved family spec; provides ``small_molecules`` and ``water_model``.
    native_forcefield : NativeForceField or None, optional
        GROMACS force-field directory (e.g. a bundled ``amber14sb.ff``) that
        provides native water and ion topologies. When given, water returns
        ``<water_model>.itp`` and Na+/Cl- return ``ions_<water_model>.itp`` from
        it, without any OpenFF work or cache access. Default None.

    Returns
    -------
    GromacsSingleMoleculeParameterSet
        The parameter set containing paths to generated ITP files.

    """
    if native_forcefield is not None and (species.is_water or species.is_ion):
        return _native_openff_parameter_set(species, native_forcefield)

    # Lazy imports to avoid loading OpenFF when not needed
    import numpy as np  # noqa: PLC0415
    from openff.interchange import Interchange  # noqa: PLC0415
    from openff.toolkit import ForceField, Topology  # noqa: PLC0415
    from openff.units import unit  # noqa: PLC0415

    small_molecules: OpenFFSmallMolecules = spec.small_molecules

    def to_interchange(molecule, forcefield_name: str, **kwargs):
        topology = Topology.from_molecules([molecule])
        interchange = Interchange.from_smirnoff(
            force_field=ForceField(forcefield_name), topology=topology, **kwargs
        )
        interchange.box = np.eye(3) * 3.0 * unit.nanometer
        return interchange

    if species.is_water:
        entry = openff_water_cache_entry(species, spec.water_model)
        forcefield_name = f"{spec.water_model}.offxml"
        moleculetype = "SOL"

        def build():
            molecule = species.openff_molecule
            # Name the molecule SOL regardless of the user's resname (e.g. WAT),
            # so the generated ITP is always SOL_SOL.itp.
            molecule.name = "SOL"
            return to_interchange(molecule, forcefield_name), molecule.name

        _ensure_openff_entry(entry, species, moleculetype, build)
        return GromacsSingleMoleculeParameterSet(
            moleculetype=moleculetype,
            smiles=species.smiles,
            component="water",
            forcefield=forcefield_name,
            water_model=spec.water_model,
            topology_merge="smirnoff",
            itp=entry.itp,
            parameter_itp=entry.params,
            forcefield_itp=None,
        )

    if species.is_ion:
        entry = openff_ion_cache_entry(species, small_molecules.forcefield)
        moleculetype = entry.name

        def build():
            molecule = species.openff_molecule
            molecule.name = moleculetype
            molecule.assign_partial_charges("formal_charge")
            return to_interchange(molecule, small_molecules.forcefield), molecule.name

        _ensure_openff_entry(entry, species, moleculetype, build)
        return GromacsSingleMoleculeParameterSet(
            moleculetype=moleculetype,
            smiles=species.smiles,
            component="ion",
            forcefield=small_molecules.forcefield,
            topology_merge="smirnoff",
            itp=entry.itp,
            parameter_itp=entry.params,
            forcefield_itp=None,
        )

    entry = openff_small_molecule_cache_entry(species, small_molecules)
    charge_method = small_molecules.resolved_charge_method

    def build():
        molecule = species.openff_molecule
        if molecule.n_conformers == 0:
            raise ValueError(
                "Molecule has no conformers. Generate at least one conformer "
                "before calling parametrization so partial charges can be assigned."
            )
        molecule.assign_partial_charges(charge_method)
        # charge_from_molecules keeps OpenFF from reassigning charges with another method.
        interchange = to_interchange(
            molecule, small_molecules.forcefield, charge_from_molecules=[molecule]
        )
        return interchange, molecule.name

    _ensure_openff_entry(entry, species, species.hash, build)
    return GromacsSingleMoleculeParameterSet(
        moleculetype=species.hash,
        smiles=species.smiles,
        component="small_molecule",
        forcefield=small_molecules.forcefield,
        charge_method=charge_method,
        topology_merge="smirnoff",
        itp=entry.itp,
        parameter_itp=entry.params,
        forcefield_itp=None,
    )


def resolve_parametrizer(
    spec, engine: str = "gromacs", native_forcefield: NativeForceField | None = None
) -> Callable[[SingleMoleculeSpecies], GromacsSingleMoleculeParameterSet]:
    """Return the parametrizer for the small molecules, water and ions of a build.

    Parameters
    ----------
    spec : CharmmParametrization or AmberParametrization or OpenFFParametrization
        Resolved family spec of the build. Its small-molecule force-field kind
        selects the function in :data:`DISPATCH_ENGINE_PARAMETRIZE`.
    engine : str, optional
        MD engine. Default ``"gromacs"``.
    native_forcefield : NativeForceField or None, optional
        Force-field directory providing native water/ion topologies (the
        bundled protein force field of a protein build).

    Returns
    -------
    callable
        ``f(species) -> parameter set``.

    """
    if spec.small_molecules is None:
        raise ValueError(
            f"The {spec.family} parametrization of this build has no small-molecule component."
        )
    kind = spec.small_molecules.kind
    function = DISPATCH_ENGINE_PARAMETRIZE.get(engine, {}).get(kind)
    if function is None:
        raise NotImplementedError(f"Cannot parametrize with {kind} for engine {engine}.")
    return partial(function, spec=spec, native_forcefield=native_forcefield)


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

    # SMIRNOFF atom types are private per parameter set; a name defined twice with
    # different parameters would silently take the first definition.
    smirnoff_prm_files = sorted(
        {str(x.parameter_itp) for x in parameters if x.topology_merge == "smirnoff"}
    )
    check_atomtype_collisions(smirnoff_prm_files, {})

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


def _read_include_layout(
    lines: list[str],
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return line indices of the forcefield, water, ``[ system ]``, ``[ molecules ]`` anchors.

    Each index is None if absent. The water anchor is the first force-field
    include after ``forcefield.itp``.
    """
    ff_include_idx = None
    water_include_idx = None
    system_idx = None
    molecules_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#include") and '"' in stripped:
            include_path = stripped.split('"')[1]
            if ".ff/" not in include_path:
                continue
            if "forcefield.itp" in include_path:
                if ff_include_idx is None:
                    ff_include_idx = i
            elif ff_include_idx is not None and water_include_idx is None:
                water_include_idx = i
        elif stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            if section == "system" and system_idx is None:
                system_idx = i
            elif section == "molecules" and molecules_idx is None:
                molecules_idx = i
    return ff_include_idx, water_include_idx, system_idx, molecules_idx


def generate_gromacs_topology_with_protein(
    protein_top_path: Path,
    ff_dir: Path,
    species: list,
    parameters: list[GromacsSingleMoleculeParameterSet],
    counts: list[int],
    system_name: str,
    out_path: Path = Path("topology.top"),
) -> None:
    """Write one GROMACS topology for a pdb2gmx protein packed with solution molecules.

    Builds a single composite topology in strict include order: the bundled
    protein force field, the merged small-molecule parameters, the protein
    moleculetypes and position restraints from pdb2gmx, the small-molecule
    moleculetypes, and finally the water and ion topologies the pdb2gmx
    topology already includes. The ``[ molecules ]`` section keeps the protein
    chain entries and appends the solution molecules in the order given, so it
    matches a coordinate file ordered protein → small molecules → water → ions.

    The small-molecule parameters are merged according to the parameter sets'
    ``topology_merge`` kind:

    - **CGenFF** (CHARMM force field): ``[ atomtypes ]`` already defined by the
      force field are de-duplicated after a compatibility check, and
      ``[ defaults ]`` is dropped because the force field bundles CGenFF.
    - **SMIRNOFF** (e.g. Sage with ``amber14sb.ff``): each file's
      ``[ defaults ]`` must match the force field's ``forcefield.itp``
      numerically before it is dropped, and any atom-type name collision with
      the force field (or a conflicting duplicate between solutes) raises.

    Mixing CGenFF and SMIRNOFF solutes raises ``ValueError``.

    ``native`` sets (water and supported ions) use the moleculetypes the pdb2gmx
    force-field includes already provide; only the other sets contribute an
    ``.itp`` include and merged parameters.

    Parameters
    ----------
    protein_top_path : Path
        Local pdb2gmx topology whose force-field ``#include`` lines resolve from
        its own directory (force field already bundled alongside it).
    ff_dir : Path
        The bundled force-field directory, scanned for atom types (and, for
        SMIRNOFF solutes, its ``forcefield.itp`` ``[ defaults ]``).
    species : list
        Non-protein solution species, in the coordinate/packing order.
    parameters : list[GromacsSingleMoleculeParameterSet]
        Parameter sets aligned with ``species``.
    counts : list[int]
        Actual molecule count for each species, aligned with ``species``.
    system_name : str
        Name written into the ``[ system ]`` section (inherited from pdb2gmx).
    out_path : Path, optional
        Destination topology path. Its parent receives the merged parameter file
        and copied small-molecule ITPs. Default ``topology.top`` in the cwd.

    """
    out_path = Path(out_path)
    out_dir = out_path.parent

    lines = protein_top_path.read_text().splitlines(keepends=True)
    ff_include_idx, water_include_idx, system_idx, molecules_idx = _read_include_layout(lines)
    if ff_include_idx is None:
        raise RuntimeError(
            f"No force-field include found in {protein_top_path}; cannot build composite topology."
        )
    if molecules_idx is None:
        raise RuntimeError(
            f"No [ molecules ] section found in {protein_top_path}; "
            "cannot build composite topology."
        )

    # Native water/ions reuse the moleculetypes the pdb2gmx force-field includes
    # already provide; every other set contributes an ITP and merged parameters.
    solute_parameters = [par for par in parameters if par.topology_merge != "native"]
    merge_kinds = {par.topology_merge for par in solute_parameters}
    if len(merge_kinds) > 1:
        raise ValueError(
            "Cannot mix small-molecule parametrizations in one protein topology: "
            f"{sorted(merge_kinds)}."
        )
    merge_kind = merge_kinds.pop() if merge_kinds else None
    label = {"smirnoff": "SMIRNOFF", "cgenff": "CGenFF", None: ""}[merge_kind]

    extra_params_include = ""
    if solute_parameters:
        forcefield_types = collect_forcefield_atomtype_definitions(ff_dir)
        prm_files = sorted({str(par.parameter_itp) for par in solute_parameters})
        if merge_kind == "smirnoff":
            # SMIRNOFF files carry their own [ defaults ] and private atom types:
            # the force field's [ defaults ] must be equivalent before it is dropped,
            # and no atom type may shadow a force-field or another solute's type.
            for prm in prm_files:
                check_defaults_compatible(ff_dir / "forcefield.itp", prm)
            check_atomtype_collisions(prm_files, forcefield_types)
            merged = merge_extra_parameter_itps(prm_files, drop_defaults=True)
        else:
            validate_cgenff_atomtype_compatibility(prm_files, ff_dir, forcefield_types)
            merged = merge_extra_parameter_itps(
                prm_files, exclude_atomtypes=forcefield_types, drop_defaults=True
            )
        (out_dir / "extra_params.itp").write_text(f"; merged from {prm_files}\n{merged}")
        extra_params_include = '#include "extra_params.itp"\n'

    # Copy each unique small-molecule ITP next to the topology and include it once.
    # SMIRNOFF ITPs name atoms by element only; the copies get the unique names
    # that topology_atom_names() also gives the coordinates.
    itps: dict[str, Path] = {}
    for par in solute_parameters:
        itps.setdefault(par.itp.name, par.itp)
    for itp_name, itp_path in itps.items():
        if merge_kind == "smirnoff":
            names = [name for name, _ in read_itp_atoms(itp_path)]
            write_itp_with_atom_names(itp_path, out_dir / itp_name, unique_atom_names(names))
        else:
            shutil.copy(itp_path, out_dir / itp_name)
    smallmol_include_block = "".join(f'#include "{name}"\n' for name in itps)

    # Copy up to [ molecules ], inserting merged parameters after the force field,
    # small-molecule moleculetypes before water, and replacing the [ system ] title.
    new_lines: list[str] = []
    system_title_written = False
    for i in range(molecules_idx):
        line = lines[i]
        if i == water_include_idx and smallmol_include_block:
            new_lines.append(f"; Include {label} small-molecule topologies\n")
            new_lines.append(smallmol_include_block)
        stripped = line.strip()
        if (
            system_idx is not None
            and i > system_idx
            and not system_title_written
            and stripped
            and not stripped.startswith(";")
        ):
            new_lines.append(f"{system_name}\n")
            system_title_written = True
            continue
        new_lines.append(line)
        if i == ff_include_idx and extra_params_include:
            new_lines.append(f"; Include merged {label} parameters\n")
            new_lines.append(extra_params_include)

    # No water include: append the small-molecule moleculetypes here instead.
    if smallmol_include_block and water_include_idx is None:
        new_lines.append(f"; Include {label} small-molecule topologies\n")
        new_lines.append(smallmol_include_block)

    protein_mol_lines = []
    for i in range(molecules_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        protein_mol_lines.append(stripped)

    new_lines.append("[ molecules ]\n")
    new_lines.append("; Compound        #mols\n")
    for mol_line in protein_mol_lines:
        new_lines.append(mol_line + "\n")
    for spec, par, count in zip(species, parameters, counts, strict=True):
        if count <= 0:
            continue
        new_lines.append(f"{par.moleculetype:<20s} {count}  ; {spec.resname}, {spec.smiles}\n")

    out_path.write_text("".join(new_lines))


# Parametrizers per engine and small-molecule force-field kind. Each is called as
# ``f(species, spec=<resolved family spec>, native_forcefield=<NativeForceField|None>)``.
DISPATCH_ENGINE_PARAMETRIZE = {
    "gromacs": {
        "cgenff": parametrize_cgenff_gromacs,
        "openff": parametrize_openff_gromacs,
    }
}
DISPATCH_TOPOLOGY_BUILD = {"gromacs": generate_gromacs_topology}
