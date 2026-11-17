# ABOUTME: Selection-based position and COM-distance restraints for pdb2gmx protein topologies
# ABOUTME: Maps selections to atoms; writes posre itps/includes, index groups and COM distances
"""Selection-based restraints for pdb2gmx protein topologies.

Selections are evaluated on the pdb2gmx structure (force-field atom names)
with the chain IDs and residue numbers of the input structure.

A :class:`~mdfactory.models.species.PositionRestraint` is mapped to atom
indices local to each protein moleculetype and written as one
``[ position_restraints ]`` file per restraint and moleculetype. Each file is
included right after pdb2gmx's own ``#ifdef POSRES`` block, guarded by
``#ifdef POSRES_<NAME>``.

A :class:`~mdfactory.models.species.ComDistanceRestraint` becomes two index
groups. Its centre-of-mass distance is measured in the built system and stored
in ``com_restraints.json``, from which the production pull block is rendered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from loguru import logger

from ..models.species import ComDistanceRestraint, PositionRestraint
from ..run_schedules.gromacs import INDEX_FILE
from ..utils.topology_utilities import iter_includes, iter_topology_entries
from .protein import _pdb_residue_key, _run_gmx, check_gmx_available

if TYPE_CHECKING:
    import MDAnalysis as mda

    from ..models.parametrization import GromacsProteinParameterSet


class ProteinMoleculetype(NamedTuple):
    """A protein moleculetype of a pdb2gmx topology."""

    #: Moleculetype name, e.g. ``Protein_chain_A``.
    name: str
    #: File that defines the moleculetype (``topol.top`` or a chain itp).
    path: Path
    #: Atom names of its ``[ atoms ]`` section, in order.
    atom_names: tuple[str, ...]
    #: Atom masses (u) of its ``[ atoms ]`` section, in order.
    masses: tuple[float, ...]


def _read_residue_sequence(pdb_path: Path) -> list[tuple[str, int, str]]:
    """Return the ``(chain, resid, insertion code)`` of each PDB residue in file order."""
    residues: list[tuple[str, int, str]] = []
    previous = None
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        key = (*_pdb_residue_key(line), line[17:20].strip())
        if key != previous:
            residues.append(key[:3])
            previous = key
    return residues


def restraint_selection_universe(structure_file: Path, reference_pdb: Path) -> mda.Universe:
    """Return the pdb2gmx structure with the reference PDB's chain IDs and residue numbers.

    pdb2gmx neither adds nor removes residues, so residues are matched by order.

    Parameters
    ----------
    structure_file : Path
        pdb2gmx output structure (``processed.gro``).
    reference_pdb : Path
        The structure pdb2gmx was run on (the cleaned input PDB).

    Returns
    -------
    MDAnalysis.Universe
        Universe with ``chainIDs``, ``resids`` and ``icodes`` from *reference_pdb*.

    Raises
    ------
    ValueError
        If the two structures have different numbers of residues.

    """
    import MDAnalysis as mda

    universe = mda.Universe(str(structure_file))
    residues = _read_residue_sequence(reference_pdb)
    if len(residues) != universe.residues.n_residues:
        raise ValueError(
            f"{structure_file} has {universe.residues.n_residues} residues but "
            f"{reference_pdb} has {len(residues)}; cannot transfer chain IDs and residue "
            "numbers for the restraint selections."
        )
    chains, resids, icodes = zip(*residues)
    universe.residues.resids = np.asarray(resids)
    universe.add_TopologyAttr("icodes", list(icodes))
    universe.add_TopologyAttr(
        "chainIDs", np.asarray(chains, dtype=object)[universe.atoms.resindices]
    )
    return universe


def select_restraint_atoms(universe: mda.Universe, name: str, selection: str) -> np.ndarray:
    """Return the sorted 0-based indices of the atoms *selection* matches.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Selection universe from :func:`restraint_selection_universe`.
    name : str
        Restraint name, for the error messages.
    selection : str
        MDAnalysis selection.

    Raises
    ------
    ValueError
        If the selection is invalid or matches no atom.

    """
    from MDAnalysis.exceptions import SelectionError

    try:
        atoms = universe.select_atoms(selection)
    except (SelectionError, ValueError) as exc:
        raise ValueError(f"Restraint '{name}': invalid selection {selection!r}: {exc}") from exc
    if atoms.n_atoms == 0:
        raise ValueError(
            f"Restraint '{name}': selection {selection!r} matches no protein atom. Atom "
            "names are the force field's (pdb2gmx output); chain IDs (chainID) and "
            "residue numbers (resid) are the input structure's."
        )
    return np.sort(atoms.indices)


def read_protein_moleculetypes(topology_file: Path) -> list[ProteinMoleculetype]:
    """Return the protein moleculetypes of a pdb2gmx topology in ``[ molecules ]`` order.

    Moleculetypes are read from *topology_file* and its local (non-``.ff``)
    includes, i.e. the per-chain itps.

    Raises
    ------
    ValueError
        If a protein molecule has no definition or a count other than 1.

    """
    topology_file = Path(topology_file)
    lines = topology_file.read_text().splitlines()
    files = [topology_file] + [
        topology_file.parent / include
        for _, include in iter_includes(lines)
        if ".ff/" not in include and (topology_file.parent / include).is_file()
    ]

    # name -> (file, atom names, masses)
    definitions: dict[str, tuple[Path, list[str], list[float]]] = {}
    for path in files:
        current = None
        for entry in iter_topology_entries(path.read_text().splitlines()):
            if entry.section == "moleculetype":
                current = definitions[entry.code.split()[0]] = (path, [], [])
            elif entry.section == "atoms" and current is not None:
                # The mass column is optional for non-protein moleculetypes.
                fields = entry.code.split()
                current[1].append(fields[4])
                current[2].append(float(fields[7]) if len(fields) > 7 else float("nan"))

    moleculetypes = []
    for entry in iter_topology_entries(lines):
        if entry.section != "molecules" or not entry.code.startswith("Protein"):
            continue
        name, count = entry.code.split()[:2]
        if name not in definitions:
            raise ValueError(f"Protein moleculetype '{name}' is not defined in {topology_file}.")
        if int(count) != 1:
            raise ValueError(
                f"Protein moleculetype '{name}' is listed {count} times in {topology_file}; "
                "restraints need exactly one copy."
            )
        path, atom_names, masses = definitions[name]
        moleculetypes.append(ProteinMoleculetype(name, path, tuple(atom_names), tuple(masses)))
    return moleculetypes


def check_atom_names(
    names, moleculetypes: list[ProteinMoleculetype], structure: str = "pdb2gmx structure"
) -> None:
    """Raise ``ValueError`` unless *names* match the moleculetypes' atoms one by one.

    Parameters
    ----------
    names : sequence of str
        Atom names of the protein atoms of a structure, in order.
    moleculetypes : list of ProteinMoleculetype
        Protein moleculetypes in ``[ molecules ]`` order.
    structure : str, optional
        Name of the structure in the error messages.

    """
    topology_names = [name for moltype in moleculetypes for name in moltype.atom_names]
    structure_names = list(names)
    if len(topology_names) != len(structure_names):
        raise ValueError(
            f"The protein moleculetypes have {len(topology_names)} atoms but the "
            f"{structure} has {len(structure_names)}."
        )
    for index, (expected, found) in enumerate(zip(topology_names, structure_names)):
        if expected != found:
            raise ValueError(
                f"Atom {index + 1} is '{found}' in the {structure} but '{expected}' "
                "in the topology; cannot map restraint selections."
            )


def split_by_moleculetype(
    indices: np.ndarray, moleculetypes: list[ProteinMoleculetype]
) -> list[tuple[ProteinMoleculetype, np.ndarray]]:
    """Split global protein atom indices into 0-based indices local to each moleculetype.

    Moleculetypes without a selected atom are omitted.
    """
    ends = np.cumsum([len(moltype.atom_names) for moltype in moleculetypes])
    owner = np.searchsorted(ends, indices, side="right")
    result = []
    for position, moltype in enumerate(moleculetypes):
        start = ends[position] - len(moltype.atom_names)
        local = indices[owner == position] - start
        if local.size:
            result.append((moltype, local))
    return result


def format_position_restraints(restraint: PositionRestraint, local_indices: np.ndarray) -> str:
    """Return the ``[ position_restraints ]`` itp text for 0-based moleculetype-local indices."""
    header = [
        f"; Position restraint '{restraint.name}' written by mdfactory",
        f"; selection: {restraint.selection}",
        f"; active with define = -D{restraint.macro}",
        "",
        "[ position_restraints ]",
    ]
    if restraint.type == "harmonic":
        header.append(";  ai  funct  fcx  fcy  fcz   (kJ mol^-1 nm^-2)")
        k = repr(restraint.k)
        rows = [f"{i + 1:6d}      1  {k}  {k}  {k}" for i in local_indices]
    else:
        header.append(";  ai  funct  g  r (nm)  k (kJ mol^-1 nm^-2)   (g = 1: sphere)")
        rows = [
            f"{i + 1:6d}      2  1  {restraint.radius!r}  {restraint.k!r}" for i in local_indices
        ]
    return "\n".join(header + rows) + "\n"


def restraint_itp_name(restraint: PositionRestraint, moleculetype: str) -> str:
    """Return the itp file name of *restraint* for *moleculetype*."""
    return f"posre_{restraint.name.lower()}_{moleculetype}.itp"


def insert_restraint_includes(
    path: Path, moleculetype: str, includes: list[tuple[str, str]]
) -> None:
    """Include restraint itps right after pdb2gmx's ``POSRES`` block of *moleculetype*.

    Parameters
    ----------
    path : Path
        File that defines *moleculetype*; modified in place.
    moleculetype : str
        Moleculetype name.
    includes : list of tuple of str
        ``(macro, itp file name)`` pairs, included in this order.

    Raises
    ------
    RuntimeError
        If the moleculetype has no ``#ifdef POSRES`` block to anchor the includes.

    """
    lines = path.read_text().splitlines(keepends=True)
    # (header line index, moleculetype name) of each [ moleculetype ] in the file.
    definitions: list[tuple[int, str]] = []
    header = None
    for entry in iter_topology_entries(lines, headers=True):
        if entry.section != "moleculetype":
            continue
        if entry.is_header:
            header = entry.index
        elif header is not None:
            definitions.append((header, entry.code.split()[0]))
            header = None
    start = next((index for index, name in definitions if name == moleculetype), None)
    endif = None
    if start is not None:
        end = next((index for index, _ in definitions if index > start), len(lines))
        ifdef = next(
            (i for i in range(start, end) if lines[i].split() == ["#ifdef", "POSRES"]), None
        )
        if ifdef is not None:
            endif = next((i for i in range(ifdef, end) if lines[i].strip() == "#endif"), None)
    if endif is None:
        raise RuntimeError(
            f"No pdb2gmx '#ifdef POSRES' block found for moleculetype '{moleculetype}' in "
            f"{path}; cannot include the selection restraints."
        )
    block = ["\n", "; Selection-based position restraints (mdfactory)\n"]
    for macro, itp in includes:
        block += [f"#ifdef {macro}\n", f'#include "{itp}"\n', "#endif\n"]
    lines[endif + 1 : endif + 1] = block
    path.write_text("".join(lines))


#: Resolved COM restraint groups: restraint name -> 0-based protein atom indices of
#: the two selections.
ComGroups = dict[str, tuple[np.ndarray, np.ndarray]]


def apply_restraints(
    params: GromacsProteinParameterSet,
    restraints: list[PositionRestraint | ComDistanceRestraint],
    reference_pdb: Path,
) -> tuple[GromacsProteinParameterSet, ComGroups]:
    """Apply a protein's restraints to its pdb2gmx output.

    Position restraints are written as itps and included in the topology. The
    selections of COM-distance restraints are resolved to atom indices, so a bad
    selection fails before the system is built around the protein.

    Parameters
    ----------
    params : GromacsProteinParameterSet
        pdb2gmx output; its topology files are modified in place.
    restraints : list
        The protein's restraints; nothing changes when empty.
    reference_pdb : Path
        The structure pdb2gmx was run on, for chain IDs and residue numbers.

    Returns
    -------
    tuple
        *params* with the position-restraint itps added to the include files,
        and the :data:`ComGroups` of the COM-distance restraints.

    Raises
    ------
    ValueError
        If a selection is invalid or empty, or the two groups of a COM-distance
        restraint share atoms.

    """
    if not restraints:
        return params, {}
    universe = restraint_selection_universe(params.structure_file, reference_pdb)
    moleculetypes = read_protein_moleculetypes(params.topology_file)
    check_atom_names(universe.atoms.names, moleculetypes)

    output_dir = params.topology_file.parent
    written: list[Path] = []
    includes: dict[ProteinMoleculetype, list[tuple[str, str]]] = {}
    com_groups: ComGroups = {}
    for restraint in restraints:
        if isinstance(restraint, ComDistanceRestraint):
            com_groups[restraint.name] = _resolve_com_groups(universe, restraint)
            continue
        indices = select_restraint_atoms(universe, restraint.name, restraint.selection)
        for moltype, local in split_by_moleculetype(indices, moleculetypes):
            itp = output_dir / restraint_itp_name(restraint, moltype.name)
            itp.write_text(format_position_restraints(restraint, local))
            written.append(itp)
            includes.setdefault(moltype, []).append((restraint.macro, itp.name))
        logger.info(
            f"Restraint '{restraint.name}' ({restraint.type}, k = {restraint.k}): "
            f"{indices.size} atoms selected by {restraint.selection!r}."
        )
    for moltype, entries in includes.items():
        insert_restraint_includes(moltype.path, moltype.name, entries)
    if written:
        params = params.model_copy(
            update={"topology_include_files": [*params.topology_include_files, *written]}
        )
    return params, com_groups


def _resolve_com_groups(
    universe: mda.Universe, restraint: ComDistanceRestraint
) -> tuple[np.ndarray, np.ndarray]:
    """Return the atom indices of both selections of a COM-distance restraint."""
    first, second = (
        select_restraint_atoms(universe, restraint.name, selection)
        for selection in restraint.selections
    )
    shared = np.intersect1d(first, second)
    if shared.size:
        raise ValueError(
            f"Restraint '{restraint.name}': the two selections share {shared.size} atoms; "
            "the groups of a COM-distance restraint must be disjoint."
        )
    logger.info(
        f"Restraint '{restraint.name}' (com_distance, k = {restraint.k}): "
        f"{first.size} and {second.size} atoms."
    )
    return first, second


# --- COM-distance restraints: build-time measurement and files -------------------------

#: Build-time facts of the COM-distance restraints, read by the mdp renderer.
COM_RESTRAINTS_FILE = "com_restraints.json"
#: GROMACS limit for ``pull-coord geometry = distance``, as a fraction of the shortest box edge.
MAX_DISTANCE_BOX_FRACTION = 0.49


class ComMeasurement(NamedTuple):
    """COM distance and periodic reference atoms of a COM-distance restraint."""

    #: Distance between the two centres of mass in nm (minimum image).
    distance: float
    #: Group atom closest to its centre of mass, 1-based in the whole system.
    pbcatoms: tuple[int, int]


def measure_com_distance(
    positions: np.ndarray,
    masses: np.ndarray,
    box: np.ndarray,
    groups: tuple[np.ndarray, np.ndarray],
) -> ComMeasurement:
    """Measure the mass-weighted COM distance of two groups of protein atoms.

    Parameters
    ----------
    positions : numpy.ndarray
        Protein atom positions in Å, shape ``(n, 3)``.
    masses : numpy.ndarray
        Protein atom masses, shape ``(n,)``.
    box : numpy.ndarray
        MDAnalysis box ``[a, b, c, alpha, beta, gamma]`` (Å, degrees).
    groups : tuple of numpy.ndarray
        0-based indices of the two groups. Each group must be whole.

    Returns
    -------
    ComMeasurement
        Minimum-image distance in nm and the 1-based pbc atom of each group.

    """
    from MDAnalysis.lib.distances import calc_bonds

    coms, pbcatoms = [], []
    for group in groups:
        com = np.average(positions[group], axis=0, weights=masses[group])
        nearest = group[np.argmin(np.linalg.norm(positions[group] - com, axis=1))]
        coms.append(com)
        pbcatoms.append(int(nearest) + 1)
    distance = float(calc_bonds(coms[0][None], coms[1][None], box=box)[0]) / 10.0
    return ComMeasurement(distance, (pbcatoms[0], pbcatoms[1]))


def reference_distance(restraint: ComDistanceRestraint, measured: float) -> float:
    """Return the pull ``init`` of *restraint*: its reference distance plus any width (nm)."""
    reference = restraint.distance if restraint.distance is not None else measured
    return reference + (restraint.width or 0.0)


def format_index_groups(groups: dict[str, np.ndarray]) -> str:
    """Return ``.ndx`` text for *groups* (name -> 0-based atom indices)."""
    lines = []
    for name, indices in groups.items():
        lines.append(f"[ {name} ]")
        numbers = [str(i + 1) for i in indices]
        lines += [" ".join(numbers[i : i + 15]) for i in range(0, len(numbers), 15)]
    return "\n".join(lines) + "\n"


def write_com_restraint_files(
    directory: Path,
    restraints: list[ComDistanceRestraint],
    com_groups: ComGroups,
    structure: Path,
    topology: Path,
) -> Path:
    """Measure the COM-distance restraints and write the index and restraint files.

    Writes ``index.ndx`` (the ``gmx make_ndx`` default groups plus the two
    groups of each restraint) and ``com_restraints.json`` (groups, pbc atoms
    and measured distance per restraint) into *directory*.

    Parameters
    ----------
    directory : Path
        Build directory.
    restraints : list of ComDistanceRestraint
        The protein's COM-distance restraints.
    com_groups : ComGroups
        Their groups from :func:`apply_restraints`.
    structure : Path
        Built system; its first atoms are the protein in pdb2gmx order.
    topology : Path
        Topology of the built system, for the protein atom masses.

    Returns
    -------
    Path
        The written index file.

    Raises
    ------
    ValueError
        If the protein atoms of *structure* do not match the topology, or a
        reference distance (plus width) reaches 0.49 times the shortest box edge.

    """
    import MDAnalysis as mda

    directory = Path(directory)
    moleculetypes = read_protein_moleculetypes(topology)
    masses = np.array([m for moltype in moleculetypes for m in moltype.masses])
    universe = mda.Universe(str(structure))
    protein = universe.atoms[: masses.size]
    check_atom_names(protein.names, moleculetypes, structure=f"protein of {structure}")
    box = universe.dimensions
    max_distance = MAX_DISTANCE_BOX_FRACTION * float(min(box[:3])) / 10.0

    entries, groups = {}, {}
    for restraint in restraints:
        measured = measure_com_distance(protein.positions, masses, box, com_groups[restraint.name])
        init = reference_distance(restraint, measured.distance)
        if init >= max_distance:
            raise ValueError(
                f"Restraint '{restraint.name}': pull distance {init:.3f} nm is not below "
                f"{MAX_DISTANCE_BOX_FRACTION} x the shortest box edge ({max_distance:.3f} nm); "
                "use a larger box (padding) or a shorter distance."
            )
        names = restraint.group_names
        groups.update(zip(names, com_groups[restraint.name]))
        entries[restraint.name] = {
            "groups": list(names),
            "pbcatoms": list(measured.pbcatoms),
            "measured_distance": measured.distance,
        }
        logger.info(
            f"Restraint '{restraint.name}': COM distance {measured.distance:.3f} nm in "
            f"{structure.name}; pull reference {init:.3f} nm."
        )

    index = directory / INDEX_FILE
    gmx = str(check_gmx_available())
    _run_gmx(
        "make_ndx",
        [gmx, "make_ndx", "-f", str(Path(structure).resolve()), "-o", str(index.resolve())],
        directory,
        "q\n",
        timeout=120,
    )
    with open(index, "a") as f:
        f.write(format_index_groups(groups))
    (directory / COM_RESTRAINTS_FILE).write_text(json.dumps(entries, indent=2) + "\n")
    return index


def read_com_restraints(directory: Path) -> dict[str, dict]:
    """Return the ``com_restraints.json`` entries of a build directory.

    Raises
    ------
    FileNotFoundError
        If the build directory has no ``com_restraints.json``.

    """
    path = Path(directory) / COM_RESTRAINTS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: the COM-distance restraints are measured at build time; "
            "rebuild the system to render its production mdp."
        )
    return json.loads(path.read_text())
