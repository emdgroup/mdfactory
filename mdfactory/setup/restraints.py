# ABOUTME: Selection-based position restraints for pdb2gmx protein topologies
# ABOUTME: Maps MDAnalysis selections to moleculetype-local atoms; writes posre itps and includes
"""Selection-based position restraints for pdb2gmx protein topologies.

A :class:`~mdfactory.models.species.PositionRestraint` names protein atoms with
an MDAnalysis selection. The selection is evaluated on the pdb2gmx structure
(force-field atom names) with the chain IDs and residue numbers of the input
structure, mapped to atom indices local to each protein moleculetype, and
written as one ``[ position_restraints ]`` file per restraint and moleculetype.
Each file is included right after pdb2gmx's own ``#ifdef POSRES`` block,
guarded by ``#ifdef POSRES_<NAME>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from loguru import logger

from ..utils.topology_utilities import iter_includes, iter_topology_entries
from .protein import _pdb_residue_key

if TYPE_CHECKING:
    import MDAnalysis as mda

    from ..models.parametrization import GromacsProteinParameterSet
    from ..models.species import PositionRestraint


class ProteinMoleculetype(NamedTuple):
    """A protein moleculetype of a pdb2gmx topology."""

    #: Moleculetype name, e.g. ``Protein_chain_A``.
    name: str
    #: File that defines the moleculetype (``topol.top`` or a chain itp).
    path: Path
    #: Atom names of its ``[ atoms ]`` section, in order.
    atom_names: tuple[str, ...]


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


def select_restraint_atoms(universe: mda.Universe, restraint: PositionRestraint) -> np.ndarray:
    """Return the sorted 0-based indices of the atoms *restraint* selects.

    Raises
    ------
    ValueError
        If the selection is invalid or matches no atom.

    """
    from MDAnalysis.exceptions import SelectionError

    try:
        atoms = universe.select_atoms(restraint.selection)
    except (SelectionError, ValueError) as exc:
        raise ValueError(
            f"Restraint '{restraint.name}': invalid selection {restraint.selection!r}: {exc}"
        ) from exc
    if atoms.n_atoms == 0:
        raise ValueError(
            f"Restraint '{restraint.name}': selection {restraint.selection!r} matches no "
            "protein atom. Atom names are the force field's (pdb2gmx output); chain IDs "
            "(chainID) and residue numbers (resid) are the input structure's."
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

    definitions: dict[str, tuple[Path, list[str]]] = {}
    for path in files:
        current: list[str] | None = None
        for entry in iter_topology_entries(path.read_text().splitlines()):
            if entry.section == "moleculetype":
                current = []
                definitions[entry.code.split()[0]] = (path, current)
            elif entry.section == "atoms" and current is not None:
                current.append(entry.code.split()[4])

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
        path, atom_names = definitions[name]
        moleculetypes.append(ProteinMoleculetype(name, path, tuple(atom_names)))
    return moleculetypes


def check_atom_names(universe: mda.Universe, moleculetypes: list[ProteinMoleculetype]) -> None:
    """Raise ``ValueError`` unless the structure's atoms match the moleculetypes atom by atom."""
    topology_names = [name for moltype in moleculetypes for name in moltype.atom_names]
    structure_names = list(universe.atoms.names)
    if len(topology_names) != len(structure_names):
        raise ValueError(
            f"The protein moleculetypes have {len(topology_names)} atoms but the "
            f"pdb2gmx structure has {len(structure_names)}."
        )
    for index, (expected, found) in enumerate(zip(topology_names, structure_names)):
        if expected != found:
            raise ValueError(
                f"Atom {index + 1} is '{found}' in the pdb2gmx structure but '{expected}' "
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


def apply_position_restraints(
    params: GromacsProteinParameterSet,
    restraints: list[PositionRestraint],
    reference_pdb: Path,
) -> GromacsProteinParameterSet:
    """Write the restraint itps of a pdb2gmx output and include them in its topology.

    Parameters
    ----------
    params : GromacsProteinParameterSet
        pdb2gmx output; its topology files are modified in place.
    restraints : list of PositionRestraint
        Restraints to apply; nothing changes when empty.
    reference_pdb : Path
        The structure pdb2gmx was run on, for chain IDs and residue numbers.

    Returns
    -------
    GromacsProteinParameterSet
        *params* with the new itps added to the include files.

    """
    if not restraints:
        return params
    universe = restraint_selection_universe(params.structure_file, reference_pdb)
    moleculetypes = read_protein_moleculetypes(params.topology_file)
    check_atom_names(universe, moleculetypes)

    output_dir = params.topology_file.parent
    written: list[Path] = []
    includes: dict[ProteinMoleculetype, list[tuple[str, str]]] = {}
    for restraint in restraints:
        indices = select_restraint_atoms(universe, restraint)
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
    return params.model_copy(
        update={"topology_include_files": [*params.topology_include_files, *written]}
    )
