# ABOUTME: Molecular topology file manipulation utilities
# ABOUTME: Handles CGenFF/GROMACS topology conversion, mol2 generation from RDKit, and ITP merging
"""Molecular topology file manipulation utilities."""

import os
import subprocess
import warnings
from collections import defaultdict
from collections.abc import Collection
from itertools import groupby
from pathlib import Path

from rdkit import Chem


def _get_sybyl_atom_type(atom):  # noqa: PLR0911
    """Return the Sybyl atom type string for an RDKit atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom object.

    Returns
    -------
    str
        Sybyl atom type (e.g. "C.3", "N.ar", "O.2").

    """
    atomic_num = atom.GetAtomicNum()
    symbol = atom.GetSymbol()
    is_aromatic = atom.GetIsAromatic()
    hyb = atom.GetHybridization()

    # Hydrogen
    if atomic_num == 1:
        return "H"

    # Carbon
    if atomic_num == 6:
        if is_aromatic:
            return "C.ar"
        if hyb == Chem.rdchem.HybridizationType.SP2:
            # C.cat: guanidinium-like carbon bonded to 3 nitrogens
            n_nitrogen_nbrs = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 7)
            if n_nitrogen_nbrs == 3:
                return "C.cat"
            return "C.2"
        if hyb == Chem.rdchem.HybridizationType.SP:
            return "C.1"
        return "C.3"

    # Nitrogen
    if atomic_num == 7:
        if is_aromatic:
            return "N.ar"
        # N.4: positively charged sp3 nitrogen with 4 heavy/H neighbors
        if atom.GetFormalCharge() > 0 and hyb == Chem.rdchem.HybridizationType.SP3:
            return "N.4"
        # N.am: amide nitrogen (sp2 N single-bonded to a C that has a C=O or C=S)
        if hyb == Chem.rdchem.HybridizationType.SP2:
            mol = atom.GetOwningMol()
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() != 6:
                    continue
                n_c_bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                if not n_c_bond or n_c_bond.GetBondTypeAsDouble() >= 2:
                    continue
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() == atom.GetIdx() or nbr2.GetAtomicNum() not in (8, 16):
                        continue
                    c_o_bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), nbr2.GetIdx())
                    if c_o_bond and c_o_bond.GetBondTypeAsDouble() == 2:
                        return "N.am"
            # N.pl3: trigonal planar nitrogen — covers two cases:
            # 1. All single bonds (e.g. aniline-like N conjugated with aromatic ring)
            # 2. Part of a guanidinium group (C bonded to 3 N's → C.cat)
            has_double = any(
                mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble() >= 2
                for nbr in atom.GetNeighbors()
            )
            if not has_double:
                return "N.pl3"
            # Check if double-bonded to a guanidinium carbon (C.cat)
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                if (
                    nbr.GetAtomicNum() == 6
                    and bond.GetBondTypeAsDouble() >= 2
                    and sum(1 for n in nbr.GetNeighbors() if n.GetAtomicNum() == 7) == 3
                ):
                    return "N.pl3"
            return "N.2"
        if hyb == Chem.rdchem.HybridizationType.SP:
            return "N.1"
        return "N.3"

    # Oxygen
    if atomic_num == 8:
        # Negatively charged oxygen (carboxylate, phosphate, sulfonate, etc.)
        if atom.GetFormalCharge() < 0:
            return "O.co2"
        if hyb == Chem.rdchem.HybridizationType.SP2 and atom.GetDegree() == 1:
            # Check if the neighboring C also has an O- (carboxylate partner)
            mol = atom.GetOwningMol()
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6:
                    for nbr2 in nbr.GetNeighbors():
                        if (
                            nbr2.GetIdx() != atom.GetIdx()
                            and nbr2.GetAtomicNum() == 8
                            and nbr2.GetFormalCharge() < 0
                        ):
                            return "O.co2"
            return "O.2"
        # Aromatic O in 5-membered rings (furan): Sybyl convention is O.2, not O.ar
        if is_aromatic:
            return "O.2"
        return "O.3"

    # Sulfur
    if atomic_num == 16:
        # Aromatic S in 5-membered rings (thiophene): Sybyl convention is S.2, not S.ar
        if is_aromatic:
            return "S.2"
        if hyb == Chem.rdchem.HybridizationType.SP2:
            return "S.2"
        # Sulfoxide (S.O) and sulfone (S.O2): count double-bonded oxygens
        mol = atom.GetOwningMol()
        dbl_o = sum(
            1
            for nbr in atom.GetNeighbors()
            if nbr.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble() == 2
        )
        if dbl_o == 1:
            return "S.O"
        if dbl_o >= 2:
            return "S.O2"
        return "S.3"

    # Phosphorus
    if atomic_num == 15:
        return "P.3"

    # Halogens and ions — use bare element symbol
    if atomic_num in (9, 17, 35, 53):  # F, Cl, Br, I
        return symbol

    # Metals / ions — no specific Sybyl subtype
    warnings.warn(
        f"No specific Sybyl atom type for element '{symbol}' "
        f"(atomic number {atomic_num}). Using bare symbol '{symbol}'.",
        stacklevel=2,
    )
    return symbol


def _get_tripos_bond_type(bond):  # noqa: PLR0911
    """Return the Tripos bond type string for an RDKit bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond object.

    Returns
    -------
    str
        Tripos bond type ("1", "2", "3", "ar", or "am").

    """
    if bond.GetIsAromatic():
        return "ar"
    # Carboxylate C-O resonance bonds are labeled "ar" (not phosphate P-O)
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    o_atom = c_atom = None
    if a1.GetAtomicNum() == 8 and a2.GetAtomicNum() == 6:
        o_atom, c_atom = a1, a2
    elif a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 8:
        o_atom, c_atom = a2, a1
    if o_atom is not None and _get_sybyl_atom_type(o_atom) == "O.co2":
        return "ar"
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        # Check for amide bond: single bond between N.am and the carbonyl C
        n_atom = c_atom = None
        if a1.GetAtomicNum() == 7 and a2.GetAtomicNum() == 6:
            n_atom, c_atom = a1, a2
        elif a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 7:
            n_atom, c_atom = a2, a1
        if n_atom is not None and c_atom is not None and _get_sybyl_atom_type(n_atom) == "N.am":
            # Only the bond to the carbonyl carbon is "am", not other N-C bonds
            mol = bond.GetOwningMol()
            for nbr in c_atom.GetNeighbors():
                if nbr.GetIdx() != n_atom.GetIdx() and nbr.GetAtomicNum() == 8:
                    c_o_bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx())
                    if c_o_bond and c_o_bond.GetBondTypeAsDouble() == 2:
                        return "am"
        return "1"
    if bt == Chem.rdchem.BondType.DOUBLE:
        return "2"
    if bt == Chem.rdchem.BondType.TRIPLE:
        return "3"
    warnings.warn(
        f"Unexpected RDKit bond type '{bt}' between atoms "
        f"{bond.GetBeginAtomIdx()} and {bond.GetEndAtomIdx()}. "
        f"Defaulting to single bond ('1').",
        stacklevel=2,
    )
    return "1"


def write_mol2_from_rdkit(rdkit_mol, mol2_path, title="XXX"):
    """Write a Tripos mol2 file directly from an RDKit molecule.

    Produces ``@<TRIPOS>MOLECULE``, ``@<TRIPOS>ATOM``, and ``@<TRIPOS>BOND``
    sections with Sybyl atom types and Tripos bond types suitable for CGenFF
    parametrization.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.Mol
        RDKit molecule with at least one conformer.
    mol2_path : str or Path
        Output file path.
    title : str, optional
        Molecule title written in the MOLECULE section. Default is ``"XXX"``.

    Raises
    ------
    ValueError
        If `rdkit_mol` has no conformers.

    """
    mol2_path = Path(mol2_path)

    if rdkit_mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule must have at least one conformer.")

    conf = rdkit_mol.GetConformer()
    num_atoms = rdkit_mol.GetNumAtoms()
    num_bonds = rdkit_mol.GetNumBonds()

    lines = []

    # MOLECULE section
    lines.append("@<TRIPOS>MOLECULE")
    lines.append(title)
    lines.append(f" {num_atoms} {num_bonds} 0 0 0")
    lines.append("SMALL")
    lines.append("GASTEIGER")  # Charge label; values are zero — CGenFF assigns its own
    lines.append("")

    # ATOM section
    lines.append("@<TRIPOS>ATOM")
    for i in range(num_atoms):
        atom = rdkit_mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        sybyl = _get_sybyl_atom_type(atom)
        # Atom name: bare element symbol (uppercase)
        name = atom.GetSymbol().upper()
        if len(name) == 1:
            name_padded = f"{name:<2s}"
        else:
            name_padded = name
        lines.append(
            f"{i + 1:>7d} {name_padded:<4s}"
            f"{pos.x:>10.4f}{pos.y:>10.4f}{pos.z:>10.4f}"
            f" {sybyl:<8s}1  UNL1        0.0000"
        )

    # BOND section
    lines.append("@<TRIPOS>BOND")
    for i in range(num_bonds):
        bond = rdkit_mol.GetBondWithIdx(i)
        bt = _get_tripos_bond_type(bond)
        lines.append(
            f"{i + 1:>6d}{bond.GetBeginAtomIdx() + 1:>6d}{bond.GetEndAtomIdx() + 1:>6d} {bt:>4s}"
        )

    with open(mol2_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def run_cgenff_to_gmx(mol2_file):
    """Run the CGenFF-to-GROMACS conversion script on a mol2 file.

    Parameters
    ----------
    mol2_file : str
        Path to the mol2 file.

    Returns
    -------
    tuple of (int, str, str)
        ``(return_code, stdout, stderr)`` from the subprocess.

    Raises
    ------
    FileNotFoundError
        If `mol2_file` does not exist.
    RuntimeError
        If the ``SILCSBIODIR`` environment variable is not set, or the
        subprocess fails.

    """
    # Ensure the mol2 file exists
    if not Path(mol2_file).is_file():
        raise FileNotFoundError(f"Mol2 file '{mol2_file}' not found.")

    # Check if SILCSBIODIR is set
    silcsbiodir = os.environ.get("SILCSBIODIR")
    if not silcsbiodir:
        raise RuntimeError("Error: SILCSBIODIR environment variable is not set.")

    # Construct the command
    script_path = os.path.join(silcsbiodir, "cgenff", "cgenff_batch.sh")
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"CGenFF batch script not found at '{script_path}'. "
            f"Verify that SILCSBIODIR ('{silcsbiodir}') is correct and CGenFF is installed."
        )

    command = ["/bin/bash", script_path, f"lig={mol2_file}", "charmm=false"]

    try:
        # Run the script
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ,  # Pass the current environment
        )

        # Capture the output
        stdout, stderr = process.communicate()

        # Get the return code
        return_code = process.returncode

        return return_code, stdout, stderr

    except Exception as e:
        raise RuntimeError("cgenff failed.") from e


def count_contiguous_strings(str_list: list[str]) -> list[tuple[str, int]]:
    """Generate from a list of strings a list of tuples with the string
    as the first element and the number of continuous repetitions as
    second element. For example, ['a', 'a', 'b', 'a'] would yield
    [('a', 2), ('b', 1), ('a', 1)].

    Parameters
    ----------
    str_list : list[str]
        List of strings

    Returns
    -------
    list[tuple[str, int]]
        Contiguous counts of strings.

    """
    return [(x, len(list(group))) for x, group in groupby(str_list)]


def extract_reusable_parts_from_cgenff_gmx_top(
    fname: str | Path,
    replace_name: str = "ABC",
    must_have_moleculetype=True,
) -> tuple[str, str, list[tuple[str, str]]]:
    """Extract reusable moleculetype and parameter sections from a GROMACS topology file
    generated by CGenFF.

    This function parses a topology file, identifies the molecule definition and parameter sections,
    and returns the moleculetype section (with a customizable molecule name) and the concatenated
    parameter sections.

    Parameters
    ----------
    fname : str or Path
        Path to the GROMACS topology file to be parsed.
    replace_name : str, optional
        The string to replace the default molecule name ("XXX") in the molecule section.
        Default is "ABC".
    must_have_moleculetype: bool
        Whether the file MUST have a moleculetype section. Default is True.

    Returns
    -------
    tuple[str, str, list[tuple[str, str]]]
        A tuple containing:
        - The molecule section as a string, with the molecule name replaced.
        - The concatenated parameter sections as a single string.
        - The individual sections

    Raises
    ------
    RuntimeError
        If the required sections cannot be found in the file.

    Notes
    -----
    - The function expects the molecule name to be "XXX" and replaces it with `replace_name`.
    - Parameter sections extracted include: atomtypes, bondtypes, pairtypes,
      angletypes, dihedraltypes, and nonbond_params.

    """
    with open(fname, "r") as fb:
        lines = fb.readlines()
    start = None
    stop = None
    molline = 0

    parameter_sections = [
        "defaults",
        "atomtypes",
        "bondtypes",
        "pairtypes",
        "angletypes",
        "dihedraltypes",
        "nonbond_params",
    ]
    current_section = None
    current_content = []
    sections = []

    for il, line in enumerate(lines):
        if "moleculetype" in line:
            start = il
            molline = il + 2
            assert lines[molline].startswith("XXX")
        if "Include Position restraint file" in line:
            stop = il
            break

        stripped_line = line.strip()
        # Check if this is a section header
        if stripped_line.startswith("[") and stripped_line.endswith("]"):
            # Save previous section if it was a parameter section
            if current_section and current_section in parameter_sections:
                sections.append((current_section, "\n".join(current_content)))
                # sections[current_section] = "\n".join(current_content)

            # Extract section name
            section_name = stripped_line[1:-1].strip()

            # Start new section if it's a parameter section
            if section_name in parameter_sections:
                current_section = section_name
                current_content = [stripped_line]  # Include the section header
            else:
                current_section = None
                current_content = []
        # Add line to current section if we're in a parameter section
        elif current_section and current_section in parameter_sections:
            current_content.append(stripped_line)

    # Don't forget the last section
    if current_section and current_section in parameter_sections:
        sections.append((current_section, "\n".join(current_content)))
        # sections[current_section] = "\n".join(current_content)

    parameter_str = "".join(x[1] for x in sections)

    if len(lines) or must_have_moleculetype:
        lines[molline] = lines[molline].replace("XXX", replace_name)
    if must_have_moleculetype and (start is None or stop is None):
        raise RuntimeError("Could not find sections in file.")
    return "".join(lines[start:stop]), parameter_str, sections


def _atomtype_signature(line: str) -> tuple[str, ...]:
    """Return the physical parameter columns from a GROMACS atom-type line."""
    fields = line.split(";", 1)[0].split()
    if len(fields) < 6:
        raise ValueError(f"Malformed GROMACS atom-type definition: {line!r}")
    signature = []
    for value in fields[-5:]:
        try:
            signature.append(f"{float(value):.12g}")
        except ValueError:
            signature.append(value.lower())
    return tuple(signature)


def collect_forcefield_atomtype_definitions(
    ff_dir: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Return atom-type names and physical parameters from a force-field directory.

    Honors ``#ifdef``/``#ifndef``/``#else``/``#endif`` the way ``grompp`` does.
    mdfactory builds without hydrogen-mass repartitioning, so no force-field
    macros are defined and the ``#ifdef HEAVY_H`` water-hydrogen variants are
    inactive. Without this, a type such as ``HT`` (defined once per branch with
    a different mass) would read as an inconsistent redefinition.
    """
    defined_macros: set[str] = set()
    definitions: dict[str, tuple[str, ...]] = {}
    for itp in sorted(Path(ff_dir).glob("*.itp")):
        in_atomtypes = False
        branch_active: list[bool] = []
        with open(itp) as fb:
            for line in fb:
                stripped = line.strip()
                if not stripped or stripped.startswith(";"):
                    continue
                if stripped.startswith("#"):
                    directive, _, rest = stripped.partition(" ")
                    if directive == "#ifdef":
                        branch_active.append(rest.strip() in defined_macros)
                    elif directive == "#ifndef":
                        branch_active.append(rest.strip() not in defined_macros)
                    elif directive == "#else" and branch_active:
                        branch_active[-1] = not branch_active[-1]
                    elif directive == "#endif" and branch_active:
                        branch_active.pop()
                    continue
                if not all(branch_active):
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_atomtypes = stripped[1:-1].strip() == "atomtypes"
                    continue
                if in_atomtypes:
                    name = stripped.split()[0]
                    signature = _atomtype_signature(stripped)
                    previous = definitions.get(name)
                    if previous is not None and previous != signature:
                        raise ValueError(f"Force field defines atom type '{name}' inconsistently.")
                    definitions[name] = signature
    return definitions


def read_itp_atoms(itp_path: Path) -> list[tuple[str, float]]:
    """Return ``(atom name, mass)`` for each atom of the first moleculetype in an ITP.

    Parameters
    ----------
    itp_path : Path
        GROMACS ``.itp`` file containing a single-molecule ``[ atoms ]`` section.

    Returns
    -------
    list of tuple of (str, float)
        Atom names and masses in topology order.

    Raises
    ------
    ValueError
        If the file has no ``[ atoms ]`` section or an atom line lacks a mass.

    """
    atoms: list[tuple[str, float]] = []
    in_atoms = False
    for raw in Path(itp_path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if in_atoms:
                break
            in_atoms = line[1:-1].strip().lower() == "atoms"
            continue
        if in_atoms:
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f"Atom line without a mass in {itp_path}: {raw!r}")
            atoms.append((fields[4], float(fields[7])))
    if not atoms:
        raise ValueError(f"No [ atoms ] section found in {itp_path}.")
    return atoms


def unique_atom_names(names: list[str]) -> list[str]:
    """Return atom names that are unique within one molecule.

    Names that are already unique are returned unchanged (e.g. CGenFF). SMIRNOFF
    topologies name atoms by element only (``C``, ``C``, ``O``, ``H``, ...);
    those get a per-name counter (``C1``, ``C2``, ``O1``, ``H1``, ...). PDB
    readers such as OpenMM's drop atoms whose name repeats within a residue, so
    coordinates must carry unique names.

    Parameters
    ----------
    names : list of str
        Atom names in topology order.

    Returns
    -------
    list of str
        Unique atom names in the same order.

    Raises
    ------
    ValueError
        If numbering does not make the names unique.

    """
    if len(set(names)) == len(names):
        return list(names)
    counters: dict[str, int] = defaultdict(int)
    numbered = []
    for name in names:
        counters[name] += 1
        numbered.append(f"{name}{counters[name]}")
    if len(set(numbered)) != len(numbered):
        raise ValueError(f"Could not derive unique atom names from {names}.")
    return numbered


def write_itp_with_atom_names(src: str | Path, dest: str | Path, names: list[str]) -> None:
    """Copy an ITP, replacing the atom names of its first ``[ atoms ]`` section.

    Parameters
    ----------
    src : str or Path
        Source ``.itp`` with a single-molecule ``[ atoms ]`` section.
    dest : str or Path
        Destination path.
    names : list of str
        New atom names in topology order; must match the number of atoms.

    Raises
    ------
    ValueError
        If the number of names does not match the ``[ atoms ]`` entries.

    """
    lines = Path(src).read_text().splitlines(keepends=True)
    in_atoms = False
    done = False
    index = 0
    for i, raw in enumerate(lines):
        line = raw.split(";", 1)[0].strip()
        if done or not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if in_atoms:
                done = True
                continue
            in_atoms = line[1:-1].strip().lower() == "atoms"
            continue
        if in_atoms:
            if index >= len(names):
                raise ValueError(f"{src} has more atoms than the {len(names)} names given.")
            fields = line.split()
            fields[4] = names[index]
            comment = raw.split(";", 1)[1].rstrip("\n") if ";" in raw else ""
            lines[i] = " ".join(fields) + (f" ;{comment}" if comment else "") + "\n"
            index += 1
    if index != len(names):
        raise ValueError(f"{src} has {index} atoms but {len(names)} names were given.")
    Path(dest).write_text("".join(lines))


def validate_cgenff_atomtype_compatibility(
    prm_files: list[str | Path],
    ff_dir: str | Path,
    forcefield_types: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """Reject CGenFF atom types that conflict with the selected CHARMM bundle.

    *forcefield_types* may pass the already-parsed result of
    :func:`collect_forcefield_atomtype_definitions` to avoid re-reading *ff_dir*.
    """
    if forcefield_types is None:
        forcefield_types = collect_forcefield_atomtype_definitions(ff_dir)
    for prm in prm_files:
        *_, sections = extract_reusable_parts_from_cgenff_gmx_top(prm, must_have_moleculetype=False)
        for section_name, section_text in sections:
            if section_name != "atomtypes":
                continue
            for line in section_text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith((";", "[")):
                    continue
                name = stripped.split()[0]
                existing = forcefield_types.get(name)
                if existing is None:
                    continue
                generated = _atomtype_signature(stripped)
                if generated != existing:
                    raise ValueError(
                        f"CGenFF atom type '{name}' conflicts with the selected CHARMM "
                        f"force-field bundle ({generated} != {existing}). Use a CGenFF "
                        "installation matching the registered charmm36m bundle."
                    )


DEFAULTS_FUDGE_TOLERANCE = 1e-6


def read_topology_defaults(path: str | Path) -> tuple[int, int, str, float, float] | None:
    """Return the ``[ defaults ]`` entry of a GROMACS topology or include file.

    Parameters
    ----------
    path : str or Path
        Topology, ``forcefield.itp``, or parameter ``.itp`` file. Includes are
        not followed.

    Returns
    -------
    tuple or None
        ``(nbfunc, comb_rule, gen_pairs, fudgeLJ, fudgeQQ)``, with ``gen_pairs``
        lower-cased, or None if the file has no ``[ defaults ]`` section.

    Raises
    ------
    ValueError
        If the ``[ defaults ]`` section has no entry or a malformed one.

    """
    in_defaults = False
    with open(path) as fb:
        for raw in fb:
            line = raw.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                if in_defaults:
                    break
                in_defaults = line[1:-1].strip().lower() == "defaults"
                continue
            if not in_defaults:
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"Malformed [ defaults ] entry in {path}: {raw.strip()!r}")
            try:
                return (
                    int(fields[0]),
                    int(fields[1]),
                    fields[2].lower(),
                    float(fields[3]),
                    float(fields[4]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Malformed [ defaults ] entry in {path}: {raw.strip()!r}"
                ) from exc
    if in_defaults:
        raise ValueError(f"[ defaults ] section in {path} has no entry.")
    return None


def check_defaults_compatible(
    reference_path: str | Path,
    other_path: str | Path,
    tolerance: float = DEFAULTS_FUDGE_TOLERANCE,
) -> None:
    """Raise unless two files declare equivalent ``[ defaults ]`` entries.

    ``nbfunc``, ``comb-rule`` and ``gen-pairs`` must match exactly; ``fudgeLJ``
    and ``fudgeQQ`` must agree within *tolerance*, because force fields write
    the 1-4 scaling with different precision (``0.833333`` versus
    ``0.83333333333333333``).

    Parameters
    ----------
    reference_path : str or Path
        File whose ``[ defaults ]`` is kept, e.g. the force field's ``forcefield.itp``.
    other_path : str or Path
        File whose ``[ defaults ]`` would be dropped, e.g. a SMIRNOFF parameter file.
    tolerance : float, optional
        Absolute tolerance for the fudge factors. Default is 1e-6.

    Raises
    ------
    ValueError
        If either file lacks ``[ defaults ]`` or the entries differ.

    """
    reference = read_topology_defaults(reference_path)
    other = read_topology_defaults(other_path)
    if reference is None or other is None:
        missing = reference_path if reference is None else other_path
        raise ValueError(f"No [ defaults ] section found in {missing}.")
    exact_match = reference[:3] == other[:3]
    fudge_match = all(abs(a - b) <= tolerance for a, b in zip(reference[3:], other[3:]))
    if not (exact_match and fudge_match):
        raise ValueError(
            "Incompatible [ defaults ] (nbfunc, comb-rule, gen-pairs, fudgeLJ, fudgeQQ): "
            f"{reference_path} has {reference}, but {other_path} has {other}. "
            "The small-molecule parameters cannot be combined with this force field."
        )


def check_atomtype_collisions(
    prm_files: list[str | Path],
    forcefield_types: dict[str, tuple[str, ...]],
) -> None:
    """Reject small-molecule atom types that clash with the force field or each other.

    Unlike CGenFF, whose atom types are meant to be shared with the CHARMM force
    field, SMIRNOFF atom types are private per molecule. Any name that the force
    field already defines is therefore a collision. A name defined by two
    parameter files is accepted only if both give the same parameters.

    Parameters
    ----------
    prm_files : list of str or Path
        Parameter files with an ``[ atomtypes ]`` section.
    forcefield_types : dict
        Force-field atom types from :func:`collect_forcefield_atomtype_definitions`.

    Raises
    ------
    ValueError
        On a collision with a force-field atom type, or a name defined with
        different parameters in two parameter files.

    """
    seen: dict[str, tuple[tuple[str, ...], str]] = {}
    for prm in prm_files:
        *_, sections = extract_reusable_parts_from_cgenff_gmx_top(prm, must_have_moleculetype=False)
        for section_name, section_text in sections:
            if section_name != "atomtypes":
                continue
            for line in section_text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith((";", "[")):
                    continue
                name = stripped.split()[0]
                if name in forcefield_types:
                    raise ValueError(
                        f"Atom type '{name}' from {prm} is already defined by the force "
                        "field. Rename the small-molecule residue to avoid the collision."
                    )
                signature = _atomtype_signature(stripped)
                previous = seen.get(name)
                if previous is not None and previous[0] != signature:
                    raise ValueError(
                        f"Atom type '{name}' is defined with different parameters in "
                        f"{previous[1]} and {prm}. Give the species distinct residue names."
                    )
                seen.setdefault(name, (signature, str(prm)))


def merge_extra_parameter_itps(
    prm_files: list[str | Path],
    exclude_atomtypes: Collection[str] | None = None,
    drop_defaults: bool = False,
):
    """Merge parameter sections from multiple CGenFF topology files.

    Extract parameter sections (atomtypes, bondtypes, etc.) from each file,
    deduplicate lines, and concatenate into a single parameter string.

    Parameters
    ----------
    prm_files : list of str or Path
        Paths to GROMACS topology files containing parameter sections.
    exclude_atomtypes : collection of str or None, optional
        Atom type names to drop from the merged ``[ atomtypes ]`` section
        because a bundled force field already defines them. The section header
        is kept. Default is None (keep all atom types).
    drop_defaults : bool, optional
        If True, drop the ``[ defaults ]`` section entirely, because a bundled
        force field already provides it. Default is False.

    Returns
    -------
    str
        Merged, deduplicated parameter sections as a single string.

    """
    section_dict = defaultdict(list)
    for prm in prm_files:
        *_, sections = extract_reusable_parts_from_cgenff_gmx_top(prm, must_have_moleculetype=False)
        for sn, sv in sections:
            if drop_defaults and sn == "defaults":
                continue
            sv_lines = [
                line for line in sv.split("\n") if len(line) > 0 and not line.startswith(";")
            ]
            for svl in sv_lines:
                if (
                    exclude_atomtypes
                    and sn == "atomtypes"
                    and not svl.startswith("[")
                    and svl.split()[0] in exclude_atomtypes
                ):
                    continue
                if svl not in section_dict[sn]:
                    section_dict[sn].append(svl)
    params_clean = ""
    for _, lst in section_dict.items():
        params_clean += "\n".join(lst) + "\n"
    return params_clean
