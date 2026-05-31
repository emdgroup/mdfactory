# ABOUTME: Protein preparation utilities for the proteinbox simulation type
# ABOUTME: Wraps PDBFixer for cleaning and gmx pdb2gmx for topology generation
"""Protein preparation utilities for the proteinbox simulation type."""

import shutil
import subprocess
from pathlib import Path

from loguru import logger

from ..models.parametrization import GromacsProteinParameterSet, Pdb2gmxConfig


def check_gmx_available() -> Path:
    """Verify that the GROMACS gmx binary is available on PATH.

    Returns
    -------
    Path
        Path to the gmx binary.

    Raises
    ------
    RuntimeError
        If gmx is not found on PATH.

    """
    gmx_path = shutil.which("gmx")
    if gmx_path is None:
        raise RuntimeError(
            "GROMACS 'gmx' binary not found on PATH. "
            "Ensure GROMACS is installed and 'module load gromacs' has been run."
        )
    return Path(gmx_path)


def clean_pdb(pdb_path: Path, output_path: Path) -> Path:
    """Clean a PDB file using PDBFixer: remove heterogens, add missing heavy atoms.

    Parameters
    ----------
    pdb_path : Path
        Input PDB file.
    output_path : Path
        Where to write the cleaned PDB.

    Returns
    -------
    Path
        Path to the cleaned PDB file.

    """
    from pdbfixer import PDBFixer  # noqa: PLC0415

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    from openmm.app import PDBFile  # noqa: PLC0415

    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    logger.info(f"PDB cleaned: {pdb_path} -> {output_path}")
    return output_path


def run_pdb2gmx(
    pdb_path: Path,
    config: Pdb2gmxConfig,
    disulfide_bonds: list[tuple[int, int]],
    protonation_states: dict[str, str],
    output_dir: Path,
) -> GromacsProteinParameterSet:
    """Run gmx pdb2gmx to generate GROMACS topology from a protein PDB.

    Parameters
    ----------
    pdb_path : Path
        Cleaned PDB file.
    config : Pdb2gmxConfig
        Force field and water model configuration.
    disulfide_bonds : list[tuple[int, int]]
        Residue ID pairs for disulfide bonds.
    protonation_states : dict[str, str]
        Residue-specific protonation state overrides.
    output_dir : Path
        Directory for output files.

    Returns
    -------
    GromacsProteinParameterSet
        Paths to generated topology, structure, and position restraint files.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_file = output_dir / "processed.gro"
    topology_file = output_dir / "topol.top"
    posre_file = output_dir / "posre.itp"

    cmd = [
        "gmx", "pdb2gmx",
        "-f", str(pdb_path),
        "-o", str(structure_file),
        "-p", str(topology_file),
        "-i", str(posre_file),
        "-ff", config.forcefield,
        "-water", config.water_model,
    ]

    if config.ignh:
        cmd.append("-ignh")

    if config.merge_all:
        cmd.extend(["-merge", "all"])

    if disulfide_bonds:
        cmd.append("-ss")

    logger.info(f"Running pdb2gmx: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        text=True,
        input="",
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"gmx pdb2gmx failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    logger.info("pdb2gmx completed successfully.")

    # Verify expected outputs exist
    for path in [structure_file, topology_file, posre_file]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected pdb2gmx output not found: {path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    total_charge = extract_charge_from_topology(topology_file)
    logger.info(f"Protein net charge from topology: {total_charge}")

    return GromacsProteinParameterSet(
        topology_file=topology_file,
        structure_file=structure_file,
        position_restraint_file=posre_file,
        forcefield=config.forcefield,
        water_model=config.water_model,
        total_charge=total_charge,
    )


def extract_charge_from_topology(top_path: Path) -> int:
    """Parse the topology file to determine the net system charge.

    Reads the [ atoms ] section(s) and sums the charge column.

    Parameters
    ----------
    top_path : Path
        Path to the GROMACS .top file.

    Returns
    -------
    int
        Net formal charge (rounded to nearest integer).

    """
    total_charge = 0.0
    in_atoms = False

    with open(top_path) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("#include"):
                # Follow includes for itp files in the same directory
                include_path = stripped.split('"')[1] if '"' in stripped else None
                if include_path and not include_path.endswith(".ff/"):
                    itp_path = top_path.parent / include_path
                    if itp_path.is_file():
                        total_charge += _sum_charges_from_itp(itp_path)
                continue

            if stripped.startswith("["):
                section = stripped.strip("[] ").lower()
                in_atoms = section == "atoms"
                continue

            if in_atoms and stripped and not stripped.startswith(";"):
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        total_charge += float(parts[6])
                    except (ValueError, IndexError):
                        pass

    return round(total_charge)


def _sum_charges_from_itp(itp_path: Path) -> float:
    """Sum charges from the [ atoms ] section of an .itp file."""
    total = 0.0
    in_atoms = False

    with open(itp_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                section = stripped.strip("[] ").lower()
                in_atoms = section == "atoms"
                continue
            if in_atoms and stripped and not stripped.startswith(";"):
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        total += float(parts[6])
                    except (ValueError, IndexError):
                        pass
    return total


def update_topology_molecules(
    top_path: Path, n_water: int, num_na: int, num_cl: int, water_name: str = "SOL"
) -> None:
    """Update the [ molecules ] section of a topology file after solvation/ionization.

    Appends water and ion molecule counts to the topology.

    Parameters
    ----------
    top_path : Path
        Path to the GROMACS .top file (modified in-place).
    n_water : int
        Number of water molecules.
    num_na : int
        Number of Na+ ions.
    num_cl : int
        Number of Cl- ions.
    water_name : str
        Residue name for water (default "SOL").

    """
    with open(top_path, "a") as f:
        if n_water > 0:
            f.write(f"{water_name:<20s} {n_water}\n")
        if num_na > 0:
            f.write(f"{'NA':<20s} {num_na}\n")
        if num_cl > 0:
            f.write(f"{'CL':<20s} {num_cl}\n")

    logger.info(
        f"Topology updated: {n_water} water, {num_na} Na+, {num_cl} Cl-"
    )


def validate_with_grompp(topology: Path, structure: Path, mdp: Path, cwd: Path) -> None:
    """Run gmx grompp as a validation check for topology/coordinate consistency.

    Parameters
    ----------
    topology : Path
        Path to topology.top file.
    structure : Path
        Path to structure .gro file.
    mdp : Path
        Path to MDP file (e.g., em.mdp).
    cwd : Path
        Working directory for the command.

    Raises
    ------
    RuntimeError
        If grompp reports errors.

    """
    tpr_out = cwd / "check.tpr"
    cmd = [
        "gmx", "grompp",
        "-f", str(mdp),
        "-c", str(structure),
        "-p", str(topology),
        "-o", str(tpr_out),
        "-maxwarn", "0",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        input="",
        capture_output=True,
        timeout=60,
    )

    # Clean up the check tpr
    if tpr_out.is_file():
        tpr_out.unlink()

    if result.returncode != 0:
        raise RuntimeError(
            f"gmx grompp validation failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    logger.info("grompp validation passed.")
