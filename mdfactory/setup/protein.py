# ABOUTME: Protein preparation utilities for the proteinbox simulation type
# ABOUTME: Wraps PDBFixer for cleaning and gmx pdb2gmx for topology generation
"""Protein preparation utilities for the proteinbox simulation type."""

import io
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

from loguru import logger

from ..models.parametrization import GromacsProteinParameterSet, Pdb2gmxConfig

_DISULFIDE_CUTOFF_ANGSTROM = 2.2

FORCEFIELD_REGISTRY: dict[str, dict[str, str]] = {
    "charmm36m": {
        "url": "http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-feb2026_cgenff-5.0.ff.tgz",
        "dirname": "charmm36-feb2026_cgenff-5.0.ff",
        "description": "CHARMM36m protein + CGenFF 5.0 (Feb 2026)",
    },
    "charmm36m-ljpme": {
        "url": "http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-feb2026_ljpme_cgenff-5.0.ff.tgz",
        "dirname": "charmm36-feb2026_ljpme_cgenff-5.0.ff",
        "description": "CHARMM36m + CGenFF 5.0, LJ-PME variant (Feb 2026)",
    },
}


def get_forcefield_dir() -> Path:
    """Return the directory where downloaded force fields are stored."""
    from ..settings import settings  # noqa: PLC0415

    return settings.gromacs_forcefield_dir


def get_gromacs_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment with configured GROMACS force fields on GMXLIB."""
    from ..settings import settings  # noqa: PLC0415

    return settings.gromacs_env(base_env)


def _resolve_protonation_residue_name(residue_name: str, forcefield: str | None) -> str:
    """Return the residue name to write into the PDB for a protonation override."""
    residue_name = residue_name.upper()
    if forcefield and forcefield.lower().startswith("charmm"):
        residue_name = {
            "HID": "HSD",
            "HIE": "HSE",
            "HIP": "HSP",
        }.get(residue_name, residue_name)

    if len(residue_name) != 3:
        raise ValueError(
            f"Protonation residue name '{residue_name}' cannot be written to PDB "
            "without shifting columns. Use a 3-character residue name supported by "
            "the selected force field."
        )
    return residue_name


def _distance(coords_a: tuple[float, float, float], coords_b: tuple[float, float, float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(coords_a, coords_b, strict=True)) ** 0.5


def _read_cysteine_sg_atoms(pdb_path: Path) -> dict[int, tuple[float, float, float]]:
    """Read cysteine SG atom coordinates keyed by residue id."""
    atoms: dict[int, tuple[float, float, float]] = {}
    duplicate_resids = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            if atom_name != "SG" or resname not in {"CYS", "CYM", "CYX"}:
                continue

            resid = int(line[22:26].strip())
            coords = (
                float(line[30:38].strip()),
                float(line[38:46].strip()),
                float(line[46:54].strip()),
            )
            if resid in atoms:
                duplicate_resids.add(resid)
            atoms[resid] = coords

    if duplicate_resids:
        raise ValueError(
            "Disulfide bond overrides use residue ids only and cannot disambiguate "
            f"duplicate cysteine residue ids: {sorted(duplicate_resids)}."
        )
    return atoms


def _build_disulfide_prompt_input(
    pdb_path: Path,
    disulfide_bonds: list[tuple[int, int]],
    cutoff_angstrom: float = _DISULFIDE_CUTOFF_ANGSTROM,
) -> str:
    """Build deterministic yes/no input for pdb2gmx disulfide prompts."""
    requested = {tuple(sorted(pair)) for pair in disulfide_bonds}
    for pair in requested:
        if pair[0] == pair[1]:
            raise ValueError(f"Invalid disulfide bond with identical residues: {pair}")

    sg_atoms = _read_cysteine_sg_atoms(pdb_path)
    missing_resids = sorted(
        {resid for pair in requested for resid in pair if resid not in sg_atoms}
    )
    if missing_resids:
        raise ValueError(
            f"Requested disulfide residues are not cysteine SG atoms in {pdb_path}: "
            f"{missing_resids}"
        )

    candidate_pairs = []
    resids = sorted(sg_atoms)
    for i, resid_a in enumerate(resids):
        for resid_b in resids[i + 1 :]:
            if _distance(sg_atoms[resid_a], sg_atoms[resid_b]) <= cutoff_angstrom:
                candidate_pairs.append((resid_a, resid_b))

    missing_pairs = sorted(requested - set(candidate_pairs))
    if missing_pairs:
        raise ValueError(
            "Requested disulfide bonds were not detected as close CYS SG pairs "
            f"within {cutoff_angstrom:.1f} A in {pdb_path}: {missing_pairs}"
        )

    answers = ["y" if pair in requested else "n" for pair in candidate_pairs]
    logger.info(f"pdb2gmx disulfide candidates: {candidate_pairs}; requested: {sorted(requested)}")
    return "\n".join(answers) + "\n"


def _apply_protonation_states(
    pdb_path: Path,
    protonation_states: dict[str, str],
    output_dir: Path,
    forcefield: str | None = None,
) -> Path:
    """Apply protonation state overrides by renaming residues in the PDB.

    pdb2gmx selects protonation states based on residue names. This function renames
    matching residues in the PDB so pdb2gmx applies the requested 3-character residue
    names without interactive prompts. Common AMBER histidine names are translated to
    CHARMM names when a CHARMM force field is selected.

    Parameters
    ----------
    pdb_path : Path
        Input PDB file (absolute path).
    protonation_states : dict[str, str]
        Mapping of "RESNAME<resid>" to target residue name, e.g.
        {"HIS15": "HID", "GLU35": "GLH"}.
    output_dir : Path
        Directory for the modified PDB.
    forcefield : str | None
        Selected pdb2gmx force field, used for force-field-specific aliases.

    Returns
    -------
    Path
        Path to the modified PDB file.

    """
    import re

    output_pdb = output_dir / "protonation_adjusted.pdb"

    # Parse overrides: "HIS15" -> (original_resname_prefix="HIS", resid=15, new_name="HID")
    overrides = []
    for key, new_name in protonation_states.items():
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", key)
        if not match:
            raise ValueError(
                f"Invalid protonation state key '{key}'. "
                "Expected format: RESNAME<resid>, e.g. 'HIS15'."
            )
        resname_prefix = match.group(1).upper()
        resid = int(match.group(2))
        overrides.append(
            (resname_prefix, resid, _resolve_protonation_residue_name(new_name, forcefield))
        )

    with open(pdb_path) as f_in, open(output_pdb, "w") as f_out:
        for line in f_in:
            output_line = line
            if output_line.startswith(("ATOM", "HETATM")) and len(output_line) >= 26:
                current_resname = output_line[17:20].strip()
                try:
                    current_resid = int(output_line[22:26].strip())
                except ValueError:
                    f_out.write(output_line)
                    continue

                for prefix, target_resid, new_name in overrides:
                    if current_resname.startswith(prefix) and current_resid == target_resid:
                        padded = f"{new_name:<3s}"
                        output_line = output_line[:17] + padded + output_line[20:]
                        break
            f_out.write(output_line)

    logger.info(f"Protonation states applied: {protonation_states}")
    return output_pdb


def check_gmx_available() -> Path:
    """Verify that the GROMACS gmx binary is available.

    Checks the configured path from settings first, then falls back to PATH.

    Returns
    -------
    Path
        Path to the gmx binary.

    Raises
    ------
    RuntimeError
        If gmx is not found.

    """
    from ..settings import settings  # noqa: PLC0415

    configured = settings.gromacs_gmx_path
    if configured is not None:
        if configured.is_file():
            return configured
        raise RuntimeError(
            f"Configured GROMACS path '{configured}' does not exist. "
            "Check [gromacs] GMX_PATH in your mdfactory config."
        )

    gmx_path = shutil.which("gmx")
    if gmx_path is None:
        raise RuntimeError(
            "GROMACS 'gmx' binary not found on PATH. "
            "Either set [gromacs] GMX_PATH in mdfactory config, "
            "or ensure 'gmx' is on PATH (e.g. 'module load gromacs')."
        )
    return Path(gmx_path)


def download_forcefield(name: str) -> Path:
    """Download a force field from the registry to the local data directory.

    Parameters
    ----------
    name : str
        Registry key (e.g. "charmm36m").

    Returns
    -------
    Path
        Path to the extracted force field directory.

    Raises
    ------
    ValueError
        If the name is not in the registry.

    """
    if name not in FORCEFIELD_REGISTRY:
        raise ValueError(
            f"Force field '{name}' is not in the download registry. "
            f"Available: {sorted(FORCEFIELD_REGISTRY.keys())}."
        )

    entry = FORCEFIELD_REGISTRY[name]
    ff_dir = get_forcefield_dir()
    ff_dir.mkdir(parents=True, exist_ok=True)

    target = ff_dir / entry["dirname"]
    if target.is_dir():
        logger.info(f"Force field '{name}' already present at {target}")
        return target

    logger.info(f"Downloading force field '{name}' from {entry['url']}...")
    response = urlopen(entry["url"], timeout=120)  # noqa: S310
    data = io.BytesIO(response.read())

    with tarfile.open(fileobj=data, mode="r:gz") as tar:
        tar.extractall(path=ff_dir, filter="data")

    if not target.is_dir():
        raise RuntimeError(
            f"Download succeeded but expected directory '{entry['dirname']}' "
            f"not found in {ff_dir}. Tarball may have unexpected structure."
        )

    logger.info(f"Force field '{name}' installed to {target}")
    return target


def download_all_forcefields() -> list[Path]:
    """Download all force fields in the registry. Idempotent."""
    paths = []
    for name in FORCEFIELD_REGISTRY:
        paths.append(download_forcefield(name))
    return paths


def _get_gmx_search_paths() -> list[Path]:
    """Return the directories GROMACS searches for force fields."""
    search_paths: list[Path] = []

    gmx_bin = str(check_gmx_available())
    result = subprocess.run(
        [gmx_bin, "--version"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    for line in result.stdout.splitlines():
        if "Data prefix" in line:
            data_prefix = Path(line.split(":", 1)[1].strip())
            top_dir = data_prefix / "share" / "gromacs" / "top"
            if top_dir.is_dir():
                search_paths.append(top_dir)
            break

    gmxlib = get_gromacs_env().get("GMXLIB", "")
    for p in gmxlib.split(":"):
        if p and Path(p).is_dir():
            search_paths.append(Path(p))

    search_paths.append(Path.cwd())
    return search_paths


def resolve_forcefield(forcefield: str) -> str:
    """Resolve a friendly force field name to the actual directory stem.

    If the name is a registry key (e.g. "charmm36m"), returns the actual
    directory name (e.g. "charmm36-feb2026_cgenff-5.0"). Otherwise returns
    the name unchanged (for built-in FFs like "charmm27").

    """
    if forcefield in FORCEFIELD_REGISTRY:
        return FORCEFIELD_REGISTRY[forcefield]["dirname"].removesuffix(".ff")
    return forcefield


def check_forcefield_available(forcefield: str) -> None:
    """Verify that the requested force field is findable by GROMACS.

    Parameters
    ----------
    forcefield : str
        Force field name as passed to ``gmx pdb2gmx -ff``, or a registry
        key (e.g. "charmm36m").

    Raises
    ------
    ValueError
        If the force field is not found and not in the registry.

    """
    # Resolve friendly name to actual dirname
    resolved = resolve_forcefield(forcefield)
    ff_dirname = f"{resolved}.ff"

    search_paths = _get_gmx_search_paths()

    for search_dir in search_paths:
        if (search_dir / ff_dirname).is_dir():
            return

    available = set()
    for search_dir in search_paths:
        if search_dir.is_dir():
            for entry in search_dir.iterdir():
                if entry.is_dir() and entry.name.endswith(".ff"):
                    available.add(entry.stem)

    raise ValueError(
        f"Force field '{forcefield}' not found. "
        f"Searched: {[str(p) for p in search_paths]}.\n"
        f"Available force fields: {sorted(available)}.\n"
        f"Downloadable: {sorted(FORCEFIELD_REGISTRY.keys())}.\n"
        "Install the force field to a search path, set [gromacs] FORCEFIELD_DIR, "
        "or download registered force fields with `mdfactory config init`."
    )


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
    gmx_bin = str(check_gmx_available())
    check_forcefield_available(config.forcefield)
    resolved_ff = resolve_forcefield(config.forcefield)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all paths to absolute to avoid cwd confusion with subprocess
    pdb_path = pdb_path.resolve()
    output_dir = output_dir.resolve()

    structure_file = output_dir / "processed.gro"
    topology_file = output_dir / "topol.top"
    posre_file = output_dir / "posre.itp"

    cmd = [
        gmx_bin,
        "pdb2gmx",
        "-f",
        str(pdb_path),
        "-o",
        str(structure_file),
        "-p",
        str(topology_file),
        "-i",
        str(posre_file),
        "-ff",
        resolved_ff,
        "-water",
        config.water_model,
    ]

    if config.ignh:
        cmd.append("-ignh")

    if config.merge_all:
        cmd.extend(["-merge", "all"])

    # Apply protonation state overrides by renaming residues in the PDB
    # before pdb2gmx processes it (standard approach for CHARMM/AMBER FFs)
    if protonation_states:
        pdb_path = _apply_protonation_states(
            pdb_path, protonation_states, output_dir, forcefield=config.forcefield
        )
        cmd[cmd.index("-f") + 1] = str(pdb_path)

    pdb2gmx_input = ""
    if disulfide_bonds:
        cmd.append("-ss")
        pdb2gmx_input = _build_disulfide_prompt_input(pdb_path, disulfide_bonds)

    logger.info(f"Running pdb2gmx: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        text=True,
        input=pdb2gmx_input,
        capture_output=True,
        timeout=120,
        env=get_gromacs_env(),
        check=False,
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
                # Follow includes only for local itp files (protein chains,
                # position restraints), not force field library files
                include_path = stripped.split('"')[1] if '"' in stripped else None
                if include_path and ".ff/" not in include_path:
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

    logger.info(f"Topology updated: {n_water} water, {num_na} Na+, {num_cl} Cl-")


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
    gmx_bin = str(check_gmx_available())
    tpr_out = cwd / "check.tpr"
    cmd = [
        gmx_bin,
        "grompp",
        "-f",
        str(mdp),
        "-c",
        str(structure),
        "-p",
        str(topology),
        "-o",
        str(tpr_out),
        "-maxwarn",
        "0",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        input="",
        capture_output=True,
        timeout=60,
        env=get_gromacs_env(),
        check=False,
    )

    # Clean up grompp artifacts
    if tpr_out.is_file():
        tpr_out.unlink()
    mdout_path = cwd / "mdout.mdp"
    if mdout_path.is_file():
        mdout_path.unlink()

    if result.returncode != 0:
        raise RuntimeError(
            f"gmx grompp validation failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    logger.info("grompp validation passed.")
