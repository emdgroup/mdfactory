# ABOUTME: Protein preparation utilities for the proteinbox simulation type
# ABOUTME: Wraps PDBFixer cleaning, gmx pdb2gmx topology generation, and GROMACS version gates
"""Protein preparation utilities for the proteinbox simulation type."""

import hashlib
import io
import itertools
import math
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

from loguru import logger

from ..models.parametrization import GromacsProteinParameterSet
from ..settings import settings

_DISULFIDE_CUTOFF_ANGSTROM = 2.2
_CYSTEINE_RESNAMES = {"CYS", "CYM", "CYX"}

FORCEFIELD_REGISTRY: dict[str, dict[str, str]] = {
    "charmm36m": {
        "url": "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-feb2026_cgenff-5.0.ff.tgz",
        "dirname": "charmm36-feb2026_cgenff-5.0.ff",
        "sha256": "c2b0c70e21cced150528ee25d205e5bd3367a5e55e85c2ff0403911980528151",
    },
    "charmm36m-ljpme": {
        "url": "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-feb2026_ljpme_cgenff-5.0.ff.tgz",
        "dirname": "charmm36-feb2026_ljpme_cgenff-5.0.ff",
        "sha256": "b2221dcba73066b6b5c30a52b9894216d03ea6c3e68206d9ce43857dd8116c21",
    },
}


# AMBER-style protonation residue names mapped to their CHARMM equivalents.
# Histidine tautomers map to 3-character names pdb2gmx accepts; protonated
# GLU/ASP map to 4-character CHARMM names that cannot fit a PDB residue-name
# column and are rejected below.
_CHARMM_PROTONATION_ALIASES = {
    "HID": "HSD",
    "HIE": "HSE",
    "HIP": "HSP",
    "GLH": "GLUP",
    "ASH": "ASPP",
    "LYN": "LSN",
}

# Histidine tautomers share a titratable identity but not a common name prefix,
# so a "HIS<n>" override must still match a residue already named HID/HSE/etc.
_HISTIDINE_RESNAMES = {"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"}


def _resolve_protonation_residue_name(residue_name: str, forcefield: str | None) -> str:
    """Return the residue name to write into the PDB for a protonation override."""
    residue_name = residue_name.upper()
    if forcefield and forcefield.lower().startswith("charmm"):
        residue_name = _CHARMM_PROTONATION_ALIASES.get(residue_name, residue_name)

    if len(residue_name) != 3:
        raise ValueError(
            f"Protonation residue name '{residue_name}' cannot be written to a PDB "
            "residue-name column (needs exactly 3 characters). Protonated GLU/ASP in "
            "CHARMM use 4-character names (GLUP/ASPP) that cannot be set by residue "
            "renaming; only 3-character states such as histidine tautomers "
            "(HID/HIE/HIP -> HSD/HSE/HSP) or neutral lysine (LYN -> LSN) are supported."
        )
    return residue_name


def _override_matches_residue(resname_prefix: str, current_resname: str) -> bool:
    """Return True if a protonation override applies to the residue currently present."""
    if current_resname.startswith(resname_prefix):
        return True
    return resname_prefix in _HISTIDINE_RESNAMES and current_resname in _HISTIDINE_RESNAMES


_ResidueKey = tuple[str, int, str]


def _pdb_residue_key(line: str) -> _ResidueKey:
    """Return the (chain, resid, insertion code) key of a PDB ATOM/HETATM line."""
    return (line[21].strip(), int(line[22:26]), line[26].strip())


def _parse_residue_reference(reference: str) -> tuple[str | None, str, int, str]:
    """Parse a possibly chain-qualified PDB residue reference.

    References have the form ``RESNAME<resid>`` or
    ``CHAIN:RESNAME<resid><insertion-code>``. For example, ``HIS15``,
    ``A:HIS15``, and ``H:HIS100A`` are valid.
    """
    match = re.fullmatch(r"(?:([A-Za-z0-9]):)?([A-Za-z]+)(\d+)([A-Za-z]?)", reference)
    if not match:
        raise ValueError(
            f"Invalid residue reference '{reference}'. Expected format "
            "RESNAME<resid> or CHAIN:RESNAME<resid><insertion-code>, "
            "e.g. 'HIS15', 'A:HIS15', or 'H:HIS100A'."
        )
    insertion_code = match.group(4)
    return match.group(1), match.group(2).upper(), int(match.group(3)), insertion_code


def _format_residue_key(key: _ResidueKey, resname: str) -> str:
    """Format a concrete PDB residue key as a configuration reference."""
    chain, resid, insertion_code = key
    residue = f"{resname}{resid}{insertion_code}"
    return f"{chain}:{residue}" if chain else residue


def _read_cysteine_sg_atoms(pdb_path: Path) -> dict[_ResidueKey, tuple[float, float, float]]:
    """Read cysteine SG atoms in PDB encounter order, keyed by residue identity."""
    atoms: dict[_ResidueKey, tuple[float, float, float]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            if atom_name != "SG" or resname not in _CYSTEINE_RESNAMES:
                continue

            key = _pdb_residue_key(line)
            coords = (
                float(line[30:38].strip()),
                float(line[38:46].strip()),
                float(line[46:54].strip()),
            )
            if key in atoms:
                raise ValueError(
                    "Multiple cysteine SG atoms share the residue identifier "
                    f"'{_format_residue_key(key, resname)}' in {pdb_path}."
                )
            atoms[key] = coords

    return atoms


def _unique_residue_match(
    label: str,
    reference: str,
    residues: dict[_ResidueKey, str],
    not_found: str,
    accepts: Callable[[str], bool] = lambda _name: True,
) -> _ResidueKey:
    """Return the single residue key matching a parsed reference, else raise ValueError."""
    chain, _, resid, insertion_code = _parse_residue_reference(reference)
    candidates = {
        key: name
        for key, name in residues.items()
        if key[1] == resid
        and (chain is None or key[0] == chain)
        and key[2] == insertion_code
        and accepts(name)
    }
    if not candidates:
        raise ValueError(not_found)
    if len(candidates) > 1:
        choices = sorted(_format_residue_key(key, name) for key, name in candidates.items())
        raise ValueError(
            f"{label} '{reference}' is ambiguous across chains or insertion codes. "
            f"Use one of these fully qualified references: {choices}."
        )
    return next(iter(candidates))


def _resolve_cysteine_reference(
    reference: str,
    sg_atoms: dict[_ResidueKey, tuple[float, float, float]],
    pdb_path: Path,
) -> _ResidueKey:
    """Resolve a cysteine reference (chain optional) to a unique PDB residue key."""
    label = "Disulfide residue reference"
    resname = _parse_residue_reference(reference)[1]
    if resname not in _CYSTEINE_RESNAMES:
        raise ValueError(
            f"{label} '{reference}' must be a cysteine (one of {sorted(_CYSTEINE_RESNAMES)})."
        )
    return _unique_residue_match(
        label,
        reference,
        dict.fromkeys(sg_atoms, resname),
        f"{label} '{reference}' is not a cysteine SG atom in {pdb_path}.",
    )


def _build_disulfide_prompt_input(
    pdb_path: Path,
    disulfide_bonds: list[tuple[str, str]],
    cutoff_angstrom: float = _DISULFIDE_CUTOFF_ANGSTROM,
) -> str:
    """Build deterministic yes/no input for pdb2gmx disulfide prompts."""
    sg_atoms = _read_cysteine_sg_atoms(pdb_path)

    requested: set[frozenset[_ResidueKey]] = set()
    for a, b in disulfide_bonds:
        key_a = _resolve_cysteine_reference(a, sg_atoms, pdb_path)
        key_b = _resolve_cysteine_reference(b, sg_atoms, pdb_path)
        if key_a == key_b:
            raise ValueError(f"Invalid disulfide bond with identical residues: {a}-{b}")
        requested.add(frozenset((key_a, key_b)))

    # pdb2gmx presents -ss prompts in PDB atom encounter order. Do not sort these
    # keys: doing so can attach each y/n answer to a different chain's prompt.
    candidate_pairs = [
        (key_a, key_b)
        for key_a, key_b in itertools.combinations(sg_atoms, 2)
        if math.dist(sg_atoms[key_a], sg_atoms[key_b]) <= cutoff_angstrom
    ]

    candidate_identities = {frozenset(pair) for pair in candidate_pairs}
    missing_pairs = requested - candidate_identities
    if missing_pairs:
        formatted_missing = sorted(tuple(sorted(pair)) for pair in missing_pairs)
        raise ValueError(
            "Requested disulfide bonds were not detected as close CYS SG pairs "
            f"within {cutoff_angstrom:.1f} A in {pdb_path}: {formatted_missing}"
        )

    answers = ["y" if frozenset(pair) in requested else "n" for pair in candidate_pairs]
    formatted_requested = sorted(tuple(sorted(pair)) for pair in requested)
    logger.info(
        f"pdb2gmx disulfide candidates: {candidate_pairs}; requested: {formatted_requested}"
    )
    return "\n".join(answers) + "\n"


def _read_pdb_residues(pdb_path: Path) -> dict[_ResidueKey, str]:
    """Read physical PDB residues keyed by chain, residue id, and insertion code."""
    residues: dict[_ResidueKey, str] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
                continue
            try:
                key = _pdb_residue_key(line)
            except ValueError:
                continue
            resname = line[17:20].strip()
            previous = residues.setdefault(key, resname)
            if previous != resname:
                raise ValueError(
                    f"Residue identifier '{_format_residue_key(key, resname)}' has "
                    f"inconsistent residue names '{previous}' and '{resname}' in {pdb_path}."
                )
    return residues


def _resolve_protonation_reference(
    reference: str, residues: dict[_ResidueKey, str], pdb_path: Path
) -> _ResidueKey:
    """Resolve a protonation reference to one physical PDB residue."""
    resname_prefix = _parse_residue_reference(reference)[1]
    return _unique_residue_match(
        "Protonation state override",
        reference,
        residues,
        f"Protonation state override '{reference}' matched no residue in {pdb_path}. "
        "Check the residue number, insertion code, chain, and name against the input PDB.",
        accepts=lambda name: _override_matches_residue(resname_prefix, name),
    )


def _apply_protonation_states(
    pdb_path: Path,
    protonation_states: dict[str, str],
    output_dir: Path,
    forcefield: str | None = None,
) -> Path:
    """Rename residues in the PDB so pdb2gmx applies the requested protonation states.

    ``protonation_states`` maps ``RESNAME<resid>`` (or ``CHAIN:RESNAME<resid>``) to a
    3-character target name; AMBER names are translated for CHARMM force fields.
    Overrides that match no residue, or resolve to the same residue, raise ValueError.
    Returns the path to the modified PDB in ``output_dir``.
    """
    output_pdb = output_dir / "protonation_adjusted.pdb"

    residues = _read_pdb_residues(pdb_path)
    resolved_overrides: dict[_ResidueKey, tuple[str, str]] = {}
    for reference, new_name in protonation_states.items():
        residue_key = _resolve_protonation_reference(reference, residues, pdb_path)
        target_name = _resolve_protonation_residue_name(new_name, forcefield)
        if residue_key in resolved_overrides:
            previous_reference = resolved_overrides[residue_key][0]
            raise ValueError(
                f"Protonation state overrides '{previous_reference}' and '{reference}' "
                "resolve to the same residue. Specify it only once."
            )
        resolved_overrides[residue_key] = (reference, target_name)

    with open(pdb_path) as f_in, open(output_pdb, "w") as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 27:
                try:
                    override = resolved_overrides.get(_pdb_residue_key(line))
                except ValueError:
                    override = None
                if override is not None:
                    f_out.write(line[:17] + f"{override[1]:<3s}" + line[20:])
                    continue
            f_out.write(line)

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


# Force fields that need a minimum GROMACS version, keyed by the ``.ff`` directory
# stem, as (year, minor). amber14sb.ff ships with GROMACS 2026 and relies on the
# LEaP dihedral atom reordering that grompp gained in 2026; older versions ignore
# ``_FF_AMBER_LEAP_ATOM_REORDERING`` silently. Single source of truth for the
# build-time gate and the simulate-time grompp script check.
GMX_MIN_VERSION_BY_FORCEFIELD: dict[str, tuple[int, int]] = {
    "amber14sb": (2026, 0),
}

_GMX_VERSION_LINE = re.compile(r"^\s*GROMACS version:\s*(\S+)", re.MULTILINE)
_GMX_VERSION_NUMBER = re.compile(r"^(\d+)(?:\.(\d+))?(?!\d)")


def parse_gmx_version(text: str) -> tuple[int, int]:
    """Parse a GROMACS version from ``gmx --version`` output or a version string.

    Parameters
    ----------
    text : str
        Full ``gmx --version`` output (the ``GROMACS version:`` line is used) or
        a bare version string such as ``2026.3-conda_forge``, ``2024.3``,
        ``2026.0-dev-20250101-abcdef`` or ``2026-beta1``.

    Returns
    -------
    tuple of int
        ``(year, minor)``; a missing minor version (``2026-beta1``) is 0.

    Raises
    ------
    ValueError
        If no GROMACS version can be read from *text*.

    """
    line = _GMX_VERSION_LINE.search(text)
    version = line.group(1) if line else text.strip()
    match = _GMX_VERSION_NUMBER.match(version)
    if match is None:
        raise ValueError(f"Could not parse a GROMACS version from {text.strip()[:200]!r}.")
    return int(match.group(1)), int(match.group(2) or 0)


def format_gmx_version(version: tuple[int, int]) -> str:
    """Format a ``(year, minor)`` GROMACS version, omitting a zero minor version."""
    year, minor = version
    return f"{year}" if minor == 0 else f"{year}.{minor}"


def min_gmx_version_for_forcefield(forcefield: str) -> tuple[int, int] | None:
    """Return the minimum GROMACS version a force field needs, or None if any works.

    Parameters
    ----------
    forcefield : str
        Force-field name or ``.ff`` directory name (e.g. ``amber14sb`` or
        ``amber14sb.ff``).

    """
    return GMX_MIN_VERSION_BY_FORCEFIELD.get(forcefield.removesuffix(".ff"))


def min_gmx_version_for_topology(top_path: Path) -> tuple[int, int] | None:
    """Return the minimum GROMACS version the force field included by a topology needs.

    Reads the ``#include "<name>.ff/forcefield.itp"`` lines of *top_path* (not
    following includes) and returns the highest requirement among them.

    Parameters
    ----------
    top_path : Path
        GROMACS topology, e.g. a simulation directory's ``topology.top``.

    Returns
    -------
    tuple of int or None
        ``(year, minor)``, or None if the file does not exist or its force
        field has no version requirement.

    """
    top_path = Path(top_path)
    if not top_path.is_file():
        return None
    requirements = []
    with open(top_path) as fb:
        for line in fb:
            stripped = line.strip()
            if not stripped.startswith("#include") or '"' not in stripped:
                continue
            parts = Path(stripped.split('"')[1]).parts
            if len(parts) < 2 or parts[-1] != "forcefield.itp" or not parts[-2].endswith(".ff"):
                continue
            required = min_gmx_version_for_forcefield(parts[-2])
            if required is not None:
                requirements.append(required)
    return max(requirements) if requirements else None


def get_gmx_version(gmx_bin: str | Path) -> tuple[tuple[int, int], str]:
    """Run ``gmx --version`` and return the parsed and the reported version.

    Returns
    -------
    tuple
        ``((year, minor), reported)`` where *reported* is the raw version string,
        e.g. ``2026.3-conda_forge``.

    Raises
    ------
    RuntimeError
        If the binary cannot be run or reports no parseable version.

    """
    try:
        result = subprocess.run(
            [str(gmx_bin), "--version"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not run '{gmx_bin} --version': {exc}") from exc
    output = result.stdout + result.stderr
    try:
        version = parse_gmx_version(output)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not read the GROMACS version of '{gmx_bin}' "
            f"(exit code {result.returncode}): {exc}"
        ) from exc
    line = _GMX_VERSION_LINE.search(output)
    reported = line.group(1) if line else format_gmx_version(version)
    return version, reported


def require_gmx_version(gmx_bin: str | Path, forcefield: str) -> None:
    """Fail early if *gmx_bin* is too old for *forcefield*.

    Force fields without an entry in :data:`GMX_MIN_VERSION_BY_FORCEFIELD` are
    accepted without running the binary.

    Raises
    ------
    RuntimeError
        If the binary is older than the force field requires.

    """
    required = min_gmx_version_for_forcefield(forcefield)
    if required is None:
        return
    version, reported = get_gmx_version(gmx_bin)
    if version < required:
        raise RuntimeError(
            f"Force field '{forcefield}' requires GROMACS >= {format_gmx_version(required)}, "
            f"but '{gmx_bin}' is GROMACS {reported}. Use a newer GROMACS, e.g. the pixi "
            "'gmx' environment (`pixi run -e gmx ...`), or point [gromacs] GMX_PATH in "
            "the mdfactory config at a GROMACS >= "
            f"{format_gmx_version(required)} binary."
        )
    logger.info(
        f"GROMACS {reported} satisfies the '{forcefield}' requirement "
        f">= {format_gmx_version(required)}."
    )


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
    ff_dir = settings.gromacs_forcefield_dir
    ff_dir.mkdir(parents=True, exist_ok=True)

    target = ff_dir / entry["dirname"]
    if target.is_dir():
        logger.info(f"Force field '{name}' already present at {target}")
        return target

    logger.info(f"Downloading force field '{name}' from {entry['url']}...")
    response = urlopen(entry["url"], timeout=120)  # noqa: S310
    payload = response.read()

    expected_sha256 = entry.get("sha256")
    if expected_sha256:
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Force field '{name}' download failed integrity check.\n"
                f"Expected SHA-256: {expected_sha256}\n"
                f"Actual SHA-256:   {actual_sha256}\n"
                "The download may be corrupted or tampered with. If the upstream file was "
                "legitimately updated, update the pinned 'sha256' in FORCEFIELD_REGISTRY."
            )

    data = io.BytesIO(payload)
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
    return [download_forcefield(name) for name in FORCEFIELD_REGISTRY]


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

    gmxlib = settings.gromacs_env().get("GMXLIB", "")
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


def _restrict_missing_residues_to_internal_gaps(fixer) -> None:
    """Drop missing terminal residues from ``fixer.missingResidues``; warn about all gaps.

    ``PDBFixer.findMissingResidues`` compares the coordinates with the SEQRES
    records and lists every unresolved residue, and ``addMissingAtoms`` then
    builds all of them. Unresolved terminal tails (tags, disordered termini)
    are not part of the deposited structure and would be built as extended
    stubs pointing into the solvent, so they are dropped. Internal gaps are
    kept and modeled: leaving them open would make pdb2gmx bond the residues
    on either side of the gap across the missing segment.

    Parameters
    ----------
    fixer : pdbfixer.PDBFixer
        Fixer on which ``findMissingResidues`` has been called. Modified in place.

    """
    chains = list(fixer.topology.chains())
    internal = {}
    for (chain_index, position), names in sorted(fixer.missingResidues.items()):
        chain = chains[chain_index]
        residues = list(chain.residues())
        seq = "-".join(names)
        if position == 0:
            first = residues[0]
            logger.warning(
                f"Chain {chain.id}: {len(names)} unresolved N-terminal residue(s) {seq} "
                f"before {first.name}{first.id} are listed in SEQRES but not modeled."
            )
        elif position == len(residues):
            last = residues[-1]
            logger.warning(
                f"Chain {chain.id}: {len(names)} unresolved C-terminal residue(s) {seq} "
                f"after {last.name}{last.id} are listed in SEQRES but not modeled."
            )
        else:
            before, after = residues[position - 1], residues[position]
            logger.warning(
                f"Chain {chain.id}: modeling {len(names)} missing internal residue(s) {seq} "
                f"between {before.name}{before.id} and {after.name}{after.id} "
                "with PDBFixer; check the loop geometry."
            )
            internal[(chain_index, position)] = names
    fixer.missingResidues = internal


def clean_pdb(pdb_path: Path, output_path: Path) -> Path:
    """Clean a PDB file using PDBFixer.

    Removes all heterogens (ligands, cofactors, metals, crystal waters), models
    residues missing from internal gaps of a chain (per the SEQRES records),
    and adds missing heavy atoms. Unresolved terminal residues are not
    modeled. Every missing segment is reported as a warning.

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
    _restrict_missing_residues_to_internal_gaps(fixer)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    from openmm.app import PDBFile  # noqa: PLC0415

    # keepIds=True preserves the input chain IDs and residue numbers so that
    # user-supplied chain, disulfide, and protonation references (which name
    # residues by their original numbering) still resolve after cleaning.
    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    logger.info(f"PDB cleaned: {pdb_path} -> {output_path}")
    return output_path


def _parse_protein_molecule_names(top_path: Path) -> list[str]:
    """Return protein moleculetype names from a topology's [ molecules ] section.

    pdb2gmx names each protein moleculetype ``Protein_chain_<ID>`` (or ``Protein``
    when chains are merged), listing each once. Water and ion entries are ignored.
    """
    names: list[str] = []
    in_molecules = False
    with open(top_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.startswith("["):
                in_molecules = stripped.strip("[] ").lower() == "molecules"
                continue
            if in_molecules and stripped.split()[0].startswith("Protein"):
                names.append(stripped.split()[0])
    return names


def _run_gmx(
    name: str, cmd: list[str], cwd: Path, stdin: str, timeout: int
) -> subprocess.CompletedProcess:
    """Run a gmx command with the configured force-field environment; raise on failure."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        input=stdin,
        capture_output=True,
        timeout=timeout,
        env=settings.gromacs_env(),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gmx {name} failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_pdb2gmx(
    pdb_path: Path,
    forcefield: str,
    water_model: str,
    ignore_hydrogens: bool,
    merge_all: bool,
    disulfide_bonds: list[tuple[str, str]],
    protonation_states: dict[str, str],
    output_dir: Path,
    chains: list[str] | None = None,
) -> GromacsProteinParameterSet:
    """Run gmx pdb2gmx to generate GROMACS topology from a protein PDB.

    Parameters
    ----------
    pdb_path : Path
        Cleaned PDB file.
    forcefield : str
        Protein force field as recognized by ``gmx pdb2gmx -ff`` or a registry key.
    water_model : str
        Water model name passed to ``gmx pdb2gmx -water``.
    ignore_hydrogens : bool
        Pass ``-ignh`` so pdb2gmx regenerates the hydrogens.
    merge_all : bool
        Pass ``-merge all`` to build all chains as one moleculetype.
    disulfide_bonds : list[tuple[str, str]]
        Cysteine residue pairs for disulfide bonds, e.g. [("CYS6", "CYS127")].
    protonation_states : dict[str, str]
        Residue-specific protonation state overrides.
    output_dir : Path
        Directory for output files.
    chains : list[str] | None
        Declared chain identifiers. When given, the chains pdb2gmx produces must
        match exactly. When omitted, the PDB must yield a single protein chain.

    Returns
    -------
    GromacsProteinParameterSet
        Paths to the generated topology, structure, and per-chain include files.

    """
    chains = chains or []
    gmx_bin = str(check_gmx_available())
    check_forcefield_available(forcefield)
    resolved_ff = resolve_forcefield(forcefield)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all paths to absolute to avoid cwd confusion with subprocess
    pdb_path = pdb_path.resolve()
    output_dir = output_dir.resolve()

    structure_file = output_dir / "processed.gro"
    topology_file = output_dir / "topol.top"
    posre_file = output_dir / "posre.itp"

    # pdb2gmx runs with cwd=output_dir; pass output filenames relative so the
    # generated topology contains a relative `#include "posre.itp"` rather than an
    # absolute path. The force-field #includes it writes are absolute and are made
    # relative later by bundle_forcefield_into_topology.
    cmd = [
        gmx_bin,
        "pdb2gmx",
        "-f",
        str(pdb_path),
        "-o",
        structure_file.name,
        "-p",
        topology_file.name,
        "-i",
        posre_file.name,
        "-ff",
        resolved_ff,
        "-water",
        water_model,
    ]

    if ignore_hydrogens:
        cmd.append("-ignh")

    if merge_all:
        cmd.extend(["-merge", "all"])

    # Apply protonation state overrides by renaming residues in the PDB
    # before pdb2gmx processes it (standard approach for CHARMM/AMBER FFs)
    if protonation_states:
        pdb_path = _apply_protonation_states(
            pdb_path, protonation_states, output_dir, forcefield=forcefield
        )
        cmd[cmd.index("-f") + 1] = str(pdb_path)

    pdb2gmx_input = ""
    if disulfide_bonds:
        cmd.append("-ss")
        pdb2gmx_input = _build_disulfide_prompt_input(pdb_path, disulfide_bonds)

    logger.info(f"Running pdb2gmx: {' '.join(cmd)}")

    result = _run_gmx("pdb2gmx", cmd, output_dir, pdb2gmx_input, timeout=120)

    logger.info("pdb2gmx completed successfully.")

    # Verify expected outputs exist. The position-restraint file is named per chain
    # for multi-chain inputs, so it is checked via the include collection below
    # rather than by a fixed name here.
    for path in [structure_file, topology_file]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected pdb2gmx output not found: {path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    # Each chain becomes its own Protein_chain_<ID> moleculetype, listed once in
    # [ molecules ]. Declared chains must match exactly; an undeclared multi-chain
    # PDB is an error so subunits are never silently dropped or merged.
    produced_names = _parse_protein_molecule_names(topology_file)
    produced_chains = [
        name[len("Protein_chain_") :]
        for name in produced_names
        if name.startswith("Protein_chain_")
    ]

    if chains:
        if set(produced_chains) != set(chains):
            raise ValueError(
                f"Declared protein chains {sorted(chains)} do not match the chains "
                f"pdb2gmx produced {sorted(produced_chains)}. Update protein.chains to "
                "list exactly the chain identifiers present in the PDB."
            )
    elif len(produced_names) > 1:
        raise ValueError(
            f"pdb2gmx produced multiple protein chains {sorted(produced_chains)}, but "
            "none were declared. List each subunit explicitly via protein.chains."
        )

    # Local includes: per-chain moleculetype files and position restraints. Force-field
    # includes live in the bundled .ff directory and are handled separately.
    include_files = [
        path
        for pattern in ("topol_Protein_chain_*.itp", "posre*.itp")
        for path in sorted(output_dir.glob(pattern))
    ]

    total_charge = extract_charge_from_topology(topology_file)
    logger.info(f"Protein net charge from topology: {total_charge}")

    return GromacsProteinParameterSet(
        topology_file=topology_file,
        structure_file=structure_file,
        topology_include_files=include_files,
        chains=produced_chains,
        forcefield=forcefield,
        water_model=water_model,
        total_charge=total_charge,
    )


def _sum_atoms_column(path: Path, column: int, follow_includes: bool = True) -> float:
    """Sum a 0-based ``[ atoms ]`` column over a topology and its local includes.

    Local ``#include`` files (protein chains, position restraints) are followed one
    level; force-field library includes (``.ff/``) are skipped.
    """
    total = 0.0
    in_atoms = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#include"):
            include = stripped.split('"')[1] if '"' in stripped else None
            if follow_includes and include and ".ff/" not in include:
                itp_path = path.parent / include
                if itp_path.is_file():
                    total += _sum_atoms_column(itp_path, column, follow_includes=False)
        elif stripped.startswith("["):
            in_atoms = stripped.strip("[] ").lower() == "atoms"
        elif in_atoms and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) > column:
                try:
                    total += float(parts[column])
                except ValueError:
                    pass
    return total


def extract_charge_from_topology(top_path: Path) -> int:
    """Return the net charge (rounded to an integer) from a topology's ``[ atoms ]`` sections."""
    return round(_sum_atoms_column(top_path, 6))


def extract_mass_from_topology(top_path: Path) -> float:
    """Return the total mass in Daltons from a topology's ``[ atoms ]`` sections."""
    return _sum_atoms_column(top_path, 7)


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


def bundle_forcefield_into_topology(top_path: Path) -> str | None:
    """Copy the force field referenced by #include lines next to the topology.

    pdb2gmx writes force-field ``#include`` lines that point either at the absolute
    location where it found the force field (its GMXLIB entry) or, for built-in
    GROMACS force fields, at a relative path such as ``charmm27.ff/forcefield.itp``.
    Copy that ``.ff`` directory next to the topology so the includes resolve from
    the topology's own directory without GMXLIB and the build directory stays
    portable. Absolute includes are also rewritten to the bundled relative form;
    relative includes already resolve once the directory sits alongside them and
    are left untouched. Mirrors the CGenFF strategy in ``generate_gromacs_topology``.

    Parameters
    ----------
    top_path : Path
        Path to the GROMACS .top file (modified in-place).

    Returns
    -------
    str or None
        Name of the bundled force-field directory (e.g. "charmm36-....ff"), or
        None if the topology has no force-field includes.

    Raises
    ------
    RuntimeError
        If a relative ``.ff`` include cannot be resolved against any GROMACS
        search path, so it cannot be bundled for a portable topology.

    """
    lines = top_path.read_text().splitlines(keepends=True)
    ff_dir_name: str | None = None
    copied_sources: set[Path] = set()
    search_paths: list[Path] | None = None
    rewrote_line = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#include") or '"' not in stripped:
            continue
        include_path = stripped.split('"')[1]
        if ".ff/" not in include_path:
            continue

        parts = Path(include_path).parts
        ff_index = next(idx for idx, part in enumerate(parts) if part.endswith(".ff"))
        ff_dir_name = parts[ff_index]
        remainder = "/".join(parts[ff_index + 1 :])
        is_absolute = Path(include_path).is_absolute()

        if is_absolute:
            ff_source = Path(*parts[: ff_index + 1])
        else:
            # Built-in force fields emit relative includes; resolve the .ff directory
            # against the GROMACS search paths so it can be bundled.
            if search_paths is None:
                search_paths = _get_gmx_search_paths()
            ff_source = next(
                (sp / ff_dir_name for sp in search_paths if (sp / ff_dir_name).is_dir()),
                None,
            )
            if ff_source is None:
                raise RuntimeError(
                    f"Force-field directory '{ff_dir_name}' referenced by {top_path} was "
                    f"not found in any GROMACS search path: {[str(p) for p in search_paths]}. "
                    "Cannot bundle it for a portable topology."
                )

        # A topology has several includes into the same .ff directory
        # (forcefield.itp, water, ions); copy the directory only once.
        if ff_source not in copied_sources:
            shutil.copytree(ff_source, top_path.parent / ff_dir_name, dirs_exist_ok=True)
            copied_sources.add(ff_source)

        # Relative includes already resolve from the topology's directory; only
        # absolute includes need rewriting to the bundled relative form.
        if is_absolute:
            lines[i] = line.replace(include_path, f"{ff_dir_name}/{remainder}")
            rewrote_line = True

    if rewrote_line:
        top_path.write_text("".join(lines))
    if ff_dir_name is not None:
        logger.info(f"Bundled force field '{ff_dir_name}' into {top_path.parent}")

    return ff_dir_name


_NET_CHARGE_WARNING = "system with net charge"


def _grompp_warnings(output: str) -> list[str]:
    """Return the text of each ``WARNING <n> [...]`` block in grompp output."""
    blocks = re.split(r"^WARNING \d+ \[", output, flags=re.MULTILINE)[1:]
    return [block.split("\n\n", 1)[0] for block in blocks]


def validate_with_grompp(
    topology: Path, structure: Path, mdp: Path, cwd: Path, allow_net_charge: bool = False
) -> None:
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
    allow_net_charge : bool, optional
        Accept grompp's warning about Ewald electrostatics in a system with net
        charge, for builds that deliberately skip neutralization. Any other
        warning still fails validation. Default is False.

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
        str(1 if allow_net_charge else 0),
    ]
    # A single grompp call: gmx_mpi aborts through MPI on failure, which can take
    # down an enclosing srun step, so never rely on a failing first attempt.
    try:
        result = _run_gmx("grompp validation", cmd, cwd, "", timeout=60)
    finally:
        tpr_out.unlink(missing_ok=True)
        (cwd / "mdout.mdp").unlink(missing_ok=True)

    if allow_net_charge:
        grompp_warnings = _grompp_warnings(result.stdout + result.stderr)
        unexpected = [w for w in grompp_warnings if _NET_CHARGE_WARNING not in " ".join(w.split())]
        if unexpected:
            raise RuntimeError(
                "gmx grompp validation reported warnings other than the accepted "
                "net-charge warning:\n" + "\n".join(unexpected)
            )
        if grompp_warnings:
            logger.warning(
                "System has a net charge (neutralization disabled); accepted grompp's "
                "net-charge warning."
            )

    logger.info("grompp validation passed.")
