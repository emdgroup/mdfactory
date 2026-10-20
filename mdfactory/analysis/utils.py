# ABOUTME: Shared analysis utilities for simulation discovery and metadata extraction
# ABOUTME: Provides simulation path scanning and species composition flattening
"""Shared analysis utilities for simulation discovery and metadata extraction."""

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from mdfactory.models.input import BuildInput
from mdfactory.utils.utilities import load_yaml_file

from .constants import STATUS_ORDER


def discover_simulations(
    base_dir: Path | str,
    trajectory_file: str = "prod.xtc",
    structure_file: str = "system.pdb",
    min_status: str | None = None,
) -> pd.DataFrame:
    """Discover simulation directories and create Simulation instances.

    Scans a base directory for subdirectories containing simulation files
    (structure and YAML BuildInput) and creates Simulation instances for
    each discovered simulation.

    Parameters
    ----------
    base_dir : Path | str
        Base directory to scan for simulations
    trajectory_file : str
        Name of trajectory file (default: prod.xtc)
    structure_file : str
        Name of structure file (default: system.pdb)
    min_status : str | None
        Minimum status to include. One of: "build", "equilibrated",
        "production", "completed". If None, defaults to "production"
        for backward compatibility (requires trajectory to exist).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['hash', 'path', 'simulation', 'status'] where:
        - hash: Simulation hash (primary identifier)
        - path: Absolute path to simulation directory
        - simulation: Simulation instance
        - status: Simulation status string

    Raises
    ------
    ValueError
        If multiple valid YAML files found in a directory, or invalid min_status

    """
    from .simulation import Simulation  # Import here to avoid circular dependency

    # Default to "production" for backward compatibility
    if min_status is None:
        min_status = "production"

    if min_status not in STATUS_ORDER:
        raise ValueError(f"Invalid min_status '{min_status}'. Must be one of: {STATUS_ORDER}")

    min_status_idx = STATUS_ORDER.index(min_status)

    base_path = Path(base_dir)
    data = []
    # NOTE: include base_path itself in addition to its subdirectories
    dirs = list(base_path.iterdir()) + [base_path]

    for d in dirs:
        if not d.is_dir():
            continue
        if not (d / structure_file).exists():
            continue

        # Try to find valid BuildInput YAML
        yaml_files = list(d.glob("*.yaml"))
        if not yaml_files:
            continue

        valid_inputs = []
        for yaml_file in yaml_files:
            try:
                valid_inputs.append(BuildInput(**load_yaml_file(yaml_file)))
            except Exception:
                continue

        if len(valid_inputs) == 0:
            continue
        elif len(valid_inputs) > 1:
            raise ValueError(f"Multiple valid YAML files found in {d}")

        build_input = valid_inputs[0]

        # Create Simulation instance (trajectory optional for build-only)
        try:
            simulation = Simulation(
                d,
                build_input=build_input,
                structure_file=structure_file,
                trajectory_file=trajectory_file if min_status != "build" else None,
            )
        except (FileNotFoundError, ValueError):
            continue

        # Check status meets minimum threshold
        status = simulation.status
        status_idx = STATUS_ORDER.index(status)
        if status_idx < min_status_idx:
            continue

        data.append(
            {
                "hash": build_input.hash,
                "path": d.resolve(),
                "simulation": simulation,
                "status": status,
            }
        )

    return pd.DataFrame(data, columns=["hash", "path", "simulation", "status"])


def flatten_species_composition(
    build_input: BuildInput,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten species composition into a flat dictionary.

    Extracts species counts and fractions as separate columns with
    naming convention: {prefix}{resname}_{metric}

    Parameters
    ----------
    build_input : BuildInput
        BuildInput instance to extract metadata from
    prefix : str
        Optional prefix for column names (default: "")

    Returns
    -------
    dict[str, any]
        Flattened dict with keys:
        - simulation_type: str
        - total_count: int
        - {prefix}{resname}_count: int
        - {prefix}{resname}_fraction: float

    """
    flat = {
        "simulation_type": build_input.simulation_type,
        "total_count": build_input.system.total_count,
    }

    # Add per-species counts and fractions
    for species in build_input.system.species:
        resname = species.resname
        flat[f"{prefix}{resname}_count"] = species.count
        flat[f"{prefix}{resname}_fraction"] = species.fraction

    return flat


def flatten_system_parameters(build_input: BuildInput) -> dict[str, Any]:
    """Flatten system-specific parameters (z_padding, target_density, etc.).

    Useful for analyses focusing on system configuration rather than composition.

    Parameters
    ----------
    build_input : BuildInput
        BuildInput instance to extract metadata from

    Returns
    -------
    dict[str, any]
        Flattened dict with system-specific parameters

    """
    flat = {
        "simulation_type": build_input.simulation_type,
        "engine": build_input.engine,
        "parametrization": build_input.parametrization,
    }

    # Add system-specific parameters from metadata
    system_specific = build_input.metadata.get("system_specific", {})
    flat.update(system_specific)

    return flat


def make_chemistry_extractor(
    species_groups: dict[str, list[str]],
) -> Callable:
    """Create a flatten function that extracts chemistry based on species groupings.

    Allows flexible mapping of resnames to output groups, with support for
    merging multiple resnames into one group (e.g., protonation states).

    Parameters
    ----------
    species_groups : dict[str, list[str]]
        Mapping of output group name to list of resnames to include.
        Single-item lists extract that resname directly.
        Multi-item lists merge those resnames (summing counts/fractions).

        Example:
            {
                "HL": ["HL"],           # single resname
                "CHL": ["CHL"],         # single resname
                "IL": ["ILN", "ILP"],   # merged group
            }

    Returns
    -------
    callable
        A function(build_input) -> dict that extracts chemistry.

        Output columns for single-resname groups:
        - {group}_count, {group}_fraction, {group}_smiles

        Output columns for merged groups:
        - {group}_count, {group}_fraction (summed totals)
        - {resname}_count, {resname}_fraction, {resname}_smiles (per member)

    Examples
    --------
    >>> extractor = make_chemistry_extractor({"HL": ["HL"], "IL": ["ILN", "ILP"]})
    >>> chemistry = extractor(build_input)
    >>> # Returns: HL_count, HL_fraction, HL_smiles,
    >>> #          IL_count, IL_fraction, ILN_count, ILN_fraction, ILN_smiles, ...

    """

    def extractor(build_input: BuildInput) -> dict[str, Any]:
        species = build_input.system.species

        # Build resname -> species lookup
        resname_to_species = {sp.resname: sp for sp in species}

        result = {}

        for group_name, resnames in species_groups.items():
            if len(resnames) == 1:
                # Single resname group - straightforward extraction
                resname = resnames[0]
                sp = resname_to_species.get(resname)
                if sp:
                    result[f"{group_name}_count"] = sp.count
                    result[f"{group_name}_fraction"] = sp.fraction
                    result[f"{group_name}_smiles"] = sp.smiles
                else:
                    result[f"{group_name}_count"] = 0
                    result[f"{group_name}_fraction"] = 0.0
                    result[f"{group_name}_smiles"] = None
            else:
                # Merged group - sum totals and keep individual columns
                total_count = 0
                total_fraction = 0.0

                for resname in resnames:
                    sp = resname_to_species.get(resname)
                    if sp:
                        total_count += sp.count
                        total_fraction += sp.fraction
                        result[f"{resname}_count"] = sp.count
                        result[f"{resname}_fraction"] = sp.fraction
                        result[f"{resname}_smiles"] = sp.smiles
                    else:
                        result[f"{resname}_count"] = 0
                        result[f"{resname}_fraction"] = 0.0
                        result[f"{resname}_smiles"] = None

                result[f"{group_name}_count"] = total_count
                result[f"{group_name}_fraction"] = total_fraction

        return result

    return extractor


# Default LNP chemistry configuration
LNP_SPECIES_GROUPS = {
    "HL": ["HL"],
    "CHL": ["CHL"],
    "IL": ["ILN", "ILP"],
}

# Pre-built extractor for standard LNP simulations
extract_lnp_chemistry = make_chemistry_extractor(LNP_SPECIES_GROUPS)


def extract_all_species(build_input: BuildInput) -> dict[str, Any]:
    """Extract all species data from BuildInput without any grouping or filtering.

    Automatically extracts count, fraction, and SMILES for every species
    defined in the YAML file. No configuration needed.

    Parameters
    ----------
    build_input : BuildInput
        BuildInput instance to extract chemistry from

    Returns
    -------
    dict[str, any]
        Dict with keys for each species resname:
        - {resname}_count: int
        - {resname}_fraction: float
        - {resname}_smiles: str | None
        - total_species_count: int (number of species types)
        - total_molecule_count: int (sum of all counts)

    Examples
    --------
    >>> chemistry = extract_all_species(build_input)
    >>> # For a system with ILI, HLI, CHL species:
    >>> # Returns: ILI_count, ILI_fraction, ILI_smiles,
    >>> #          HLI_count, HLI_fraction, HLI_smiles,
    >>> #          CHL_count, CHL_fraction, CHL_smiles,
    >>> #          total_species_count, total_molecule_count

    """
    from loguru import logger

    try:
        species = build_input.system.species
    except AttributeError as e:
        logger.warning(f"Failed to extract species from build_input: {e}")
        return {"error": str(e)}

    if not species:
        logger.warning("No species found in build_input")
        return {
            "total_species_count": 0,
            "total_molecule_count": 0,
        }

    result = {}
    counts = []

    for sp in species:
        resname = sp.resname
        result[f"{resname}_count"] = sp.count
        result[f"{resname}_fraction"] = sp.fraction
        result[f"{resname}_smiles"] = getattr(sp, "smiles", None)
        counts.append(sp.count)

    result["total_species_count"] = len(species)
    # A concentration-based species has no count until build time, so the
    # molecule total is unknown from the BuildInput alone; report None rather
    # than crash when summing.
    result["total_molecule_count"] = None if any(c is None for c in counts) else sum(counts)

    return result


def get_chemistry_extractor(
    mode: str = "all",
    species_groups: dict[str, list[str]] | None = None,
) -> Callable:
    """Get a chemistry extractor function based on mode.

    Convenience function for selecting between extraction modes.

    Parameters
    ----------
    mode : str
        Extraction mode:
        - "all": Extract all species from YAML (no filtering/grouping)
        - "lnp": Use LNP-specific grouping (HL, CHL, IL with ILN+ILP merged)
        - "custom": Use custom species_groups (requires species_groups param)
    species_groups : dict[str, list[str]] | None
        Required when mode="custom". Mapping of group names to resnames.

    Returns
    -------
    callable
        A function(build_input) -> dict that extracts chemistry.

    Raises
    ------
    ValueError
        If mode="custom" but species_groups not provided, or unknown mode.

    Examples
    --------
    >>> extractor = get_chemistry_extractor("all")
    >>> extractor = get_chemistry_extractor("lnp")
    >>> extractor = get_chemistry_extractor("custom", {"IL": ["MC3N", "MC3P"]})

    """
    if mode == "all":
        return extract_all_species
    elif mode == "lnp":
        return extract_lnp_chemistry
    elif mode == "custom":
        if species_groups is None:
            raise ValueError("species_groups required when mode='custom'")
        return make_chemistry_extractor(species_groups)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'all', 'lnp', or 'custom'.")


def system_chemistry(simulation, **kwargs) -> pd.DataFrame:
    """Extract species composition as a long-format DataFrame.

    One row per species in the simulation. Reads only from BuildInput
    metadata (no trajectory data required).

    Parameters
    ----------
    simulation : Simulation
        Simulation instance (only build_input is accessed)
    **kwargs
        Absorbed for compatibility with analysis function contract

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - resname: str - Residue name
        - smiles: str | None - SMILES string (None for base Species)
        - count: int - Molecule count
        - fraction: float - Mole fraction
        - simulation_type: str - e.g. "bilayer", "mixedbox"

        For ``protein_mixedbox``, rows additionally contain requested
        concentration, resolved/final counts, and achieved concentration.

    """
    build_input = simulation.build_input
    achieved_species = []
    build_metadata_path = getattr(simulation, "path", None)
    if build_metadata_path is not None:
        build_metadata_path = Path(build_metadata_path) / "build_metadata.json"
        if build_metadata_path.is_file():
            import json  # noqa: PLC0415

            build_metadata = json.loads(build_metadata_path.read_text())
            achieved_species = build_metadata.get("species", [])
    rows = []
    if build_input.simulation_type == "protein_mixedbox":
        # The protein anchors the box but is not part of system.species (packing
        # is solution-only). List it as a species row here to match proteinbox
        # output; it consumes no achieved_species index.
        protein = build_input.system.protein
        rows.append(
            {
                "resname": protein.resname,
                "smiles": None,
                "count": protein.count,
                "fraction": protein.fraction,
                "simulation_type": build_input.simulation_type,
                "concentration": None,
                "resolved_count": None,
                "final_count": None,
                "achieved_concentration": None,
            }
        )
    for species_index, sp in enumerate(build_input.system.species):
        achieved = achieved_species[species_index] if species_index < len(achieved_species) else {}
        row = {
            "resname": sp.resname,
            "smiles": getattr(sp, "smiles", None),
            "count": sp.count,
            "fraction": sp.fraction,
            "simulation_type": build_input.simulation_type,
        }
        if build_input.simulation_type == "protein_mixedbox":
            row.update(
                {
                    "concentration": getattr(sp, "concentration", None),
                    "resolved_count": achieved.get("resolved_count"),
                    "final_count": achieved.get("final_count"),
                    "achieved_concentration": achieved.get("achieved_concentration_molar"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
