# ABOUTME: MDP file parser, modifier, and writer for rescue tier parameter adjustment
# ABOUTME: Supports binary-division rescue strategy (halving step sizes, doubling nsteps)
"""MDP file utilities for adaptive rescue retry.

Provides functions to parse GROMACS MDP parameter files, modify critical
parameters for rescue tiers (binary division strategy), and write modified
files back to disk preserving structure and comments.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

#: Parameters modified per stage during rescue.  Each maps parameter name
#: to the operation applied at each tier: ``"halve"`` divides by 2,
#: ``"double"`` multiplies by 2.
RESCUE_PARAMS: dict[str, dict[str, str]] = {
    "EM": {"emstep": "halve", "nsteps": "double"},
    "NVT": {"dt": "halve", "nsteps": "double"},
    "NPT": {"dt": "halve", "nsteps": "double"},
}

#: Stages eligible for rescue retry (Production is never rescued).
RESCUE_ELIGIBLE_STAGES: frozenset[str] = frozenset(RESCUE_PARAMS.keys())


def parse_mdp(path: Path) -> list[tuple[str, str | None, str | None]]:
    """Parse an MDP file preserving structure for round-trip modification.

    Each line is returned as a tuple ``(raw_line, key, value)`` where
    ``key`` and ``value`` are ``None`` for comment/blank lines.

    Parameters
    ----------
    path : Path
        Path to the MDP file.

    Returns
    -------
    list[tuple[str, str | None, str | None]]
        List of ``(raw_line, normalized_key, value_string)`` tuples.
        Keys are normalized to lowercase with dashes replaced by underscores.

    """
    lines: list[tuple[str, str | None, str | None]] = []
    content = path.read_text()

    for raw_line in content.splitlines():
        # Strip comment portion (everything after ';')
        stripped = raw_line.split(";")[0].strip()
        if "=" not in stripped:
            lines.append((raw_line, None, None))
            continue

        raw_key, val = stripped.split("=", 1)
        norm_key = raw_key.strip().lower().replace("-", "_")
        lines.append((raw_line, norm_key, val.strip()))

    return lines


def get_mdp_value(parsed: list[tuple[str, str | None, str | None]], key: str) -> str | None:
    """Extract the value of a specific key from parsed MDP data.

    Parameters
    ----------
    parsed : list
        Output of :func:`parse_mdp`.
    key : str
        Normalized key name (lowercase, underscores).

    Returns
    -------
    str or None
        The value string, or ``None`` if the key is not found.

    """
    for _, k, v in parsed:
        if k == key:
            return v
    return None


def modify_mdp_value(
    parsed: list[tuple[str, str | None, str | None]],
    key: str,
    new_value: str,
) -> list[tuple[str, str | None, str | None]]:
    """Return a copy of parsed MDP with one key's value replaced.

    The raw line is reconstructed to preserve the key's original casing
    and surrounding whitespace as much as possible.

    Parameters
    ----------
    parsed : list
        Output of :func:`parse_mdp`.
    key : str
        Normalized key to modify.
    new_value : str
        New value string.

    Returns
    -------
    list
        Modified parsed list (new copy).

    """
    result = []
    for raw_line, k, v in parsed:
        if k == key:
            # Reconstruct: keep everything before '=' and any trailing comment
            parts = raw_line.split(";", 1)
            assignment = parts[0]
            comment = f";{parts[1]}" if len(parts) > 1 else ""

            eq_idx = assignment.index("=")
            prefix = assignment[: eq_idx + 1]  # "key    ="
            # Preserve spacing after '=' if possible
            after_eq = assignment[eq_idx + 1 :]
            leading_spaces = len(after_eq) - len(after_eq.lstrip())
            spacing = " " * max(leading_spaces, 1)

            new_raw = f"{prefix}{spacing}{new_value}"
            if comment:
                # Pad to roughly original width for alignment
                new_raw = f"{new_raw:<{len(assignment)}}{comment}"
            result.append((new_raw, k, new_value))
        else:
            result.append((raw_line, k, v))
    return result


def write_mdp(parsed: list[tuple[str, str | None, str | None]], path: Path) -> None:
    """Write parsed (possibly modified) MDP data back to a file.

    Parameters
    ----------
    parsed : list
        Parsed MDP data (from :func:`parse_mdp` or :func:`modify_mdp_value`).
    path : Path
        Output file path.

    """
    lines = [raw_line for raw_line, _, _ in parsed]
    path.write_text("\n".join(lines) + "\n")


def apply_rescue_tier(sim_dir: Path, stage: str, tier: int) -> Path:
    """Apply rescue-tier parameter modifications to a stage's MDP file.

    Reads the stage's standard MDP file, applies binary-division
    modifications for the given tier, and writes the result to a
    rescue-specific filename (e.g. ``em_rescue_t1.mdp``).

    Parameters
    ----------
    sim_dir : Path
        Simulation directory containing the MDP file.
    stage : str
        Stage name (``"EM"``, ``"NVT"``, ``"NPT"``).
    tier : int
        Rescue tier (1-based). Each tier halves/doubles parameters
        relative to the original (not the previous tier).

    Returns
    -------
    Path
        Path to the written rescue MDP file.

    Raises
    ------
    ValueError
        If the stage is not rescue-eligible or the tier is < 1.
    FileNotFoundError
        If the source MDP file does not exist.

    """
    if stage not in RESCUE_PARAMS:
        raise ValueError(f"Stage {stage!r} is not rescue-eligible. Valid: {sorted(RESCUE_PARAMS)}")
    if tier < 1:
        raise ValueError(f"Rescue tier must be >= 1, got {tier}")

    from .stages import STAGE_BY_NAME

    spec = STAGE_BY_NAME[stage]
    source_mdp = sim_dir / spec.mdp_file

    if not source_mdp.exists():
        raise FileNotFoundError(f"Source MDP not found: {source_mdp}")

    parsed = parse_mdp(source_mdp)
    params = RESCUE_PARAMS[stage]

    for param, operation in params.items():
        current_val = get_mdp_value(parsed, param)
        if current_val is None:
            logger.warning(f"MDP parameter {param!r} not found in {source_mdp}, skipping")
            continue

        try:
            numeric = float(current_val)
        except ValueError:
            logger.warning(f"Cannot parse {param}={current_val!r} as number, skipping")
            continue

        if operation == "halve":
            new_val = numeric / (2**tier)
        elif operation == "double":
            new_val = numeric * (2**tier)
        else:
            raise ValueError(f"Unknown operation: {operation!r}")

        # Format: use int if the result is a whole number, else float
        if new_val == int(new_val) and "." not in current_val:
            formatted = str(int(new_val))
        else:
            formatted = f"{new_val:g}"

        logger.warning(f"RESCUE tier {tier} for {stage}: {param} = {current_val} → {formatted}")
        parsed = modify_mdp_value(parsed, param, formatted)

    # Write rescue MDP with tier-specific name
    rescue_name = f"{spec.mdp_file.rsplit('.', 1)[0]}_rescue_t{tier}.mdp"
    rescue_path = sim_dir / rescue_name
    write_mdp(parsed, rescue_path)

    return rescue_path
