# ABOUTME: Run schedules generated at build time from BuildInput + SimulationSettings
# ABOUTME: Exposes the mdp writer and the re-render API for existing build directories
"""Run schedules generated at build time.

The GROMACS mdp files are rendered from the build input and its simulation
settings (:mod:`mdfactory.run_schedules.gromacs`). :func:`apply_simulation_settings`
re-renders them in existing build directories.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from mdfactory.models.input import BuildInput, write_build_record
from mdfactory.models.simulation_settings import (
    STAGES,
    SimulationSettings,
    load_simulation_settings,
    merge_simulation_settings,
)

from .gromacs import STAGE_FILES, render_run_schedule, write_mdp_files

__all__ = [
    "STAGE_FILES",
    "SettingsUpdate",
    "apply_simulation_settings",
    "render_run_schedule",
    "write_mdp_files",
]


@dataclass
class SettingsUpdate:
    """Outcome of re-rendering one build directory.

    Attributes
    ----------
    directory : Path
        The build directory.
    status : {"updated", "unchanged", "refused"}
        ``updated``: settings changed and the mdp files were (or, in a dry
        run, would be) rewritten. ``unchanged``: the new settings equal the
        recorded ones. ``refused``: nothing was written, see ``reason``.
    changed_stages : list of str
        Stages whose mdp parameters change.
    written : list of Path
        Files written (or that would be written in a dry run).
    reason : str or None
        Why the directory was refused.

    """

    directory: Path
    status: Literal["updated", "unchanged", "refused"]
    changed_stages: list[str] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)
    reason: str | None = None


def _stage_outputs(directory: Path, stage: str) -> list[Path]:
    """Return the existing GROMACS outputs of *stage* in *directory*."""
    # Imported here: the build path must not import the orchestration package.
    from mdfactory.orchestration.stages import STAGE_REGISTRY

    spec = next(spec for spec in STAGE_REGISTRY if spec.mdp_file == STAGE_FILES[stage])
    names = [f"{spec.deffnm}.{ext}" for ext in ("tpr", "cpt", "log", "edr")]
    names += [spec.gro_out] if spec.gro_out else []
    names += list(spec.traj_files)
    return [directory / name for name in names if (directory / name).exists()]


class _Refused(Exception):
    """A directory cannot be re-rendered; the message is the reason."""


def _load_record(directory: Path) -> tuple[Path, dict[str, Any], BuildInput]:
    record = directory / f"{directory.name}.yaml"
    if not record.is_file():
        raise _Refused(f"no build record {record.name}")
    data = yaml.safe_load(record.read_text())
    try:
        current = BuildInput(**data)
    except (TypeError, ValueError) as exc:
        raise _Refused(f"invalid build record: {exc}") from exc
    if current.hash != directory.name:
        raise _Refused(f"the build record's hash {current.hash} does not match the directory name")
    return record, data, current


def _update_directory(
    directory: Path, settings: Mapping[str, Any], dry_run: bool
) -> SettingsUpdate:
    record, data, current = _load_record(directory)
    merged = merge_simulation_settings(current.simulation_settings, settings)
    try:
        updated = BuildInput(**{**data, "simulation_settings": merged})
    except (TypeError, ValueError) as exc:
        raise _Refused(f"invalid settings: {exc}") from exc
    if updated.hash != current.hash:
        raise _Refused(
            "temperature and pressure are part of the build identity; build a new "
            "system for another state point"
        )
    if updated.simulation_settings == current.simulation_settings:
        return SettingsUpdate(directory, "unchanged")

    old, new = render_run_schedule(current), render_run_schedule(updated)
    changed = [stage for stage in STAGES if old[stage] != new[stage]]
    outputs = {stage: _stage_outputs(directory, stage) for stage in STAGES}
    blocked = {stage: outputs[stage] for stage in changed if outputs[stage]}
    if blocked:
        details = "; ".join(
            f"{stage}: {', '.join(p.name for p in paths)}" for stage, paths in blocked.items()
        )
        raise _Refused(
            f"changed stages already have outputs ({details}); remove them before re-rendering"
        )

    stages = tuple(stage for stage in STAGES if not outputs[stage])
    written = [directory / STAGE_FILES[stage] for stage in stages] + [record]
    if not dry_run:
        write_mdp_files(updated, directory, stages)
        write_build_record(updated, directory)
    return SettingsUpdate(directory, "updated", changed_stages=changed, written=written)


def _try_update(directory: Path, settings: Mapping[str, Any], dry_run: bool) -> SettingsUpdate:
    try:
        return _update_directory(directory, settings, dry_run)
    except _Refused as exc:
        return SettingsUpdate(directory, "refused", reason=str(exc))


def apply_simulation_settings(
    build_dirs: Iterable[Path],
    settings: SimulationSettings | Mapping[str, Any] | Path,
    *,
    dry_run: bool = False,
) -> list[SettingsUpdate]:
    """Re-render the mdp files of existing build directories with new settings.

    The new settings are merged over each directory's recorded settings
    (``<hash>.yaml``) and validated against its build. A directory is refused
    when the merge changes temperature or pressure (the identity), or when a
    stage whose parameters change already has outputs. Otherwise the mdp
    files of every stage without outputs and the record are rewritten; the
    hash stays the same.

    Parameters
    ----------
    build_dirs : iterable of Path
        Build directories, each named after its hash.
    settings : SimulationSettings, mapping or Path
        Partial settings, or a settings YAML file. A
        :class:`SimulationSettings` contributes only its explicitly set fields.
    dry_run : bool, optional
        Report what would change without writing anything.

    Returns
    -------
    list of SettingsUpdate
        One entry per directory.

    """
    if isinstance(settings, (str, Path)):
        settings = load_simulation_settings(Path(settings))
    mapping = merge_simulation_settings(None, settings)
    SimulationSettings.model_validate(mapping)
    return [_try_update(Path(d).resolve(), mapping, dry_run) for d in build_dirs]
