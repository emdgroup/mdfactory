# ABOUTME: Stage composition functions for GROMACS MD pipeline
# ABOUTME: Chains grompp + mdrun for EM → NVT → NPT → Production
"""GROMACS MD simulation stage composition.

Provides :class:`StageSpec`, :data:`STAGE_REGISTRY`, :data:`STAGE_BY_NAME`,
and the generic :func:`run_stage` that drives all four pipeline stages.
Named convenience wrappers (:func:`run_em_stage` etc.) delegate to
:func:`run_stage` for backward-compatible call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from parsl import AppFuture


@dataclass(frozen=True)
class StageSpec:
    """Descriptor for a single GROMACS MD pipeline stage.

    Parameters
    ----------
    name : str
        Human-readable stage name used in CLI and logs (e.g. ``"EM"``).
    deffnm : str
        Prefix passed to ``gmx mdrun -deffnm`` (e.g. ``"min"``).
    mdp_file : str
        MDP parameter file name (e.g. ``"em.mdp"``).
    gro_in : str
        Input structure file for ``gmx grompp -c`` (e.g. ``"system.pdb"``).
    gro_out : str
        Expected output ``.gro`` file; empty string when the stage writes a
        trajectory instead (Production stage).
    tpr_file : str
        Preprocessed run-input file (e.g. ``"min.tpr"``).
    cpt_file : str
        Checkpoint file written by mdrun (e.g. ``"min.cpt"``).
    prereq_cpt : str or None
        Checkpoint from the *prior* stage that is required both as the
        ``gmx grompp -t`` input and for workflow-integrity validation.
        ``None`` for stages with no velocity continuation (EM, NVT).
    traj_files : tuple of str
        Accepted trajectory output files checked after mdrun — non-empty only
        for the Production stage: ``("prod.xtc", "prod.trr")``.
    ref_file : str or None
        Reference structure for ``gmx grompp -r`` (position restraints).
        ``None`` when no position restraints are used.
    maxwarn : int
        Maximum grompp warnings to allow (``-maxwarn``).
    supports_pme_gpu : bool
        ``False`` for EM (``steep`` integrator) — GROMACS rejects ``-pme gpu``
        for non-dynamical integrators.

    """

    name: str
    deffnm: str
    mdp_file: str
    gro_in: str
    gro_out: str
    tpr_file: str
    cpt_file: str
    prereq_cpt: "str | None"
    traj_files: "tuple[str, ...]"
    ref_file: "str | None"
    maxwarn: int
    supports_pme_gpu: bool


#: Canonical ordered list of all pipeline stages.
STAGE_REGISTRY: list[StageSpec] = [
    StageSpec(
        name="EM",
        deffnm="min",
        mdp_file="em.mdp",
        gro_in="system.pdb",
        gro_out="min.gro",
        tpr_file="min.tpr",
        cpt_file="min.cpt",
        prereq_cpt=None,
        traj_files=(),
        ref_file=None,
        maxwarn=1,
        supports_pme_gpu=False,
    ),
    StageSpec(
        name="NVT",
        deffnm="nvt",
        mdp_file="nvt.mdp",
        gro_in="min.gro",
        gro_out="nvt.gro",
        tpr_file="nvt.tpr",
        cpt_file="nvt.cpt",
        prereq_cpt=None,
        traj_files=(),
        ref_file="min.gro",
        maxwarn=1,
        supports_pme_gpu=True,
    ),
    StageSpec(
        name="NPT",
        deffnm="npt",
        mdp_file="npt.mdp",
        gro_in="nvt.gro",
        gro_out="npt.gro",
        tpr_file="npt.tpr",
        cpt_file="npt.cpt",
        prereq_cpt="nvt.cpt",
        traj_files=(),
        ref_file="nvt.gro",
        maxwarn=2,
        supports_pme_gpu=True,
    ),
    StageSpec(
        name="Production",
        deffnm="prod",
        mdp_file="md.mdp",
        gro_in="npt.gro",
        gro_out="",
        tpr_file="prod.tpr",
        cpt_file="prod.cpt",
        prereq_cpt="npt.cpt",
        traj_files=("prod.xtc", "prod.trr"),
        ref_file=None,
        maxwarn=2,
        supports_pme_gpu=True,
    ),
]

#: Lookup from stage name to :class:`StageSpec` — the single source of truth
#: for all stage-specific file names, dependencies, and flags.
STAGE_BY_NAME: dict[str, StageSpec] = {s.name: s for s in STAGE_REGISTRY}


class ResourceHints(NamedTuple):
    """Resource hints extracted from a stage executor config for mdrun.

    Parameters
    ----------
    ntasks : int
        Explicit thread count (0 = auto-detect from SLURM env var).
    disable_gpu : bool
        ``True`` when the stage config has no GPU resource (``gres`` is
        ``None`` or does not contain ``"gpu"``).
    gmx_binary : str
        GROMACS binary selection: ``"auto"``, ``"gmx"``, or ``"gmx_mpi"``.

    """

    ntasks: int
    disable_gpu: bool
    gmx_binary: str  # "auto" | "gmx" | "gmx_mpi"


def _extract_resource_hints(stage_config: Any) -> ResourceHints:
    """Extract mdrun resource hints from a stage-specific executor config.

    Parameters
    ----------
    stage_config : SlurmExecutorConfig or None
        Per-stage config returned by :meth:`SlurmExecutorConfig.get_stage_config`.
        May be ``None`` for local / unconfigured runs.

    Returns
    -------
    ResourceHints
        Named tuple with ``ntasks``, ``disable_gpu``, and ``gmx_binary``.

    Notes
    -----
    Uses :func:`getattr` with safe fallbacks so the function remains
    callable with any config object that does not yet declare all fields.

    """
    if stage_config is None:
        return ResourceHints(ntasks=0, disable_gpu=False, gmx_binary="auto")
    ntasks = getattr(stage_config, "cpus_per_node", 0) or 0
    gres = getattr(stage_config, "gres", None)
    disable_gpu = gres is None or "gpu" not in str(gres).lower()
    gmx_binary = getattr(stage_config, "gmx_binary", "auto")
    return ResourceHints(ntasks=ntasks, disable_gpu=disable_gpu, gmx_binary=gmx_binary)


def run_stage(
    spec: StageSpec,
    sim_dir: Path,
    prev_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
    stage_config: Any = None,
) -> "AppFuture":
    """Execute a single GROMACS pipeline stage.

    All four pipeline stages share an identical structural skeleton — resolve
    working directory, extract resource hints, optionally skip grompp on
    restart, submit grompp → mdrun with explicit dependency chaining.  Only
    the file names and flags differ, so this generic function covers all cases
    driven by the :class:`StageSpec` descriptor.

    Parameters
    ----------
    spec : StageSpec
        Stage descriptor from :data:`STAGE_REGISTRY`.  Supplies all
        stage-specific file names, flags, and resource hints.
    sim_dir : Path
        Simulation directory (must be absolute-resolvable).
    prev_future : AppFuture or None
        Future from the preceding stage.  ``None`` when this is the first
        stage in the pipeline or when resuming from checkpoint and the
        preceding output already exists on disk.
    grompp_app : callable
        Result of :func:`~mdfactory.orchestration.apps.get_grompp_app`.
    mdrun_app : callable
        Result of :func:`~mdfactory.orchestration.apps.get_mdrun_app`.
    restart_from_cpt : str, optional
        Path to an existing checkpoint file.  When non-empty the grompp step
        is skipped and mdrun is invoked with ``-cpi <file> -append``.
    stage_config : SlurmExecutorConfig or None, optional
        Per-stage resource config.  Resource hints (thread count, GPU disable)
        are extracted and forwarded to the mdrun bash app.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    work_dir = str(sim_dir.resolve())
    hints = _extract_resource_hints(stage_config)

    if restart_from_cpt:
        # TPR exists from the interrupted run — skip grompp, resume mdrun.
        mdrun_inputs = [prev_future] if prev_future is not None else []
        return mdrun_app(
            deffnm=spec.deffnm,
            work_dir=work_dir,
            restart_from_cpt=restart_from_cpt,
            ntasks=hints.ntasks,
            disable_gpu=hints.disable_gpu,
            gmx_binary=hints.gmx_binary,
            gro_out=spec.gro_out,
            traj_files=spec.traj_files,
            inputs=mdrun_inputs,
        )

    grompp_inputs = [prev_future] if prev_future is not None else []

    grompp_kwargs: dict[str, Any] = {
        "mdp_file": spec.mdp_file,
        "gro_file": spec.gro_in,
        "top_file": "topology.top",
        "tpr_file": spec.tpr_file,
        "work_dir": work_dir,
        "maxwarn": spec.maxwarn,
        "inputs": grompp_inputs,
    }
    if spec.ref_file:
        grompp_kwargs["ref_file"] = spec.ref_file
    if spec.prereq_cpt:
        grompp_kwargs["cpt_file"] = spec.prereq_cpt

    grompp_fut = grompp_app(**grompp_kwargs)

    return mdrun_app(
        deffnm=spec.deffnm,
        work_dir=work_dir,
        ntasks=hints.ntasks,
        disable_gpu=hints.disable_gpu,
        pme_gpu=spec.supports_pme_gpu,
        gmx_binary=hints.gmx_binary,
        gro_out=spec.gro_out,
        traj_files=spec.traj_files,
        inputs=[grompp_fut],
    )


# ---- Convenience one-liner wrappers (backward-compatible named API) ----

def run_em_stage(sim_dir: Path, grompp_app, mdrun_app, stage_config: Any = None) -> "AppFuture":
    """Execute energy minimization stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory (contains system.pdb, topology.top, em.mdp).
    grompp_app : callable
        Result of get_grompp_app().
    mdrun_app : callable
        Result of get_mdrun_app().
    stage_config : SlurmExecutorConfig or None, optional
        Stage-specific resource config from
        :meth:`~mdfactory.orchestration.config.SlurmExecutorConfig.get_stage_config`.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    return run_stage(
        STAGE_BY_NAME["EM"], sim_dir, None, grompp_app, mdrun_app, stage_config=stage_config
    )


def run_nvt_stage(
    sim_dir: Path,
    em_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
    stage_config: Any = None,
) -> "AppFuture":
    """Execute NVT equilibration stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    em_future : AppFuture or None
        Future from EM stage (dependency). If None, assumes min.gro exists
        (checkpoint resume scenario).
    grompp_app : callable
        Result of get_grompp_app().
    mdrun_app : callable
        Result of get_mdrun_app().
    restart_from_cpt : str, optional
        Path to an existing NVT checkpoint file.  When non-empty the grompp
        step is skipped and mdrun is invoked with ``-cpi <file> -append``.
    stage_config : SlurmExecutorConfig or None, optional
        Per-stage resource config.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    return run_stage(
        STAGE_BY_NAME["NVT"],
        sim_dir,
        em_future,
        grompp_app,
        mdrun_app,
        restart_from_cpt=restart_from_cpt,
        stage_config=stage_config,
    )


def run_npt_stage(
    sim_dir: Path,
    nvt_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
    stage_config: Any = None,
) -> "AppFuture":
    """Execute NPT equilibration stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    nvt_future : AppFuture or None
        Future from NVT stage (dependency). If None, assumes nvt.gro and
        nvt.cpt exist (checkpoint resume scenario).
    grompp_app : callable
        Result of get_grompp_app().
    mdrun_app : callable
        Result of get_mdrun_app().
    restart_from_cpt : str, optional
        Path to an existing NPT checkpoint file.  When non-empty the grompp
        step is skipped and mdrun is invoked with ``-cpi <file> -append``.
    stage_config : SlurmExecutorConfig or None, optional
        Per-stage resource config.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    return run_stage(
        STAGE_BY_NAME["NPT"],
        sim_dir,
        nvt_future,
        grompp_app,
        mdrun_app,
        restart_from_cpt=restart_from_cpt,
        stage_config=stage_config,
    )


def run_production_stage(
    sim_dir: Path,
    npt_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
    stage_config: Any = None,
) -> "AppFuture":
    """Execute production MD stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    npt_future : AppFuture or None
        Future from NPT stage (dependency). If None, assumes npt.gro and
        npt.cpt exist (checkpoint resume scenario).
    grompp_app : callable
        Result of get_grompp_app().
    mdrun_app : callable
        Result of get_mdrun_app().
    restart_from_cpt : str, optional
        Path to an existing Production checkpoint file.  When non-empty the
        grompp step is skipped and mdrun is invoked with ``-cpi <file> -append``.
    stage_config : SlurmExecutorConfig or None, optional
        Per-stage resource config.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    return run_stage(
        STAGE_BY_NAME["Production"],
        sim_dir,
        npt_future,
        grompp_app,
        mdrun_app,
        restart_from_cpt=restart_from_cpt,
        stage_config=stage_config,
    )
