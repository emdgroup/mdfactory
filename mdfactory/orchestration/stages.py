# ABOUTME: Stage composition functions for GROMACS MD pipeline
# ABOUTME: Chains grompp + mdrun for EM → NVT → NPT → Production
"""GROMACS MD simulation stage composition.

Provides high-level stage functions that chain gmx grompp and gmx mdrun
for the 4-stage GROMACS pipeline: EM → NVT → NPT → Production.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parsl import AppFuture


def run_em_stage(sim_dir: Path, grompp_app, mdrun_app) -> "AppFuture":
    """Execute energy minimization stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory (contains system.pdb, topology.top, em.mdp).
    grompp_app : callable
        Result of get_grompp_app().
    mdrun_app : callable
        Result of get_mdrun_app().

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    work_dir = str(sim_dir.resolve())

    # Step 1: Preprocessing (no dependencies)
    grompp_fut = grompp_app(
        mdp_file="em.mdp",
        gro_file="system.pdb",
        top_file="topology.top",
        tpr_file="min.tpr",
        work_dir=work_dir,
        maxwarn=1,
        inputs=[],
    )

    # Step 2: Simulation (depends on grompp)
    mdrun_fut = mdrun_app(
        deffnm="min",
        work_dir=work_dir,
        inputs=[grompp_fut],
    )

    return mdrun_fut


def run_nvt_stage(
    sim_dir: Path,
    em_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
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
        step is skipped (the TPR already exists) and mdrun is invoked with
        ``-cpi <file> -append`` to continue from the saved state.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    work_dir = str(sim_dir.resolve())

    if restart_from_cpt:
        # TPR exists from the interrupted run — skip grompp, resume mdrun.
        mdrun_inputs = [em_future] if em_future is not None else []
        return mdrun_app(
            deffnm="nvt",
            work_dir=work_dir,
            restart_from_cpt=restart_from_cpt,
            inputs=mdrun_inputs,
        )

    # Build inputs list: include em_future only if provided
    grompp_inputs = [em_future] if em_future is not None else []

    grompp_fut = grompp_app(
        mdp_file="nvt.mdp",
        gro_file="min.gro",
        ref_file="min.gro",
        top_file="topology.top",
        tpr_file="nvt.tpr",
        work_dir=work_dir,
        maxwarn=1,
        inputs=grompp_inputs,
    )

    return mdrun_app(
        deffnm="nvt",
        work_dir=work_dir,
        inputs=[grompp_fut],
    )


def run_npt_stage(
    sim_dir: Path,
    nvt_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
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

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    work_dir = str(sim_dir.resolve())

    if restart_from_cpt:
        # TPR exists from the interrupted run — skip grompp, resume mdrun.
        mdrun_inputs = [nvt_future] if nvt_future is not None else []
        return mdrun_app(
            deffnm="npt",
            work_dir=work_dir,
            restart_from_cpt=restart_from_cpt,
            inputs=mdrun_inputs,
        )

    # Build inputs list: include nvt_future only if provided
    grompp_inputs = [nvt_future] if nvt_future is not None else []

    grompp_fut = grompp_app(
        mdp_file="npt.mdp",
        gro_file="nvt.gro",
        ref_file="nvt.gro",
        cpt_file="nvt.cpt",
        top_file="topology.top",
        tpr_file="npt.tpr",
        work_dir=work_dir,
        maxwarn=2,
        inputs=grompp_inputs,
    )

    return mdrun_app(
        deffnm="npt",
        work_dir=work_dir,
        inputs=[grompp_fut],
    )


def run_production_stage(
    sim_dir: Path,
    npt_future: "AppFuture | None",
    grompp_app,
    mdrun_app,
    restart_from_cpt: str = "",
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
        grompp step is skipped (prod.tpr already exists) and mdrun is invoked
        with ``-cpi <file> -append`` to continue the interrupted trajectory.

    Returns
    -------
    AppFuture
        Future that resolves when mdrun completes.

    """
    work_dir = str(sim_dir.resolve())

    if restart_from_cpt:
        # prod.tpr exists from the interrupted run — skip grompp, resume mdrun.
        mdrun_inputs = [npt_future] if npt_future is not None else []
        return mdrun_app(
            deffnm="prod",
            work_dir=work_dir,
            restart_from_cpt=restart_from_cpt,
            inputs=mdrun_inputs,
        )

    # Build inputs list: include npt_future only if provided
    grompp_inputs = [npt_future] if npt_future is not None else []

    grompp_fut = grompp_app(
        mdp_file="md.mdp",
        gro_file="npt.gro",
        cpt_file="npt.cpt",
        top_file="topology.top",
        tpr_file="prod.tpr",
        work_dir=work_dir,
        maxwarn=2,
        inputs=grompp_inputs,
    )

    return mdrun_app(
        deffnm="prod",
        work_dir=work_dir,
        inputs=[grompp_fut],
    )


def run_full_pipeline(sim_dir: Path, grompp_app, mdrun_app) -> "AppFuture":
    """Execute full 4-stage GROMACS pipeline with automatic dependencies.

    Chains EM → NVT → NPT → Production with Parsl data dependencies.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    grompp_app : callable
        Result of get_grompp_app().
    mdrun_app : callable
        Result of get_mdrun_app().

    Returns
    -------
    AppFuture
        Future that resolves when production stage completes.

    """
    em_fut = run_em_stage(sim_dir, grompp_app, mdrun_app)
    nvt_fut = run_nvt_stage(sim_dir, em_fut, grompp_app, mdrun_app)
    npt_fut = run_npt_stage(sim_dir, nvt_fut, grompp_app, mdrun_app)
    prod_fut = run_production_stage(sim_dir, npt_fut, grompp_app, mdrun_app)

    return prod_fut
