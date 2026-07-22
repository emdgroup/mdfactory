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

    # Step 1: Preprocessing
    grompp_fut = grompp_app(
        mdp_file="em.mdp",
        gro_file="system.pdb",
        top_file="topology.top",
        tpr_file="min.tpr",
        work_dir=work_dir,
        maxwarn=1,
    )

    # Step 2: Simulation (depends on grompp)
    mdrun_fut = mdrun_app(
        deffnm="min",
        work_dir=work_dir,
        nt=12,
        inputs=[grompp_fut],
    )

    return mdrun_fut


def run_nvt_stage(sim_dir: Path, em_future: "AppFuture", grompp_app, mdrun_app) -> "AppFuture":
    """Execute NVT equilibration stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    em_future : AppFuture
        Future from EM stage (dependency).
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

    # Wait for EM to complete before starting
    grompp_fut = grompp_app(
        mdp_file="nvt.mdp",
        gro_file="min.gro",
        ref_file="min.gro",
        top_file="topology.top",
        tpr_file="nvt.tpr",
        work_dir=work_dir,
        maxwarn=1,
        inputs=[em_future],
    )

    mdrun_fut = mdrun_app(
        deffnm="nvt",
        work_dir=work_dir,
        nt=12,
        inputs=[grompp_fut],
    )

    return mdrun_fut


def run_npt_stage(sim_dir: Path, nvt_future: "AppFuture", grompp_app, mdrun_app) -> "AppFuture":
    """Execute NPT equilibration stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    nvt_future : AppFuture
        Future from NVT stage (dependency).
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

    grompp_fut = grompp_app(
        mdp_file="npt.mdp",
        gro_file="nvt.gro",
        ref_file="nvt.gro",
        cpt_file="nvt.cpt",
        top_file="topology.top",
        tpr_file="npt.tpr",
        work_dir=work_dir,
        maxwarn=2,
        inputs=[nvt_future],
    )

    mdrun_fut = mdrun_app(
        deffnm="npt",
        work_dir=work_dir,
        nt=12,
        inputs=[grompp_fut],
    )

    return mdrun_fut


def run_production_stage(
    sim_dir: Path, npt_future: "AppFuture", grompp_app, mdrun_app
) -> "AppFuture":
    """Execute production MD stage.

    Parameters
    ----------
    sim_dir : Path
        Simulation directory.
    npt_future : AppFuture
        Future from NPT stage (dependency).
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

    grompp_fut = grompp_app(
        mdp_file="md.mdp",
        gro_file="npt.gro",
        cpt_file="npt.cpt",
        top_file="topology.top",
        tpr_file="prod.tpr",
        work_dir=work_dir,
        maxwarn=2,
        inputs=[npt_future],
    )

    mdrun_fut = mdrun_app(
        deffnm="prod",
        work_dir=work_dir,
        nt=12,
        inputs=[grompp_fut],
    )

    return mdrun_fut


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
