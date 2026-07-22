# ABOUTME: Parsl application definitions for build orchestration
# ABOUTME: Wraps run_build_from_dict as a @python_app with runtime decoration
"""Parsl application definitions for build orchestration."""


def _build_system_impl(build_input_dict: dict) -> dict:
    """Run a single system build inside a Parsl worker.

    All heavy imports (OpenMM, MDAnalysis, OpenFF) happen inside the
    worker process — only the plain dict crosses the serialization boundary.

    Parameters
    ----------
    build_input_dict : dict
        Serialized BuildInput as a plain dictionary. May contain a special
        ``_build_dir`` key specifying the output directory.

    Returns
    -------
    dict
        Result dictionary with hash, status, and directory.

    """
    import os
    from pathlib import Path

    from mdfactory.models.input import BuildInput
    from mdfactory.workflows import run_build_from_dict

    # Extract and remove internal keys before validation
    input_copy = {k: v for k, v in build_input_dict.items() if not k.startswith("_")}
    model = BuildInput(**input_copy)
    build_dir = Path(build_input_dict.get("_build_dir", model.hash))
    build_dir.mkdir(parents=True, exist_ok=True)
    original_dir = os.getcwd()
    try:
        os.chdir(build_dir)
        run_build_from_dict(model)
    finally:
        os.chdir(original_dir)
    return {"hash": model.hash, "status": "success", "directory": str(build_dir.resolve())}


def get_build_app():
    """Create and return the Parsl python_app for building systems.

    The ``@python_app`` decorator is applied here (not at module level)
    because it requires an active Parsl DataFlowKernel.

    Returns
    -------
    callable
        A Parsl ``@python_app`` wrapping the build implementation.

    Raises
    ------
    ImportError
        If parsl is not installed.

    """
    try:
        from parsl import python_app  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "parsl is required for build orchestration. "
            "Install with `pip install 'mdfactory[parsl]'`."
        ) from exc
    return python_app(_build_system_impl)


def get_grompp_app():
    """Create and return the Parsl bash_app for GROMACS preprocessing.

    The ``@bash_app`` decorator is applied here (not at module level)
    because it requires an active Parsl DataFlowKernel.

    Returns
    -------
    callable
        A Parsl ``@bash_app`` wrapping gmx grompp.

    Raises
    ------
    ImportError
        If parsl is not installed.

    """
    try:
        from parsl import bash_app  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "parsl is required for simulation orchestration. "
            "Install with `pip install 'mdfactory[parsl]'`."
        ) from exc

    @bash_app
    def run_grompp(
        mdp_file: str,
        gro_file: str,
        top_file: str,
        tpr_file: str,
        work_dir: str,
        ref_file: str = "",
        cpt_file: str = "",
        maxwarn: int = 1,
        stdout=None,
        stderr=None,
        inputs=None,
    ):
        """Generate GROMACS .tpr file via grompp.

        Parameters
        ----------
        mdp_file : str
            MDP parameter file (em.mdp, nvt.mdp, npt.mdp, md.mdp).
        gro_file : str
            Input structure file (system.pdb, min.gro, etc).
        top_file : str
            Topology file (topology.top).
        tpr_file : str
            Output TPR file (min.tpr, nvt.tpr, etc).
        work_dir : str
            Absolute path to simulation directory.
        ref_file : str, optional
            Reference structure for position restraints (-r flag).
        cpt_file : str, optional
            Checkpoint file for velocities (-t flag).
        maxwarn : int
            Maximum number of warnings to allow.
        stdout : str, optional
            File path for stdout.
        stderr : str, optional
            File path for stderr.
        inputs : list, optional
            Parsl input dependencies.

        Returns
        -------
        str
            Bash script to execute.

        """
        ref_flag = f"-r {ref_file}" if ref_file else ""
        cpt_flag = f"-t {cpt_file}" if cpt_file else ""

        return f"""
        set -euo pipefail  # Exit on error, undefined vars, pipe failures

        cd {work_dir}

        # Auto-detect GROMACS binary (gmx_mpi for MPI builds, gmx for thread-MPI)
        if command -v gmx_mpi &> /dev/null; then
            GMX_BIN="gmx_mpi"
        elif command -v gmx &> /dev/null; then
            GMX_BIN="gmx"
        else
            echo "ERROR: Neither gmx nor gmx_mpi found in PATH" >&2
            exit 1
        fi

        echo "GROMACS grompp: {mdp_file} -> {tpr_file} (using $GMX_BIN)" >&2

        # Run grompp with explicit error checking
        if ! $GMX_BIN grompp -f {mdp_file} -c {gro_file} -p {top_file} -o {tpr_file} \\
            {ref_flag} {cpt_flag} -maxwarn {maxwarn}; then
            echo "ERROR: grompp failed for {tpr_file}" >&2
            echo "Check that input files ({mdp_file}, {gro_file}, {top_file}) are valid" >&2
            exit 1
        fi

        # Verify TPR was created
        if [ ! -f {tpr_file} ]; then
            echo "ERROR: grompp succeeded but {tpr_file} was not created" >&2
            exit 1
        fi

        echo "GROMACS grompp: SUCCESS - {tpr_file} created" >&2
        """

    return run_grompp


def get_mdrun_app():
    """Create and return the Parsl bash_app for GROMACS simulation.

    The ``@bash_app`` decorator is applied here (not at module level)
    because it requires an active Parsl DataFlowKernel.

    Returns
    -------
    callable
        A Parsl ``@bash_app`` wrapping gmx mdrun.

    Raises
    ------
    ImportError
        If parsl is not installed.

    """
    try:
        from parsl import bash_app  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "parsl is required for simulation orchestration. "
            "Install with `pip install 'mdfactory[parsl]'`."
        ) from exc

    @bash_app
    def run_mdrun(
        deffnm: str,
        work_dir: str,
        stdout=None,
        stderr=None,
        inputs=None,
    ):
        """Run GROMACS simulation with auto-detected resources.

        Thread count and GPU usage are auto-detected from SLURM environment
        variables, with sensible fallbacks for local execution.

        Resource detection priority:
        - Thread count: SLURM_CPUS_PER_TASK > OMP_NUM_THREADS > default(12)
        - GPU: Auto-detected from CUDA_VISIBLE_DEVICES

        Parameters
        ----------
        deffnm : str
            Default filename prefix (min, nvt, npt, prod).
        work_dir : str
            Absolute path to simulation directory.
        stdout : str, optional
            File path for stdout.
        stderr : str, optional
            File path for stderr.
        inputs : list, optional
            Parsl input dependencies.

        Returns
        -------
        str
            Bash script to execute.

        """
        return f"""
        set -euo pipefail  # Exit on error, undefined vars, pipe failures

        cd {work_dir}

        # Auto-detect GROMACS binary (gmx_mpi for MPI builds, gmx for thread-MPI)
        if command -v gmx_mpi &> /dev/null; then
            GMX_BIN="gmx_mpi"
        elif command -v gmx &> /dev/null; then
            GMX_BIN="gmx"
        else
            echo "ERROR: Neither gmx nor gmx_mpi found in PATH" >&2
            exit 1
        fi

        # Auto-detect thread count from SLURM allocation
        # Priority: SLURM_CPUS_PER_TASK > OMP_NUM_THREADS > default(12)
        NTHR=${{SLURM_CPUS_PER_TASK:-${{OMP_NUM_THREADS:-12}}}}

        echo "GROMACS mdrun: Starting {deffnm} with $NTHR threads (using $GMX_BIN)" >&2

        # Auto-detect GPU availability from CUDA_VISIBLE_DEVICES
        if [ -n "${{CUDA_VISIBLE_DEVICES}}" ] && [ "${{CUDA_VISIBLE_DEVICES}}" != "NoDevFiles" ]; then
            # GPU mode: use first GPU with MPI+OpenMP hybrid parallelization
            GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
            echo "GROMACS mdrun: Running on GPU $GPU_ID with $NTHR OpenMP threads" >&2

            if ! $GMX_BIN mdrun -deffnm {deffnm} -ntmpi 1 -ntomp $NTHR -gpu_id $GPU_ID -nb gpu -pme gpu; then
                echo "ERROR: mdrun failed for {deffnm} (GPU mode)" >&2
                echo "Check md.log for details" >&2
                exit 1
            fi
        else
            # CPU-only mode: pure thread-MPI parallelization
            echo "GROMACS mdrun: Running on CPU with $NTHR threads" >&2

            if ! $GMX_BIN mdrun -deffnm {deffnm} -nt $NTHR; then
                echo "ERROR: mdrun failed for {deffnm} (CPU mode)" >&2
                echo "Check md.log for details" >&2
                exit 1
            fi
        fi

        # Verify expected output files were created
        # Different stages produce different outputs
        if [[ "{deffnm}" == "min" ]]; then
            EXPECTED_OUTPUT="min.gro"
        elif [[ "{deffnm}" == "prod" ]]; then
            EXPECTED_OUTPUT="prod.xtc"
        else
            EXPECTED_OUTPUT="{deffnm}.gro"
        fi

        if [ ! -f "$EXPECTED_OUTPUT" ]; then
            echo "ERROR: mdrun completed but $EXPECTED_OUTPUT was not created" >&2
            exit 1
        fi

        echo "GROMACS mdrun: SUCCESS - {deffnm} completed" >&2
        """

    return run_mdrun
