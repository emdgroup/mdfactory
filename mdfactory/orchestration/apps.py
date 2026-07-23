# ABOUTME: Parsl application definitions for build orchestration
# ABOUTME: Wraps run_build_from_dict as a @python_app with runtime decoration
"""Parsl application definitions for build orchestration."""


def _get_gromacs_detect_script() -> str:
    """Return bash script for GROMACS binary auto-detection.

    Returns
    -------
    str
        Bash script that sets GMX_BIN variable to gmx or gmx_mpi.

    Notes
    -----
    Prefers gmx (thread-MPI) for local execution, falls back to gmx_mpi.
    This avoids PMI/srun issues when using LocalProvider with MPI builds.
    """
    return """
# Auto-detect GROMACS binary
# Prefer gmx (thread-MPI) for local execution, fallback to gmx_mpi for cluster
if command -v gmx &> /dev/null; then
    GMX_BIN="gmx"
elif command -v gmx_mpi &> /dev/null; then
    GMX_BIN="gmx_mpi"
else
    echo "ERROR: Neither gmx nor gmx_mpi found in PATH" >&2
    exit 1
fi
"""


def _require_bash_app():
    """Import and return Parsl's ``bash_app`` decorator.

    Factored out of :func:`get_grompp_app` and :func:`get_mdrun_app` to avoid
    repeating the same try/except import guard in every factory function.

    Returns
    -------
    callable
        The ``parsl.bash_app`` decorator.

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
    return bash_app


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


def _build_grompp_script(
    mdp_file: str,
    gro_file: str,
    top_file: str,
    tpr_file: str,
    work_dir: str,
    ref_file: str = "",
    cpt_file: str = "",
    maxwarn: int = 1,
) -> str:
    """Build the bash script for ``gmx grompp``.

    Pure function — no Parsl dependency.  Fully testable without a
    DataFlowKernel by calling it directly with test arguments.

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

        {_get_gromacs_detect_script()}

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


def _build_mdrun_script(
    deffnm: str,
    work_dir: str,
    restart_from_cpt: str = "",
    ntasks: int = 0,
    disable_gpu: bool = False,
    pme_gpu: bool = True,
    gro_out: str = "",
    traj_files: "tuple[str, ...]" = (),
) -> str:
    """Build the bash script for ``gmx mdrun``.

    Pure function — no Parsl dependency.  Fully testable without a
    DataFlowKernel by calling it directly with test arguments.

    Output validation is spec-driven via ``gro_out`` / ``traj_files``:
    no hardcoded ``deffnm`` string comparisons are needed.

    Parameters
    ----------
    deffnm : str
        Default filename prefix (min, nvt, npt, prod).
    work_dir : str
        Absolute path to simulation directory.
    restart_from_cpt : str, optional
        Path to checkpoint file (.cpt) to resume from. When non-empty,
        ``-cpi <file> -append`` flags are added.
    ntasks : int, optional
        Explicit thread count override.  When > 0 used directly instead
        of reading ``$SLURM_CPUS_PER_TASK``.
    disable_gpu : bool, optional
        When ``True``, force CPU-only execution.
    pme_gpu : bool, optional
        When ``True`` and a GPU is available, run PME on GPU (``-pme gpu``).
        Set to ``False`` for stages whose integrator is non-dynamical (EM),
        which GROMACS rejects with PME-GPU.
    gro_out : str, optional
        Expected output ``.gro`` file to verify after mdrun.  Empty string
        means no ``.gro`` check (e.g. Production stage writes trajectory).
    traj_files : tuple of str, optional
        Trajectory output files to verify — at least one must exist.
        Non-empty only for Production: ``("prod.xtc", "prod.trr")``.

    Returns
    -------
    str
        Bash script to execute.

    """
    # Build checkpoint restart flags once; inject literally into the script.
    cpt_flags = f"-cpi {restart_from_cpt} -append " if restart_from_cpt else ""
    cpt_msg = f" (resuming from {restart_from_cpt})" if restart_from_cpt else ""

    # Thread-count: explicit override wins over SLURM/OMP auto-detection.
    nthr_line = (
        f"NTHR={ntasks}"
        if ntasks > 0
        else "NTHR=${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-$(nproc)}}"
    )

    # GPU guard: short-circuit with literal "false" when caller requests CPU-only.
    gpu_condition = (
        "false"
        if disable_gpu
        else '[ -n "${CUDA_VISIBLE_DEVICES}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]'
    )

    # PME flag: spec-driven — EM uses "-pme cpu" (steep integrator), others "-pme gpu".
    pme_flag = "-pme gpu" if pme_gpu else "-pme cpu"

    # Output-verification block: spec-driven, no hardcoded deffnm comparisons.
    if traj_files:
        traj_conds = " && ".join(f'[ ! -f "{f}" ]' for f in traj_files)
        traj_list = " or ".join(traj_files)
        output_check = f"""
        # Verify trajectory output (accept any of the expected formats)
        if {traj_conds}; then
            echo "ERROR: mdrun completed but none of ({traj_list}) was created" >&2
            exit 1
        fi"""
    elif gro_out:
        output_check = f"""
        # Verify structure output
        if [ ! -f "{gro_out}" ]; then
            echo "ERROR: mdrun completed but {gro_out} was not created" >&2
            exit 1
        fi"""
    else:
        output_check = ""

    return f"""
        set -euo pipefail  # Exit on error, undefined vars, pipe failures

        cd {work_dir}

        {_get_gromacs_detect_script()}

        # Thread count: explicit override ({ntasks}) > SLURM > OMP > default(nproc)
        {nthr_line}
        # Re-export so gmx_mpi -ntomp and OMP_NUM_THREADS always agree.
        # Parsl's SlurmProvider sets OMP_NUM_THREADS=<cores_per_node> in the
        # worker environment; if a stage override uses fewer threads the two
        # values conflict and GROMACS aborts with a fatal error.
        export OMP_NUM_THREADS=$NTHR

        # Detect build type: gmx_mpi is a pure MPI build (no thread-MPI support).
        # -nt / -ntmpi are thread-MPI flags; MPI builds use -ntomp for OpenMP threads
        # and rely on the MPI launcher (mpirun/srun) for rank count.
        if [ "$GMX_BIN" = "gmx_mpi" ]; then
            IS_MPI_BUILD=true
        else
            IS_MPI_BUILD=false
        fi

        echo "GROMACS mdrun: Starting {deffnm} with $NTHR threads (using $GMX_BIN){cpt_msg}" >&2

        # Auto-detect GPU availability from CUDA_VISIBLE_DEVICES
        if {gpu_condition}; then
            # GPU mode: non-bonded on GPU; PME on GPU or CPU depending on integrator.
            GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
            echo "GROMACS mdrun: Running on GPU $GPU_ID with $NTHR OpenMP threads ({pme_flag})" >&2

            if $IS_MPI_BUILD; then
                # MPI build: single-rank execution (no MPI launcher invoked).
                # -ntomp sets OpenMP threads; add srun/mpirun here for multi-rank.
                if ! $GMX_BIN mdrun -deffnm {deffnm} {cpt_flags}-ntomp $NTHR -gpu_id $GPU_ID -nb gpu {pme_flag}; then
                    echo "ERROR: mdrun failed for {deffnm} (GPU/MPI mode)" >&2
                    echo "Check md.log for details" >&2
                    exit 1
                fi
            else
                # Thread-MPI build: pin to 1 MPI rank, $NTHR OpenMP threads
                if ! $GMX_BIN mdrun -deffnm {deffnm} {cpt_flags}-ntmpi 1 -ntomp $NTHR -gpu_id $GPU_ID -nb gpu {pme_flag}; then
                    echo "ERROR: mdrun failed for {deffnm} (GPU/thread-MPI mode)" >&2
                    echo "Check md.log for details" >&2
                    exit 1
                fi
            fi
        else
            # CPU-only mode
            echo "GROMACS mdrun: Running on CPU with $NTHR threads" >&2

            if $IS_MPI_BUILD; then
                # MPI build: single-rank execution; -ntomp sets OpenMP threads per rank
                if ! $GMX_BIN mdrun -deffnm {deffnm} {cpt_flags}-ntomp $NTHR; then
                    echo "ERROR: mdrun failed for {deffnm} (CPU/MPI mode)" >&2
                    echo "Check md.log for details" >&2
                    exit 1
                fi
            else
                # Thread-MPI build: -nt sets total threads
                if ! $GMX_BIN mdrun -deffnm {deffnm} {cpt_flags}-nt $NTHR; then
                    echo "ERROR: mdrun failed for {deffnm} (CPU/thread-MPI mode)" >&2
                    echo "Check md.log for details" >&2
                    exit 1
                fi
            fi
        fi
        {output_check}
        echo "GROMACS mdrun: SUCCESS - {deffnm} completed" >&2
        """


def get_grompp_app():
    """Create and return the Parsl bash_app for GROMACS preprocessing.

    The ``@bash_app`` decorator is applied here (not at module level)
    because it requires an active Parsl DataFlowKernel.

    The underlying bash script is built by :func:`_build_grompp_script`, a
    pure function that can be tested independently of Parsl.

    Returns
    -------
    callable
        A Parsl ``@bash_app`` wrapping gmx grompp.

    Raises
    ------
    ImportError
        If parsl is not installed.

    """
    bash_app = _require_bash_app()

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
        return _build_grompp_script(mdp_file, gro_file, top_file, tpr_file, work_dir, ref_file, cpt_file, maxwarn)

    return run_grompp


def get_mdrun_app():
    """Create and return the Parsl bash_app for GROMACS simulation.

    The ``@bash_app`` decorator is applied here (not at module level)
    because it requires an active Parsl DataFlowKernel.

    The underlying bash script is built by :func:`_build_mdrun_script`, a
    pure function that can be tested independently of Parsl.  Output
    validation is spec-driven via ``gro_out`` / ``traj_files`` — no
    hardcoded ``deffnm`` string comparisons in the generated bash.

    Returns
    -------
    callable
        A Parsl ``@bash_app`` wrapping gmx mdrun.

    Raises
    ------
    ImportError
        If parsl is not installed.

    """
    bash_app = _require_bash_app()

    @bash_app
    def run_mdrun(
        deffnm: str,
        work_dir: str,
        restart_from_cpt: str = "",
        ntasks: int = 0,
        disable_gpu: bool = False,
        pme_gpu: bool = True,
        gro_out: str = "",
        traj_files: "tuple[str, ...]" = (),
        stdout=None,
        stderr=None,
        inputs=None,
    ):
        """Run GROMACS simulation with auto-detected resources.

        Thread count and GPU usage are auto-detected from SLURM environment
        variables, with sensible fallbacks for local execution.  Explicit
        ``ntasks`` / ``disable_gpu`` / ``pme_gpu`` arguments override the
        auto-detection, enabling per-stage resource configuration.

        Output validation is spec-driven: pass ``gro_out`` for stages that
        write a ``.gro`` file, or ``traj_files`` for Production (which writes
        ``.xtc``/``.trr``).

        Resource detection priority:
        - Thread count: ``ntasks`` arg (if >0) > SLURM_CPUS_PER_TASK >
          OMP_NUM_THREADS > default(nproc)
        - GPU: disabled when ``disable_gpu=True`` or CUDA_VISIBLE_DEVICES unset
        - PME-on-GPU: only when ``pme_gpu=True`` AND GPU is available.
          Must be ``False`` for non-dynamical integrators (e.g. EM/steep).

        Parameters
        ----------
        deffnm : str
            Default filename prefix (min, nvt, npt, prod).
        work_dir : str
            Absolute path to simulation directory.
        restart_from_cpt : str, optional
            Path to checkpoint file (.cpt) to resume from.
        ntasks : int, optional
            Explicit thread count override.  When > 0, overrides auto-detect.
        disable_gpu : bool, optional
            Force CPU-only execution.
        pme_gpu : bool, optional
            Run PME on GPU when a GPU is available. Set to ``False`` for EM.
        gro_out : str, optional
            Expected output ``.gro`` file to verify after mdrun.
        traj_files : tuple of str, optional
            Trajectory output files to verify (at least one must exist).
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
        return _build_mdrun_script(
            deffnm, work_dir, restart_from_cpt, ntasks, disable_gpu, pme_gpu, gro_out, traj_files
        )

    return run_mdrun
