# ABOUTME: Parsl application definitions for build orchestration
# ABOUTME: Wraps run_build_from_dict as a @python_app with runtime decoration
"""Parsl application definitions for build orchestration."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Layer 1: Single-responsibility decision resolvers
# ---------------------------------------------------------------------------


def _resolve_thread_count_expr(ntasks: int) -> str:
    """Return the bash expression used to assign NTHR.

    Parameters
    ----------
    ntasks : int
        Explicit thread count.  When > 0, returns a literal decimal string
        so that no shell expansion is needed.  When 0, returns a three-level
        fallback expression: SLURM_CPUS_PER_TASK > OMP_NUM_THREADS > nproc.

    Returns
    -------
    str
        Literal integer string or shell parameter expansion.

    Notes
    -----
    The three-level fallback
    ``${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-$(nproc)}}`` is load-bearing.
    Parsl's SlurmProvider injects OMP_NUM_THREADS=<cores_per_node> into worker
    environments; the two-level form ``${SLURM_CPUS_PER_TASK:-$(nproc)}``
    would silently use *all* cores on the node when SLURM_CPUS_PER_TASK is
    absent, causing a thread-count mismatch that GROMACS aborts on.

    """
    if ntasks > 0:
        return str(ntasks)
    return "${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-$(nproc)}}"


def _resolve_is_mpi(gmx_binary: str) -> "bool | None":
    """Return whether the GROMACS binary is an MPI build.

    Parameters
    ----------
    gmx_binary : str
        One of ``"gmx"``, ``"gmx_mpi"``, or ``"auto"``.

    Returns
    -------
    bool or None
        ``True`` for ``gmx_mpi``, ``False`` for ``gmx``, ``None`` for auto
        (binary unknown until runtime).

    """
    if gmx_binary == "gmx_mpi":
        return True
    if gmx_binary == "gmx":
        return False
    return None


def _resolve_thread_flags(is_mpi: "bool | None", has_gpu: bool) -> str:
    """Return the thread-count flags for the mdrun invocation.

    Parameters
    ----------
    is_mpi : bool or None
        ``True`` for MPI build, ``False`` for thread-MPI build, ``None`` when
        the binary is unknown (auto mode).
    has_gpu : bool
        Whether GPU execution is active.

    Returns
    -------
    str
        Thread flag string to embed literally in the mdrun command line.

    """
    if is_mpi is None:
        return "$MDRUN_THREAD_FLAGS"
    if is_mpi:
        return "-ntomp $NTHR"
    if has_gpu:
        return "-ntmpi 1 -ntomp $NTHR"
    return "-nt $NTHR"


def _resolve_gpu_flags(has_gpu: bool, pme_gpu: bool) -> str:
    """Return the GPU offload flags for the mdrun invocation.

    Parameters
    ----------
    has_gpu : bool
        Whether GPU execution is active.
    pme_gpu : bool
        Whether PME should run on GPU.  Must be ``False`` for non-dynamical
        integrators (e.g. EM/steep) which GROMACS rejects with PME-GPU.

    Returns
    -------
    str
        GPU flag string, or ``""`` for CPU-only runs.

    """
    if not has_gpu:
        return ""
    pme_flag = "-pme gpu" if pme_gpu else "-pme cpu"
    return f"-nb gpu {pme_flag} -gpu_id $GPU_ID"


def _resolve_restart_flags(restart_from_cpt: str) -> str:
    """Return checkpoint restart flags for the mdrun invocation.

    Parameters
    ----------
    restart_from_cpt : str
        Path to checkpoint file, or ``""`` for a fresh run.

    Returns
    -------
    str
        ``"-cpi <file> -append"`` when a checkpoint is given, else ``""``.

    """
    if not restart_from_cpt:
        return ""
    return f"-cpi {restart_from_cpt} -append"


def _resolve_binary_token(gmx_binary: str) -> str:
    """Return the executable token for use in the mdrun command line.

    Parameters
    ----------
    gmx_binary : str
        One of ``"gmx"``, ``"gmx_mpi"``, or ``"auto"``.

    Returns
    -------
    str
        Literal executable name (``"gmx"`` / ``"gmx_mpi"``) or the shell
        variable reference ``"$GMX_BIN"`` for auto mode.

    """
    if gmx_binary == "gmx":
        return "gmx"
    if gmx_binary == "gmx_mpi":
        return "gmx_mpi"
    return "$GMX_BIN"


# ---------------------------------------------------------------------------
# Layer 2: Command assembler
# ---------------------------------------------------------------------------


def _assemble_mdrun_command(
    binary: str,
    deffnm: str,
    restart_flags: str,
    thread_flags: str,
    gpu_flags: str,
) -> str:
    """Compose the single concrete mdrun command line.

    All inputs are already-resolved strings (literals or shell variable
    references). Joins non-empty parts with single spaces; no branching.

    Parameters
    ----------
    binary : str
        Executable token: ``"gmx"``, ``"gmx_mpi"``, or ``"$GMX_BIN"``.
    deffnm : str
        Default filename prefix (min, nvt, npt, prod).
    restart_flags : str
        Checkpoint flags (``"-cpi <file> -append"`` or ``""``).
    thread_flags : str
        Thread-count flags (``"-ntomp $NTHR"``, ``"-nt $NTHR"``, etc.).
    gpu_flags : str
        GPU offload flags (``"-nb gpu -pme gpu -gpu_id $GPU_ID"`` or ``""``).

    Returns
    -------
    str
        The complete mdrun command as a single line.

    Examples
    --------
    >>> _assemble_mdrun_command("gmx", "min", "", "-nt $NTHR", "")
    'gmx mdrun -deffnm min -nt $NTHR'
    >>> _assemble_mdrun_command("gmx_mpi", "prod", "-cpi prod.cpt -append", "-ntomp $NTHR", "")
    'gmx_mpi mdrun -deffnm prod -cpi prod.cpt -append -ntomp $NTHR'

    """
    parts = [binary, "mdrun", f"-deffnm {deffnm}"]
    for flag in (restart_flags, thread_flags, gpu_flags):
        if flag:
            parts.append(flag)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Layer 3: Script section builders
# ---------------------------------------------------------------------------


def _build_env_preamble(work_dir: str, nthr_expr: str, has_gpu: bool) -> str:
    """Emit the environment-setup lines for the bash script.

    No conditional branching — all lines are unconditional assignments.

    Parameters
    ----------
    work_dir : str
        Absolute path to simulation directory (``cd`` target).
    nthr_expr : str
        Expression to assign to NTHR (from :func:`_resolve_thread_count_expr`).
    has_gpu : bool
        When ``True``, include the ``GPU_ID`` assignment.

    Returns
    -------
    str
        Multi-line preamble string with no leading or trailing newline.

    Notes
    -----
    Emit order:

    1. ``set -euo pipefail``
    2. ``cd <work_dir>``
    3. ``NTHR=<nthr_expr>``
    4. ``GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}`` — only when ``has_gpu=True``
    5. ``export OMP_NUM_THREADS=$NTHR``

    The GPU_ID assignment precedes the export so the export line is always the
    final line of the preamble section.

    """
    lines = [
        "set -euo pipefail",
        f"cd {work_dir}",
        f"NTHR={nthr_expr}",
    ]
    if has_gpu:
        # Use :- instead of %%,* so CUDA_VISIBLE_DEVICES unset → empty string
        # rather than aborting under set -u on CPU-only hosts.
        lines.append("GPU_ID=${CUDA_VISIBLE_DEVICES:-}")
    lines.append("export OMP_NUM_THREADS=$NTHR")
    return "\n".join(lines)


def _build_gmx_detect_block(gmx_extra: str = "", gmx_mpi_extra: str = "") -> str:
    """Emit the core GROMACS binary detection if/elif/else/fi block.

    Assigns ``$GMX_BIN`` to ``gmx`` or ``gmx_mpi`` depending on what is
    found in ``PATH``, and exits with an error if neither is available.
    Optional extra shell assignments (e.g. setting ``$MDRUN_THREAD_FLAGS``)
    are appended with a semicolon to the respective branch lines.

    Parameters
    ----------
    gmx_extra : str, optional
        Additional assignment appended to the ``gmx`` branch, e.g.
        ``'MDRUN_THREAD_FLAGS="-nt $NTHR"'``.  Empty string means no extra.
    gmx_mpi_extra : str, optional
        Additional assignment appended to the ``gmx_mpi`` branch.

    Returns
    -------
    str
        Multi-line if/elif/else/fi block with no leading or trailing newline.

    Notes
    -----
    This is the single source of truth for the detection skeleton shared by
    :func:`_build_binary_detection_preamble` (mdrun) and
    :func:`_get_gromacs_detect_script` (grompp).

    """
    gmx_line = f"    GMX_BIN=gmx{'; ' + gmx_extra if gmx_extra else ''}"
    gmx_mpi_line = f"    GMX_BIN=gmx_mpi{'; ' + gmx_mpi_extra if gmx_mpi_extra else ''}"
    return (
        "if command -v gmx &>/dev/null; then\n"
        f"{gmx_line}\n"
        "elif command -v gmx_mpi &>/dev/null; then\n"
        f"{gmx_mpi_line}\n"
        "else\n"
        '    echo "ERROR: Neither gmx nor gmx_mpi found in PATH" >&2; exit 1\n'
        "fi"
    )


def _build_binary_detection_preamble(has_gpu: bool) -> str:
    """Emit the GROMACS binary detection block for auto mode.

    This is the only permitted runtime branching in the mdrun script.  It
    assigns both ``$GMX_BIN`` and ``$MDRUN_THREAD_FLAGS`` so that the mdrun
    command line can reference both and remain a single unconditional line.

    Parameters
    ----------
    has_gpu : bool
        When ``True``, the thread-MPI branch uses ``-ntmpi 1 -ntomp $NTHR``
        (GPU requires thread-MPI pinning); when ``False``, uses ``-nt $NTHR``
        (let GROMACS allocate threads freely on CPU).

    Returns
    -------
    str
        Multi-line if/elif/else/fi block with no leading or trailing newline.

    Notes
    -----
    The block is absent when ``gmx_binary`` is set to ``"gmx"`` or
    ``"gmx_mpi"`` — only auto mode needs runtime binary detection.
    Delegates to :func:`_build_gmx_detect_block` for the shared skeleton.

    """
    tmpi_flags = '"-ntmpi 1 -ntomp $NTHR"' if has_gpu else '"-nt $NTHR"'
    return _build_gmx_detect_block(
        gmx_extra=f"MDRUN_THREAD_FLAGS={tmpi_flags}",
        gmx_mpi_extra='MDRUN_THREAD_FLAGS="-ntomp $NTHR"',
    )


def _build_output_check(gro_out: str, traj_files: "tuple[str, ...]") -> str:
    """Emit output existence verification for the bash script.

    Parameters
    ----------
    gro_out : str
        Expected output ``.gro`` file, or ``""`` when not applicable.
    traj_files : tuple of str
        Expected trajectory files (accept any of them), or ``()`` when not
        applicable.  Non-empty for the Production stage only.

    Returns
    -------
    str
        An if-block that exits with code 1 when outputs are missing, or ``""``
        when no output verification is required.

    """
    if traj_files:
        traj_conds = " && ".join(f'[ ! -f "{f}" ]' for f in traj_files)
        traj_list = " or ".join(traj_files)
        return (
            f"if {traj_conds}; then\n"
            f'    echo "ERROR: mdrun completed but none of ({traj_list}) was created" >&2\n'
            f"    exit 1\n"
            f"fi"
        )
    if gro_out:
        return (
            f'if [ ! -f "{gro_out}" ]; then\n'
            f'    echo "ERROR: mdrun completed but {gro_out} was not created" >&2\n'
            f"    exit 1\n"
            f"fi"
        )
    return ""


def _get_gromacs_detect_script() -> str:
    """Return bash snippet for GROMACS binary auto-detection (grompp use).

    Wraps :func:`_build_gmx_detect_block` with a comment header and
    surrounding newlines so it embeds cleanly inside the grompp f-string.

    Returns
    -------
    str
        Bash block that sets ``$GMX_BIN`` to ``gmx`` or ``gmx_mpi``,
        with a leading blank line, comment, and trailing newline.

    Notes
    -----
    Prefers ``gmx`` (thread-MPI) for local execution; falls back to
    ``gmx_mpi``.  This avoids PMI/srun issues when using LocalProvider
    with MPI builds.  Only ``$GMX_BIN`` is set here — no thread-flag
    assignment is needed for grompp.

    """
    block = _build_gmx_detect_block()
    return (
        f"\n# Auto-detect GROMACS binary (prefer gmx thread-MPI, fall back to gmx_mpi)\n{block}\n"
    )


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
    gmx_binary: str = "auto",
) -> str:
    """Build the bash script for ``gmx mdrun``.

    Pure function — no Parsl dependency.  Fully testable without a
    DataFlowKernel by calling it directly with test arguments.

    All conditional logic is resolved in Python before any bash is generated.
    The resulting script contains no if-clauses in the command section; only
    variable assignments and a single unconditional ``mdrun`` command.  The
    only permitted runtime branch is the binary-detection preamble emitted
    when ``gmx_binary="auto"``.

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
        When ``True`` and GPU mode is active, run PME on GPU (``-pme gpu``).
        Set to ``False`` for stages whose integrator is non-dynamical (EM),
        which GROMACS rejects with PME-GPU.
    gro_out : str, optional
        Expected output ``.gro`` file to verify after mdrun.  Empty string
        means no ``.gro`` check (e.g. Production stage writes trajectory).
    traj_files : tuple of str, optional
        Trajectory output files to verify — at least one must exist.
        Non-empty only for Production: ``("prod.xtc", "prod.trr")``.
    gmx_binary : str, optional
        GROMACS binary selection: ``"gmx"`` (thread-MPI build),
        ``"gmx_mpi"`` (pure MPI build), or ``"auto"`` (detect at runtime).
        Defaults to ``"auto"`` for backward compatibility.

    Returns
    -------
    str
        Bash script to execute.

    """
    # Layer 1: resolve all decisions in Python before any bash is generated.
    has_gpu = not disable_gpu
    is_mpi = _resolve_is_mpi(gmx_binary)
    nthr_expr = _resolve_thread_count_expr(ntasks)
    thread_flags = _resolve_thread_flags(is_mpi, has_gpu)
    gpu_flags = _resolve_gpu_flags(has_gpu, pme_gpu)
    restart_flags = _resolve_restart_flags(restart_from_cpt)
    binary = _resolve_binary_token(gmx_binary)
    cpt_msg = f" (resuming from {restart_from_cpt})" if restart_from_cpt else ""

    # Layer 2: assemble the single unconditional mdrun command.
    command = _assemble_mdrun_command(binary, deffnm, restart_flags, thread_flags, gpu_flags)

    # Layer 3 + 4: assemble script sections; join with newlines.
    sections = [_build_env_preamble(work_dir, nthr_expr, has_gpu)]
    if gmx_binary == "auto":
        sections.append(_build_binary_detection_preamble(has_gpu))
    sections.append(
        f'echo "GROMACS mdrun: Starting {deffnm} with $NTHR threads (using {binary}){cpt_msg}" >&2'
    )
    sections.append(command)
    output_check = _build_output_check(gro_out, traj_files)
    if output_check:
        sections.append(output_check)
    sections.append(f'echo "GROMACS mdrun: SUCCESS - {deffnm} completed" >&2')
    return "\n".join(sections)


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
        return _build_grompp_script(
            mdp_file, gro_file, top_file, tpr_file, work_dir, ref_file, cpt_file, maxwarn
        )

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
        gmx_binary: str = "auto",
        stdout=None,
        stderr=None,
        inputs=None,
    ):
        """Run GROMACS simulation with resolved resources.

        All conditional logic (binary selection, GPU mode, thread count, restart
        flags) is resolved in Python by :func:`_build_mdrun_script` before any
        bash is generated.  The resulting bash script contains a single
        unconditional ``mdrun`` command.

        Output validation is spec-driven: pass ``gro_out`` for stages that
        write a ``.gro`` file, or ``traj_files`` for Production (which writes
        ``.xtc``/``.trr``).

        Resource resolution priority:
        - Thread count: ``ntasks`` arg (if >0) > SLURM_CPUS_PER_TASK >
          OMP_NUM_THREADS > nproc
        - GPU: active when ``disable_gpu=False`` (GPU mode decided at submission
          time, not at runtime).
        - Binary: ``gmx_binary`` selects the executable; ``"auto"`` enables
          runtime detection via a single isolated if-block.
        - PME-on-GPU: only when ``pme_gpu=True`` AND GPU mode is active.
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
            Run PME on GPU when GPU mode is active. Set to ``False`` for EM.
        gro_out : str, optional
            Expected output ``.gro`` file to verify after mdrun.
        traj_files : tuple of str, optional
            Trajectory output files to verify (at least one must exist).
        gmx_binary : str, optional
            GROMACS binary: ``"gmx"``, ``"gmx_mpi"``, or ``"auto"``.
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
            deffnm,
            work_dir,
            restart_from_cpt,
            ntasks,
            disable_gpu,
            pme_gpu,
            gro_out,
            traj_files,
            gmx_binary,
        )

    return run_mdrun
