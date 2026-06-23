# ABOUTME: Interactive SLURM configuration wizard using questionary.
# ABOUTME: Leverages cluster autodiscovery to guide users through executor setup.
"""Interactive SLURM configuration wizard.

Provides a terminal-based wizard that queries the local SLURM cluster
(via :func:`~mdfactory.performance.cluster.discover_cluster`) and guides
the user through selecting accounts, partitions, and resource limits.
The result is a :class:`~mdfactory.orchestration.config.SlurmExecutorConfig`
that can be saved as YAML and used with Parsl-based workflows.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from mdfactory.orchestration.config import SlurmExecutorConfig
from mdfactory.performance.cluster import ClusterInfo, Partition, discover_cluster

console = Console()


def _import_questionary():
    """Import questionary with a clear error message if not installed.

    ``questionary`` ships in the ``[parsl]`` optional extra. Importing it
    lazily (rather than at module level) means ``import mdfactory.orchestration``
    succeeds for users who never touch the TUI, matching the lazy-import
    pattern used for ``parsl`` and ``rich``.

    Returns
    -------
    module
        The questionary module.

    Raises
    ------
    ImportError
        If questionary is not installed.

    """
    try:
        import questionary
    except ImportError as exc:
        raise ImportError(
            "questionary is required for the SLURM TUI. "
            "Install with: pip install 'mdfactory[parsl]'"
        ) from exc
    return questionary


def _default_worker_init() -> str:
    """Auto-detect the pixi environment and return an appropriate worker_init.

    On HPC clusters with a shared filesystem, workers need the same pixi
    environment as the submitting process.  This function locates the
    project root via ``mdfactory.__file__`` and returns a pixi shell-hook
    command if the environment exists.

    Returns
    -------
    str
        A shell command to activate the pixi environment, or ``""`` if
        no pixi environment is detected.
    """
    try:
        import mdfactory as _mdf

        project_root = Path(_mdf.__file__).parent.parent
        pixi_env = project_root / ".pixi" / "envs" / "default"
        if pixi_env.exists():
            return f'eval "$(pixi shell-hook --manifest-path {project_root} -e default)"'
    except Exception:
        pass
    return ""


class UserCancelledError(Exception):
    """Raised when the user cancels an interactive prompt."""


def _require(value: str | None, label: str) -> str:
    """Return *value* or raise if the user cancelled the prompt.

    Parameters
    ----------
    value : str or None
        Return value from a ``questionary`` ``.ask()`` call.
    label : str
        Human-readable label used in the error message.

    Returns
    -------
    str
        The validated, non-None value.

    Raises
    ------
    UserCancelledError
        If *value* is None (user pressed Ctrl-C / Ctrl-D).
    """
    if value is None:
        raise UserCancelledError(f"Prompt cancelled at: {label}")
    return value


# ---------------------------------------------------------------------------
# Partition / node helpers
# ---------------------------------------------------------------------------


def _format_node_type_summary(partition: Partition) -> str:
    """Build a one-line summary of node hardware for a partition label.

    Parameters
    ----------
    partition : Partition
        Partition whose node types to summarise.

    Returns
    -------
    str
        Human-readable summary, e.g. ``"96 cpu, 512 GB, 4×a100"``
    """
    parts: list[str] = []
    for nt in partition.node_types:
        items = [f"{nt.cpus} cpu", f"{nt.memory_mb // 1024} GB"]
        for count, gpu_type in nt.gpu_specs:
            gpu_label = f"{count}×{gpu_type}" if gpu_type else f"{count}×gpu"
            items.append(gpu_label)
        parts.append(", ".join(items))
    return " | ".join(parts)


def _partition_has_gpus(partition: Partition) -> bool:
    """Check whether any node type in *partition* has GPUs."""
    return any(len(nt.gpu_specs) > 0 for nt in partition.node_types)


def _suggest_gres(partition: Partition) -> str:
    """Suggest a ``--gres`` string from the first GPU-equipped node type.

    Parameters
    ----------
    partition : Partition
        Selected partition.

    Returns
    -------
    str
        Suggested GRES string, e.g. ``"gpu:a100:1"``, or empty string.
    """
    for nt in partition.node_types:
        for count, gpu_type in nt.gpu_specs:
            if gpu_type:
                return f"gpu:{gpu_type}:1"
            return "gpu:1"
    return ""


def _suggest_mem(partition: Partition) -> str:
    """Suggest a ``--mem`` value from the first node type.

    Parameters
    ----------
    partition : Partition
        Selected partition.

    Returns
    -------
    str
        Suggested memory string in GB, e.g. ``"64G"``.
    """
    if partition.node_types:
        mem_gb = partition.node_types[0].memory_mb // 1024
        return f"{mem_gb}G"
    return "16G"


def _default_walltime(partition: Partition) -> str:
    """Pick a sensible default walltime from the partition.

    Parameters
    ----------
    partition : Partition
        Selected partition.

    Returns
    -------
    str
        Walltime string (falls back to ``"2:00:00"``).
    """
    mt = partition.max_time
    if mt and mt not in ("unknown", "infinite"):
        return mt
    return "2:00:00"


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _select_with_custom(
    message: str,
    choices: list[str],
    default: str = "",
) -> str:
    """Present a selection list with a "Custom…" escape hatch.

    Parameters
    ----------
    message : str
        Prompt message shown to the user.
    choices : list[str]
        Pre-defined choices (e.g. ``["1h", "2h", "4h"]``).
    default : str
        Pre-selected value in the list.

    Returns
    -------
    str
        The selected or custom-entered value.

    Raises
    ------
    UserCancelledError
        If the user cancels.
    """
    questionary = _import_questionary()
    _CUSTOM = "Custom…"
    all_choices = [*choices, _CUSTOM]
    selected = _require(
        questionary.select(message, choices=all_choices, default=default).ask(),
        message,
    )
    if selected == _CUSTOM:
        return _require(
            questionary.text(f"{message} (enter value):").ask(),
            message,
        )
    return selected


# ---------------------------------------------------------------------------
# Interactive prompts — cluster-assisted
# ---------------------------------------------------------------------------


def _configure_with_cluster(cluster: ClusterInfo) -> SlurmExecutorConfig:
    """Run the interactive wizard using autodiscovered cluster info.

    Parameters
    ----------
    cluster : ClusterInfo
        Cluster information from ``discover_cluster()``.

    Returns
    -------
    SlurmExecutorConfig
        Fully populated SLURM executor config.

    Raises
    ------
    UserCancelledError
        If the user cancels any prompt.
    """
    questionary = _import_questionary()
    # --- Account ---
    if cluster.accounts:
        account = _require(
            questionary.select(
                "SLURM account:",
                choices=cluster.accounts,
                default=cluster.default_account or cluster.accounts[0],
            ).ask(),
            "account",
        )
    else:
        account = _require(
            questionary.text("SLURM account (no accounts discovered):").ask(),
            "account",
        )

    # --- Partition ---
    up_partitions = [p for p in cluster.partitions if p.state == "up"]
    if not up_partitions:
        console.print("⚠ No partitions in 'up' state — showing all partitions.")
        up_partitions = list(cluster.partitions)

    if not up_partitions:
        raise UserCancelledError("No partitions available on the cluster.")

    partition_choices = [
        questionary.Choice(
            title=f"{p.name}  ({p.total_nodes} nodes — {_format_node_type_summary(p)})",
            value=p.name,
        )
        for p in up_partitions
    ]
    default_partition = next((p.name for p in up_partitions if p.is_default), up_partitions[0].name)

    partition_name = _require(
        questionary.select(
            "SLURM partition:",
            choices=partition_choices,
            default=default_partition,
        ).ask(),
        "partition",
    )

    partition = next(p for p in up_partitions if p.name == partition_name)

    # --- Display node types ---
    console.print(f"\n  Partition '{partition_name}' node types:")
    for nt in partition.node_types:
        gpu_info = ""
        if nt.gpu_specs:
            gpu_parts = [f"{c}×{t}" if t else f"{c}×gpu" for c, t in nt.gpu_specs]
            gpu_info = f", GPUs: {', '.join(gpu_parts)}"
        console.print(
            f"    {nt.count} nodes — {nt.cpus} CPUs, {nt.memory_mb // 1024} GB RAM{gpu_info}"
        )
    console.print()

    # --- Walltime ---
    walltime = _select_with_custom(
        "Walltime (--time):",
        choices=["30m", "1h", "2h", "4h", "8h", "12h", "1d"],
        default="2h",
    )

    # --- CPUs per node ---
    max_cpus = partition.node_types[0].cpus if partition.node_types else 128
    cpu_choices = [str(c) for c in [1, 2, 4, 8, 16, 32, 64, max_cpus] if c <= max_cpus]
    # Deduplicate (max_cpus might equal one of the fixed values)
    cpu_choices = list(dict.fromkeys(cpu_choices))
    cpus_per_node = int(
        _select_with_custom(
            "CPUs per node (--cpus-per-task):",
            choices=cpu_choices,
            default="4" if "4" in cpu_choices else cpu_choices[0],
        )
    )

    # --- GPU ---
    gres: str | None = None
    if _partition_has_gpus(partition):
        gres_default = _suggest_gres(partition)
        gres_input = _require(
            questionary.text(
                "GRES (--gres), leave empty to skip:",
                default=gres_default,
            ).ask(),
            "gres",
        )
        gres = gres_input.strip() or None

    # --- Memory ---
    mem_gb = partition.node_types[0].memory_mb // 1024 if partition.node_types else 128
    mem_choices = [f"{m}G" for m in [10, 20, 50, 100, 200, 500, mem_gb] if m <= mem_gb]
    mem_choices = list(dict.fromkeys(mem_choices))
    mem_selected = _select_with_custom(
        "Memory per node (--mem):",
        choices=mem_choices,
        default="20G" if "20G" in mem_choices else mem_choices[0],
    )
    mem: str | None = mem_selected.strip() or None

    # --- Max blocks ---
    max_blocks = int(
        _select_with_custom(
            "Max simultaneous SLURM jobs (max_blocks):",
            choices=["1", "2", "4", "8", "16", "32"],
            default="4",
        )
    )

    # --- Worker init ---
    default_init = _default_worker_init()
    worker_init = _require(
        questionary.text(
            "Worker init script (activates environment on compute nodes):",
            default=default_init,
        ).ask(),
        "worker_init",
    )

    # --- QOS ---
    qos: str | None = None
    if cluster.qos_policies:
        use_qos = questionary.confirm("Configure QOS?", default=False).ask()
        if use_qos is None:
            raise UserCancelledError("Prompt cancelled at: qos confirm")
        if use_qos:
            qos = _require(
                questionary.select("QOS policy:", choices=cluster.qos_policies).ask(),
                "qos",
            )

    # --- Constraint ---
    constraint_input = _require(
        questionary.text(
            "Node feature constraint (--constraint), leave empty to skip:",
            default="",
        ).ask(),
        "constraint",
    )
    constraint: str | None = constraint_input.strip() or None

    return SlurmExecutorConfig(
        account=account,
        partition=partition_name,
        walltime=walltime,
        cpus_per_node=cpus_per_node,
        gres=gres,
        mem=mem,
        qos=qos,
        constraint=constraint,
        max_blocks=max_blocks,
        worker_init=worker_init,
    )


# ---------------------------------------------------------------------------
# Interactive prompts — manual fallback
# ---------------------------------------------------------------------------


def _configure_manual() -> SlurmExecutorConfig:
    """Run the interactive wizard with manual text entry (no cluster info).

    Returns
    -------
    SlurmExecutorConfig
        Fully populated SLURM executor config.

    Raises
    ------
    UserCancelledError
        If the user cancels any prompt.
    """
    questionary = _import_questionary()
    account = _require(questionary.text("SLURM account:").ask(), "account")
    partition = _require(questionary.text("SLURM partition:", default="gpu").ask(), "partition")
    walltime = _require(questionary.text("Walltime (--time):", default="2:00:00").ask(), "walltime")
    cpus_per_node = int(
        _require(questionary.text("CPUs per node:", default="12").ask(), "cpus_per_node")
    )

    gres_input = _require(
        questionary.text("GRES (--gres), leave empty to skip:", default="").ask(),
        "gres",
    )
    gres: str | None = gres_input.strip() or None

    mem_input = _require(
        questionary.text("Memory per node (--mem), leave empty to skip:", default="").ask(),
        "mem",
    )
    mem: str | None = mem_input.strip() or None

    qos_input = _require(
        questionary.text("QOS (--qos), leave empty to skip:", default="").ask(),
        "qos",
    )
    qos: str | None = qos_input.strip() or None

    max_blocks = int(
        _require(
            questionary.text("Max simultaneous SLURM jobs (max_blocks):", default="4").ask(),
            "max_blocks",
        )
    )

    default_init = _default_worker_init()
    worker_init = _require(
        questionary.text(
            "Worker init script (activates environment on compute nodes):",
            default=default_init,
        ).ask(),
        "worker_init",
    )

    return SlurmExecutorConfig(
        account=account,
        partition=partition,
        walltime=walltime,
        cpus_per_node=cpus_per_node,
        gres=gres,
        mem=mem,
        qos=qos,
        max_blocks=max_blocks,
        worker_init=worker_init,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_slurm_interactive() -> SlurmExecutorConfig:
    """Interactive SLURM configuration wizard.

    Attempts to autodiscover the SLURM cluster. If discovery succeeds,
    the wizard presents select menus populated with real partitions,
    accounts, and hardware specs. Otherwise it falls back to free-text
    prompts.

    Returns
    -------
    SlurmExecutorConfig
        A validated SLURM executor config ready for use with Parsl.

    Raises
    ------
    UserCancelledError
        If the user cancels any prompt (Ctrl-C / Ctrl-D).
    """
    questionary = _import_questionary()
    console.print("Querying SLURM cluster...")
    cluster = discover_cluster()

    if cluster is not None:
        console.print(
            f"✓ Cluster discovered: {len(cluster.partitions)} partitions, "
            f"{len(cluster.accounts)} accounts\n"
        )
        return _configure_with_cluster(cluster)

    console.print("⚠ SLURM not detected (sinfo unavailable). Falling back to manual entry.\n")
    proceed = questionary.confirm("Enter SLURM configuration manually?", default=True).ask()
    if not proceed:
        raise UserCancelledError("User declined manual SLURM configuration.")
    return _configure_manual()


def save_slurm_config_yaml(config: SlurmExecutorConfig, path: Path) -> None:
    """Write a SLURM executor config to a YAML file.

    Parameters
    ----------
    config : SlurmExecutorConfig
        The config to serialise.
    path : Path
        Destination file path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(exclude_none=True)
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    console.print(f"✓ SLURM config written to {path}")


def configure_and_save_slurm() -> SlurmExecutorConfig:
    """Run the interactive wizard and save the result to YAML.

    Combines :func:`configure_slurm_interactive` with a file-save prompt.

    Returns
    -------
    SlurmExecutorConfig
        The config that was saved.

    Raises
    ------
    UserCancelledError
        If the user cancels any prompt.
    """
    questionary = _import_questionary()
    try:
        config = configure_slurm_interactive()
    except UserCancelledError:
        console.print("Configuration cancelled.")
        raise

    save_path = _require(
        questionary.text("Save config to:", default="slurm_executor.yaml").ask(),
        "save path",
    )

    save_slurm_config_yaml(config, Path(save_path))
    return config
