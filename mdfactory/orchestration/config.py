# ABOUTME: Pydantic models for Parsl executor configuration
# ABOUTME: Supports local and SLURM providers with YAML serialization
"""Executor configuration models for Parsl workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import parsl


def _import_parsl():
    """Import parsl with a clear error message if not installed.

    Returns
    -------
    module
        The parsl module.

    Raises
    ------
    ImportError
        If parsl is not installed.

    """
    try:
        import parsl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "parsl is required for build orchestration. "
            "Install with `pip install 'mdfactory[parsl]'`."
        ) from exc
    return parsl


class ExecutorConfig(BaseModel):
    """Base executor configuration for Parsl workflows.

    Parameters
    ----------
    provider : str
        Execution provider type ("local" or "slurm").
    worker_init : str
        Shell commands to run before starting workers (module loads, activation).
    working_directory : Path or None
        Working directory for the executor.
    max_workers_per_node : int
        Maximum number of parallel tasks per compute node.
    max_blocks : int
        Maximum number of execution blocks (for local: process groups).

    """

    provider: Literal["local", "slurm"] = "local"
    worker_init: str = ""
    working_directory: Path | None = None
    max_workers_per_node: int = 1
    max_blocks: int = 1

    def to_parsl_config(self) -> "parsl.Config":
        """Build a Parsl Config with HighThroughputExecutor + LocalProvider.

        Returns
        -------
        parsl.Config
            Configured Parsl Config object.

        """
        parsl = _import_parsl()
        from parsl.executors import HighThroughputExecutor
        from parsl.providers import LocalProvider

        provider = LocalProvider(
            worker_init=self.worker_init,
            min_blocks=0,
            init_blocks=1,
            max_blocks=self.max_blocks,
            parallelism=1,
        )
        executor = HighThroughputExecutor(
            label="local",
            provider=provider,
            max_workers_per_node=self.max_workers_per_node,
            working_dir=str(self.working_directory) if self.working_directory else None,
        )
        return parsl.Config(executors=[executor])

    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutorConfig | SlurmExecutorConfig":
        """Load executor configuration from a YAML file.

        Parameters
        ----------
        path : Path
            Path to the YAML configuration file.

        Returns
        -------
        ExecutorConfig or SlurmExecutorConfig
            The appropriate config model based on the ``provider`` field.

        """
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Executor config YAML is empty or invalid (expected a mapping): {path}"
            )
        provider = data.get("provider", "local")
        if provider == "slurm":
            return SlurmExecutorConfig(**data)
        return cls(**data)


class SlurmExecutorConfig(ExecutorConfig):
    """SLURM executor configuration for Parsl workflows.

    Field names mirror sbatch flags where possible. The only Parsl-specific
    field is ``max_workers_per_node`` (number of parallel tasks per node).

    Parameters
    ----------
    account : str
        SLURM account (``--account``).
    partition : str
        SLURM partition (``--partition``).
    walltime : str
        Wall-clock time limit (``--time``).
    nodes : int
        Number of nodes per SLURM job (``--nodes``).
    cpus_per_node : int
        CPUs per node (``--cpus-per-task`` equivalent).
    gres : str or None
        Generic resource specification (``--gres``), e.g. ``"gpu:l40s:1"``.
    mem : str or None
        Memory per node (``--mem``), e.g. ``"32G"``.
    qos : str or None
        Quality of service (``--qos``).
    constraint : str or None
        Node feature constraint (``--constraint``).
    max_blocks : int
        Maximum number of simultaneous SLURM jobs. Each block is one
        SLURM job; set this to control how many run in parallel.
    scheduler_options : str
        Additional raw ``#SBATCH`` lines for anything not covered above.

    Examples
    --------
    .. code-block:: yaml

        provider: slurm
        account: hpc_chem
        partition: gpu
        walltime: "4:00:00"
        nodes: 1
        cpus_per_node: 12
        gres: "gpu:l40s:1"
        max_blocks: 5
        max_workers_per_node: 1
        worker_init: |
          eval "$(pixi shell-hook -e default)"

    """

    provider: Literal["slurm"] = "slurm"
    account: str
    partition: str = "cpu"
    walltime: str = Field(default="2h", validate_default=True)
    nodes: int = 1
    cpus_per_node: int = 12
    gres: str | None = None
    mem: str | None = None
    qos: str | None = None
    constraint: str | None = None
    scheduler_options: str = ""

    @field_validator("walltime", mode="before")
    @classmethod
    def _normalize_walltime(cls, v: str) -> str:
        """Normalize walltime string on construction."""
        from mdfactory.performance.slurm_config import normalize_slurm_time

        return normalize_slurm_time(v)

    @classmethod
    def from_cluster(
        cls,
        *,
        needs_gpu: bool = False,
        min_cpus: int = 1,
        min_mem_gb: int = 1,
        **extra_fields: Any,
    ) -> "SlurmExecutorConfig":
        """Create an instance with SLURM fields auto-populated from the cluster.

        Three-tier precedence (highest wins):

        1. Explicit keyword arguments passed by the caller.
        2. ``[slurm]`` section in ``config.ini`` — read via
           ``mdfactory.settings.settings``.
        3. Live ``sinfo`` / ``sacctmgr`` autodiscovery via
           ``mdfactory.performance.cluster``.

        Parameters
        ----------
        needs_gpu : bool
            Select a GPU-capable partition when ``True``.
        min_cpus : int
            Minimum CPUs per node required (passed to ``select_partition()``).
        min_mem_gb : int
            Minimum memory per node in GB (passed to ``select_partition()``).
        **extra_fields
            Additional fields forwarded to the constructor.  Base SLURM
            fields (``account``, ``partition``, ``qos``, ``constraint``)
            may be passed here to override autodiscovery.

        Returns
        -------
        SlurmExecutorConfig
            A fully initialised instance.

        Raises
        ------
        RuntimeError
            If SLURM is unavailable *and* the required ``account`` or
            ``partition`` value cannot be resolved from config.
        """
        from mdfactory.performance.cluster import discover_cluster, select_partition
        from mdfactory.settings import settings

        cluster = discover_cluster()

        # --- account: explicit kwarg > config.ini > autodiscovery ---
        resolved_account: str | None = extra_fields.pop("account", None)
        if resolved_account is None:
            resolved_account = settings.slurm_account
        if resolved_account is None:
            if cluster is None:
                raise RuntimeError(
                    "SLURM autodiscovery failed and no account configured. "
                    "Set [slurm] ACCOUNT in config.ini or pass account= explicitly."
                )
            resolved_account = cluster.default_account
            if resolved_account is None:
                raise RuntimeError(
                    "No SLURM account available. "
                    "Set [slurm] ACCOUNT in config.ini or pass account= explicitly."
                )

        # --- partition: explicit kwarg > config.ini (cpu/gpu) > autodiscovery ---
        resolved_partition: str | None = extra_fields.pop("partition", None)
        if resolved_partition is None:
            resolved_partition = (
                settings.slurm_partition_gpu if needs_gpu else settings.slurm_partition_cpu
            )
        if resolved_partition is None:
            if cluster is None:
                raise RuntimeError(
                    "SLURM autodiscovery failed and no partition configured. "
                    "Set [slurm] PARTITION_CPU or PARTITION_GPU in config.ini "
                    "or pass partition= explicitly."
                )
            selected = select_partition(
                cluster,
                needs_gpu=needs_gpu,
                min_cpus=min_cpus,
                min_mem_gb=min_mem_gb,
            )
            if selected is None:
                raise RuntimeError(
                    f"No suitable partition found for requirements: "
                    f"needs_gpu={needs_gpu}, cpus={min_cpus}, mem={min_mem_gb}GB"
                )
            resolved_partition = selected.name

        # --- qos: explicit kwarg > config.ini ---
        resolved_qos: str | None = extra_fields.pop("qos", None)
        if resolved_qos is None:
            resolved_qos = settings.slurm_qos

        # --- constraint: explicit kwarg only ---
        resolved_constraint: str | None = extra_fields.pop("constraint", None)

        return cls(
            account=resolved_account,
            partition=resolved_partition,
            qos=resolved_qos,
            constraint=resolved_constraint,
            **extra_fields,
        )

    def to_parsl_config(self) -> "parsl.Config":
        """Build a Parsl Config with HighThroughputExecutor + SlurmProvider.

        Returns
        -------
        parsl.Config
            Configured Parsl Config object.

        """
        parsl = _import_parsl()
        from parsl.executors import HighThroughputExecutor
        from parsl.providers import SlurmProvider

        # Build scheduler_options from structured fields + raw extras
        opts = []
        if self.gres:
            opts.append(f"#SBATCH --gres={self.gres}")
        if self.mem:
            opts.append(f"#SBATCH --mem={self.mem}")
        if self.qos:
            opts.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            opts.append(f"#SBATCH --constraint={self.constraint}")
        if self.scheduler_options:
            opts.append(self.scheduler_options)
        scheduler_options = "\n".join(opts)

        provider = SlurmProvider(
            account=self.account,
            partition=self.partition,
            walltime=self.walltime,
            nodes_per_block=self.nodes,
            cores_per_node=self.cpus_per_node,
            worker_init=self.worker_init,
            scheduler_options=scheduler_options,
            min_blocks=0,
            init_blocks=1,
            max_blocks=self.max_blocks,
            parallelism=1,
        )
        executor = HighThroughputExecutor(
            label="slurm",
            provider=provider,
            max_workers_per_node=self.max_workers_per_node,
            working_dir=str(self.working_directory) if self.working_directory else None,
        )
        return parsl.Config(executors=[executor])
