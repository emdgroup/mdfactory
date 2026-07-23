# ABOUTME: Pydantic models for Parsl executor configuration
# ABOUTME: Supports local and SLURM providers with YAML serialization
"""Executor configuration models for Parsl workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from mdfactory.performance.slurm_config import BaseSlurmConfig

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
    available_accelerators : int or list[str]
        GPU pinning for Parsl workers. An integer count (or explicit list of
        device IDs) makes Parsl assign each worker a distinct accelerator and
        set ``CUDA_VISIBLE_DEVICES`` accordingly. Defaults to ``0`` (no pinning,
        all workers share the node's uncontrolled GPU context). Set this when
        running ``max_workers_per_node > 1`` on GPU nodes to avoid silent
        wrong-GPU contention.
    run_dir : Path
        Directory where Parsl writes its ``runinfo`` logs and database.
        Defaults to ``~/.parsl/mdfactory`` so logs land in a predictable,
        controllable location instead of scattering ``runinfo/`` into the
        current working directory (typically the login-node home on HPC).
    retries : int
        Number of times Parsl re-runs a failed app before propagating the
        exception. Defaults to 3 to handle transient SLURM / network faults.
        Set to 0 to disable retries.

    """

    provider: Literal["local", "slurm"] = "local"
    worker_init: str = ""
    working_directory: Path | None = None
    max_workers_per_node: int = 1
    max_blocks: int = 1
    available_accelerators: int | list[str] = 0
    run_dir: Path = Field(default_factory=lambda: Path("~/.parsl/mdfactory").expanduser())
    retries: int = Field(default=3, ge=0, description="Number of times Parsl retries a failed app.")

    @field_validator("run_dir", mode="before")
    @classmethod
    def _expand_run_dir(cls, v: Any) -> Any:
        """Expand ``~`` in user-supplied run_dir values."""
        return Path(v).expanduser() if v is not None else v

    @field_serializer("run_dir", "working_directory")
    def _serialize_paths(self, v: Path | None) -> str | None:
        """Serialize Path fields as plain strings for YAML compatibility."""
        return str(v) if v is not None else None

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
            available_accelerators=self.available_accelerators,
            working_dir=str(self.working_directory) if self.working_directory else None,
        )
        return parsl.Config(
            executors=[executor],
            run_dir=str(self.run_dir),
            retries=self.retries,
        )

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


class SlurmExecutorConfig(ExecutorConfig, BaseSlurmConfig):
    """SLURM executor configuration for Parsl workflows.

    Inherits the four cross-cutting SLURM fields (``account``, ``partition``,
    ``qos``, ``constraint``) and the authoritative ``from_cluster()`` factory
    from :class:`~mdfactory.performance.slurm_config.BaseSlurmConfig`, and the
    Parsl executor fields (``run_dir``, ``available_accelerators``, …) from
    :class:`ExecutorConfig`. Only Parsl/SLURM-job-specific fields are declared
    here.

    Field names mirror sbatch flags where possible. The only Parsl-specific
    field is ``max_workers_per_node`` (number of parallel tasks per node).

    Notes
    -----
    ``BaseSlurmConfig`` is frozen (immutable), but executor configs are mutated
    in places (e.g. the TUI wizard), so this subclass explicitly sets
    ``frozen=False`` to match :class:`ExecutorConfig`'s mutable behaviour.

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
        CPUs requested per node. Wired to Parsl ``SlurmProvider(cores_per_node=...)``,
        which Parsl uses to size the worker pool (roughly ``--ntasks`` /
        cores-per-node) — it is **not** the sbatch ``--cpus-per-task`` flag. For
        MPI+OpenMP workloads (e.g. GROMACS), OpenMP threads-per-rank must be set
        separately via ``scheduler_options`` / ``worker_init``.
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
        Additional raw ``#SBATCH`` lines for anything not covered above
        (allocation-level flags injected into the job script).
    launch_options : str
        Extra ``srun`` flags forwarded to Parsl's ``SrunLauncher(overrides=...)``
        for task-placement / binding control (e.g.
        ``"--cpu-bind=cores --distribution=block:block"`` for NUMA-local CPU
        binding in MPI+OpenMP workers). Empty string leaves Parsl's default
        launcher untouched.

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

    model_config = ConfigDict(frozen=False)

    provider: Literal["slurm"] = "slurm"
    # account, partition, qos, constraint inherited from BaseSlurmConfig.
    # from_cluster() inherited from BaseSlurmConfig — extra fields below are
    # forwarded to the constructor via resolve_slurm_fields().
    walltime: str = Field(default="2h", validate_default=True)
    nodes: int = 1
    cpus_per_node: int = 12
    gres: str | None = None
    mem: str | None = None
    scheduler_options: str = ""
    launch_options: str = ""

    @field_validator("walltime", mode="before")
    @classmethod
    def _normalize_walltime(cls, v: str) -> str:
        """Normalize walltime string on construction."""
        from mdfactory.performance.slurm_config import normalize_slurm_time

        return normalize_slurm_time(v)

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

        # Only override Parsl's default launcher when srun-level flags are given,
        # so NUMA/task-placement binding is opt-in (relevant for MPI+OpenMP work).
        provider_kwargs: dict[str, Any] = {}
        if self.launch_options:
            from parsl.launchers import SrunLauncher

            provider_kwargs["launcher"] = SrunLauncher(overrides=self.launch_options)

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
            **provider_kwargs,
        )
        executor = HighThroughputExecutor(
            label="slurm",
            provider=provider,
            max_workers_per_node=self.max_workers_per_node,
            available_accelerators=self.available_accelerators,
            working_dir=str(self.working_directory) if self.working_directory else None,
        )
        return parsl.Config(
            executors=[executor],
            run_dir=str(self.run_dir),
            retries=self.retries,
        )
