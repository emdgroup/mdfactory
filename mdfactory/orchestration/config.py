# ABOUTME: Pydantic models for Parsl executor configuration
# ABOUTME: Supports local and SLURM providers with YAML serialization
"""Executor configuration models for Parsl workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import parsl
import yaml
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider, SlurmProvider
from pydantic import BaseModel


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

    def to_parsl_config(self) -> parsl.Config:
        """Build a Parsl Config with HighThroughputExecutor + LocalProvider.

        Returns
        -------
        parsl.Config
            Configured Parsl Config object.

        """
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
    partition: str = "gpu"
    walltime: str = "2:00:00"
    nodes: int = 1
    cpus_per_node: int = 12
    gres: str | None = None
    mem: str | None = None
    qos: str | None = None
    constraint: str | None = None
    max_workers_per_node: int = 1
    max_blocks: int = 1
    scheduler_options: str = ""

    def to_parsl_config(self) -> parsl.Config:
        """Build a Parsl Config with HighThroughputExecutor + SlurmProvider.

        Returns
        -------
        parsl.Config
            Configured Parsl Config object.

        """
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
