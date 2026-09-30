# ABOUTME: HPC performance optimization package for mdfactory.
# ABOUTME: Cluster autodiscovery, CPU affinity, benchmarking, and GPU MPS management.
"""HPC performance optimization utilities.

Modules
-------
cluster
    SLURM cluster autodiscovery — query partitions, node types, accounts, and QOS.
slurm_config
    SLURM configuration models shared across all submission backends.
    ``BaseSlurmConfig`` provides 3-tier autodiscovery for account and partition.
    ``SlurmConfig`` is the submitit backend configuration.
benchmark
    Scalability benchmark sweep — short GROMACS trials across resource configs.
"""

from mdfactory.performance import cluster
from mdfactory.performance.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark_sweep,
)
from mdfactory.performance.slurm_config import BaseSlurmConfig, SlurmConfig

__all__ = [
    "cluster",
    "BaseSlurmConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark_sweep",
    "SlurmConfig",
]
