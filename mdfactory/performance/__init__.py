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
"""

from mdfactory.performance.slurm_config import BaseSlurmConfig, SlurmConfig

__all__ = ["BaseSlurmConfig", "SlurmConfig"]
