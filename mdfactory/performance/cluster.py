# ABOUTME: SLURM cluster autodiscovery — query partitions, node types, accounts, QOS.
# ABOUTME: Parses sinfo/sacctmgr output into structured dataclasses for resource-aware scheduling.
"""SLURM cluster autodiscovery.

Query the local SLURM scheduler and return a structured representation of
available resources (partitions, node types, accounts, QOS policies, GPU types).

Functions
---------
discover_cluster
    Main entry point — returns ``ClusterInfo`` or ``None`` if SLURM is unavailable.
select_partition
    Heuristic partition selection given resource requirements.

Examples
--------
>>> from mdfactory.performance.cluster import discover_cluster, select_partition
>>> cluster = discover_cluster()
>>> if cluster is not None:
...     gpu_part = select_partition(cluster, needs_gpu=True)
"""

from __future__ import annotations

import functools
import os
import shutil

from pydantic import BaseModel, ConfigDict, Field

from mdfactory.utils.utilities import run_command


class NodeType(BaseModel):
    """Hardware specification of a node type within a partition.

    Parameters
    ----------
    cpus : int
        Number of CPU cores per node.
    memory_mb : int
        Memory in megabytes per node.
    gpu_specs : tuple of (int, str | None)
        GPU specifications as (count, type) tuples. Empty if CPU-only.
        Multiple entries represent different GPU types on the same node.
        Type can be None if SLURM reports GPU count without type info.
    features : tuple of str
        SLURM feature/constraint tags on this node type (immutable).
    count : int
        Number of nodes with this exact configuration.
    """

    model_config = ConfigDict(frozen=True)

    cpus: int
    memory_mb: int
    gpu_specs: tuple[tuple[int, str | None], ...] = Field(default_factory=tuple)
    features: tuple[str, ...] = Field(default_factory=tuple)
    count: int = 1


class Partition(BaseModel):
    """A SLURM partition with its node types and limits.

    Parameters
    ----------
    name : str
        Partition name (e.g., ``"gpu"``, ``"cpu"``).
    state : str
        Partition-level state: ``"up"`` if any node is schedulable, otherwise
        the last observed unhealthy state (e.g., ``"down"``, ``"drained"``).
    max_time : str
        Maximum walltime (SLURM format, e.g., ``"3-00:00:00"``).
    default_time : str
        Default walltime assigned when user does not specify one.
        Populated from sinfo ``%L``; equals ``max_time`` on legacy output.
    node_types : list of NodeType
        Distinct hardware configurations available in this partition.
    total_nodes : int
        Total number of nodes in the partition.
    is_default : bool
        Whether this is the cluster's default partition.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    state: str
    max_time: str
    default_time: str
    node_types: list[NodeType] = Field(default_factory=list)
    total_nodes: int = 0
    is_default: bool = False


class ClusterInfo(BaseModel):
    """Structured representation of a SLURM cluster's resources.

    Parameters
    ----------
    partitions : list of Partition
        All discovered partitions.
    accounts : list of str
        SLURM accounts available to the current user.
    qos_policies : list of str
        Available QOS policy names.
    default_account : str or None
        The user's default account, if determinable.
    """

    model_config = ConfigDict(frozen=True)

    partitions: list[Partition] = Field(default_factory=list)
    accounts: list[str] = Field(default_factory=list)
    qos_policies: list[str] = Field(default_factory=list)
    default_account: str | None = None


def _parse_gres(gres_str: str) -> list[tuple[int, str | None]]:
    """Parse SLURM GRES string to extract GPU count and type for all GPU entries.

    Parameters
    ----------
    gres_str : str
        GRES field from sinfo (e.g., ``"gpu:a100:4"``, ``"gpu:2"``, ``"(null)"``).
        May contain multiple GPU entries separated by commas.

    Returns
    -------
    list of tuple (int, str or None)
        List of (gpu_count, gpu_type) for each GPU entry. Empty list when no GPUs.

    Examples
    --------
    >>> _parse_gres("gpu:a100:4")
    [(4, 'a100')]
    >>> _parse_gres("gpu:b200:8(S:0-1),gpu:1g.23gb:7(S:1)")
    [(8, 'b200'), (7, '1g.23gb')]
    >>> _parse_gres("(null)")
    []
    """
    if not gres_str or gres_str == "(null)":
        return []

    gpu_entries = []

    # Handle multiple GRES entries separated by commas
    for raw_entry in gres_str.split(","):
        entry = raw_entry.strip()
        if not entry.startswith("gpu"):
            continue

        # Strip socket binding suffix like "(S:0-1)" or "(IDX:0-3)"
        if "(" in entry:
            entry = entry.split("(")[0]

        parts = entry.split(":")
        if len(parts) == 3:
            # gpu:type:count
            _, gpu_type, count_str = parts
            gpu_entries.append((int(count_str), gpu_type))
        elif len(parts) == 2:
            # gpu:count (no type specified)
            _, count_str = parts
            # Check if second part is a number or a type
            try:
                count = int(count_str)
                gpu_entries.append((count, None))
            except ValueError:
                # gpu:type with implicit count of 1
                gpu_entries.append((1, count_str))

    return gpu_entries


def _parse_time_limit(time_str: str) -> str:
    """Normalize SLURM time limit strings.

    Parameters
    ----------
    time_str : str
        Time limit from sinfo (e.g., ``"3-00:00:00"``, ``"infinite"``, ``"2:00:00"``).

    Returns
    -------
    str
        Cleaned time string.
    """
    if not time_str or time_str == "n/a":
        return "unknown"
    return time_str.strip()


def _parse_memory_mb(mem_str: str) -> int:
    """Parse memory string from sinfo to megabytes.

    Parameters
    ----------
    mem_str : str
        Memory field from sinfo (numeric, in MB by default).

    Returns
    -------
    int
        Memory in MB. Returns 0 on parse failure.
    """
    try:
        # sinfo %m gives memory in MB as an integer
        cleaned = mem_str.strip().rstrip("+")
        return int(cleaned)
    except (ValueError, AttributeError):
        return 0


def _parse_features(features_str: str) -> tuple[str, ...]:
    """Parse SLURM features/constraints string.

    Parameters
    ----------
    features_str : str
        Features field from sinfo (comma-separated or ``"(null)"``).

    Returns
    -------
    tuple of str
        Feature strings (immutable).
    """
    if not features_str or features_str == "(null)":
        return ()
    return tuple(f.strip() for f in features_str.split(",") if f.strip())


def _parse_sinfo(output: str) -> list[Partition]:
    """Parse sinfo output into Partition objects.

    Expects output from:
        sinfo -N --noheader -o "%P %n %c %m %G %f %l %L %T"

    Fields: Partition, NodeName, CPUs, Memory(MB), GRES, Features,
            MaxTimeLimit, DefaultTimeLimit, State

    Also supports the legacy 8-field format (without %L) for backward
    compatibility — in that case ``default_time`` equals ``max_time``.

    Parameters
    ----------
    output : str
        Raw sinfo output.

    Returns
    -------
    list of Partition
        Parsed partition list with deduplicated node types.
    """
    # Collect data per partition
    partition_data: dict[str, dict] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 9:
            # 9-field format: %P %n %c %m %G %f %l %L %T
            partition_name = parts[0]
            cpus_str = parts[2]
            mem_str = parts[3]
            gres_str = parts[4]
            features_str = parts[5]
            max_time = parts[6]
            default_time = parts[7]
            state = parts[8]
        elif len(parts) == 8:
            # Legacy 8-field format (no %L): default_time = max_time
            partition_name = parts[0]
            cpus_str = parts[2]
            mem_str = parts[3]
            gres_str = parts[4]
            features_str = parts[5]
            max_time = parts[6]
            default_time = parts[6]
            state = parts[7]
        else:
            continue

        # Handle default partition marker (trailing asterisk)
        is_default = partition_name.endswith("*")
        if is_default:
            partition_name = partition_name.rstrip("*")

        # Parse node specs
        try:
            cpus = int(cpus_str)
        except ValueError:
            continue

        memory_mb = _parse_memory_mb(mem_str)
        gpu_entries = _parse_gres(gres_str)
        features = _parse_features(features_str)

        if partition_name not in partition_data:
            partition_data[partition_name] = {
                "max_time": _parse_time_limit(max_time),
                "default_time": _parse_time_limit(default_time),
                "node_types": {},  # Maps node_key -> count
                "node_count": 0,
                "is_default": is_default,
                "has_healthy_node": False,
                "last_unhealthy_state": "down",
            }

        # Track if any line marks this as default
        if is_default:
            partition_data[partition_name]["is_default"] = True

        # Determine if node is operational (can schedule jobs or report accurate config)
        # Exclude only truly broken/invalid nodes: down, drained, inval, fail, maint, unknown
        state_lower = state.lower()
        is_broken = any(
            broken in state_lower
            for broken in ("down", "drained", "inval", "fail", "maint", "unknown")
        )
        is_operational = not is_broken

        # Partition is "up" if ANY operational node exists
        if is_operational:
            partition_data[partition_name]["has_healthy_node"] = True
        else:
            partition_data[partition_name]["last_unhealthy_state"] = state

        # Count all nodes (including allocated, reserved, etc.) for stable totals
        partition_data[partition_name]["node_count"] += 1

        # Only collect node specs from operational nodes
        # (broken nodes may report incorrect/stale hardware configs)
        if is_operational:
            # Convert GPU entries to a sorted tuple for consistent hashing
            # Sort by type name for deterministic ordering
            gpu_specs = tuple(sorted(gpu_entries, key=lambda x: x[1] or ""))

            # Use a hashable representation for deduplication
            node_key = (cpus, memory_mb, gpu_specs, features)

            # Track count of nodes with this exact configuration
            if node_key not in partition_data[partition_name]["node_types"]:
                partition_data[partition_name]["node_types"][node_key] = 0
            partition_data[partition_name]["node_types"][node_key] += 1

    # Build Partition objects
    partitions = []
    for name, data in partition_data.items():
        # Partition state: "up" if any node is healthy, otherwise report
        # the last observed unhealthy state
        if data["has_healthy_node"]:
            partition_state = "up"
        else:
            partition_state = data["last_unhealthy_state"]

        node_types = [
            NodeType(
                cpus=cpus,
                memory_mb=mem,
                gpu_specs=gpu_specs,
                features=feats,
                count=count,
            )
            for (cpus, mem, gpu_specs, feats), count in data["node_types"].items()
        ]
        # Sort node types by CPU count, memory, then GPU specs for deterministic ordering
        node_types.sort(key=lambda n: (n.cpus, n.memory_mb, n.gpu_specs))

        partitions.append(
            Partition(
                name=name,
                state=partition_state,
                max_time=data["max_time"],
                default_time=data["default_time"],
                node_types=node_types,
                total_nodes=data["node_count"],
                is_default=data["is_default"],
            )
        )

    # Sort partitions: default first, then alphabetical
    partitions.sort(key=lambda p: (not p.is_default, p.name))
    return partitions


def _parse_accounts(output: str) -> list[str]:
    """Parse sacctmgr account output.

    Parameters
    ----------
    output : str
        Raw sacctmgr output (one account per line, parsable2 format).

    Returns
    -------
    list of str
        Unique account names, sorted.
    """
    accounts = set()
    for line in output.splitlines():
        account = line.strip()
        if account:
            accounts.add(account)
    return sorted(accounts)


def _parse_qos(output: str) -> list[str]:
    """Parse sacctmgr QOS output.

    Parameters
    ----------
    output : str
        Raw sacctmgr output (pipe-separated: Name|MaxWall|MaxTRES).

    Returns
    -------
    list of str
        QOS policy names, sorted.
    """
    qos_names = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Format: Name|MaxWall|MaxTRES
        parts = line.split("|")
        if parts and parts[0].strip():
            qos_names.add(parts[0].strip())
    return sorted(qos_names)


def _discover_partitions() -> list[Partition] | None:
    """Query sinfo for partition and node information.

    Returns
    -------
    list of Partition or None
        Parsed partitions, or None if sinfo is unavailable.
    """
    output = run_command(
        ["sinfo", "-N", "--noheader", "-o", "%P %n %c %m %G %f %l %L %T"],
        timeout=30,
        graceful=True,
    )
    if output is None:
        return None
    return _parse_sinfo(output)


def _discover_accounts() -> list[str] | None:
    """Query sacctmgr for the current user's accounts.

    Returns
    -------
    list of str or None
        Account names, or None if sacctmgr is unavailable.
    """
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    if not user:
        return None
    output = run_command(
        [
            "sacctmgr",
            "show",
            "assoc",
            f"user={user}",
            "format=Account",
            "--noheader",
            "--parsable2",
        ],
        timeout=30,
        graceful=True,
    )
    if output is None:
        return None
    return _parse_accounts(output)


def _discover_qos() -> list[str] | None:
    """Query sacctmgr for available QOS policies.

    Returns
    -------
    list of str or None
        QOS names, or None if sacctmgr is unavailable.
    """
    output = run_command(
        [
            "sacctmgr",
            "show",
            "qos",
            "format=Name,MaxWall,MaxTRES",
            "--noheader",
            "--parsable2",
        ],
        timeout=30,
        graceful=True,
    )
    if output is None:
        return None
    return _parse_qos(output)


def _discover_default_account() -> str | None:
    """Query sacctmgr for the current user's default SLURM account.

    Returns
    -------
    str or None
        The user's default account, or None if unavailable.
    """
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    if not user:
        return None
    output = run_command(
        [
            "sacctmgr",
            "show",
            "user",
            user,
            "format=DefaultAccount",
            "--noheader",
            "--parsable2",
        ],
        timeout=30,
        graceful=True,
    )
    if output is None:
        return None
    account = output.strip().splitlines()[0].strip() if output.strip() else None
    return account if account else None


@functools.lru_cache(maxsize=1)
def discover_cluster() -> ClusterInfo | None:
    """Query SLURM and return structured cluster information.

    Calls ``sinfo`` and ``sacctmgr`` to discover partitions, node types,
    accounts, and QOS policies. Returns None gracefully when SLURM commands
    are not available (e.g., running on a laptop).

    Results are cached for the session (cluster topology doesn't change
    mid-session). Call ``discover_cluster.cache_clear()`` to force re-query.

    Returns
    -------
    ClusterInfo or None
        Structured cluster information, or None if SLURM is unavailable.

    Examples
    --------
    >>> cluster = discover_cluster()
    >>> if cluster is not None:
    ...     for p in cluster.partitions:
    ...         print(f"{p.name}: {p.total_nodes} nodes")
    """
    # sinfo is the minimum requirement — if it's not available, we're not
    # on a SLURM cluster
    if shutil.which("sinfo") is None:
        return None

    partitions = _discover_partitions()
    if partitions is None:
        return None

    # Accounts and QOS are best-effort (sacctmgr may be restricted)
    accounts = _discover_accounts() or []
    qos_policies = _discover_qos() or []

    # Query the real SLURM default account; fall back to first available
    default_account = _discover_default_account()
    if default_account is None and accounts:
        default_account = accounts[0]

    return ClusterInfo(
        partitions=partitions,
        accounts=accounts,
        qos_policies=qos_policies,
        default_account=default_account,
    )


def select_partition(
    cluster: ClusterInfo,
    *,
    needs_gpu: bool = False,
    min_cpus: int = 1,
    min_mem_gb: int = 1,
) -> Partition | None:
    """Heuristic partition selection given resource requirements.

    Selects the best-matching partition from the cluster based on hardware
    needs. Prefers partitions that are ``"up"`` and have nodes meeting the
    specified requirements.

    Parameters
    ----------
    cluster : ClusterInfo
        Cluster information from ``discover_cluster()``.
    needs_gpu : bool
        If True, only consider partitions with GPU-equipped nodes.
    min_cpus : int
        Minimum CPUs per node required.
    min_mem_gb : int
        Minimum memory per node in GB.

    Returns
    -------
    Partition or None
        Best matching partition, or None if no partition meets requirements.

    Examples
    --------
    >>> cluster = discover_cluster()
    >>> gpu_partition = select_partition(cluster, needs_gpu=True, min_cpus=8)
    """
    min_mem_mb = min_mem_gb * 1024
    candidates: list[Partition] = []

    for partition in cluster.partitions:
        # Skip partitions with no schedulable nodes
        if partition.state.lower() != "up":
            continue

        # Check if any node type meets requirements
        has_qualifying_node = False
        for node in partition.node_types:
            if node.cpus < min_cpus:
                continue
            if node.memory_mb < min_mem_mb:
                continue
            # Check if node has any GPUs
            has_gpu = len(node.gpu_specs) > 0
            if needs_gpu and not has_gpu:
                continue
            has_qualifying_node = True
            break

        if has_qualifying_node:
            candidates.append(partition)

    if not candidates:
        return None

    # Prefer: default partition > most nodes > alphabetical
    candidates.sort(key=lambda p: (not p.is_default, -p.total_nodes, p.name))
    return candidates[0]
