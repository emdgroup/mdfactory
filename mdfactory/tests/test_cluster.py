# ABOUTME: Unit tests for mdfactory.performance.cluster (SLURM autodiscovery).
# ABOUTME: Uses mocked sinfo/sacctmgr output — no SLURM required to run.
"""Tests for SLURM cluster autodiscovery."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mdfactory.performance.cluster import (
    ClusterInfo,
    NodeType,
    Partition,
    _parse_accounts,
    _parse_gres,
    _parse_qos,
    _parse_sinfo,
    discover_cluster,
    select_partition,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic sinfo / sacctmgr output
# ---------------------------------------------------------------------------

# 9-field format: Partition Node CPUs Mem GRES Features MaxTime DefTime State
SINFO_OUTPUT_MIXED = """\
cpu* node001 128 512000 (null) epyc9555,avx512 3-00:00:00 1-00:00:00 idle
cpu* node002 128 512000 (null) epyc9555,avx512 3-00:00:00 1-00:00:00 mixed
cpu* node003 128 512000 (null) epyc9555,avx512 3-00:00:00 1-00:00:00 allocated
gpu node010 64 256000 gpu:a100:4 a100,nvlink 1-00:00:00 4:00:00 idle
gpu node011 64 256000 gpu:a100:4 a100,nvlink 1-00:00:00 4:00:00 idle
gpu node012 96 512000 gpu:h100:8 h100,nvlink 1-00:00:00 4:00:00 mixed
bigmem node020 256 2048000 (null) bigmem,epyc 7-00:00:00 1-00:00:00 idle
"""

SINFO_OUTPUT_SINGLE_PARTITION = """\
compute node001 64 128000 (null) (null) 2-00:00:00 1-00:00:00 idle
compute node002 64 128000 (null) (null) 2-00:00:00 1-00:00:00 idle
"""

SINFO_OUTPUT_GPU_ONLY = """\
gpu-short node001 32 64000 gpu:v100:2 v100 4:00:00 2:00:00 idle
gpu-short node002 32 64000 gpu:v100:2 v100 4:00:00 2:00:00 idle
gpu-long node003 64 128000 gpu:a100:4 a100 2-00:00:00 4:00:00 idle
"""

SINFO_OUTPUT_NO_TYPE_GPU = """\
gpu node001 64 256000 gpu:4 (null) 1-00:00:00 4:00:00 idle
"""

# First node drained, rest healthy → partition should be "up"
SINFO_OUTPUT_MIXED_HEALTH = """\
cpu node001 64 128000 (null) (null) 2-00:00:00 1-00:00:00 drained
cpu node002 64 128000 (null) (null) 2-00:00:00 1-00:00:00 idle
cpu node003 64 128000 (null) (null) 2-00:00:00 1-00:00:00 idle
"""

# All nodes unhealthy → partition should report unhealthy state
SINFO_OUTPUT_ALL_DOWN = """\
cpu node001 64 128000 (null) (null) 2-00:00:00 1-00:00:00 down
cpu node002 64 128000 (null) (null) 2-00:00:00 1-00:00:00 drained
"""

# Legacy 8-field format (no %L default time) — backward compatibility
SINFO_OUTPUT_LEGACY_8FIELD = """\
compute node001 64 128000 (null) (null) 2-00:00:00 idle
compute node002 64 128000 (null) (null) 2-00:00:00 idle
"""

SACCTMGR_ACCOUNTS = """\
myproject
shared-account
default-account
"""

SACCTMGR_QOS = """\
normal||
high|1-00:00:00|cpu=128,mem=512G
gpu|12:00:00|cpu=64,gres/gpu=4
"""


# ---------------------------------------------------------------------------
# Tests: GRES parsing
# ---------------------------------------------------------------------------


class TestParseGres:
    """Test GPU GRES string parsing."""

    def test_gpu_with_type_and_count(self):
        assert _parse_gres("gpu:a100:4") == [(4, "a100")]

    def test_gpu_with_count_only(self):
        assert _parse_gres("gpu:2") == [(2, None)]

    def test_gpu_with_type_only(self):
        assert _parse_gres("gpu:h100") == [(1, "h100")]

    def test_null_gres(self):
        assert _parse_gres("(null)") == []

    def test_empty_string(self):
        assert _parse_gres("") == []

    def test_multi_gres_with_gpu(self):
        assert _parse_gres("mps:shared,gpu:a100:4") == [(4, "a100")]

    def test_non_gpu_gres(self):
        assert _parse_gres("mps:shared") == []

    def test_multiple_gpu_types(self):
        """Test multiple GPU entries (e.g., MIG slices)."""
        result = _parse_gres("gpu:b200:7,gpu:1g.23gb:7")
        assert len(result) == 2
        assert (7, "1g.23gb") in result
        assert (7, "b200") in result

    def test_socket_binding_stripped(self):
        """Test that socket binding suffixes are removed."""
        assert _parse_gres("gpu:l40s:4(S:0-1)") == [(4, "l40s")]
        assert _parse_gres("gpu:b200:8(S:0-1),gpu:1g.23gb:7(S:1)") == [(8, "b200"), (7, "1g.23gb")]


# ---------------------------------------------------------------------------
# Tests: sinfo parsing
# ---------------------------------------------------------------------------


class TestParseSinfo:
    """Test sinfo output parsing into Partition objects."""

    def test_mixed_cluster(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)

        # Should find 3 partitions
        assert len(partitions) == 3
        names = [p.name for p in partitions]
        assert "cpu" in names
        assert "gpu" in names
        assert "bigmem" in names

    def test_default_partition_marker(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)

        cpu_part = next(p for p in partitions if p.name == "cpu")
        assert cpu_part.is_default is True

        gpu_part = next(p for p in partitions if p.name == "gpu")
        assert gpu_part.is_default is False

    def test_default_partition_sorted_first(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        assert partitions[0].name == "cpu"
        assert partitions[0].is_default is True

    def test_cpu_partition_node_types(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        cpu_part = next(p for p in partitions if p.name == "cpu")

        # All 3 nodes have same spec → 1 unique node type
        assert len(cpu_part.node_types) == 1
        nt = cpu_part.node_types[0]
        assert nt.cpus == 128
        assert nt.memory_mb == 512000
        assert nt.gpu_specs == ()  # No GPUs
        assert "epyc9555" in nt.features
        assert nt.count == 3  # 3 nodes with this config

    def test_gpu_partition_multiple_node_types(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        gpu_part = next(p for p in partitions if p.name == "gpu")

        # 2 distinct node types: a100 (64 core) and h100 (96 core)
        assert len(gpu_part.node_types) == 2

        a100_node = next(n for n in gpu_part.node_types if (4, "a100") in n.gpu_specs)
        assert a100_node.cpus == 64
        assert a100_node.gpu_specs == ((4, "a100"),)
        assert a100_node.count == 2  # 2 a100 nodes

        h100_node = next(n for n in gpu_part.node_types if (8, "h100") in n.gpu_specs)
        assert h100_node.cpus == 96
        assert h100_node.gpu_specs == ((8, "h100"),)
        assert h100_node.count == 1  # 1 h100 node

    def test_total_node_count(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)

        cpu_part = next(p for p in partitions if p.name == "cpu")
        assert cpu_part.total_nodes == 3

        gpu_part = next(p for p in partitions if p.name == "gpu")
        assert gpu_part.total_nodes == 3

        bigmem_part = next(p for p in partitions if p.name == "bigmem")
        assert bigmem_part.total_nodes == 1

    def test_time_limit_parsed(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        cpu_part = next(p for p in partitions if p.name == "cpu")
        assert cpu_part.max_time == "3-00:00:00"

    def test_single_partition(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_SINGLE_PARTITION)
        assert len(partitions) == 1
        assert partitions[0].name == "compute"
        assert partitions[0].total_nodes == 2

    def test_gpu_without_type(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_NO_TYPE_GPU)
        assert len(partitions) == 1
        nt = partitions[0].node_types[0]
        assert nt.gpu_specs == ((4, None),)  # 4 GPUs, no type specified

    def test_default_time_parsed_separately(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        cpu_part = next(p for p in partitions if p.name == "cpu")
        assert cpu_part.max_time == "3-00:00:00"
        assert cpu_part.default_time == "1-00:00:00"

        gpu_part = next(p for p in partitions if p.name == "gpu")
        assert gpu_part.max_time == "1-00:00:00"
        assert gpu_part.default_time == "4:00:00"

    def test_legacy_8field_format(self):
        """Parser handles legacy 8-field sinfo output (no %L)."""
        partitions = _parse_sinfo(SINFO_OUTPUT_LEGACY_8FIELD)
        assert len(partitions) == 1
        assert partitions[0].name == "compute"
        # default_time falls back to max_time
        assert partitions[0].default_time == partitions[0].max_time

    def test_partition_state_up_when_any_node_healthy(self):
        """Partition with mixed healthy/unhealthy nodes should be 'up'."""
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED_HEALTH)
        assert len(partitions) == 1
        assert partitions[0].state == "up"

    def test_partition_state_unhealthy_when_all_nodes_down(self):
        """Partition with no healthy nodes reports unhealthy state."""
        partitions = _parse_sinfo(SINFO_OUTPUT_ALL_DOWN)
        assert len(partitions) == 1
        assert partitions[0].state != "up"
        assert partitions[0].state in ("down", "drained")

    def test_empty_output(self):
        partitions = _parse_sinfo("")
        assert partitions == []

    def test_malformed_lines_skipped(self):
        output = "this is not valid sinfo output\n" + SINFO_OUTPUT_SINGLE_PARTITION
        partitions = _parse_sinfo(output)
        # Should still parse valid lines
        assert len(partitions) == 1


# ---------------------------------------------------------------------------
# Tests: account / QOS parsing
# ---------------------------------------------------------------------------


class TestParseAccounts:
    """Test sacctmgr account output parsing."""

    def test_multiple_accounts(self):
        accounts = _parse_accounts(SACCTMGR_ACCOUNTS)
        assert accounts == ["default-account", "myproject", "shared-account"]

    def test_empty_output(self):
        assert _parse_accounts("") == []

    def test_whitespace_handling(self):
        assert _parse_accounts("  acc1  \n  acc2  \n") == ["acc1", "acc2"]

    def test_deduplication(self):
        assert _parse_accounts("acc1\nacc1\nacc2") == ["acc1", "acc2"]


class TestParseQos:
    """Test sacctmgr QOS output parsing."""

    def test_multiple_qos(self):
        qos = _parse_qos(SACCTMGR_QOS)
        assert qos == ["gpu", "high", "normal"]

    def test_empty_output(self):
        assert _parse_qos("") == []

    def test_pipe_separated_format(self):
        qos = _parse_qos("standard|2-00:00:00|cpu=256\n")
        assert qos == ["standard"]


# ---------------------------------------------------------------------------
# Tests: discover_cluster integration (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDiscoverCluster:
    """Test discover_cluster with mocked subprocess calls."""

    def setup_method(self):
        """Clear LRU cache between tests."""
        discover_cluster.cache_clear()

    def test_returns_none_without_sinfo(self):
        with patch("mdfactory.performance.cluster.shutil.which", return_value=None):
            result = discover_cluster()
        assert result is None

    def test_returns_cluster_info_with_sinfo(self):
        """Test full integration when SLURM commands return data."""
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        accounts = _parse_accounts(SACCTMGR_ACCOUNTS)
        qos = _parse_qos(SACCTMGR_QOS)

        with (
            patch("mdfactory.performance.cluster.shutil.which", return_value="/usr/bin/sinfo"),
            patch("mdfactory.performance.cluster._discover_partitions", return_value=partitions),
            patch("mdfactory.performance.cluster._discover_accounts", return_value=accounts),
            patch("mdfactory.performance.cluster._discover_qos", return_value=qos),
            patch(
                "mdfactory.performance.cluster._discover_default_account", return_value="myproject"
            ),
        ):
            result = discover_cluster()

        assert result is not None
        assert isinstance(result, ClusterInfo)
        assert len(result.partitions) == 3
        assert len(result.accounts) == 3
        assert len(result.qos_policies) == 3
        assert result.default_account == "myproject"

    def test_default_account_falls_back_to_first(self):
        """When default account query fails, fall back to first account."""
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        accounts = _parse_accounts(SACCTMGR_ACCOUNTS)
        qos = _parse_qos(SACCTMGR_QOS)

        with (
            patch("mdfactory.performance.cluster.shutil.which", return_value="/usr/bin/sinfo"),
            patch("mdfactory.performance.cluster._discover_partitions", return_value=partitions),
            patch("mdfactory.performance.cluster._discover_accounts", return_value=accounts),
            patch("mdfactory.performance.cluster._discover_qos", return_value=qos),
            patch("mdfactory.performance.cluster._discover_default_account", return_value=None),
        ):
            result = discover_cluster()

        assert result is not None
        # Falls back to first alphabetical account
        assert result.default_account == "default-account"

    def test_graceful_without_sacctmgr(self):
        """When sacctmgr fails, still return partitions."""
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)

        with (
            patch("mdfactory.performance.cluster.shutil.which", return_value="/usr/bin/sinfo"),
            patch("mdfactory.performance.cluster._discover_partitions", return_value=partitions),
            patch("mdfactory.performance.cluster._discover_accounts", return_value=None),
            patch("mdfactory.performance.cluster._discover_qos", return_value=None),
            patch("mdfactory.performance.cluster._discover_default_account", return_value=None),
        ):
            result = discover_cluster()

        assert result is not None
        assert len(result.partitions) == 3
        assert result.accounts == []
        assert result.qos_policies == []
        assert result.default_account is None

    def test_caching(self):
        """Second call returns cached result without re-querying."""
        partitions = _parse_sinfo(SINFO_OUTPUT_SINGLE_PARTITION)

        with (
            patch("mdfactory.performance.cluster.shutil.which", return_value="/usr/bin/sinfo"),
            patch(
                "mdfactory.performance.cluster._discover_partitions", return_value=partitions
            ) as mock_partitions,
            patch("mdfactory.performance.cluster._discover_accounts", return_value=None),
            patch("mdfactory.performance.cluster._discover_qos", return_value=None),
            patch("mdfactory.performance.cluster._discover_default_account", return_value=None),
        ):
            result1 = discover_cluster()
            result2 = discover_cluster()

        assert result1 is result2
        # _discover_partitions called only once due to caching
        assert mock_partitions.call_count == 1

    def test_returns_none_when_sinfo_fails(self):
        """If sinfo exists but returns error, return None."""
        with (
            patch("mdfactory.performance.cluster.shutil.which", return_value="/usr/bin/sinfo"),
            patch("mdfactory.performance.cluster._discover_partitions", return_value=None),
        ):
            result = discover_cluster()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: select_partition
# ---------------------------------------------------------------------------


class TestSelectPartition:
    """Test heuristic partition selection."""

    @pytest.fixture()
    def cluster(self) -> ClusterInfo:
        """Build a ClusterInfo from the mixed sinfo output."""
        partitions = _parse_sinfo(SINFO_OUTPUT_MIXED)
        return ClusterInfo(
            partitions=partitions,
            accounts=["myproject"],
            qos_policies=["normal"],
            default_account="myproject",
        )

    def test_select_default_cpu_partition(self, cluster: ClusterInfo):
        result = select_partition(cluster)
        assert result is not None
        assert result.name == "cpu"

    def test_select_gpu_partition(self, cluster: ClusterInfo):
        result = select_partition(cluster, needs_gpu=True)
        assert result is not None
        assert result.name == "gpu"

    def test_select_with_high_cpu_requirement(self, cluster: ClusterInfo):
        # Need 200+ CPUs → only bigmem (256) qualifies
        result = select_partition(cluster, min_cpus=200)
        assert result is not None
        assert result.name == "bigmem"

    def test_select_with_high_memory_requirement(self, cluster: ClusterInfo):
        # Need 1TB+ → only bigmem qualifies
        result = select_partition(cluster, min_mem_gb=1500)
        assert result is not None
        assert result.name == "bigmem"

    def test_returns_none_when_impossible(self, cluster: ClusterInfo):
        # Need 1000 CPUs — nobody has that
        result = select_partition(cluster, min_cpus=1000)
        assert result is None

    def test_returns_none_gpu_when_no_gpu_partition(self):
        partitions = _parse_sinfo(SINFO_OUTPUT_SINGLE_PARTITION)
        cluster = ClusterInfo(partitions=partitions)
        result = select_partition(cluster, needs_gpu=True)
        assert result is None

    def test_prefers_default_partition(self, cluster: ClusterInfo):
        # Both cpu and bigmem meet min_cpus=1 — prefer cpu (default)
        result = select_partition(cluster, min_cpus=1)
        assert result is not None
        assert result.name == "cpu"

    def test_gpu_with_min_cpus(self, cluster: ClusterInfo):
        # Need GPU + 90 CPUs → only h100 node qualifies (96 cpus)
        result = select_partition(cluster, needs_gpu=True, min_cpus=90)
        assert result is not None
        assert result.name == "gpu"

    def test_skips_down_partitions(self):
        """Partitions where all nodes are down are not selectable."""
        partitions = _parse_sinfo(SINFO_OUTPUT_ALL_DOWN)
        cluster = ClusterInfo(partitions=partitions)
        result = select_partition(cluster)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: dataclass properties
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Test dataclass construction and immutability."""

    def test_node_type_frozen(self):
        from pydantic import ValidationError

        nt = NodeType(cpus=64, memory_mb=256000, gpu_specs=((4, "a100"),), count=1)
        with pytest.raises(ValidationError):
            nt.cpus = 128  # type: ignore[misc]

    def test_partition_frozen(self):
        from pydantic import ValidationError

        p = Partition(name="test", state="up", max_time="1-00:00:00", default_time="1:00:00")
        with pytest.raises(ValidationError):
            p.name = "other"  # type: ignore[misc]

    def test_cluster_info_frozen(self):
        from pydantic import ValidationError

        ci = ClusterInfo()
        with pytest.raises(ValidationError):
            ci.default_account = "hack"  # type: ignore[misc]

    def test_node_type_defaults(self):
        nt = NodeType(cpus=32, memory_mb=64000)
        assert nt.gpu_specs == ()  # No GPUs by default
        assert nt.features == ()
        assert nt.count == 1  # Default count

    def test_node_type_features_immutable(self):
        nt = NodeType(
            cpus=64, memory_mb=256000, gpu_specs=((4, "a100"),), features=("a100", "nvlink")
        )
        assert nt.features == ("a100", "nvlink")
        with pytest.raises(TypeError):
            nt.features[0] = "other"  # type: ignore[index]

    def test_cluster_info_defaults(self):
        ci = ClusterInfo()
        assert ci.partitions == []
        assert ci.accounts == []
        assert ci.qos_policies == []
        assert ci.default_account is None
