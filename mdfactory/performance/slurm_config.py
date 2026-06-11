# ABOUTME: SLURM configuration models shared across all submission backends.
# ABOUTME: BaseSlurmConfig with 3-tier autodiscovery; SlurmConfig for submitit analysis jobs.
"""SLURM configuration models for all submission backends.

Provides a shared ``BaseSlurmConfig`` Pydantic model with four cross-cutting
SLURM fields (``account``, ``partition``, ``qos``, ``constraint``) and a single
authoritative ``from_cluster()`` factory that implements three-tier precedence:

1. Explicit keyword arguments passed by the caller
2. ``[slurm]`` section in ``config.ini``
3. Live ``sinfo`` / ``sacctmgr`` autodiscovery via ``mdfactory.performance.cluster``

Subclasses add backend-specific fields and inherit ``from_cluster()`` for free.

Classes
-------
BaseSlurmConfig
    Abstract base for all SLURM-facing config objects.
SlurmConfig
    Configuration for submitit-based analysis submission.

Functions
---------
normalize_slurm_time
    Convert human-friendly time strings (``"2h"``, ``"30m"``) to
    ``HH:MM:SS`` / ``D-HH:MM:SS`` format accepted by SLURM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_slurm_time(value: str) -> str:
    """Normalize SLURM time strings to ``HH:MM:SS`` or ``D-HH:MM:SS`` format.

    Accepts human-friendly shorthand as well as all canonical SLURM formats.

    Parameters
    ----------
    value : str
        Raw time string, e.g. ``"2h"``, ``"30m"``, ``"90"`` (minutes),
        ``"1d"``, ``"01:00:00"``, ``"3-00:00:00"``.

    Returns
    -------
    str
        Normalized time string.  Strings that already contain ``:`` are
        returned unchanged (pass-through for HH:MM:SS / D-HH:MM:SS).

    Examples
    --------
    >>> normalize_slurm_time("2h")
    '02:00:00'
    >>> normalize_slurm_time("30m")
    '00:30:00'
    >>> normalize_slurm_time("90")
    '01:30:00'
    >>> normalize_slurm_time("1d")
    '1-00:00:00'
    >>> normalize_slurm_time("01:00:00")
    '01:00:00'
    """
    raw = value.strip()
    if ":" in raw:
        return raw
    lowered = raw.lower()
    if lowered.endswith("d"):
        days = int(lowered[:-1])
        return f"{days}-00:00:00"
    if lowered.endswith("h"):
        hours = int(lowered[:-1])
        return f"{hours:02d}:00:00"
    if lowered.endswith("m"):
        minutes = int(lowered[:-1])
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:00"
    if lowered.isdigit():
        minutes = int(lowered)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:00"
    return raw


class BaseSlurmConfig(BaseModel):
    """SLURM resource specification shared by all submission backends.

    Owns the four cross-cutting SLURM fields and the single authoritative
    ``from_cluster()`` factory method.  Subclasses add backend-specific
    fields (e.g. ``cpus_per_task`` for submitit, ``cpus_per_node`` for Parsl)
    and inherit ``from_cluster()`` without writing any additional code.

    Parameters
    ----------
    account : str
        SLURM account string (``--account``).
    partition : str
        SLURM partition name (``--partition``).  Defaults to ``"cpu"``.
    qos : str or None
        Quality-of-service policy name (``--qos``).  ``None`` lets SLURM
        apply its own default.
    constraint : str or None
        Node feature constraint string (``--constraint``).

    Notes
    -----
    This model is *frozen*: instances are immutable after construction.
    Use Pydantic's ``model_copy(update={...})`` to derive a modified copy.
    """

    model_config = ConfigDict(frozen=True)

    account: str
    partition: str = "cpu"
    qos: str | None = None
    constraint: str | None = None

    @classmethod
    def from_cluster(
        cls,
        *,
        needs_gpu: bool = False,
        min_cpus: int = 1,
        min_mem_gb: int = 1,
        **extra_fields: Any,
    ) -> "BaseSlurmConfig":
        """Create an instance with SLURM fields auto-populated from the cluster.

        Three-tier precedence (highest wins):

        1. Explicit keyword arguments passed by the caller (e.g.
           ``account="mygroup"`` or ``partition="gpu"``).
        2. ``[slurm]`` section in ``config.ini`` — read via
           ``mdfactory.settings.settings``.
        3. Live ``sinfo`` / ``sacctmgr`` autodiscovery via
           ``mdfactory.performance.cluster``.

        The lazy import of ``mdfactory.performance.cluster`` keeps this
        module importable on non-SLURM machines.

        Parameters
        ----------
        needs_gpu : bool
            Select a GPU-capable partition when ``True``.  Used both for
            partition autodiscovery and for choosing the correct
            ``PARTITION_GPU`` config key.
        min_cpus : int
            Minimum CPUs per node required (passed to ``select_partition()``).
        min_mem_gb : int
            Minimum memory per node in GB (passed to ``select_partition()``).
        **extra_fields
            Backend-specific fields forwarded to the subclass constructor.
            For ``SlurmConfig``: ``time``, ``cpus_per_task``, ``mem_gb``,
            ``job_name_prefix``.
            Base-class fields (``account``, ``partition``, ``qos``,
            ``constraint``) may also be passed here to override autodiscovery.

        Returns
        -------
        BaseSlurmConfig
            A fully initialised subclass instance (the concrete type is
            determined by ``cls``).

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
                settings.slurm_partition_gpu
                if needs_gpu
                else settings.slurm_partition_cpu
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

        # --- constraint: explicit kwarg only (no config.ini equivalent) ---
        resolved_constraint: str | None = extra_fields.pop("constraint", None)

        return cls(
            account=resolved_account,
            partition=resolved_partition,
            qos=resolved_qos,
            constraint=resolved_constraint,
            **extra_fields,
        )


class SlurmConfig(BaseSlurmConfig):
    """SLURM configuration for submitit-based analysis submission.

    Backend: submitit — one SLURM job per analysis task.

    The ``time`` field is normalised on construction via
    ``normalize_slurm_time()``, so ``"2h"``, ``"120"`` (minutes), and
    ``"02:00:00"`` are all accepted and stored as ``"02:00:00"``.

    Parameters
    ----------
    account : str
        SLURM account (inherited from ``BaseSlurmConfig``).
    partition : str
        SLURM partition (inherited).  Defaults to ``"cpu"``.
    qos : str or None
        QOS policy (inherited).
    constraint : str or None
        Node constraint (inherited).
    time : str
        Job time limit.  Maps to SLURM ``--time``.
        Accepts human-friendly strings (``"2h"``, ``"30m"``, ``"1d"``) as
        well as ``MM``, ``HH:MM:SS``, and ``D-HH:MM:SS`` formats.
    cpus_per_task : int
        CPUs allocated per task.  Maps to SLURM ``--cpus-per-task``.
        **Not** the same as ``--cpus-per-node`` used by Parsl executor configs.
    mem_gb : int
        Memory per task in gigabytes.  Maps to SLURM ``--mem``.
    job_name_prefix : str
        Prefix for SLURM job names submitted via submitit.

    Examples
    --------
    Minimal construction (explicit account):

    >>> cfg = SlurmConfig(account="mygroup")
    >>> cfg.time
    '02:00:00'

    Autodiscovery on a SLURM cluster:

    >>> cfg = SlurmConfig.from_cluster(time="4h", cpus_per_task=8, mem_gb=16)
    """

    time: str = Field(default="2h", validate_default=True)
    cpus_per_task: int = 4
    """Maps to SLURM ``--cpus-per-task`` (NOT ``--cpus-per-node``)."""
    mem_gb: int = 8
    job_name_prefix: str = "mdfactory-analysis"

    @field_validator("time", mode="before")
    @classmethod
    def _normalize_time(cls, v: str) -> str:
        """Normalize time string on construction."""
        return normalize_slurm_time(v)

    @classmethod
    def from_yaml(cls, path: Path) -> "SlurmConfig":
        """Load a ``SlurmConfig`` from a YAML file.

        Parameters
        ----------
        path : Path
            Path to a YAML file whose top-level keys match ``SlurmConfig``
            field names.

        Returns
        -------
        SlurmConfig
            Loaded and validated instance.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        pydantic.ValidationError
            If the YAML content fails validation.

        Examples
        --------
        .. code-block:: yaml

           # slurm.yaml
           account: mygroup
           partition: cpu
           time: 4h
           cpus_per_task: 8
           mem_gb: 16

        >>> cfg = SlurmConfig.from_yaml(Path("slurm.yaml"))
        >>> cfg.time
        '04:00:00'
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SLURM config YAML not found: {path}")
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data or {})
