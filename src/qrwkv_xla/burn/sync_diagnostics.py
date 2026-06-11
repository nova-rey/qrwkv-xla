from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SyncReadiness:
    distributed_training_ready: bool
    missing_predicates: tuple[str, ...]


@dataclass(frozen=True)
class CollectiveSyncProbe:
    enabled: bool
    method: str | None
    local_value: float | None
    reduced_value: float | None
    expected_value: float | None
    verified: bool
    error: str | None = None

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChecksumSyncVerification:
    method: str
    local: float
    global_min: float | None
    global_max: float | None
    verified: bool
    error: str | None = None


def evaluate_distributed_training_readiness(
    *,
    distributed_example_sharding_verified: bool,
    collective_sync_probe_verified: bool,
    gradient_sync_enabled: bool,
    gradient_sync_verified: bool,
    parameter_sync_verified: bool,
    optimizer_state_sync_verified: bool,
    loss_is_global: bool,
    checkpoint_fingerprint_match: bool,
    checkpoint_fingerprint_match_required: bool = True,
    process_count: int = 2,
) -> SyncReadiness:
    predicates = {
        "jax_process_count_gt_1": process_count > 1,
        "distributed_example_sharding_verified": distributed_example_sharding_verified,
        "collective_sync_probe_verified": collective_sync_probe_verified,
        "gradient_sync_enabled": gradient_sync_enabled,
        "gradient_sync_verified": gradient_sync_verified,
        "parameter_sync_verified": parameter_sync_verified,
        "optimizer_state_sync_verified": optimizer_state_sync_verified,
        "loss_is_global": loss_is_global,
    }
    if checkpoint_fingerprint_match_required:
        predicates["checkpoint_fingerprint_match"] = checkpoint_fingerprint_match
    missing = tuple(name for name, value in predicates.items() if not value)
    return SyncReadiness(
        distributed_training_ready=not missing,
        missing_predicates=missing,
    )


def run_collective_sync_probe(
    *,
    process_index: int,
    process_count: int,
    enabled: bool,
) -> CollectiveSyncProbe:
    local_value = float(process_index + 1)
    expected = float(process_count * (process_count + 1) / 2)
    if not enabled:
        return CollectiveSyncProbe(
            enabled=False,
            method=None,
            local_value=local_value,
            reduced_value=None,
            expected_value=expected,
            verified=False,
        )
    try:
        gathered = _process_allgather(
            jnp.asarray(local_value, dtype=jnp.float32),
            process_count=process_count,
        )
        reduced = float(np.sum(np.asarray(gathered, dtype=np.float32)))
        return CollectiveSyncProbe(
            enabled=True,
            method="jax.experimental.multihost_utils.process_allgather_sum",
            local_value=local_value,
            reduced_value=reduced,
            expected_value=expected,
            verified=bool(np.isclose(reduced, expected)),
        )
    except Exception as exc:
        return CollectiveSyncProbe(
            enabled=True,
            method="jax.experimental.multihost_utils.process_allgather_sum",
            local_value=local_value,
            reduced_value=None,
            expected_value=expected,
            verified=False,
            error=str(exc),
        )


def average_pytree_across_processes(
    tree: Any,
    *,
    process_count: int,
) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: _mean_array_across_processes(leaf, process_count=process_count),
        tree,
    )


def global_mean_scalar(
    value: Any,
    *,
    process_count: int,
) -> jax.Array:
    return _mean_array_across_processes(value, process_count=process_count)


def pytree_numeric_checksum(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        total = total + jnp.sum(arr)
    return float(total)


def verify_checksum_sync_across_processes(
    local_checksum: float,
    *,
    process_count: int,
) -> ChecksumSyncVerification:
    method = "jax.experimental.multihost_utils.process_allgather_minmax"
    try:
        gathered = _process_allgather(
            jnp.asarray(local_checksum, dtype=jnp.float32),
            process_count=process_count,
        )
        values = np.asarray(gathered, dtype=np.float32).reshape((-1,))
        global_min = float(np.min(values))
        global_max = float(np.max(values))
        return ChecksumSyncVerification(
            method=method,
            local=float(local_checksum),
            global_min=global_min,
            global_max=global_max,
            verified=bool(np.isclose(global_min, global_max)),
        )
    except Exception as exc:
        return ChecksumSyncVerification(
            method=method,
            local=float(local_checksum),
            global_min=None,
            global_max=None,
            verified=False,
            error=str(exc),
        )


def _mean_array_across_processes(
    leaf: Any,
    *,
    process_count: int,
) -> jax.Array:
    gathered = _process_allgather(jnp.asarray(leaf), process_count=process_count)
    return jnp.mean(jnp.asarray(gathered), axis=0)


def _process_allgather(value: jax.Array, *, process_count: int) -> jax.Array:
    from jax.experimental import multihost_utils

    gathered = multihost_utils.process_allgather(value, tiled=False)
    arr = jnp.asarray(gathered)
    if int(arr.shape[0]) != int(process_count):
        raise RuntimeError(
            "process_allgather did not return one value per JAX process: "
            f"expected {process_count}, got shape {arr.shape}"
        )
    return arr
