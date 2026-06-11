from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
) -> SyncReadiness:
    predicates = {
        "distributed_example_sharding_verified": distributed_example_sharding_verified,
        "collective_sync_probe_verified": collective_sync_probe_verified,
        "gradient_sync_enabled": gradient_sync_enabled,
        "gradient_sync_verified": gradient_sync_verified,
        "parameter_sync_verified": parameter_sync_verified,
        "optimizer_state_sync_verified": optimizer_state_sync_verified,
        "loss_is_global": loss_is_global,
        "checkpoint_fingerprint_match": checkpoint_fingerprint_match,
    }
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
        from jax.experimental import multihost_utils

        gathered = multihost_utils.process_allgather(
            jnp.asarray(local_value, dtype=jnp.float32),
            tiled=False,
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
