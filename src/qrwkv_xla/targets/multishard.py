from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp

from qrwkv_xla.targets.consumption import (
    OfflineTargetBatch,
    TeacherTargetBatch,
    load_offline_target_batch,
    load_teacher_target_batch,
    mse_logits_loss,
)
from qrwkv_xla.targets.store import TeacherTargetStore

MULTISHARD_TARGET_STORE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "dataset_pipeline_ready",
    "production_storage_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
)


@dataclass(frozen=True)
class MultiShardTargetStoreSmokeResult:
    status: str
    shard_count: int
    examples_seen: int
    losses: tuple[float, ...]
    aggregate_loss: float
    loss_finite: bool
    canonical_layout: str = "metadata.json + shards/shard-XXXXX.npz"
    claims_not_made: tuple[str, ...] = MULTISHARD_TARGET_STORE_CLAIMS_NOT_MADE
    phase: str = "P106"
    scope: str = "multi_shard_target_store_smoke"

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def iter_target_store_shard_ids(store: TeacherTargetStore) -> tuple[int, ...]:
    store.validate()
    return tuple(range(store.metadata.shard_count))


def iter_offline_target_batches(
    store: TeacherTargetStore,
) -> tuple[OfflineTargetBatch, ...]:
    return tuple(
        load_offline_target_batch(store, shard_id=shard_id)
        for shard_id in iter_target_store_shard_ids(store)
    )


def iter_teacher_target_batches(
    store: TeacherTargetStore,
) -> tuple[TeacherTargetBatch, ...]:
    return tuple(
        load_teacher_target_batch(store, shard_id=shard_id)
        for shard_id in iter_target_store_shard_ids(store)
    )


def run_multishard_target_store_smoke(
    store: TeacherTargetStore,
) -> MultiShardTargetStoreSmokeResult:
    batches = iter_offline_target_batches(store)
    losses = tuple(
        float(mse_logits_loss(batch.teacher_logits, batch.teacher_logits))
        for batch in batches
    )
    examples_seen = sum(int(batch.input_ids.shape[0]) for batch in batches)
    aggregate_loss = float(jnp.mean(jnp.asarray(losses))) if losses else float("nan")
    loss_finite = bool(jnp.isfinite(jnp.asarray(losses)).all())
    return MultiShardTargetStoreSmokeResult(
        status="pass" if loss_finite else "fail",
        shard_count=store.metadata.shard_count,
        examples_seen=examples_seen,
        losses=losses,
        aggregate_loss=aggregate_loss,
        loss_finite=loss_finite,
    )
