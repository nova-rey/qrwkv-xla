from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.artifacts import FingerprintExemplarBatch
from qrwkv_xla.distill.losses import cascaded_soft_labels_loss
from qrwkv_xla.training.fingerprint_stats import select_position_logits


@dataclass(frozen=True)
class FingerprintExemplarLossConfig:
    use_record_weights: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class FingerprintExemplarLossOutput:
    loss: jax.Array
    kl_loss: jax.Array
    cross_entropy: jax.Array
    entropy: jax.Array
    mean_weight: jax.Array


def compute_fingerprint_exemplar_loss(
    student_logits: jax.Array,
    batch: FingerprintExemplarBatch,
    config: FingerprintExemplarLossConfig | None = None,
) -> FingerprintExemplarLossOutput:
    cfg = config or FingerprintExemplarLossConfig()
    if cfg.eps <= 0.0:
        raise ValueError(f"eps must be positive, got {cfg.eps}")

    student_logits = jnp.asarray(student_logits)
    if batch.target_type == "cascaded_soft_labels_v1":
        return _compute_cascaded_loss(student_logits, batch, cfg)
    if batch.teacher_probs is None:
        raise ValueError("dense exemplar batch is missing teacher_probs")
    teacher_probs = jnp.asarray(batch.teacher_probs, dtype=jnp.float32)
    batch_size = _validate_shapes(student_logits, teacher_probs)
    weights = (
        jnp.asarray(batch.weight, dtype=jnp.float32)
        if cfg.use_record_weights
        else jnp.ones((batch_size,), dtype=jnp.float32)
    )
    if weights.ndim != 1 or weights.shape[0] != batch_size:
        raise ValueError(
            "batch.weight must be rank 1 with batch dimension "
            f"{batch_size}, got {weights.shape}"
        )

    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    cross_entropy_per_record = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
    entropy_per_record = -jnp.sum(
        teacher_probs * jnp.log(teacher_probs + cfg.eps),
        axis=-1,
    )
    kl_per_record = cross_entropy_per_record - entropy_per_record

    cross_entropy = _weighted_mean(cross_entropy_per_record, weights, cfg.eps)
    entropy = _weighted_mean(entropy_per_record, weights, cfg.eps)
    kl_loss = _weighted_mean(kl_per_record, weights, cfg.eps)
    return FingerprintExemplarLossOutput(
        loss=kl_loss,
        kl_loss=kl_loss,
        cross_entropy=cross_entropy,
        entropy=entropy,
        mean_weight=jnp.mean(weights),
    )


def _compute_cascaded_loss(
    student_logits: jax.Array,
    batch: FingerprintExemplarBatch,
    config: FingerprintExemplarLossConfig,
) -> FingerprintExemplarLossOutput:
    required = (
        "top_token_ids",
        "top_log_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
        "bucket_edges",
        "bucket_mass",
    )
    missing = [name for name in required if getattr(batch, name) is None]
    if missing:
        raise ValueError(f"cascaded exemplar batch missing fields: {missing}")
    student = jnp.asarray(student_logits, dtype=jnp.float32)
    if student.ndim != 2:
        raise ValueError("compressed exemplar student_logits must be [batch, vocab]")
    weights = (
        jnp.asarray(batch.weight, dtype=jnp.float32)
        if config.use_record_weights
        else jnp.ones((student.shape[0],), dtype=jnp.float32)
    )

    def one(
        student_row,
        token_ids,
        top_log_probs,
        top_mass,
        tail_mass,
        entropy,
        buckets,
    ):
        report = cascaded_soft_labels_loss(
            student_row[None, None, :],
            token_ids[None, None, :],
            top_log_probs[None, None, :],
            jnp.ones((1, 1), dtype=jnp.float32),
            tail_mass=tail_mass[None, None],
            bucket_mass=buckets[None, None, :],
            bucket_edges=jnp.asarray(batch.bucket_edges, dtype=jnp.float32),
            top_mass=top_mass[None, None],
            teacher_entropy=entropy[None, None],
        )
        return report.total_loss, report.head_loss

    total, head = jax.vmap(one)(
        student,
        jnp.asarray(batch.top_token_ids, dtype=jnp.int32),
        jnp.asarray(batch.top_log_probs, dtype=jnp.float32),
        jnp.asarray(batch.top_mass, dtype=jnp.float32),
        jnp.asarray(batch.tail_mass, dtype=jnp.float32),
        jnp.asarray(batch.teacher_entropy, dtype=jnp.float32),
        jnp.asarray(batch.bucket_mass, dtype=jnp.float32),
    )
    entropy_values = jnp.asarray(batch.teacher_entropy, dtype=jnp.float32)
    loss = _weighted_mean(total, weights, config.eps)
    return FingerprintExemplarLossOutput(
        loss=loss,
        kl_loss=loss,
        cross_entropy=_weighted_mean(head, weights, config.eps),
        entropy=_weighted_mean(entropy_values, weights, config.eps),
        mean_weight=jnp.mean(weights),
    )


def compute_fingerprint_exemplar_loss_at_positions(
    logits: jax.Array,
    batch: FingerprintExemplarBatch,
    config: FingerprintExemplarLossConfig | None = None,
) -> FingerprintExemplarLossOutput:
    selected_logits = select_position_logits(logits, batch.position)
    return compute_fingerprint_exemplar_loss(selected_logits, batch, config)


def _validate_shapes(student_logits: jax.Array, teacher_probs: jax.Array) -> int:
    if student_logits.ndim != 2:
        raise ValueError(
            f"student_logits must be rank 2 [batch, vocab], got {student_logits.shape}"
        )
    if teacher_probs.ndim != 2:
        raise ValueError(
            "batch.teacher_probs must be rank 2 [batch, vocab], "
            f"got {teacher_probs.shape}"
        )
    if student_logits.shape != teacher_probs.shape:
        raise ValueError(
            "student_logits and batch.teacher_probs shape mismatch: "
            f"{student_logits.shape} vs {teacher_probs.shape}"
        )
    return int(student_logits.shape[0])


def _weighted_mean(values: jax.Array, weights: jax.Array, eps: float) -> jax.Array:
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), eps)
