from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.artifacts import FingerprintExemplarBatch
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
