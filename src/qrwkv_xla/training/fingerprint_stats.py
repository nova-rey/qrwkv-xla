from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FingerprintDistributionStats:
    entropy: jax.Array
    top1_margin: jax.Array
    top8_mass: jax.Array
    top32_mass: jax.Array
    tail_mass: jax.Array


def compute_fingerprint_distribution_stats(
    logits: jax.Array,
    *,
    topk_values: tuple[int, int] = (8, 32),
    eps: float = 1e-8,
) -> FingerprintDistributionStats:
    """Compute behavioral-fingerprint statistics from [batch, vocab] logits.

    Entropy uses natural-log units. Top-1 margin is probability-space
    `p_top1 - p_top2`. Tail mass is probability mass outside top-32.
    """
    logits = jnp.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2 [batch, vocab], got {logits.shape}")
    if len(topk_values) != 2:
        raise ValueError("topk_values must contain exactly two Python integers")
    top8_k, top32_k = topk_values
    if top8_k <= 0 or top32_k <= 0:
        raise ValueError(f"topk_values must be positive, got {topk_values!r}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)

    sorted_probs = jnp.flip(jnp.sort(probs, axis=-1), axis=-1)
    vocab_size = logits.shape[-1]
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1] if vocab_size >= 2 else jnp.zeros_like(top1)
    top1_margin = top1 - top2

    effective_top8 = min(top8_k, vocab_size)
    effective_top32 = min(top32_k, vocab_size)
    top8_mass = jnp.sum(sorted_probs[:, :effective_top8], axis=-1)
    top32_mass = jnp.sum(sorted_probs[:, :effective_top32], axis=-1)
    tail_mass = jnp.maximum(
        1.0 - top32_mass,
        jnp.asarray(0.0, dtype=top32_mass.dtype),
    )

    return FingerprintDistributionStats(
        entropy=entropy,
        top1_margin=top1_margin,
        top8_mass=top8_mass,
        top32_mass=top32_mass,
        tail_mass=tail_mass,
    )


def select_position_logits(logits: jax.Array, positions: jax.Array) -> jax.Array:
    logits = jnp.asarray(logits)
    positions = jnp.asarray(positions)
    if logits.ndim != 3:
        raise ValueError(
            f"logits must be rank 3 [batch, seq, vocab], got {logits.shape}"
        )
    if positions.ndim != 1:
        raise ValueError(f"positions must be rank 1 [batch], got {positions.shape}")
    if positions.shape[0] != logits.shape[0]:
        raise ValueError(
            "positions length must equal batch size: "
            f"positions={positions.shape[0]} batch={logits.shape[0]}"
        )
    batch_indices = jnp.arange(logits.shape[0])
    return logits[batch_indices, positions, :]


def compute_fingerprint_distribution_stats_at_positions(
    logits: jax.Array,
    positions: jax.Array,
    *,
    topk_values: tuple[int, int] = (8, 32),
    eps: float = 1e-8,
) -> FingerprintDistributionStats:
    selected_logits = select_position_logits(logits, positions)
    return compute_fingerprint_distribution_stats(
        selected_logits,
        topk_values=topk_values,
        eps=eps,
    )
