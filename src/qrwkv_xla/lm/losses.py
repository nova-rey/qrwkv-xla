from __future__ import annotations

import jax
import jax.numpy as jnp


def masked_next_token_cross_entropy(
    *,
    logits: jax.Array,
    labels: jax.Array,
    label_mask: jax.Array,
) -> jax.Array:
    log_probs = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
    label_ids = jnp.asarray(labels, dtype=jnp.int32)
    mask = jnp.asarray(label_mask).astype(log_probs.dtype)
    token_log_probs = jnp.take_along_axis(
        log_probs,
        label_ids[..., None],
        axis=-1,
    )[..., 0]
    token_nll = -token_log_probs
    numerator = jnp.sum(token_nll * mask)
    denominator = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=log_probs.dtype))
    return numerator / denominator
