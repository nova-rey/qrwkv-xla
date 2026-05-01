from __future__ import annotations

import jax
import jax.numpy as jnp


def init_lm_head_params(
    key: jax.Array,
    *,
    hidden_size: int,
    vocab_size: int,
    init_scale: float = 0.02,
) -> dict[str, jax.Array]:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    if init_scale <= 0.0:
        raise ValueError(f"init_scale must be > 0, got {init_scale}")
    return {
        "weight": jax.random.normal(key, (hidden_size, vocab_size)) * init_scale,
        "bias": jnp.zeros((vocab_size,), dtype=jnp.float32),
    }


def apply_lm_head(hidden: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    hidden_array = jnp.asarray(hidden)
    if hidden_array.ndim != 3:
        raise ValueError(f"hidden must have shape [B,S,H], got {hidden_array.shape}")
    weight = jnp.asarray(params["weight"])
    bias = jnp.asarray(params["bias"])
    if weight.ndim != 2:
        raise ValueError(f"lm_head weight must have shape [H,V], got {weight.shape}")
    if bias.ndim != 1:
        raise ValueError(f"lm_head bias must have shape [V], got {bias.shape}")
    if hidden_array.shape[-1] != weight.shape[0]:
        raise ValueError(
            "hidden size and lm_head weight shape mismatch: "
            f"{hidden_array.shape[-1]} != {weight.shape[0]}"
        )
    if bias.shape[0] != weight.shape[1]:
        raise ValueError(
            "lm_head bias and weight vocab shape mismatch: "
            f"{bias.shape[0]} != {weight.shape[1]}"
        )
    return jnp.einsum("bsh,hv->bsv", hidden_array, weight) + bias


def apply_tied_lm_head(
    hidden: jax.Array,
    embedding: jax.Array,
    bias: jax.Array | None = None,
) -> jax.Array:
    hidden_array = jnp.asarray(hidden)
    embedding_array = jnp.asarray(embedding)
    if hidden_array.ndim != 3:
        raise ValueError(f"hidden must have shape [B,S,H], got {hidden_array.shape}")
    if embedding_array.ndim != 2:
        raise ValueError(
            f"embedding must have shape [V,H], got {embedding_array.shape}"
        )
    if hidden_array.shape[-1] != embedding_array.shape[-1]:
        raise ValueError(
            "hidden size and embedding shape mismatch: "
            f"{hidden_array.shape[-1]} != {embedding_array.shape[-1]}"
        )
    logits = jnp.einsum("bsh,vh->bsv", hidden_array, embedding_array)
    if bias is not None:
        bias_array = jnp.asarray(bias)
        if bias_array.shape != (embedding_array.shape[0],):
            raise ValueError(
                "tied lm_head bias must have shape "
                f"{(embedding_array.shape[0],)}, got {bias_array.shape}"
            )
        logits = logits + bias_array
    return logits
