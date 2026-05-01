from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GradientClipResult:
    gradients: Any
    global_norm: jax.Array
    clipped_global_norm: jax.Array
    clip_scale: jax.Array
    was_clipped: jax.Array


def global_gradient_norm(gradients: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(gradients)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)

    sum_sq = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            continue
        sum_sq = sum_sq + jnp.sum(jnp.square(array))
    return jnp.sqrt(sum_sq)


def clip_gradients_by_global_norm(
    gradients: Any,
    *,
    max_grad_norm: float | None,
    epsilon: float = 1e-6,
) -> GradientClipResult:
    norm = global_gradient_norm(gradients)
    if max_grad_norm is None or max_grad_norm <= 0:
        return GradientClipResult(
            gradients=gradients,
            global_norm=norm,
            clipped_global_norm=norm,
            clip_scale=jnp.asarray(1.0, dtype=norm.dtype),
            was_clipped=jnp.asarray(False),
        )

    clip_scale = jnp.minimum(
        jnp.asarray(1.0, dtype=norm.dtype),
        jnp.asarray(max_grad_norm, dtype=norm.dtype) / (norm + epsilon),
    )

    def clip_leaf(leaf: Any) -> Any:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return leaf
        return leaf * clip_scale

    clipped = jax.tree_util.tree_map(clip_leaf, gradients)
    clipped_norm = global_gradient_norm(clipped)
    return GradientClipResult(
        gradients=clipped,
        global_norm=norm,
        clipped_global_norm=clipped_norm,
        clip_scale=clip_scale,
        was_clipped=clip_scale < 1.0,
    )
