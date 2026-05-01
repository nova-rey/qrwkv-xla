from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_add(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_mul_scalar(tree: Any, scalar: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x * scalar, tree)


def tree_square(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.square(x), tree)


def tree_sqrt(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.sqrt, tree)
