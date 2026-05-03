from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def replicate_to_devices(tree: Any, *, device_count: int) -> Any:
    if device_count < 1:
        raise ValueError(f"device_count must be >= 1, got {device_count}")
    return jax.tree_util.tree_map(
        lambda leaf: jnp.stack([jnp.asarray(leaf)] * device_count),
        tree,
    )


def unreplicate_from_devices(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[0], tree)
