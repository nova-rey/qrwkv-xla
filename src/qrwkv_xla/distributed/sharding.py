from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def can_shard_batch(batch_size: int, device_count: int) -> bool:
    if device_count < 1:
        return False
    return batch_size >= device_count and batch_size % device_count == 0


def shard_array_for_devices(array: jax.Array, *, device_count: int) -> jax.Array:
    if device_count < 1:
        raise ValueError(f"device_count must be >= 1, got {device_count}")
    value = jnp.asarray(array)
    if value.ndim == 0:
        raise ValueError("cannot shard scalar array; batch axis 0 is required")
    batch_size = int(value.shape[0])
    if not can_shard_batch(batch_size, device_count):
        raise ValueError(
            "batch size must be divisible by device_count and at least one item "
            f"per device, got batch_size={batch_size}, device_count={device_count}"
        )
    per_device = batch_size // device_count
    return jnp.reshape(value, (device_count, per_device, *value.shape[1:]))


def shard_batch_for_devices(batch: Any, *, device_count: int) -> Any:
    def shard_leaf(leaf: Any) -> jax.Array:
        if _is_array_like(leaf):
            return shard_array_for_devices(leaf, device_count=device_count)
        raise TypeError(
            "distributed batch pytrees may only contain array leaves; "
            f"got {type(leaf).__name__}"
        )

    return jax.tree_util.tree_map(shard_leaf, batch)


def unshard_first_device(value: Any) -> Any:
    def first_leaf(leaf: Any) -> Any:
        if _is_array_like(leaf):
            array = jnp.asarray(leaf)
            if array.ndim == 0:
                raise ValueError("cannot unshard scalar array without device axis")
            return array[0]
        raise TypeError(
            "distributed pytrees may only contain array leaves; "
            f"got {type(leaf).__name__}"
        )

    return jax.tree_util.tree_map(first_leaf, value)


def _is_array_like(value: Any) -> bool:
    return isinstance(value, (jax.Array, np.ndarray))
