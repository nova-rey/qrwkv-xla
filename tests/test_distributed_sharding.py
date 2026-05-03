from __future__ import annotations

import jax.numpy as jnp
import pytest

from qrwkv_xla.distributed.sharding import (
    can_shard_batch,
    shard_array_for_devices,
    shard_batch_for_devices,
    unshard_first_device,
)


def test_can_shard_batch_checks_divisibility() -> None:
    assert can_shard_batch(4, 2) is True
    assert can_shard_batch(3, 2) is False
    assert can_shard_batch(1, 2) is False


def test_shard_array_for_devices_reshapes_leading_axis() -> None:
    array = jnp.arange(8).reshape(4, 2)
    sharded = shard_array_for_devices(array, device_count=2)

    assert sharded.shape == (2, 2, 2)
    assert sharded[0].tolist() == [[0, 1], [2, 3]]
    assert sharded[1].tolist() == [[4, 5], [6, 7]]


def test_shard_array_for_devices_requires_divisible_batch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        shard_array_for_devices(jnp.arange(6).reshape(3, 2), device_count=2)


def test_shard_batch_for_devices_maps_nested_pytrees() -> None:
    batch = {
        "input_ids": jnp.arange(8).reshape(4, 2),
        "nested": {"mask": jnp.ones((4, 2), dtype=jnp.int32)},
    }

    sharded = shard_batch_for_devices(batch, device_count=2)

    assert sharded["input_ids"].shape == (2, 2, 2)
    assert sharded["nested"]["mask"].shape == (2, 2, 2)


def test_unshard_first_device_returns_first_slice() -> None:
    sharded = {"input_ids": jnp.arange(8).reshape(2, 2, 2)}
    unsharded = unshard_first_device(sharded)
    assert unsharded["input_ids"].shape == (2, 2)
    assert unsharded["input_ids"].tolist() == [[0, 1], [2, 3]]


def test_shard_batch_rejects_non_array_metadata() -> None:
    with pytest.raises(TypeError, match="array leaves"):
        shard_batch_for_devices(
            {"input_ids": jnp.ones((2, 2)), "meta": "nope"}, device_count=2
        )
