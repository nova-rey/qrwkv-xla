from __future__ import annotations

import jax.numpy as jnp

from qrwkv_xla.distributed.replication import (
    replicate_to_devices,
    unreplicate_from_devices,
)
from qrwkv_xla.optimizers.state import OptimizerState


def test_replicate_to_devices_adds_leading_device_axis() -> None:
    params = {"w": jnp.arange(4).reshape(2, 2)}
    replicated = replicate_to_devices(params, device_count=3)

    assert replicated["w"].shape == (3, 2, 2)
    assert replicated["w"][1].tolist() == params["w"].tolist()


def test_unreplicate_from_devices_returns_first_leaf() -> None:
    replicated = {"w": jnp.arange(12).reshape(3, 2, 2)}
    params = unreplicate_from_devices(replicated)

    assert params["w"].shape == (2, 2)
    assert params["w"].tolist() == [[0, 1], [2, 3]]


def test_nested_optimizer_state_replicates_and_unreplicates() -> None:
    state = OptimizerState(
        type="adamw",
        step=jnp.asarray(2, dtype=jnp.int32),
        slots={
            "m": {"w": jnp.ones((2, 2))},
            "v": {"w": jnp.full((2, 2), 3.0)},
        },
    )

    replicated = replicate_to_devices(state, device_count=2)
    assert replicated.step.shape == (2,)
    assert replicated.slots["m"]["w"].shape == (2, 2, 2)

    restored = unreplicate_from_devices(replicated)
    assert restored.type == "adamw"
    assert int(restored.step) == 2
    assert restored.slots["v"]["w"].tolist() == [[3.0, 3.0], [3.0, 3.0]]
