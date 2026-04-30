from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.students import RWKV7ReferenceConfig, rwkv7_reference_layer


def test_rwkv7_reference_config_rejects_invalid_values() -> None:
    invalid_cases = [
        {"vocab_size": 0},
        {"hidden_size": 0},
        {"num_layers": 0},
        {"init_scale": 0.0},
    ]

    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            RWKV7ReferenceConfig(**kwargs)


def test_rwkv7_reference_layer_output_shape() -> None:
    inputs = jnp.ones((2, 4, 3), dtype=jnp.float32)

    output = rwkv7_reference_layer(inputs, **_layer_params(3))

    assert output.shape == (2, 4, 3)


def test_rwkv7_reference_layer_is_deterministic_for_same_input_and_params() -> None:
    inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10.0
    params = _layer_params(3)

    first = rwkv7_reference_layer(inputs, **params)
    second = rwkv7_reference_layer(inputs, **params)

    np.testing.assert_allclose(np.asarray(first), np.asarray(second))


def test_rwkv7_reference_layer_output_changes_when_input_changes() -> None:
    inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10.0
    changed_inputs = inputs.at[:, 1, :].add(0.5)

    output = rwkv7_reference_layer(inputs, **_layer_params(3))
    changed_output = rwkv7_reference_layer(changed_inputs, **_layer_params(3))

    assert not np.allclose(np.asarray(output), np.asarray(changed_output))


def test_jitted_rwkv7_reference_layer_runs() -> None:
    inputs = jnp.ones((2, 4, 3), dtype=jnp.float32)
    params = _layer_params(3)
    run_layer = jax.jit(lambda x: rwkv7_reference_layer(x, **params))

    output = run_layer(inputs)

    assert output.shape == (2, 4, 3)
    assert jnp.all(jnp.isfinite(output))


def _layer_params(hidden_size: int) -> dict[str, jax.Array]:
    base = jnp.eye(hidden_size, dtype=jnp.float32)
    return {
        "wr": base * 0.2,
        "wk": base * 0.3,
        "wv": base * 0.4,
        "wg": base * 0.5,
        "wo": base * 0.6,
        "time_decay": jnp.linspace(-0.2, 0.2, hidden_size, dtype=jnp.float32),
        "time_bias": jnp.linspace(0.1, 0.3, hidden_size, dtype=jnp.float32),
    }
