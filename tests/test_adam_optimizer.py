from __future__ import annotations

import numpy as np

from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update


def test_adam_first_step_uses_bias_correction() -> None:
    params = {"w": np.array([1.0, -2.0], dtype=np.float32)}
    grads = {"w": np.array([0.5, -0.25], dtype=np.float32)}
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    state = init_optimizer_state(params, config)

    new_params, new_state, _ = optimizer_update(params, grads, state, config)

    np.testing.assert_allclose(new_params["w"], np.array([0.9, -1.9]), rtol=1e-6)
    np.testing.assert_allclose(new_state.slots["m"]["w"], np.array([0.05, -0.025]))
    np.testing.assert_allclose(
        new_state.slots["v"]["w"],
        np.array([0.00025, 0.0000625]),
        rtol=1e-6,
    )
    assert int(new_state.step) == 1


def test_adam_rejects_l2_style_weight_decay() -> None:
    params = {"w": np.array([1.0], dtype=np.float32)}
    grads = {"w": np.array([0.5], dtype=np.float32)}
    config = OptimizerConfig(type="adam", learning_rate=0.1, weight_decay=0.01)
    state = init_optimizer_state(params, OptimizerConfig(type="adam"))

    try:
        optimizer_update(params, grads, state, config)
    except ValueError as exc:
        assert "Use adamw" in str(exc)
    else:
        raise AssertionError("adam with weight_decay should fail validation")
