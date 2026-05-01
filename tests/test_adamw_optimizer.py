from __future__ import annotations

import numpy as np

from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update


def test_adamw_applies_decoupled_weight_decay_to_params() -> None:
    params = {"w": np.array([1.0, -2.0], dtype=np.float32)}
    grads = {"w": np.array([0.5, -0.25], dtype=np.float32)}
    config = OptimizerConfig(
        type="adamw",
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.1,
    )
    state = init_optimizer_state(params, config)

    new_params, new_state, _ = optimizer_update(params, grads, state, config)

    np.testing.assert_allclose(new_params["w"], np.array([0.89, -1.88]), rtol=1e-6)
    np.testing.assert_allclose(new_state.slots["m"]["w"], np.array([0.05, -0.025]))
    assert int(new_state.step) == 1
