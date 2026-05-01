from __future__ import annotations

import numpy as np

from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update


def test_sgd_update_matches_previous_param_minus_lr_grad_behavior() -> None:
    params = {"w": np.array([1.0, -2.0], dtype=np.float32)}
    grads = {"w": np.array([0.5, -0.25], dtype=np.float32)}
    config = OptimizerConfig(type="sgd", learning_rate=0.1)
    state = init_optimizer_state(params, config)

    new_params, new_state, metrics = optimizer_update(params, grads, state, config)

    np.testing.assert_allclose(new_params["w"], np.array([0.95, -1.975]))
    assert int(new_state.step) == 1
    assert float(metrics["learning_rate"]) == np.float32(0.1)
