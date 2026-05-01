from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.optimizers.config import OptimizerConfig
from qrwkv_xla.optimizers.state import OptimizerState


def init_sgd_state(params: Any) -> OptimizerState:
    del params
    return OptimizerState(type="sgd", step=jnp.asarray(0, dtype=jnp.int32), slots={})


def sgd_update(
    params: Any,
    grads: Any,
    state: OptimizerState,
    config: OptimizerConfig,
) -> tuple[Any, OptimizerState, dict[str, Any]]:
    new_params = jax.tree_util.tree_map(
        lambda param, grad: param - config.learning_rate * grad,
        params,
        grads,
    )
    new_state = OptimizerState(
        type="sgd",
        step=state.step + 1,
        slots={},
    )
    return new_params, new_state, _metrics(new_state, config)


def _metrics(state: OptimizerState, config: OptimizerConfig) -> dict[str, Any]:
    return {
        "optimizer_step": state.step,
        "learning_rate": jnp.asarray(config.learning_rate, dtype=jnp.float32),
    }
