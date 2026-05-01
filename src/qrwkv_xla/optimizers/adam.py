from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.optimizers.config import OptimizerConfig
from qrwkv_xla.optimizers.state import OptimizerState
from qrwkv_xla.optimizers.tree import tree_zeros_like


def init_adam_state(
    params: Any,
    *,
    optimizer_type: str = "adam",
) -> OptimizerState:
    if optimizer_type not in {"adam", "adamw"}:
        raise ValueError("optimizer_type must be 'adam' or 'adamw'")
    return OptimizerState(
        type=optimizer_type,
        step=jnp.asarray(0, dtype=jnp.int32),
        slots={
            "m": tree_zeros_like(params),
            "v": tree_zeros_like(params),
        },
    )


def adam_update(
    params: Any,
    grads: Any,
    state: OptimizerState,
    config: OptimizerConfig,
) -> tuple[Any, OptimizerState, dict[str, Any]]:
    if config.type not in {"adam", "adamw"}:
        raise ValueError("adam_update requires optimizer type 'adam' or 'adamw'")
    m_prev = state.slots["m"]
    v_prev = state.slots["v"]
    step = state.step + 1

    m = jax.tree_util.tree_map(
        lambda old, grad: config.beta1 * old + (1.0 - config.beta1) * grad,
        m_prev,
        grads,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: config.beta2 * old + (1.0 - config.beta2) * jnp.square(grad),
        v_prev,
        grads,
    )

    beta1_power = jnp.power(jnp.asarray(config.beta1, dtype=jnp.float32), step)
    beta2_power = jnp.power(jnp.asarray(config.beta2, dtype=jnp.float32), step)

    def update_param(param, m_leaf, v_leaf):
        m_hat = m_leaf / (1.0 - beta1_power)
        v_hat = v_leaf / (1.0 - beta2_power)
        adam_delta = m_hat / (jnp.sqrt(v_hat) + config.epsilon)
        if config.type == "adamw":
            adam_delta = adam_delta + config.weight_decay * param
        return param - config.learning_rate * adam_delta

    new_params = jax.tree_util.tree_map(update_param, params, m, v)
    new_state = OptimizerState(
        type=config.type,
        step=step,
        slots={
            "m": m,
            "v": v,
        },
    )
    return new_params, new_state, _metrics(new_state, config)


def _metrics(state: OptimizerState, config: OptimizerConfig) -> dict[str, Any]:
    return {
        "optimizer_step": state.step,
        "learning_rate": jnp.asarray(config.learning_rate, dtype=jnp.float32),
    }
