from __future__ import annotations

from numbers import Real
from typing import Any

from qrwkv_xla.optimizers.adam import adam_update, init_adam_state
from qrwkv_xla.optimizers.config import OptimizerConfig, validate_optimizer_config
from qrwkv_xla.optimizers.sgd import init_sgd_state, sgd_update
from qrwkv_xla.optimizers.state import OptimizerState


def init_optimizer_state(params: Any, config: OptimizerConfig) -> OptimizerState:
    if _config_is_concrete(config):
        validate_optimizer_config(config)
    if config.type == "sgd":
        return init_sgd_state(params)
    if config.type in {"adam", "adamw"}:
        return init_adam_state(params, optimizer_type=config.type)
    raise ValueError(f"unknown optimizer type: {config.type!r}")


def optimizer_update(
    params: Any,
    grads: Any,
    state: OptimizerState,
    config: OptimizerConfig,
) -> tuple[Any, OptimizerState, dict[str, Any]]:
    if _config_is_concrete(config):
        validate_optimizer_config(config)
    if state.type != config.type:
        raise ValueError(
            f"optimizer state type {state.type!r} does not match config {config.type!r}"
        )
    if config.type == "sgd":
        return sgd_update(params, grads, state, config)
    if config.type in {"adam", "adamw"}:
        return adam_update(params, grads, state, config)
    raise ValueError(f"unknown optimizer type: {config.type!r}")


def _config_is_concrete(config: OptimizerConfig) -> bool:
    return all(
        isinstance(value, Real)
        for value in (
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.epsilon,
            config.weight_decay,
        )
    )
