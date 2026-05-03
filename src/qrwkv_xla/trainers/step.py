from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.datasets.target_bundle import TargetBatch
from qrwkv_xla.losses import WeightedLoss, hidden_mse_loss
from qrwkv_xla.optimizers import (
    OptimizerConfig,
    init_optimizer_state,
    optimizer_update,
    validate_optimizer_config,
)
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.training.gradients import clip_gradients_by_global_norm


def batch_to_jax(batch: TargetBatch) -> dict[str, jax.Array]:
    converted: dict[str, jax.Array] = {
        "input_ids": jnp.asarray(batch.input_ids),
        "attention_mask": jnp.asarray(batch.attention_mask),
        "hidden_states": jnp.asarray(batch.hidden_states),
    }
    if batch.logits is not None:
        converted["logits"] = jnp.asarray(batch.logits)
    if batch.attention_targets is not None:
        converted["attention_targets"] = jnp.asarray(batch.attention_targets)
    return converted


def _default_distillation_loss(student_output: Any, batch: dict[str, jax.Array]):
    loss = hidden_mse_loss(
        student_output.hidden_states,
        batch["hidden_states"],
        batch.get("attention_mask"),
    )
    return WeightedLoss(total=loss, components={"loss": loss, "hidden_mse": loss})


def make_train_step(
    apply_fn: Callable[..., Any],
    distillation_loss: Callable[..., WeightedLoss] | None = None,
    optimizer_config: OptimizerConfig | None = None,
    max_grad_norm: float | None = None,
    clip_epsilon: float = 1e-6,
) -> Callable[..., Any]:
    loss_builder = distillation_loss or _default_distillation_loss
    opt_config = optimizer_config or OptimizerConfig()
    validate_optimizer_config(opt_config)

    def train_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any):
            output = apply_fn(
                params,
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            weighted_loss = loss_builder(output, batch)
            return weighted_loss.total, weighted_loss.components

        (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        clip_result = clip_gradients_by_global_norm(
            grads,
            max_grad_norm=max_grad_norm,
            epsilon=clip_epsilon,
        )
        update_config = OptimizerConfig(
            type=opt_config.type,
            learning_rate=state.learning_rate,
            beta1=opt_config.beta1,
            beta2=opt_config.beta2,
            epsilon=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
        )
        optimizer_state = state.optimizer_state
        if optimizer_state is None:
            optimizer_state = init_optimizer_state(state.params, update_config)
        new_params, new_optimizer_state, optimizer_metrics = optimizer_update(
            state.params,
            clip_result.gradients,
            optimizer_state,
            update_config,
        )
        new_state = TrainState(
            params=new_params,
            step=state.step + 1,
            learning_rate=state.learning_rate,
            optimizer_state=new_optimizer_state,
        )
        return new_state, dict(
            components,
            loss=loss,
            grad_global_norm=clip_result.global_norm,
            grad_clipped_global_norm=clip_result.clipped_global_norm,
            grad_clip_scale=clip_result.clip_scale,
            grad_was_clipped=clip_result.was_clipped,
            max_grad_norm=jnp.asarray(
                0.0 if max_grad_norm is None else max_grad_norm,
                dtype=jnp.float32,
            ),
            **optimizer_metrics,
        )

    return jax.jit(train_step)
