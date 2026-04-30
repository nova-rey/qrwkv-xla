from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.datasets.target_bundle import TargetBatch
from qrwkv_xla.losses import hidden_mse_loss
from qrwkv_xla.trainers.state import TrainState


def batch_to_jax(batch: TargetBatch) -> dict[str, jax.Array]:
    converted: dict[str, jax.Array] = {
        "input_ids": jnp.asarray(batch.input_ids),
        "attention_mask": jnp.asarray(batch.attention_mask),
        "hidden_states": jnp.asarray(batch.hidden_states),
    }
    if batch.logits is not None:
        converted["logits"] = jnp.asarray(batch.logits)
    return converted


def make_train_step(apply_fn: Callable[..., Any]) -> Callable[..., Any]:
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
            return hidden_mse_loss(
                output.hidden_states,
                batch["hidden_states"],
                batch.get("attention_mask"),
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_params = jax.tree_util.tree_map(
            lambda param, grad: param - state.learning_rate * grad,
            state.params,
            grads,
        )
        new_state = TrainState(
            params=new_params,
            step=state.step + 1,
            learning_rate=state.learning_rate,
        )
        return new_state, {"loss": loss}

    return jax.jit(train_step)
