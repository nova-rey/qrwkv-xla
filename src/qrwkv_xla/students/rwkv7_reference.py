from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.students.base import StudentOutput
from qrwkv_xla.students.lm_head import (
    apply_lm_head,
    apply_tied_lm_head,
    init_lm_head_params,
)


@dataclass(frozen=True)
class RWKV7ReferenceConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    init_scale: float = 0.02
    emit_logits: bool = False
    tie_embeddings: bool = False
    emit_mixer_outputs: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0, got {self.init_scale}")


def rwkv7_reference_layer(
    inputs: jax.Array,
    *,
    wr: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wg: jax.Array,
    wo: jax.Array,
    time_decay: jax.Array,
    time_bias: jax.Array,
    attention_mask: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    return_mixer: bool = False,
    return_state: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Run one simplified RWKV7-style recurrent layer over [B,S,H] inputs."""
    x = jnp.asarray(inputs)
    if x.ndim != 3:
        raise ValueError(f"inputs must have shape [B,S,H], got {x.shape}")

    batch_size, sequence_length, hidden_size = x.shape
    if attention_mask is None:
        mask = jnp.ones((batch_size, sequence_length, 1), dtype=x.dtype)
    else:
        mask = jnp.asarray(attention_mask, dtype=x.dtype)
        if mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [B,S], got {mask.shape}")
        mask = mask[:, :, None]

    decay = jax.nn.sigmoid(jnp.asarray(time_decay))[None, :]
    bias = jnp.asarray(time_bias)[None, :]
    xs = (jnp.swapaxes(x, 0, 1), jnp.swapaxes(mask, 0, 1))
    if initial_state is None:
        recurrent_state = jnp.zeros((batch_size, hidden_size), dtype=x.dtype)
    else:
        recurrent_state = jnp.asarray(initial_state, dtype=x.dtype)
        expected_shape = (batch_size, hidden_size)
        if recurrent_state.shape != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, "
                f"got {recurrent_state.shape}"
            )

    def step(state: jax.Array, item: tuple[jax.Array, jax.Array]):
        token, token_mask = item
        receptance = jax.nn.sigmoid(token @ wr + bias)
        key = token @ wk
        value = token @ wv
        gate = jax.nn.sigmoid(token @ wg)
        proposed_state = decay * state + key * value
        next_state = jnp.where(token_mask > 0, proposed_state, state)
        mixer = (receptance * next_state * gate) @ wo
        output = jnp.tanh(token + mixer)
        output = output * token_mask
        mixer = mixer * token_mask
        return next_state, (output, mixer)

    final_state, (outputs, mixers) = jax.lax.scan(step, recurrent_state, xs)
    outputs = jnp.swapaxes(outputs, 0, 1)
    mixers = jnp.swapaxes(mixers, 0, 1)
    if return_mixer and return_state:
        return outputs, mixers, final_state
    if return_mixer:
        return outputs, mixers
    if return_state:
        return outputs, final_state
    return outputs


@dataclass(frozen=True)
class RWKV7ReferenceStudent:
    config: RWKV7ReferenceConfig

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(key, 9)
        matrix_shape = (
            self.config.num_layers,
            self.config.hidden_size,
            self.config.hidden_size,
        )
        vector_shape = (self.config.num_layers, self.config.hidden_size)
        params = {
            "embedding": jax.random.normal(
                keys[0],
                (self.config.vocab_size, self.config.hidden_size),
            )
            * self.config.init_scale,
            "wr": jax.random.normal(keys[1], matrix_shape) * self.config.init_scale,
            "wk": jax.random.normal(keys[2], matrix_shape) * self.config.init_scale,
            "wv": jax.random.normal(keys[3], matrix_shape) * self.config.init_scale,
            "wg": jax.random.normal(keys[4], matrix_shape) * self.config.init_scale,
            "wo": jax.random.normal(keys[5], matrix_shape) * self.config.init_scale,
            "time_decay": jax.random.normal(keys[6], vector_shape)
            * self.config.init_scale,
            "time_bias": jax.random.normal(keys[7], vector_shape)
            * self.config.init_scale,
        }
        if self.config.emit_logits and not self.config.tie_embeddings:
            params["lm_head"] = init_lm_head_params(
                keys[8],
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                init_scale=self.config.init_scale,
            )
        elif self.config.emit_logits:
            params["lm_head_bias"] = jnp.zeros(
                (self.config.vocab_size,), dtype=jnp.float32
            )
        return params

    def apply(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> StudentOutput:
        token_ids = jnp.asarray(input_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,S], got {token_ids.shape}")

        x = jnp.asarray(params["embedding"])[token_ids]
        layers: list[jax.Array] = []
        mixer_layers: list[jax.Array] = []
        for layer_index in range(self.config.num_layers):
            x, mixer = rwkv7_reference_layer(
                x,
                wr=jnp.asarray(params["wr"])[layer_index],
                wk=jnp.asarray(params["wk"])[layer_index],
                wv=jnp.asarray(params["wv"])[layer_index],
                wg=jnp.asarray(params["wg"])[layer_index],
                wo=jnp.asarray(params["wo"])[layer_index],
                time_decay=jnp.asarray(params["time_decay"])[layer_index],
                time_bias=jnp.asarray(params["time_bias"])[layer_index],
                attention_mask=attention_mask,
                return_mixer=True,
            )
            layers.append(x)
            mixer_layers.append(mixer)

        hidden_states = jnp.stack(layers, axis=1)
        logits = None
        if self.config.emit_logits:
            if self.config.tie_embeddings:
                logits = apply_tied_lm_head(
                    x,
                    params["embedding"],
                    params.get("lm_head_bias"),
                )
            else:
                logits = apply_lm_head(x, params["lm_head"])
        mixer_outputs = (
            jnp.stack(mixer_layers, axis=1) if self.config.emit_mixer_outputs else None
        )
        return StudentOutput(
            hidden_states=hidden_states,
            logits=logits,
            mixer_outputs=mixer_outputs,
        )
