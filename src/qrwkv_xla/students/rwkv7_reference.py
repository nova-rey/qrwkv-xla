from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.students.base import StudentOutput


@dataclass(frozen=True)
class RWKV7ReferenceConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    init_scale: float = 0.02

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
) -> jax.Array:
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
    initial_state = jnp.zeros((batch_size, hidden_size), dtype=x.dtype)

    def step(state: jax.Array, item: tuple[jax.Array, jax.Array]):
        token, token_mask = item
        receptance = jax.nn.sigmoid(token @ wr + bias)
        key = token @ wk
        value = token @ wv
        gate = jax.nn.sigmoid(token @ wg)
        proposed_state = decay * state + key * value
        next_state = jnp.where(token_mask > 0, proposed_state, state)
        output = jnp.tanh(token + (receptance * next_state * gate) @ wo)
        output = output * token_mask
        return next_state, output

    _, outputs = jax.lax.scan(step, initial_state, xs)
    return jnp.swapaxes(outputs, 0, 1)


@dataclass(frozen=True)
class RWKV7ReferenceStudent:
    config: RWKV7ReferenceConfig

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(key, 8)
        matrix_shape = (
            self.config.num_layers,
            self.config.hidden_size,
            self.config.hidden_size,
        )
        vector_shape = (self.config.num_layers, self.config.hidden_size)
        return {
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
        for layer_index in range(self.config.num_layers):
            x = rwkv7_reference_layer(
                x,
                wr=jnp.asarray(params["wr"])[layer_index],
                wk=jnp.asarray(params["wk"])[layer_index],
                wv=jnp.asarray(params["wv"])[layer_index],
                wg=jnp.asarray(params["wg"])[layer_index],
                wo=jnp.asarray(params["wo"])[layer_index],
                time_decay=jnp.asarray(params["time_decay"])[layer_index],
                time_bias=jnp.asarray(params["time_bias"])[layer_index],
                attention_mask=attention_mask,
            )
            layers.append(x)

        hidden_states = jnp.stack(layers, axis=1)
        return StudentOutput(hidden_states=hidden_states)
