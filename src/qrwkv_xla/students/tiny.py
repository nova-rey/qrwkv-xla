from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.students.base import StudentOutput


@dataclass(frozen=True)
class TinyStudentConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class TinyStudent:
    config: TinyStudentConfig

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        embed_key, scale_key, bias_key = jax.random.split(key, 3)
        return {
            "embedding": jax.random.normal(
                embed_key,
                (self.config.vocab_size, self.config.hidden_size),
            )
            * 0.02,
            "layer_scale": jax.random.normal(
                scale_key,
                (self.config.num_layers, self.config.hidden_size),
            )
            * 0.02
            + 1.0,
            "layer_bias": jax.random.normal(
                bias_key,
                (self.config.num_layers, self.config.hidden_size),
            )
            * 0.02,
        }

    def apply(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> StudentOutput:
        del attention_mask
        token_ids = jnp.asarray(input_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,S], got {token_ids.shape}")

        embeddings = jnp.asarray(params["embedding"])[token_ids]
        layer_scale = jnp.asarray(params["layer_scale"])
        layer_bias = jnp.asarray(params["layer_bias"])
        hidden_states = jnp.tanh(
            embeddings[:, None, :, :] * layer_scale[None, :, None, :]
            + layer_bias[None, :, None, :]
        )
        return StudentOutput(hidden_states=hidden_states)
