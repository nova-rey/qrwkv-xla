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
class TinyStudentConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    emit_logits: bool = False
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class TinyStudent:
    config: TinyStudentConfig

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        embed_key, scale_key, bias_key, head_key = jax.random.split(key, 4)
        params = {
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
        if self.config.emit_logits and not self.config.tie_embeddings:
            params["lm_head"] = init_lm_head_params(
                head_key,
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
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
        logits = None
        if self.config.emit_logits:
            final_hidden = hidden_states[:, -1, :, :]
            if self.config.tie_embeddings:
                logits = apply_tied_lm_head(
                    final_hidden,
                    params["embedding"],
                    params.get("lm_head_bias"),
                )
            else:
                logits = apply_lm_head(final_hidden, params["lm_head"])
        return StudentOutput(hidden_states=hidden_states, logits=logits)
