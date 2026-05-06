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
class RWKV7RADLADSReferenceConfig:
    """Slow CPU-first RWKV7 reference aligned to RADLADS recurrent state math.

    This is intentionally partial: it mirrors the RADLADS head-wise matrix
    recurrence with clear JAX scan code, but it does not implement Qwen block
    norms, RoPE, LoRA time mixing, grouped KV heads, Triton/Pallas kernels, or
    checkpoint parity.
    """

    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 1
    init_scale: float = 0.02
    emit_logits: bool = False
    tie_embeddings: bool = False
    emit_mixer_outputs: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads, "
                f"got hidden_size={self.hidden_size} num_heads={self.num_heads}"
            )
        if self.init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0, got {self.init_scale}")

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_heads


def rwkv7_radlads_reference_initial_state(
    *,
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    for name, value in (
        ("num_layers", num_layers),
        ("num_heads", num_heads),
        ("head_size", head_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    return jnp.zeros(
        (num_layers, batch_size, num_heads, head_size, head_size),
        dtype=dtype,
    )


def rwkv7_radlads_reference_layer(
    inputs: jax.Array,
    *,
    wr: jax.Array,
    ww: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wa: jax.Array,
    wb: jax.Array,
    wg: jax.Array,
    wo: jax.Array,
    time_bias: jax.Array,
    num_heads: int,
    attention_mask: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    return_mixer: bool = False,
    return_state: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Run one clear RADLADS-style RWKV7 recurrent layer over [B,S,C] inputs.

    The recurrent state is [B, num_heads, head_size, head_size]. For each token,
    this reference projects r/w/k/v/a/b/g, zeroes v for masked padding tokens,
    normalizes the key direction used by the in-context update, applies RADLADS
    decay semantics
    ``log_w = -exp(-0.5) * sigmoid(w)``, updates the matrix state, then reads it
    with r. This follows the shape and operation order of the RADLADS recurrent
    comments and Triton kernels, but omits production model details documented
    on ``RWKV7RADLADSReferenceConfig``.
    """
    x = jnp.asarray(inputs)
    if x.ndim != 3:
        raise ValueError(f"inputs must have shape [B,S,C], got {x.shape}")
    batch_size, sequence_length, hidden_size = x.shape
    if hidden_size % num_heads != 0:
        raise ValueError(
            "inputs hidden size must be divisible by num_heads, "
            f"got hidden_size={hidden_size} num_heads={num_heads}"
        )
    head_size = hidden_size // num_heads

    if attention_mask is None:
        mask = jnp.ones((batch_size, sequence_length, 1), dtype=x.dtype)
    else:
        mask = jnp.asarray(attention_mask, dtype=x.dtype)
        if mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [B,S], got {mask.shape}")
        mask = mask[:, :, None]

    if initial_state is None:
        state = jnp.zeros(
            (batch_size, num_heads, head_size, head_size),
            dtype=jnp.float32,
        )
    else:
        state = jnp.asarray(initial_state, dtype=jnp.float32)
        expected_shape = (batch_size, num_heads, head_size, head_size)
        if state.shape != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, got {state.shape}"
            )

    xs = (jnp.swapaxes(x, 0, 1), jnp.swapaxes(mask, 0, 1))
    time_bias_heads = jnp.asarray(time_bias).reshape(num_heads, head_size)

    def project(token: jax.Array, weight: jax.Array) -> jax.Array:
        return (token @ weight).reshape(batch_size, num_heads, head_size)

    def step(carry: jax.Array, item: tuple[jax.Array, jax.Array]):
        prev_state = carry
        token, token_mask = item
        r = project(token, wr)
        w_lora = project(token, ww) + time_bias_heads[None, :, :]
        k = project(token, wk)
        token_mask_state = token_mask.reshape(batch_size, 1, 1)
        v = project(token, wv) * token_mask_state
        a = jax.nn.sigmoid(project(token, wa))
        b = project(token, wb)
        gate = jax.nn.sigmoid(project(token, wg))

        kk = k / jnp.maximum(
            jnp.linalg.norm(k, axis=-1, keepdims=True),
            jnp.asarray(1e-6, dtype=k.dtype),
        )
        log_w = -jnp.exp(jnp.asarray(-0.5, dtype=w_lora.dtype)) * jax.nn.sigmoid(w_lora)
        decay = jnp.exp(log_w)

        vk = jnp.einsum("bhi,bhj->bhij", v, k)
        ab = jnp.einsum("bhi,bhj->bhij", -kk, kk * a + b)
        next_state = prev_state * decay[:, :, None, :] + prev_state @ ab + vk

        mixed_heads = jnp.einsum("bhij,bhj->bhi", next_state, r) * gate
        mixer = mixed_heads.reshape(batch_size, hidden_size) @ wo
        output = jnp.tanh(token + mixer)
        output = output * token_mask
        mixer = mixer * token_mask
        return next_state, (output, mixer)

    final_state, (outputs, mixers) = jax.lax.scan(step, state, xs)
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
class RWKV7RADLADSReferenceStudent:
    config: RWKV7RADLADSReferenceConfig

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(key, 11)
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
            "ww": jax.random.normal(keys[2], matrix_shape) * self.config.init_scale,
            "wk": jax.random.normal(keys[3], matrix_shape) * self.config.init_scale,
            "wv": jax.random.normal(keys[4], matrix_shape) * self.config.init_scale,
            "wa": jax.random.normal(keys[5], matrix_shape) * self.config.init_scale,
            "wb": jax.random.normal(keys[6], matrix_shape) * self.config.init_scale,
            "wg": jax.random.normal(keys[7], matrix_shape) * self.config.init_scale,
            "wo": jax.random.normal(keys[8], matrix_shape) * self.config.init_scale,
            "time_bias": jax.random.normal(keys[9], vector_shape)
            * self.config.init_scale,
        }
        if self.config.emit_logits and not self.config.tie_embeddings:
            params["lm_head"] = init_lm_head_params(
                keys[10],
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                init_scale=self.config.init_scale,
            )
        elif self.config.emit_logits:
            params["lm_head_bias"] = jnp.zeros(
                (self.config.vocab_size,), dtype=jnp.float32
            )
        return params

    def init_state(self, batch_size: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
        return rwkv7_radlads_reference_initial_state(
            batch_size=batch_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_size=self.config.head_size,
            dtype=dtype,
        )

    def apply(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> StudentOutput:
        output, _ = self.apply_with_state(params, input_ids, attention_mask)
        return output

    def apply_with_state(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        initial_state: jax.Array | None = None,
    ) -> tuple[StudentOutput, jax.Array]:
        token_ids = jnp.asarray(input_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,S], got {token_ids.shape}")

        x = jnp.asarray(params["embedding"])[token_ids]
        if initial_state is None:
            recurrent_state = self.init_state(token_ids.shape[0])
        else:
            recurrent_state = jnp.asarray(initial_state, dtype=jnp.float32)
            expected_shape = (
                self.config.num_layers,
                token_ids.shape[0],
                self.config.num_heads,
                self.config.head_size,
                self.config.head_size,
            )
            if recurrent_state.shape != expected_shape:
                raise ValueError(
                    "initial_state must have shape "
                    f"{expected_shape}, got {recurrent_state.shape}"
                )

        layers: list[jax.Array] = []
        mixer_layers: list[jax.Array] = []
        final_states: list[jax.Array] = []
        for layer_index in range(self.config.num_layers):
            x, mixer, layer_state = rwkv7_radlads_reference_layer(
                x,
                wr=jnp.asarray(params["wr"])[layer_index],
                ww=jnp.asarray(params["ww"])[layer_index],
                wk=jnp.asarray(params["wk"])[layer_index],
                wv=jnp.asarray(params["wv"])[layer_index],
                wa=jnp.asarray(params["wa"])[layer_index],
                wb=jnp.asarray(params["wb"])[layer_index],
                wg=jnp.asarray(params["wg"])[layer_index],
                wo=jnp.asarray(params["wo"])[layer_index],
                time_bias=jnp.asarray(params["time_bias"])[layer_index],
                num_heads=self.config.num_heads,
                attention_mask=attention_mask,
                initial_state=recurrent_state[layer_index],
                return_mixer=True,
                return_state=True,
            )
            layers.append(x)
            mixer_layers.append(mixer)
            final_states.append(layer_state)

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
        return (
            StudentOutput(
                hidden_states=hidden_states,
                logits=logits,
                mixer_outputs=mixer_outputs,
            ),
            jnp.stack(final_states, axis=0),
        )

    def step(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        state: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> tuple[StudentOutput, jax.Array]:
        token_ids = jnp.asarray(input_ids)
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError(f"input_ids must have shape [B,1], got {token_ids.shape}")
        if attention_mask is not None and jnp.asarray(attention_mask).shape != (
            token_ids.shape[0],
            1,
        ):
            raise ValueError("attention_mask must have shape [B,1] for step")
        return self.apply_with_state(
            params,
            token_ids,
            attention_mask=attention_mask,
            initial_state=state,
        )
