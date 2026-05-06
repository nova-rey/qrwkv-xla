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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RWKV7QwenReferenceState:
    """Explicit slow-reference recurrent cache for the Qwen/RADLADS path."""

    wkv_matrix_state: jax.Array
    shift_state: jax.Array
    next_position: jax.Array

    def tree_flatten(self):
        return (
            self.wkv_matrix_state,
            self.shift_state,
            self.next_position,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        wkv_matrix_state, shift_state, next_position = children
        return cls(
            wkv_matrix_state=wkv_matrix_state,
            shift_state=shift_state,
            next_position=next_position,
        )


@dataclass(frozen=True)
class RWKV7QwenReferenceConfig:
    """Qwen/RADLADS-compatible slow JAX reference path.

    This backend adds a Qwen-style norm/attention/residual/norm/MLP/residual
    shell, RoPE, grouped KV heads, explicit position tracking, matrix WKV state,
    and shift state. It is intentionally CPU/local oriented and does not claim
    optimized kernel parity or full RADLADS numerical parity.
    """

    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 1
    num_kv_heads: int | None = None
    intermediate_size: int | None = None
    init_scale: float = 0.02
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    emit_logits: bool = False
    tie_embeddings: bool = False
    emit_mixer_outputs: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.effective_num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be > 0, got {self.effective_num_kv_heads}"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads, "
                f"got hidden_size={self.hidden_size} num_heads={self.num_heads}"
            )
        if self.num_heads % self.effective_num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads, "
                f"got num_heads={self.num_heads} "
                f"num_kv_heads={self.effective_num_kv_heads}"
            )
        if self.head_size % 2 != 0:
            raise ValueError(
                "head_size must be even for RoPE, "
                f"got hidden_size={self.hidden_size} num_heads={self.num_heads}"
            )
        if self.effective_num_kv_heads > self.num_heads:
            raise ValueError("num_kv_heads must be <= num_heads")
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0 when provided")
        if self.init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0, got {self.init_scale}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")
        if self.rms_norm_eps <= 0.0:
            raise ValueError(f"rms_norm_eps must be > 0, got {self.rms_norm_eps}")

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_hidden_size(self) -> int:
        return self.effective_num_kv_heads * self.head_size

    @property
    def effective_intermediate_size(self) -> int:
        if self.intermediate_size is None:
            return self.hidden_size * 4
        return self.intermediate_size


def rwkv7_qwen_reference_initial_state(
    *,
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    hidden_size: int,
    next_position: int | jax.Array = 0,
    dtype: jnp.dtype = jnp.float32,
) -> RWKV7QwenReferenceState:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    for name, value in (
        ("num_layers", num_layers),
        ("num_heads", num_heads),
        ("head_size", head_size),
        ("hidden_size", hidden_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    return RWKV7QwenReferenceState(
        wkv_matrix_state=jnp.zeros(
            (num_layers, batch_size, num_heads, head_size, head_size),
            dtype=dtype,
        ),
        shift_state=jnp.zeros((num_layers, batch_size, hidden_size), dtype=dtype),
        next_position=jnp.asarray(next_position, dtype=jnp.int32),
    )


def rwkv7_qwen_reference_rope(
    x: jax.Array,
    positions: jax.Array,
    *,
    theta: float = 10000.0,
) -> jax.Array:
    """Apply deterministic rotary position embedding to [..., heads, head_size]."""
    values = jnp.asarray(x)
    if values.ndim < 3:
        raise ValueError(f"x must have at least 3 dims, got {values.shape}")
    head_size = values.shape[-1]
    if head_size % 2 != 0:
        raise ValueError(f"RoPE requires even head_size, got {head_size}")
    pos = jnp.asarray(positions, dtype=values.dtype)
    half = head_size // 2
    exponent = jnp.arange(0, half, dtype=values.dtype) / jnp.asarray(
        half,
        dtype=values.dtype,
    )
    freqs = 1.0 / (jnp.asarray(theta, dtype=values.dtype) ** exponent)
    angles = pos[..., None] * freqs
    while angles.ndim < values.ndim:
        angles = jnp.expand_dims(angles, axis=-2)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    rotated = jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1)
    return rotated.reshape(values.shape)


def rwkv7_qwen_reference_group_kv(x: jax.Array, *, num_heads: int) -> jax.Array:
    values = jnp.asarray(x)
    if values.ndim != 3:
        raise ValueError(f"x must have shape [B,num_kv_heads,D], got {values.shape}")
    num_kv_heads = values.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_kv_heads, "
            f"got {num_heads} and {num_kv_heads}"
        )
    repeats = num_heads // num_kv_heads
    return jnp.repeat(values, repeats, axis=1)


@dataclass(frozen=True)
class RWKV7QwenReferenceStudent:
    config: RWKV7QwenReferenceConfig

    def init_params(self, key: jax.Array) -> dict[str, object]:
        keys = jax.random.split(key, 20)
        layer_count = self.config.num_layers
        hidden = self.config.hidden_size
        kv_hidden = self.config.kv_hidden_size
        intermediate = self.config.effective_intermediate_size
        params: dict[str, object] = {
            "token_embedding": {
                "weight": jax.random.normal(keys[0], (self.config.vocab_size, hidden))
                * self.config.init_scale,
            },
            "layers": {
                "input_layernorm": {"weight": jnp.ones((layer_count, hidden))},
                "post_attention_layernorm": {
                    "weight": jnp.ones((layer_count, hidden)),
                },
                "self_attn": {
                    "q_proj": {
                        "weight": _normal(
                            keys[1],
                            (layer_count, hidden, hidden),
                            self.config.init_scale,
                        ),
                    },
                    "k_proj": {
                        "weight": _normal(
                            keys[2],
                            (layer_count, hidden, kv_hidden),
                            self.config.init_scale,
                        ),
                    },
                    "v_proj": {
                        "weight": _normal(
                            keys[3],
                            (layer_count, hidden, kv_hidden),
                            self.config.init_scale,
                        ),
                    },
                    "w_proj": {
                        "weight": _normal(
                            keys[4],
                            (layer_count, hidden, hidden),
                            self.config.init_scale,
                        ),
                    },
                    "a_proj": {
                        "weight": _normal(
                            keys[5],
                            (layer_count, hidden, hidden),
                            self.config.init_scale,
                        ),
                    },
                    "b_proj": {
                        "weight": _normal(
                            keys[6],
                            (layer_count, hidden, hidden),
                            self.config.init_scale,
                        ),
                    },
                    "g_proj": {
                        "weight": _normal(
                            keys[7],
                            (layer_count, hidden, hidden),
                            self.config.init_scale,
                        ),
                    },
                    "o_proj": {
                        "weight": _normal(
                            keys[8],
                            (layer_count, hidden, hidden),
                            self.config.init_scale,
                        ),
                    },
                    "time_bias": _normal(
                        keys[9],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                    "time_mix": _normal(
                        keys[10],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                },
                "mlp": {
                    "gate_proj": {
                        "weight": _normal(
                            keys[11],
                            (layer_count, hidden, intermediate),
                            self.config.init_scale,
                        ),
                    },
                    "up_proj": {
                        "weight": _normal(
                            keys[12],
                            (layer_count, hidden, intermediate),
                            self.config.init_scale,
                        ),
                    },
                    "down_proj": {
                        "weight": _normal(
                            keys[13],
                            (layer_count, intermediate, hidden),
                            self.config.init_scale,
                        ),
                    },
                },
            },
            "final_layernorm": {"weight": jnp.ones((hidden,))},
        }
        if self.config.emit_logits and not self.config.tie_embeddings:
            params["lm_head"] = init_lm_head_params(
                keys[14],
                hidden_size=hidden,
                vocab_size=self.config.vocab_size,
                init_scale=self.config.init_scale,
            )
        elif self.config.emit_logits:
            params["lm_head_bias"] = jnp.zeros(
                (self.config.vocab_size,), dtype=jnp.float32
            )
        return params

    def init_state(
        self,
        batch_size: int,
        *,
        next_position: int | jax.Array = 0,
        dtype: jnp.dtype = jnp.float32,
    ) -> RWKV7QwenReferenceState:
        return rwkv7_qwen_reference_initial_state(
            batch_size=batch_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_size=self.config.head_size,
            hidden_size=self.config.hidden_size,
            next_position=next_position,
            dtype=dtype,
        )

    def apply(
        self,
        params: dict[str, object],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> StudentOutput:
        output, _state = self.apply_with_state(params, input_ids, attention_mask)
        return output

    def apply_with_state(
        self,
        params: dict[str, object],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        initial_state: RWKV7QwenReferenceState | None = None,
    ) -> tuple[StudentOutput, RWKV7QwenReferenceState]:
        token_ids = jnp.asarray(input_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,S], got {token_ids.shape}")
        batch_size, sequence_length = token_ids.shape
        mask = _attention_mask(attention_mask, batch_size, sequence_length)
        state = self.init_state(batch_size) if initial_state is None else initial_state
        self._validate_state(state, batch_size=batch_size)

        embeddings = params["token_embedding"]["weight"]  # type: ignore[index]
        x = jnp.asarray(embeddings)[token_ids]
        positions = state.next_position + jnp.arange(sequence_length, dtype=jnp.int32)

        layers = params["layers"]  # type: ignore[index]
        hidden_layers: list[jax.Array] = []
        mixer_layers: list[jax.Array] = []
        wkv_states: list[jax.Array] = []
        shift_states: list[jax.Array] = []
        for layer_index in range(self.config.num_layers):
            x, mixer, layer_wkv, layer_shift = self._layer(
                x,
                mask=mask,
                positions=positions,
                params=layers,  # type: ignore[arg-type]
                layer_index=layer_index,
                initial_wkv=state.wkv_matrix_state[layer_index],
                initial_shift=state.shift_state[layer_index],
            )
            hidden_layers.append(x)
            mixer_layers.append(mixer)
            wkv_states.append(layer_wkv)
            shift_states.append(layer_shift)

        final_norm_weight = params["final_layernorm"]["weight"]  # type: ignore[index]
        final_hidden = _rms_norm(
            x,
            jnp.asarray(final_norm_weight),
            self.config.rms_norm_eps,
        )
        logits = None
        if self.config.emit_logits:
            token_embedding = params["token_embedding"]["weight"]  # type: ignore[index]
            if self.config.tie_embeddings:
                logits = apply_tied_lm_head(
                    final_hidden,
                    jnp.asarray(token_embedding),
                    params.get("lm_head_bias"),  # type: ignore[union-attr]
                )
            else:
                logits = apply_lm_head(final_hidden, params["lm_head"])  # type: ignore[index]

        output = StudentOutput(
            hidden_states=jnp.stack(hidden_layers, axis=1),
            logits=logits,
            mixer_outputs=(
                jnp.stack(mixer_layers, axis=1)
                if self.config.emit_mixer_outputs
                else None
            ),
        )
        final_state = RWKV7QwenReferenceState(
            wkv_matrix_state=jnp.stack(wkv_states, axis=0),
            shift_state=jnp.stack(shift_states, axis=0),
            next_position=state.next_position
            + jnp.asarray(sequence_length, dtype=jnp.int32),
        )
        return output, final_state

    def step(
        self,
        params: dict[str, object],
        input_ids: jax.Array,
        state: RWKV7QwenReferenceState,
        attention_mask: jax.Array | None = None,
    ) -> tuple[StudentOutput, RWKV7QwenReferenceState]:
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

    def _validate_state(
        self,
        state: RWKV7QwenReferenceState,
        *,
        batch_size: int,
    ) -> None:
        expected_wkv = (
            self.config.num_layers,
            batch_size,
            self.config.num_heads,
            self.config.head_size,
            self.config.head_size,
        )
        expected_shift = (
            self.config.num_layers,
            batch_size,
            self.config.hidden_size,
        )
        if state.wkv_matrix_state.shape != expected_wkv:
            raise ValueError(
                f"wkv_matrix_state must have shape {expected_wkv}, "
                f"got {state.wkv_matrix_state.shape}"
            )
        if state.shift_state.shape != expected_shift:
            raise ValueError(
                f"shift_state must have shape {expected_shift}, "
                f"got {state.shift_state.shape}"
            )

    def _layer(
        self,
        x: jax.Array,
        *,
        mask: jax.Array,
        positions: jax.Array,
        params: dict[str, object],
        layer_index: int,
        initial_wkv: jax.Array,
        initial_shift: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        attn_input = _rms_norm(
            x,
            params["input_layernorm"]["weight"][layer_index],  # type: ignore[index]
            self.config.rms_norm_eps,
        )
        attn_out, final_wkv, final_shift = self._attention(
            attn_input,
            mask=mask,
            positions=positions,
            params=params["self_attn"],  # type: ignore[index]
            layer_index=layer_index,
            initial_wkv=initial_wkv,
            initial_shift=initial_shift,
        )
        residual = x + attn_out
        mlp_input = _rms_norm(
            residual,
            params["post_attention_layernorm"]["weight"][layer_index],  # type: ignore[index]
            self.config.rms_norm_eps,
        )
        mlp_out = self._mlp(
            mlp_input,
            params=params["mlp"],  # type: ignore[index]
            layer_index=layer_index,
        )
        mlp_out = mlp_out * mask
        return residual + mlp_out, attn_out, final_wkv, final_shift

    def _attention(
        self,
        x: jax.Array,
        *,
        mask: jax.Array,
        positions: jax.Array,
        params: dict[str, object],
        layer_index: int,
        initial_wkv: jax.Array,
        initial_shift: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size, _sequence_length, hidden = x.shape
        head_size = self.config.head_size
        num_heads = self.config.num_heads
        num_kv_heads = self.config.effective_num_kv_heads
        time_mix = jax.nn.sigmoid(jnp.asarray(params["time_mix"])[layer_index])
        time_bias = jnp.asarray(params["time_bias"])[layer_index].reshape(
            num_heads, head_size
        )
        xs = (
            jnp.swapaxes(x, 0, 1),
            jnp.swapaxes(mask, 0, 1),
            positions,
        )

        def project(token: jax.Array, name: str, heads: int) -> jax.Array:
            weight = params[f"{name}_proj"]["weight"][layer_index]  # type: ignore[index]
            return (token @ weight).reshape(batch_size, heads, head_size)

        def step(carry, item):
            prev_wkv, prev_shift = carry
            token, token_mask, position = item
            mixed = token + time_mix * (prev_shift - token)
            q = project(mixed, "q", num_heads)
            k = project(mixed, "k", num_kv_heads)
            v = project(mixed, "v", num_kv_heads)
            q = rwkv7_qwen_reference_rope(q, position, theta=self.config.rope_theta)
            k = rwkv7_qwen_reference_rope(k, position, theta=self.config.rope_theta)
            k = rwkv7_qwen_reference_group_kv(k, num_heads=num_heads)
            v = rwkv7_qwen_reference_group_kv(v, num_heads=num_heads)
            token_mask_state = token_mask.reshape(batch_size, 1, 1)
            v = v * token_mask_state
            w = project(mixed, "w", num_heads) + time_bias[None, :, :]
            a = jax.nn.sigmoid(project(mixed, "a", num_heads))
            b = project(mixed, "b", num_heads)
            gate = jax.nn.sigmoid(project(mixed, "g", num_heads))

            kk = k / jnp.maximum(
                jnp.linalg.norm(k, axis=-1, keepdims=True),
                jnp.asarray(1e-6, dtype=k.dtype),
            )
            log_w = -jnp.exp(jnp.asarray(-0.5, dtype=w.dtype)) * jax.nn.sigmoid(w)
            decay = jnp.exp(log_w)
            vk = jnp.einsum("bhi,bhj->bhij", v, k)
            ab = jnp.einsum("bhi,bhj->bhij", -kk, kk * a + b)
            next_wkv = prev_wkv * decay[:, :, None, :] + prev_wkv @ ab + vk

            mixed_heads = jnp.einsum("bhij,bhj->bhi", next_wkv, q) * gate
            projected = mixed_heads.reshape(batch_size, hidden)
            out_weight = params["o_proj"]["weight"][layer_index]  # type: ignore[index]
            out = (projected @ out_weight) * token_mask
            return (next_wkv, token), out

        (final_wkv, final_shift), outputs = jax.lax.scan(
            step,
            (jnp.asarray(initial_wkv, dtype=jnp.float32), jnp.asarray(initial_shift)),
            xs,
        )
        return jnp.swapaxes(outputs, 0, 1), final_wkv, final_shift

    def _mlp(
        self,
        x: jax.Array,
        *,
        params: dict[str, object],
        layer_index: int,
    ) -> jax.Array:
        gate_w = params["gate_proj"]["weight"][layer_index]  # type: ignore[index]
        up_w = params["up_proj"]["weight"][layer_index]  # type: ignore[index]
        down_w = params["down_proj"]["weight"][layer_index]  # type: ignore[index]
        return (jax.nn.silu(x @ gate_w) * (x @ up_w)) @ down_w


def _normal(key: jax.Array, shape: tuple[int, ...], scale: float) -> jax.Array:
    return jax.random.normal(key, shape) * scale


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    values = jnp.asarray(x)
    variance = jnp.mean(values * values, axis=-1, keepdims=True)
    return values * jax.lax.rsqrt(variance + eps) * weight


def _attention_mask(
    attention_mask: jax.Array | None,
    batch_size: int,
    sequence_length: int,
) -> jax.Array:
    if attention_mask is None:
        return jnp.ones((batch_size, sequence_length, 1), dtype=jnp.float32)
    mask = jnp.asarray(attention_mask, dtype=jnp.float32)
    if mask.shape != (batch_size, sequence_length):
        raise ValueError(
            "attention_mask must have shape "
            f"{(batch_size, sequence_length)}, got {mask.shape}"
        )
    return mask[:, :, None]
