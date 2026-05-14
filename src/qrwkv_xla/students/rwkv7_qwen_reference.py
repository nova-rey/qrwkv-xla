from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    use_rope: bool = True
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    emit_logits: bool = False
    tie_embeddings: bool = False
    emit_mixer_outputs: bool = False
    radlads_compatible_math: bool = False
    radlads_low_rank_decay: bool = False
    radlads_low_rank_iclr: bool = False
    radlads_value_residual_mix: bool = False
    radlads_balance_state_terms: bool = False
    radlads_attention_group_norm: bool = False
    radlads_balance_state: bool = False
    radlads_replay_mode: bool = False
    attention_qkv_bias: bool = False
    radlads_low_rank_gate: bool = False
    lora_rank_decay: int | None = None
    lora_rank_iclr: int | None = None
    lora_rank_value_residual_mix: int | None = None
    lora_rank_gate: int | None = None

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
        for name in (
            "lora_rank_decay",
            "lora_rank_iclr",
            "lora_rank_value_residual_mix",
            "lora_rank_gate",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when provided")
        if self.radlads_balance_state and not self.use_radlads_balance_state_terms:
            raise ValueError(
                "radlads_balance_state requires radlads_balance_state_terms "
                "or radlads_compatible_math"
            )

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

    @property
    def use_radlads_low_rank_decay(self) -> bool:
        return self.radlads_compatible_math or self.radlads_low_rank_decay

    @property
    def use_radlads_low_rank_iclr(self) -> bool:
        return self.radlads_compatible_math or self.radlads_low_rank_iclr

    @property
    def use_radlads_value_residual_mix(self) -> bool:
        return self.radlads_compatible_math or self.radlads_value_residual_mix

    @property
    def use_radlads_balance_state_terms(self) -> bool:
        return self.radlads_compatible_math or self.radlads_balance_state_terms

    @property
    def use_attention_qkv_bias(self) -> bool:
        return self.radlads_replay_mode or self.attention_qkv_bias

    @property
    def use_radlads_low_rank_gate(self) -> bool:
        return self.radlads_replay_mode or self.radlads_low_rank_gate

    @property
    def use_radlads_attention_group_norm(self) -> bool:
        return self.radlads_attention_group_norm

    @property
    def use_radlads_attention_output_scale(self) -> bool:
        return self.radlads_compatible_math or self.radlads_attention_group_norm

    @property
    def effective_lora_rank_decay(self) -> int:
        return self.lora_rank_decay or _radlads_lora_rank(
            self.hidden_size,
            exponent=0.5,
            multiplier=1.8,
        )

    @property
    def effective_lora_rank_iclr(self) -> int:
        return self.lora_rank_iclr or _radlads_lora_rank(
            self.hidden_size,
            exponent=0.5,
            multiplier=1.8,
        )

    @property
    def effective_lora_rank_value_residual_mix(self) -> int:
        return self.lora_rank_value_residual_mix or _radlads_lora_rank(
            self.hidden_size,
            exponent=0.5,
            multiplier=1.3,
        )

    @property
    def effective_lora_rank_gate(self) -> int:
        return self.lora_rank_gate or _radlads_lora_rank(
            self.hidden_size,
            exponent=0.8,
            multiplier=0.6,
        )


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
        extra_keys = jax.random.split(keys[15], 12)
        layer_count = self.config.num_layers
        hidden = self.config.hidden_size
        kv_hidden = self.config.kv_hidden_size
        intermediate = self.config.effective_intermediate_size
        rank_decay = self.config.effective_lora_rank_decay
        rank_iclr = self.config.effective_lora_rank_iclr
        rank_value = self.config.effective_lora_rank_value_residual_mix
        rank_gate = self.config.effective_lora_rank_gate
        self_attn: dict[str, object] = {
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
        }
        if self.config.use_attention_qkv_bias:
            self_attn["q_proj"]["bias"] = jnp.zeros(  # type: ignore[index]
                (layer_count, hidden),
                dtype=jnp.float32,
            )
            self_attn["k_proj"]["bias"] = jnp.zeros(  # type: ignore[index]
                (layer_count, kv_hidden),
                dtype=jnp.float32,
            )
            self_attn["v_proj"]["bias"] = jnp.zeros(  # type: ignore[index]
                (layer_count, kv_hidden),
                dtype=jnp.float32,
            )
        if self.config.use_radlads_low_rank_decay:
            self_attn.update(
                {
                    "w0": _normal(
                        extra_keys[0],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                    "w1": _normal(
                        extra_keys[1],
                        (layer_count, hidden, rank_decay),
                        self.config.init_scale,
                    ),
                    "w2": _normal(
                        extra_keys[2],
                        (layer_count, rank_decay, hidden),
                        self.config.init_scale,
                    ),
                }
            )
        if self.config.use_radlads_low_rank_iclr:
            self_attn.update(
                {
                    "a0": _normal(
                        extra_keys[3],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                    "a1": _normal(
                        extra_keys[4],
                        (layer_count, hidden, rank_iclr),
                        self.config.init_scale,
                    ),
                    "a2": _normal(
                        extra_keys[5],
                        (layer_count, rank_iclr, hidden),
                        self.config.init_scale,
                    ),
                }
            )
        if self.config.use_radlads_value_residual_mix:
            self_attn.update(
                {
                    "v0": _normal(
                        extra_keys[6],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                    "v1": _normal(
                        extra_keys[7],
                        (layer_count, hidden, rank_value),
                        self.config.init_scale,
                    ),
                    "v2": _normal(
                        extra_keys[8],
                        (layer_count, rank_value, hidden),
                        self.config.init_scale,
                    ),
                }
            )
        if self.config.use_radlads_balance_state_terms:
            self_attn.update(
                {
                    "k_k": _normal(
                        extra_keys[9],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                    "k_a": _normal(
                        extra_keys[10],
                        (layer_count, hidden),
                        self.config.init_scale,
                    ),
                    "r_k": _normal(
                        extra_keys[11],
                        (layer_count, self.config.num_heads, self.config.head_size),
                        self.config.init_scale,
                    ),
                }
            )
        if self.config.use_radlads_low_rank_gate:
            self_attn.update(
                {
                    "g1": _normal(
                        extra_keys[0],
                        (layer_count, hidden, rank_gate),
                        self.config.init_scale,
                    ),
                    "g2": _normal(
                        extra_keys[1],
                        (layer_count, rank_gate, hidden),
                        self.config.init_scale,
                    ),
                }
            )
        if self.config.use_radlads_attention_group_norm:
            self_attn["ln_x"] = {
                "weight": jnp.ones((layer_count, hidden), dtype=jnp.float32),
                "bias": jnp.zeros((layer_count, hidden), dtype=jnp.float32),
            }

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
                "self_attn": self_attn,
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
        diagnostics: Any | None = None,
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
        _diag_record(diagnostics, "token_embedding", x, stage="input_embeddings")
        positions = state.next_position + jnp.arange(sequence_length, dtype=jnp.int32)

        layers = params["layers"]  # type: ignore[index]
        hidden_layers: list[jax.Array] = []
        mixer_layers: list[jax.Array] = []
        wkv_states: list[jax.Array] = []
        shift_states: list[jax.Array] = []
        v_first = None
        for layer_index in range(self.config.num_layers):
            x, mixer, layer_wkv, layer_shift, v_first = self._layer(
                x,
                mask=mask,
                positions=positions,
                params=layers,  # type: ignore[arg-type]
                layer_index=layer_index,
                initial_wkv=state.wkv_matrix_state[layer_index],
                initial_shift=state.shift_state[layer_index],
                v_first=v_first,
                diagnostics=diagnostics,
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
        _diag_record(diagnostics, "final_hidden", final_hidden, stage="final_norm")
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
        _diag_record(diagnostics, "logits", logits, stage="logits")

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
        _diag_record(
            diagnostics,
            "returned_hidden_states",
            output.hidden_states,
            stage="returned_hidden_states",
        )
        _diag_record(
            diagnostics,
            "returned_wkv_matrix_state",
            final_state.wkv_matrix_state,
            stage="returned_wkv_matrix_state",
        )
        _diag_record(
            diagnostics,
            "returned_shift_state",
            final_state.shift_state,
            stage="returned_shift_state",
        )
        return output, final_state

    def step(
        self,
        params: dict[str, object],
        input_ids: jax.Array,
        state: RWKV7QwenReferenceState,
        attention_mask: jax.Array | None = None,
        diagnostics: Any | None = None,
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
            diagnostics=diagnostics,
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
        v_first: jax.Array | None,
        diagnostics: Any | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
        attn_input = _rms_norm(
            x,
            params["input_layernorm"]["weight"][layer_index],  # type: ignore[index]
            self.config.rms_norm_eps,
        )
        _diag_record(
            diagnostics,
            f"layers.{layer_index}.pre_attention_norm",
            attn_input,
            stage="pre_attention_norm",
            layer=layer_index,
        )
        attn_out, final_wkv, final_shift, next_v_first = self._attention(
            attn_input,
            mask=mask,
            positions=positions,
            params=params["self_attn"],  # type: ignore[index]
            layer_index=layer_index,
            initial_wkv=initial_wkv,
            initial_shift=initial_shift,
            v_first=v_first,
            diagnostics=diagnostics,
        )
        residual = x + attn_out
        _diag_record(
            diagnostics,
            f"layers.{layer_index}.post_attention_residual",
            residual,
            stage="post_attention_residual",
            layer=layer_index,
        )
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
        _diag_record(
            diagnostics,
            f"layers.{layer_index}.mlp_output",
            mlp_out,
            stage="mlp_output",
            layer=layer_index,
        )
        layer_output = residual + mlp_out
        _diag_record(
            diagnostics,
            f"layers.{layer_index}.layer_output",
            layer_output,
            stage="layer_output",
            layer=layer_index,
        )
        return layer_output, attn_out, final_wkv, final_shift, next_v_first

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
        v_first: jax.Array | None,
        diagnostics: Any | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
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

        def project(
            token: jax.Array,
            name: str,
            heads: int,
            *,
            time_index: int | None = None,
        ) -> jax.Array:
            proj = params[f"{name}_proj"]  # type: ignore[index]
            weight = proj["weight"][layer_index]  # type: ignore[index]
            value = token @ weight
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.{name}_projection",
                value,
                stage=f"{name}_projection",
                layer=layer_index,
                time_index=time_index,
            )
            if self.config.use_attention_qkv_bias and name in {"q", "k", "v"}:
                value = value + jnp.asarray(proj["bias"])[layer_index]  # type: ignore[index]
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.{name}_bias_add",
                    value,
                    stage=f"{name}_bias_add",
                    layer=layer_index,
                    time_index=time_index,
                )
            reshaped = value.reshape(batch_size, heads, head_size)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.{name}_head_split",
                reshaped,
                stage=f"{name}_head_split",
                layer=layer_index,
                time_index=time_index,
            )
            return reshaped

        def low_rank(
            token: jax.Array,
            prefix: str,
            *,
            time_index: int | None = None,
        ) -> jax.Array:
            base = jnp.asarray(params[f"{prefix}0"])[layer_index]
            down = jnp.asarray(params[f"{prefix}1"])[layer_index]
            up = jnp.asarray(params[f"{prefix}2"])[layer_index]
            down_proj = token @ down
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.{prefix}0",
                base,
                stage=f"{prefix}0",
                layer=layer_index,
                time_index=time_index,
            )
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.{prefix}1_projection",
                down_proj,
                stage=f"{prefix}1_projection",
                layer=layer_index,
                time_index=time_index,
            )
            activated = jnp.tanh(down_proj)
            up_proj = activated @ up
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.{prefix}2_projection",
                up_proj,
                stage=f"{prefix}2_projection",
                layer=layer_index,
                time_index=time_index,
            )
            return base[None, :] + up_proj.astype(jnp.float32)

        if (
            self.config.use_radlads_value_residual_mix
            and layer_index > 0
            and v_first is None
        ):
            raise ValueError("v_first is required for RADLADS value residual mixing")
        v_first_scan = (
            jnp.swapaxes(v_first, 0, 1)
            if v_first is not None
            else jnp.zeros((x.shape[1], batch_size, hidden), dtype=x.dtype)
        )
        xs = xs + (v_first_scan,)

        def step(carry, item, *, time_index: int | None = None):
            prev_wkv, prev_shift = carry
            token, token_mask, position, token_v_first = item
            mixed = token + time_mix * (prev_shift - token)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.mixed_input",
                mixed,
                stage="mixed_input",
                layer=layer_index,
                time_index=time_index,
            )
            projection_token = token if self.config.radlads_compatible_math else mixed
            q = project(projection_token, "q", num_heads, time_index=time_index)
            k = project(projection_token, "k", num_kv_heads, time_index=time_index)
            v = project(projection_token, "v", num_kv_heads, time_index=time_index)
            if self.config.use_rope:
                q = rwkv7_qwen_reference_rope(q, position, theta=self.config.rope_theta)
                k = rwkv7_qwen_reference_rope(k, position, theta=self.config.rope_theta)
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.rope_q",
                    q,
                    stage="rope_output_q",
                    layer=layer_index,
                    time_index=time_index,
                )
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.rope_k",
                    k,
                    stage="rope_output_k",
                    layer=layer_index,
                    time_index=time_index,
                )
            k = rwkv7_qwen_reference_group_kv(k, num_heads=num_heads)
            v = rwkv7_qwen_reference_group_kv(v, num_heads=num_heads)
            v_flat = v.reshape(batch_size, hidden)
            next_v_first = v_flat
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.v_first",
                token_v_first,
                stage="v_first",
                layer=layer_index,
                time_index=time_index,
            )
            if self.config.use_radlads_value_residual_mix and layer_index > 0:
                v_low_rank = low_rank(token, "v", time_index=time_index)
                v_mix = jax.nn.sigmoid(v_low_rank)
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.v_mix",
                    v_mix,
                    stage="mixed_value",
                    layer=layer_index,
                    time_index=time_index,
                )
                v_flat = v_flat + (token_v_first - v_flat) * v_mix
                v = v_flat.reshape(batch_size, num_heads, head_size)
            token_mask_state = token_mask.reshape(batch_size, 1, 1)
            v = v * token_mask_state
            if self.config.use_radlads_low_rank_decay:
                w = low_rank(token, "w", time_index=time_index).reshape(
                    batch_size,
                    num_heads,
                    head_size,
                )
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.w_head_split",
                    w,
                    stage="w_head_split",
                    layer=layer_index,
                    time_index=time_index,
                )
            else:
                w = (
                    project(mixed, "w", num_heads, time_index=time_index)
                    + time_bias[None, :, :]
                )
            if self.config.use_radlads_low_rank_iclr:
                a_low_rank = low_rank(token, "a", time_index=time_index)
                a = jax.nn.sigmoid(a_low_rank).reshape(
                    batch_size,
                    num_heads,
                    head_size,
                )
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.iclr_update_rate",
                    a,
                    stage="iclr_update_rate",
                    layer=layer_index,
                    time_index=time_index,
                )
            else:
                a = jax.nn.sigmoid(
                    project(mixed, "a", num_heads, time_index=time_index)
                )
            b = project(mixed, "b", num_heads, time_index=time_index)
            if self.config.use_radlads_low_rank_gate:
                g1 = jnp.asarray(params["g1"])[layer_index]
                g2 = jnp.asarray(params["g2"])[layer_index]
                g1_proj = projection_token @ g1
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.g1_projection",
                    g1_proj,
                    stage="g1_projection",
                    layer=layer_index,
                    time_index=time_index,
                )
                gate = jax.nn.sigmoid(g1_proj) @ g2
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.g2_projection",
                    gate,
                    stage="g2_projection",
                    layer=layer_index,
                    time_index=time_index,
                )
                gate = gate.reshape(batch_size, num_heads, head_size)
            else:
                gate = jax.nn.sigmoid(
                    project(projection_token, "g", num_heads, time_index=time_index)
                )
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.gate_output",
                gate,
                stage="gate_output",
                layer=layer_index,
                time_index=time_index,
            )

            if self.config.use_radlads_balance_state_terms:
                if self.config.radlads_balance_state:
                    kk = _l2_normalize(k)
                else:
                    k_k = jnp.asarray(params["k_k"])[layer_index].reshape(
                        num_heads,
                        head_size,
                    )
                    k_a = jnp.asarray(params["k_a"])[layer_index].reshape(
                        num_heads,
                        head_size,
                    )
                    kk = _l2_normalize(k * k_k[None, :, :])
                    k = k * (1.0 + (a - 1.0) * k_a[None, :, :])
            else:
                kk = _l2_normalize(k)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.k_k",
                kk,
                stage="k_k",
                layer=layer_index,
                time_index=time_index,
            )
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.k_a",
                k,
                stage="k_a",
                layer=layer_index,
                time_index=time_index,
            )
            if self.config.use_radlads_low_rank_decay:
                log_neglog_w = -0.5 - jax.nn.softplus(-w)
                log_w = -jnp.exp(log_neglog_w.astype(jnp.float32))
            else:
                log_w = -jnp.exp(jnp.asarray(-0.5, dtype=w.dtype)) * jax.nn.sigmoid(w)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.log_w",
                log_w,
                stage="low_rank_decay",
                layer=layer_index,
                time_index=time_index,
            )
            decay = jnp.exp(log_w)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.decay",
                decay,
                stage="decay_applied_weights",
                layer=layer_index,
                time_index=time_index,
            )
            if (
                self.config.use_radlads_balance_state_terms
                and self.config.radlads_balance_state
            ):
                k = k * (1.0 - decay + a)
            vk = jnp.einsum("bhi,bhj->bhij", v, k)
            if self.config.radlads_replay_mode:
                ab = jnp.einsum("bhi,bhj->bhij", -kk, kk * a)
            else:
                ab = jnp.einsum("bhi,bhj->bhij", -kk, kk * a + b)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.initial_matrix_state",
                prev_wkv,
                stage="initial_matrix_state",
                layer=layer_index,
                time_index=time_index,
            )
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.update_term",
                vk,
                stage="update_term",
                layer=layer_index,
                time_index=time_index,
            )
            next_wkv = prev_wkv * decay[:, :, None, :] + prev_wkv @ ab + vk
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.next_matrix_state",
                next_wkv,
                stage="wkv_state_after",
                layer=layer_index,
                time_index=time_index,
            )

            mixed_heads = jnp.einsum("bhij,bhj->bhi", next_wkv, q)
            projected = mixed_heads.reshape(batch_size, hidden)
            if self.config.use_radlads_attention_group_norm:
                ln_x = params["ln_x"]  # type: ignore[index]
                projected = _head_group_norm(
                    projected,
                    weight=jnp.asarray(ln_x["weight"])[layer_index],
                    bias=jnp.asarray(ln_x["bias"])[layer_index],
                    num_heads=num_heads,
                    head_size=head_size,
                )
                _diag_record(
                    diagnostics,
                    f"layers.{layer_index}.self_attn.ln_x_output",
                    projected,
                    stage="normalized_output",
                    layer=layer_index,
                    time_index=time_index,
                )
            elif self.config.use_radlads_attention_output_scale:
                projected = projected * (head_size**-0.5)
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.output_before_o_proj",
                projected,
                stage="output_before_o_proj",
                layer=layer_index,
                time_index=time_index,
            )
            projected = projected * gate.reshape(batch_size, hidden)
            out_weight = params["o_proj"]["weight"][layer_index]  # type: ignore[index]
            out = (projected @ out_weight) * token_mask
            _diag_record(
                diagnostics,
                f"layers.{layer_index}.self_attn.o_proj_output",
                out,
                stage="o_proj_output",
                layer=layer_index,
                time_index=time_index,
            )
            return (next_wkv, token), (out, next_v_first)

        if diagnostics is None:
            (final_wkv, final_shift), (outputs, v_first_outputs) = jax.lax.scan(
                step,
                (
                    jnp.asarray(initial_wkv, dtype=jnp.float32),
                    jnp.asarray(initial_shift),
                ),
                xs,
            )
        else:
            carry = (
                jnp.asarray(initial_wkv, dtype=jnp.float32),
                jnp.asarray(initial_shift),
            )
            outputs_list = []
            v_first_list = []
            for time_index in range(x.shape[1]):
                item = tuple(value[time_index] for value in xs)
                carry, emitted = step(carry, item, time_index=time_index)
                out, next_v_first = emitted
                outputs_list.append(out)
                v_first_list.append(next_v_first)
            final_wkv, final_shift = carry
            outputs = jnp.stack(outputs_list, axis=0)
            v_first_outputs = jnp.stack(v_first_list, axis=0)
        next_v_first = (
            jnp.swapaxes(v_first_outputs, 0, 1) if layer_index == 0 else v_first
        )
        return jnp.swapaxes(outputs, 0, 1), final_wkv, final_shift, next_v_first

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


def _radlads_lora_rank(
    hidden_size: int,
    *,
    exponent: float,
    multiplier: float,
) -> int:
    return max(1, round(hidden_size**exponent * multiplier / 32)) * 32


def _l2_normalize(x: jax.Array) -> jax.Array:
    return x / jnp.maximum(
        jnp.linalg.norm(x, axis=-1, keepdims=True),
        jnp.asarray(1e-6, dtype=x.dtype),
    )


def _head_group_norm(
    x: jax.Array,
    *,
    weight: jax.Array,
    bias: jax.Array,
    num_heads: int,
    head_size: int,
) -> jax.Array:
    values = jnp.asarray(x, dtype=jnp.float32).reshape(x.shape[0], num_heads, head_size)
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    normalized = (values - mean) * jax.lax.rsqrt(
        variance + jnp.asarray(head_size * 1e-5, dtype=jnp.float32)
    )
    flat = normalized.reshape(x.shape)
    return (flat * weight[None, :] + bias[None, :]).astype(x.dtype)


def _diag_record(
    diagnostics: Any | None,
    name: str,
    value: Any,
    *,
    stage: str,
    layer: int | None = None,
    time_index: int | None = None,
) -> None:
    if diagnostics is None or value is None or not hasattr(diagnostics, "record"):
        return
    diagnostics.record(
        name,
        value,
        stage=stage,
        layer=layer,
        time_index=time_index,
    )


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
