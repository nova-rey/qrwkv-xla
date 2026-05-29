from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import jax
import jax.numpy as jnp

from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.students.base import StudentOutput
from qrwkv_xla.students.wkv_runtime import WKVRuntime, normalize_wkv_runtime

TINY_DEBUG_ARCHITECTURE_ID: Final = "tiny_debug"


@dataclass(frozen=True)
class TinyDebugState:
    step: jax.Array


@dataclass(frozen=True)
class TinyDebugStudentBackend:
    vocab_contract: VocabContract
    runtime: WKVRuntime = WKVRuntime.REFERENCE
    architecture_id: str = TINY_DEBUG_ARCHITECTURE_ID

    def __post_init__(self) -> None:
        runtime = normalize_wkv_runtime(self.runtime)
        if runtime is WKVRuntime.PALLAS:
            raise ValueError("tiny_debug backend does not support pallas runtime")
        object.__setattr__(self, "runtime", runtime)

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        del key
        return {
            "scale": jnp.asarray(0.125, dtype=jnp.float32),
            "bias": jnp.asarray(0.0, dtype=jnp.float32),
        }

    def init_state(self, batch_size: int, **kwargs: Any) -> TinyDebugState:
        del batch_size, kwargs
        return TinyDebugState(step=jnp.asarray(0, dtype=jnp.int32))

    def forward_full(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        initial_state: TinyDebugState | None = None,
        **kwargs: Any,
    ) -> tuple[StudentOutput, TinyDebugState]:
        del attention_mask, kwargs
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        logits = self._logits(params, input_ids)
        hidden_states = jnp.mean(logits, axis=-1, keepdims=True)
        state = initial_state or self.init_state(batch_size=int(input_ids.shape[0]))
        next_state = TinyDebugState(
            step=state.step + jnp.asarray(input_ids.shape[1], dtype=jnp.int32)
        )
        return StudentOutput(hidden_states=hidden_states, logits=logits), next_state

    def forward_step(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        state: TinyDebugState,
        attention_mask: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[StudentOutput, TinyDebugState]:
        output, next_state = self.forward_full(
            params,
            input_ids,
            attention_mask=attention_mask,
            initial_state=state,
            **kwargs,
        )
        return output, next_state

    def export_state(self, state: TinyDebugState) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "step": int(state.step),
        }

    def import_state(
        self,
        payload: dict[str, Any],
        *,
        template: Any | None = None,
    ) -> TinyDebugState:
        del template
        if payload.get("architecture_id") != self.architecture_id:
            raise ValueError("tiny_debug state payload architecture_id mismatch")
        return TinyDebugState(step=jnp.asarray(int(payload["step"]), dtype=jnp.int32))

    def logits(self, output: StudentOutput) -> jax.Array:
        if output.logits is None:
            raise ValueError("tiny_debug output does not include logits")
        return output.logits

    def _logits(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
    ) -> jax.Array:
        vocab = jnp.arange(self.vocab_contract.vocab_size, dtype=jnp.float32)
        token_values = jnp.asarray(input_ids % self.vocab_contract.vocab_size)
        return (
            token_values[:, :, None].astype(jnp.float32) + vocab[None, None, :]
        ) * params["scale"] + params["bias"]
