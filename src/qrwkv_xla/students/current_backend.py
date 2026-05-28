from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from qrwkv_xla.parity.radlads_wkv_state_convention import (
    export_reference_state_object,
    import_reference_state_object,
)
from qrwkv_xla.students.backend import StudentBackend
from qrwkv_xla.students.base import StudentModel, StudentOutput
from qrwkv_xla.students.factory import create_student


@dataclass(frozen=True)
class CurrentQRWKVStudentBackend:
    """Behavior-preserving adapter over the current QRWKV student implementation."""

    student: StudentModel

    @classmethod
    def from_config(
        cls,
        architecture: str = "rwkv7_qwen_reference",
        *,
        vocab_size: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        emit_logits: bool = False,
        tie_embeddings: bool = False,
        emit_mixer_outputs: bool = False,
    ) -> CurrentQRWKVStudentBackend:
        return cls(
            create_student(
                architecture,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                emit_logits=emit_logits,
                tie_embeddings=tie_embeddings,
                emit_mixer_outputs=emit_mixer_outputs,
            )
        )

    def init_params(self, key: jax.Array) -> Any:
        return self.student.init_params(key)

    def init_state(self, batch_size: int, **kwargs: Any) -> Any:
        init_state = getattr(self.student, "init_state", None)
        if init_state is None:
            raise NotImplementedError(
                f"{type(self.student).__name__} does not expose init_state"
            )
        return init_state(batch_size=batch_size, **kwargs)

    def forward_full(
        self,
        params: Any,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        initial_state: Any | None = None,
        **kwargs: Any,
    ) -> tuple[StudentOutput, Any]:
        apply_with_state = getattr(self.student, "apply_with_state", None)
        if apply_with_state is None:
            if initial_state is not None:
                raise NotImplementedError(
                    f"{type(self.student).__name__} cannot accept an initial state"
                )
            return self.student.apply(params, input_ids, attention_mask), None
        return apply_with_state(
            params,
            input_ids,
            attention_mask=attention_mask,
            initial_state=initial_state,
            **kwargs,
        )

    def forward_step(
        self,
        params: Any,
        input_ids: jax.Array,
        state: Any,
        attention_mask: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[StudentOutput, Any]:
        step = getattr(self.student, "step", None)
        if step is None:
            raise NotImplementedError(
                f"{type(self.student).__name__} does not expose step"
            )
        return step(
            params,
            input_ids,
            state,
            attention_mask=attention_mask,
            **kwargs,
        )

    def export_state(self, state: Any) -> dict[str, Any]:
        return export_reference_state_object(state)

    def import_state(
        self,
        payload: dict[str, Any],
        *,
        template: Any | None = None,
    ) -> Any:
        return import_reference_state_object(payload, template=template)

    def logits(self, output: StudentOutput) -> jax.Array:
        if output.logits is None:
            raise ValueError("student output does not include logits")
        return output.logits


def create_current_qrwkv_student_backend(
    architecture: str = "rwkv7_qwen_reference",
    **kwargs: Any,
) -> StudentBackend:
    return CurrentQRWKVStudentBackend.from_config(architecture, **kwargs)
