from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.students.backend import StudentBackend
from qrwkv_xla.students.config_selection import qrwkv_student_config_from_vocab_contract
from qrwkv_xla.students.current_backend import CurrentQRWKVStudentBackend
from qrwkv_xla.students.tiny_debug_backend import (
    TINY_DEBUG_ARCHITECTURE_ID,
    TinyDebugStudentBackend,
)
from qrwkv_xla.students.wkv_runtime import WKVRuntime

CURRENT_QRWKV_ARCHITECTURE_ID: Final = "current_qrwkv"


@dataclass(frozen=True)
class StudentBackendSpec:
    architecture_id: str
    description: str


_STUDENT_BACKEND_REGISTRY: dict[str, StudentBackendSpec] = {
    CURRENT_QRWKV_ARCHITECTURE_ID: StudentBackendSpec(
        architecture_id=CURRENT_QRWKV_ARCHITECTURE_ID,
        description="Current QRWKV backend adapter over rwkv7_qwen_reference",
    ),
    TINY_DEBUG_ARCHITECTURE_ID: StudentBackendSpec(
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        description="Tiny deterministic debug backend for registry smoke tests",
    ),
}


def available_student_architectures() -> tuple[str, ...]:
    return tuple(sorted(_STUDENT_BACKEND_REGISTRY))


def student_backend_spec(architecture_id: str) -> StudentBackendSpec:
    try:
        return _STUDENT_BACKEND_REGISTRY[architecture_id]
    except KeyError as exc:
        available = ", ".join(available_student_architectures())
        raise ValueError(
            f"unknown student architecture_id {architecture_id!r}; "
            f"available: {available}"
        ) from exc


def create_student_backend(
    *,
    vocab_contract: VocabContract,
    architecture_id: str | None = None,
    base_config: Any | None = None,
    runtime: Any | None = None,
) -> StudentBackend:
    selected_architecture_id = architecture_id or CURRENT_QRWKV_ARCHITECTURE_ID
    student_backend_spec(selected_architecture_id)
    if selected_architecture_id == CURRENT_QRWKV_ARCHITECTURE_ID:
        selected = qrwkv_student_config_from_vocab_contract(
            vocab_contract,
            base_config=base_config,
            runtime=_runtime_to_wkv_runtime(runtime),
        )
        return CurrentQRWKVStudentBackend.from_config(
            selected.architecture,
            vocab_size=selected.config.vocab_size,
            hidden_size=selected.config.hidden_size,
            num_layers=selected.config.num_layers,
            num_heads=selected.config.num_heads,
            num_kv_heads=selected.config.num_kv_heads,
            emit_logits=selected.config.emit_logits,
            tie_embeddings=selected.config.tie_embeddings,
            emit_mixer_outputs=selected.config.emit_mixer_outputs,
            runtime=runtime if runtime is not None else selected.runtime,
        )
    if selected_architecture_id == TINY_DEBUG_ARCHITECTURE_ID:
        return TinyDebugStudentBackend(
            vocab_contract=vocab_contract,
            runtime=_runtime_to_wkv_runtime(runtime) or WKVRuntime.REFERENCE,
        )
    raise AssertionError(
        f"unhandled student architecture_id {selected_architecture_id!r}"
    )


def _runtime_to_wkv_runtime(runtime: Any | None) -> str | WKVRuntime | None:
    if runtime is None:
        return None
    wkv_runtime = getattr(runtime, "wkv_runtime", None)
    if wkv_runtime is not None:
        return wkv_runtime
    return runtime
