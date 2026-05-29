from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax

from qrwkv_xla.students.pallas_wkv import (
    pallas_wkv_sequence_update_fused_or_scan,
    pallas_wkv_update,
    reference_wkv_sequence_update,
    reference_wkv_update,
)
from qrwkv_xla.students.wkv_runtime import WKVRuntime, normalize_wkv_runtime


class StudentRuntime(Protocol):
    """Execution-runtime boundary for student models.

    StudentRuntime describes how a validated student architecture executes its
    WKV path. It must not encode architecture identity or teacher behavior.
    """

    name: str
    wkv_runtime: WKVRuntime

    def step(
        self,
        state: jax.Array,
        k: jax.Array,
        v: jax.Array,
        decay: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError

    def sequence(
        self,
        initial_state: jax.Array,
        k_seq: jax.Array,
        v_seq: jax.Array,
        decay_seq: jax.Array,
    ) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ReferenceJaxStudentRuntime:
    name: str = "reference_jax"
    wkv_runtime: WKVRuntime = WKVRuntime.REFERENCE

    def step(
        self,
        state: jax.Array,
        k: jax.Array,
        v: jax.Array,
        decay: jax.Array,
    ) -> jax.Array:
        return reference_wkv_update(state, k, v, decay)

    def sequence(
        self,
        initial_state: jax.Array,
        k_seq: jax.Array,
        v_seq: jax.Array,
        decay_seq: jax.Array,
    ) -> dict[str, Any]:
        return reference_wkv_sequence_update(initial_state, k_seq, v_seq, decay_seq)


@dataclass(frozen=True)
class PallasStudentRuntime:
    name: str = "pallas"
    wkv_runtime: WKVRuntime = WKVRuntime.PALLAS

    def step(
        self,
        state: jax.Array,
        k: jax.Array,
        v: jax.Array,
        decay: jax.Array,
    ) -> jax.Array:
        return pallas_wkv_update(state, k, v, decay)

    def sequence(
        self,
        initial_state: jax.Array,
        k_seq: jax.Array,
        v_seq: jax.Array,
        decay_seq: jax.Array,
    ) -> dict[str, Any]:
        return pallas_wkv_sequence_update_fused_or_scan(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )


def create_student_runtime(
    runtime: str | WKVRuntime | StudentRuntime | None = None,
) -> StudentRuntime:
    if runtime is None:
        return ReferenceJaxStudentRuntime()
    if _is_student_runtime(runtime):
        return runtime
    normalized = normalize_wkv_runtime(runtime)
    if normalized is WKVRuntime.REFERENCE:
        return ReferenceJaxStudentRuntime()
    if normalized is WKVRuntime.PALLAS:
        return PallasStudentRuntime()
    raise ValueError(f"unsupported student runtime: {runtime!r}")


def _is_student_runtime(value: object) -> bool:
    return (
        hasattr(value, "name")
        and hasattr(value, "wkv_runtime")
        and callable(getattr(value, "step", None))
        and callable(getattr(value, "sequence", None))
    )
