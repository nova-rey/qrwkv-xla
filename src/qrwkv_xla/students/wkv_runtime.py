from __future__ import annotations

from enum import StrEnum
from typing import Any


class WKVRuntime(StrEnum):
    REFERENCE = "reference"
    PALLAS = "pallas"


class PallasRuntimeUnavailableError(RuntimeError):
    """Raised when the opt-in Pallas WKV runtime is requested but unavailable."""


def normalize_wkv_runtime(value: str | WKVRuntime) -> WKVRuntime:
    try:
        return WKVRuntime(value)
    except ValueError as exc:
        allowed = ", ".join(runtime.value for runtime in WKVRuntime)
        raise ValueError(
            f"wkv_runtime must be one of {allowed}, got {value!r}"
        ) from exc


def pallas_unavailable_reason() -> str:
    try:
        import jax.experimental.pallas  # noqa: F401
    except Exception:
        return "missing_pallas_dependency_or_backend"
    return "pallas_runtime_not_implemented_yet"


def build_pallas_runtime_probe(
    *,
    requested: str | WKVRuntime = WKVRuntime.PALLAS,
    reference_default_preserved: bool = True,
) -> dict[str, Any]:
    requested_runtime = normalize_wkv_runtime(requested)
    reason = pallas_unavailable_reason()
    recommended = (
        "P82 targeted Pallas dependency/backend availability fix"
        if reason == "missing_pallas_dependency_or_backend"
        else "P82 targeted Pallas runtime scaffold completion"
    )
    return {
        "schema": "qrwkv_xla.p81_pallas_runtime_probe.v1",
        "phase": "P81",
        "default_runtime": WKVRuntime.REFERENCE.value,
        "allowed_runtimes": [runtime.value for runtime in WKVRuntime],
        "reference_default_preserved": reference_default_preserved,
        "wkv_runtime_requested": requested_runtime.value,
        "wkv_runtime_effective": "unavailable",
        "pallas_available": False,
        "fallback_used": False,
        "fallback_reason": None,
        "prototype_status": "unavailable",
        "prototype_scope": "runtime_selector_and_explicit_unavailable_probe",
        "reason": reason,
        "kernel_parity_claimed": False,
        "recommended_next_phase": recommended,
    }
