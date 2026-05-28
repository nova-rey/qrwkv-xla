from __future__ import annotations

from enum import StrEnum
from typing import Any

from qrwkv_xla.students.pallas_wkv import (
    pallas_available,
    run_pallas_wkv_parity_probe,
)


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
    available, reason = pallas_available()
    if available:
        return "pallas_available"
    return reason or "missing_pallas_dependency_or_backend"


def build_pallas_runtime_probe(
    *,
    requested: str | WKVRuntime = WKVRuntime.PALLAS,
    reference_default_preserved: bool = True,
) -> dict[str, Any]:
    requested_runtime = normalize_wkv_runtime(requested)
    available, unavailable_reason = pallas_available()
    if requested_runtime is WKVRuntime.REFERENCE:
        return {
            "schema": "qrwkv_xla.p83_pallas_wkv_parity_probe.v1",
            "phase": "P83",
            "default_runtime": WKVRuntime.REFERENCE.value,
            "allowed_runtimes": [runtime.value for runtime in WKVRuntime],
            "reference_default_preserved": reference_default_preserved,
            "wkv_runtime_requested": requested_runtime.value,
            "wkv_runtime_effective": WKVRuntime.REFERENCE.value,
            "pallas_available": available,
            "fallback_used": False,
            "fallback_reason": None,
            "prototype_status": "not_requested",
            "prototype_scope": "minimal_pallas_wkv_execution_probe",
            "parity_status": "not_requested",
            "parity_scope": "tiny_one_step_wkv_update",
            "reason": unavailable_reason,
            "kernel_parity_claimed": False,
            "recommended_next_phase": "P84 broader Pallas WKV shape/dtype parity",
        }

    probe = run_pallas_wkv_parity_probe()
    status = probe.get("prototype_status")
    effective = WKVRuntime.PALLAS.value if status == "pass" else "unavailable"
    return {
        "schema": "qrwkv_xla.p83_pallas_wkv_parity_probe.v1",
        "phase": "P83",
        "default_runtime": WKVRuntime.REFERENCE.value,
        "allowed_runtimes": [runtime.value for runtime in WKVRuntime],
        "reference_default_preserved": reference_default_preserved,
        "wkv_runtime_requested": requested_runtime.value,
        "wkv_runtime_effective": effective,
        "pallas_available": bool(probe.get("pallas_available", False)),
        "fallback_used": False,
        "fallback_reason": None,
        "prototype_status": status,
        "prototype_scope": probe.get(
            "prototype_scope", "minimal_pallas_wkv_execution_probe"
        ),
        "parity_status": probe.get("parity_status"),
        "parity_scope": probe.get("parity_scope", "tiny_one_step_wkv_update"),
        "shape_match": probe.get("shape_match"),
        "reason": probe.get("reason"),
        "probe_backend": probe.get("probe_backend"),
        "probe_shapes": probe.get("probe_shapes"),
        "finite": probe.get("finite"),
        "max_abs_error": probe.get("max_abs_error"),
        "max_rel_error": probe.get("max_rel_error"),
        "atol": probe.get("atol"),
        "rtol": probe.get("rtol"),
        "max_abs_error_vs_formula": probe.get("max_abs_error_vs_formula"),
        "output_abs_max": probe.get("output_abs_max"),
        "kernel_parity_claimed": bool(probe.get("kernel_parity_claimed", False)),
        "recommended_next_phase": probe.get(
            "recommended_next_phase", "P84 kernel runtime selection hardening"
        ),
    }
