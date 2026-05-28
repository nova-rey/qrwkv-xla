from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

WKV_UPDATE_FORMULA = "state * decay[..., None, :] + k[..., :, None] * v[..., None, :]"
P84_SEQUENCE_SCAN_PARITY = "P85 sequence/scan-style Pallas WKV parity"
P84_SHAPE_FIX = "P85 targeted Pallas WKV shape/layout parity fix"
P84_DTYPE_FIX = "P85 targeted Pallas WKV dtype parity fix"
P84_DEPENDENCY_FIX = "P85 targeted Pallas dependency/backend availability fix"
P84_RUNTIME_HARDENING = "P85 kernel runtime selection hardening"


@dataclass(frozen=True)
class PallasWKVParityCase:
    case_id: str
    batch: int
    heads: int
    dim: int
    dtype: str
    atol: float
    rtol: float
    required: bool = True


def pallas_available() -> tuple[bool, str | None]:
    try:
        import jax.experimental.pallas  # noqa: F401
    except Exception as exc:
        return False, f"missing_pallas_dependency_or_backend:{type(exc).__name__}"
    return True, None


def reference_wkv_update(
    state: jax.Array,
    k: jax.Array,
    v: jax.Array,
    decay: jax.Array,
) -> jax.Array:
    return state * decay[..., None, :] + k[..., :, None] * v[..., None, :]


def pallas_wkv_update(
    state: jax.Array,
    k: jax.Array,
    v: jax.Array,
    decay: jax.Array,
) -> jax.Array:
    available, reason = pallas_available()
    if not available:
        raise RuntimeError(reason or "missing_pallas_dependency_or_backend")

    import jax.experimental.pallas as pl

    state = jnp.asarray(state)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    decay = jnp.asarray(decay)
    _validate_probe_shapes(state, k, v, decay)

    def kernel(state_ref, k_ref, v_ref, decay_ref, out_ref):
        state_block = state_ref[:]
        k_block = k_ref[:]
        v_block = v_ref[:]
        decay_block = decay_ref[:]
        out_ref[:] = (
            state_block * decay_block[..., None, :]
            + k_block[..., :, None] * v_block[..., None, :]
        )

    run_kernel = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(state.shape, state.dtype),
        interpret=True,
        name="pallas_one_step_wkv_parity_probe",
    )
    return run_kernel(state, k, v, decay)


def run_pallas_wkv_parity_probe(
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    available, reason = pallas_available()
    if not available:
        return {
            "prototype_status": "unavailable",
            "parity_status": "unavailable",
            "parity_scope": "tiny_one_step_wkv_update",
            "pallas_available": False,
            "reason": reason or "missing_pallas_dependency_or_backend",
            "recommended_next_phase": (
                "P84 targeted Pallas dependency/backend availability fix"
            ),
            "kernel_parity_claimed": False,
            "shape_match": False,
            "finite": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "atol": atol,
            "rtol": rtol,
        }

    try:
        state, k, v, decay = _probe_inputs()
        expected = reference_wkv_update(state, k, v, decay)
        output = pallas_wkv_update(state, k, v, decay)
        output.block_until_ready()
        shape_match = output.shape == expected.shape
        finite = bool(jnp.all(jnp.isfinite(output)))
        abs_error = jnp.abs(output - expected)
        max_abs_error = float(jnp.max(jnp.abs(output - expected)))
        denom = jnp.maximum(jnp.abs(expected), jnp.asarray(1e-12, dtype=expected.dtype))
        max_rel_error = float(jnp.max(abs_error / denom))
    except Exception as exc:
        return {
            "prototype_status": "failed",
            "parity_status": "failed",
            "parity_scope": "tiny_one_step_wkv_update",
            "pallas_available": False,
            "reason": f"pallas_wkv_parity_probe_failed:{type(exc).__name__}:{exc}",
            "recommended_next_phase": (
                "P84 targeted Pallas WKV shape/layout contract fix"
            ),
            "kernel_parity_claimed": False,
            "shape_match": False,
            "finite": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "atol": atol,
            "rtol": rtol,
        }

    parity_pass = bool(
        shape_match and finite and max_abs_error <= atol and max_rel_error <= rtol
    )
    return {
        "prototype_status": "pass",
        "parity_status": "pass" if parity_pass else "fail",
        "parity_scope": "tiny_one_step_wkv_update",
        "pallas_available": True,
        "prototype_scope": "minimal_pallas_wkv_execution_probe",
        "probe_backend": "pallas_call_interpret",
        "probe_shapes": {
            "state": list(state.shape),
            "k": list(k.shape),
            "v": list(v.shape),
            "decay": list(decay.shape),
            "output": list(output.shape),
        },
        "shape_match": shape_match,
        "finite": finite,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "atol": atol,
        "rtol": rtol,
        "kernel_parity_claimed": parity_pass,
        "max_abs_error_vs_formula": max_abs_error,
        "output_abs_max": float(jnp.max(jnp.abs(output))),
        "recommended_next_phase": (
            "P84 broader Pallas WKV shape/dtype parity"
            if parity_pass
            else "P84 targeted Pallas WKV parity fix"
        ),
    }


def pallas_wkv_shape_dtype_parity_cases() -> tuple[PallasWKVParityCase, ...]:
    return (
        PallasWKVParityCase(
            "float32_b1_h1_d2",
            batch=1,
            heads=1,
            dim=2,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVParityCase(
            "float32_b1_h2_d2",
            batch=1,
            heads=2,
            dim=2,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVParityCase(
            "float32_b2_h1_d2",
            batch=2,
            heads=1,
            dim=2,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVParityCase(
            "float32_b1_h1_d4",
            batch=1,
            heads=1,
            dim=4,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVParityCase(
            "float32_b2_h2_d4",
            batch=2,
            heads=2,
            dim=4,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVParityCase(
            "bfloat16_b1_h1_d2",
            batch=1,
            heads=1,
            dim=2,
            dtype="bfloat16",
            atol=5e-2,
            rtol=5e-2,
            required=False,
        ),
        PallasWKVParityCase(
            "bfloat16_b2_h2_d4",
            batch=2,
            heads=2,
            dim=4,
            dtype="bfloat16",
            atol=5e-2,
            rtol=5e-2,
            required=False,
        ),
    )


def run_pallas_wkv_shape_dtype_parity_matrix() -> dict[str, Any]:
    cases = pallas_wkv_shape_dtype_parity_cases()
    available, reason = pallas_available()
    if not available:
        unavailable_cases = [
            {
                **asdict(case),
                "state_shape": [case.batch, case.heads, case.dim, case.dim],
                "k_shape": [case.batch, case.heads, case.dim],
                "v_shape": [case.batch, case.heads, case.dim],
                "decay_shape": [case.batch, case.heads, case.dim],
                "output_shape": None,
                "shape_match": False,
                "finite": False,
                "max_abs_error": None,
                "max_rel_error": None,
                "parity_status": "unavailable",
                "reason": reason or "missing_pallas_dependency_or_backend",
            }
            for case in cases
        ]
        return _p84_matrix_result(
            cases=unavailable_cases,
            pallas_available=False,
            wkv_runtime_effective="unavailable",
            recommended_next_phase=P84_DEPENDENCY_FIX,
            reason=reason or "missing_pallas_dependency_or_backend",
        )

    rows = [_run_pallas_wkv_shape_dtype_case(case) for case in cases]
    return _p84_matrix_result(
        cases=rows,
        pallas_available=True,
        wkv_runtime_effective="pallas",
        recommended_next_phase=_p84_recommendation(rows),
        reason=None,
    )


def run_minimal_pallas_wkv_probe() -> dict[str, Any]:
    probe = run_pallas_wkv_parity_probe()
    return {
        key: value
        for key, value in probe.items()
        if key
        in {
            "prototype_status",
            "pallas_available",
            "reason",
            "prototype_scope",
            "probe_backend",
            "probe_shapes",
            "finite",
            "max_abs_error_vs_formula",
            "output_abs_max",
            "recommended_next_phase",
        }
    }


def _probe_inputs() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    state = jnp.arange(4, dtype=jnp.float32).reshape(1, 1, 2, 2)
    k = jnp.asarray([[[2.0, 3.0]]], dtype=jnp.float32)
    v = jnp.asarray([[[5.0, 7.0]]], dtype=jnp.float32)
    decay = jnp.asarray([[[0.5, 0.25]]], dtype=jnp.float32)
    return state, k, v, decay


def _run_pallas_wkv_shape_dtype_case(case: PallasWKVParityCase) -> dict[str, Any]:
    row = {
        **asdict(case),
        "state_shape": [case.batch, case.heads, case.dim, case.dim],
        "k_shape": [case.batch, case.heads, case.dim],
        "v_shape": [case.batch, case.heads, case.dim],
        "decay_shape": [case.batch, case.heads, case.dim],
        "output_shape": None,
        "shape_match": False,
        "finite": False,
        "max_abs_error": None,
        "max_rel_error": None,
        "parity_status": "unavailable",
        "reason": None,
    }
    try:
        state, k, v, decay = _case_inputs(case)
        expected = reference_wkv_update(state, k, v, decay)
        output = pallas_wkv_update(state, k, v, decay)
        output.block_until_ready()
        shape_match = output.shape == expected.shape
        finite = bool(jnp.all(jnp.isfinite(output)))
        abs_error = jnp.abs(output.astype(jnp.float32) - expected.astype(jnp.float32))
        max_abs_error = float(jnp.max(abs_error))
        denom = jnp.maximum(jnp.abs(expected.astype(jnp.float32)), jnp.asarray(1e-12))
        max_rel_error = float(jnp.max(abs_error / denom))
        parity_pass = bool(
            shape_match
            and finite
            and max_abs_error <= case.atol
            and max_rel_error <= case.rtol
        )
        row.update(
            {
                "output_shape": list(output.shape),
                "shape_match": shape_match,
                "finite": finite,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
                "parity_status": "pass" if parity_pass else "fail",
                "reason": None if parity_pass else "case_exceeded_tolerance_or_shape",
            }
        )
    except Exception as exc:
        row.update(
            {
                "parity_status": "unavailable" if case.dtype == "bfloat16" else "fail",
                "reason": f"pallas_wkv_case_failed:{type(exc).__name__}:{exc}",
            }
        )
    return row


def _case_inputs(
    case: PallasWKVParityCase,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    dtype = _dtype_for_case(case)
    state_size = case.batch * case.heads * case.dim * case.dim
    vector_size = case.batch * case.heads * case.dim
    state = jnp.arange(state_size, dtype=jnp.float32).reshape(
        case.batch, case.heads, case.dim, case.dim
    ) / jnp.asarray(max(case.dim, 1), dtype=jnp.float32)
    k = (
        jnp.arange(vector_size, dtype=jnp.float32).reshape(
            case.batch, case.heads, case.dim
        )
        + 1.0
    )
    v = (
        jnp.arange(vector_size, dtype=jnp.float32).reshape(
            case.batch, case.heads, case.dim
        )
        + 2.0
    )
    decay = jnp.linspace(
        0.25,
        0.75,
        vector_size,
        dtype=jnp.float32,
    ).reshape(case.batch, case.heads, case.dim)
    return state.astype(dtype), k.astype(dtype), v.astype(dtype), decay.astype(dtype)


def _dtype_for_case(case: PallasWKVParityCase) -> jnp.dtype:
    if case.dtype == "float32":
        return jnp.float32
    if case.dtype == "bfloat16":
        return jnp.bfloat16
    raise ValueError(f"unsupported Pallas WKV parity dtype: {case.dtype}")


def _p84_matrix_result(
    *,
    cases: list[dict[str, Any]],
    pallas_available: bool,
    wkv_runtime_effective: str,
    recommended_next_phase: str,
    reason: str | None,
) -> dict[str, Any]:
    summary = _p84_summary(cases)
    return {
        "schema": "qrwkv_xla.p84_pallas_shape_dtype_parity_matrix.v1",
        "phase": "P84",
        "parity_scope": "broader_one_step_wkv_shape_dtype",
        "formula": WKV_UPDATE_FORMULA,
        "default_runtime": "reference",
        "wkv_runtime_requested": "pallas",
        "wkv_runtime_effective": wkv_runtime_effective,
        "pallas_available": pallas_available,
        "backend": "pallas_call_interpret" if pallas_available else None,
        "cases": cases,
        "summary": summary,
        "kernel_parity_claimed": summary["kernel_parity_claimed"],
        "reason": reason,
        "recommended_next_phase": recommended_next_phase,
    }


def _p84_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    required_cases = [case for case in cases if case.get("required", True)]
    cases_pass = sum(1 for case in cases if case.get("parity_status") == "pass")
    cases_fail = sum(1 for case in cases if case.get("parity_status") == "fail")
    cases_unavailable = sum(
        1 for case in cases if case.get("parity_status") == "unavailable"
    )
    all_required_cases_pass = bool(required_cases) and all(
        case.get("parity_status") == "pass" for case in required_cases
    )
    return {
        "cases_total": len(cases),
        "cases_pass": cases_pass,
        "cases_fail": cases_fail,
        "cases_unavailable": cases_unavailable,
        "all_required_cases_pass": all_required_cases_pass,
        "kernel_parity_claimed": all_required_cases_pass,
    }


def _p84_recommendation(cases: list[dict[str, Any]]) -> str:
    required_cases = [case for case in cases if case.get("required", True)]
    if required_cases and all(
        case.get("parity_status") == "pass" for case in required_cases
    ):
        return P84_SEQUENCE_SCAN_PARITY
    failing = [case for case in required_cases if case.get("parity_status") != "pass"]
    if any(case.get("parity_status") == "unavailable" for case in failing):
        return P84_DEPENDENCY_FIX
    if any(case.get("dtype") != "float32" for case in failing):
        return P84_DTYPE_FIX
    if failing:
        return P84_SHAPE_FIX
    return P84_RUNTIME_HARDENING


def _validate_probe_shapes(
    state: jax.Array,
    k: jax.Array,
    v: jax.Array,
    decay: jax.Array,
) -> None:
    if state.ndim != 4:
        raise ValueError(f"state must have shape [B,H,D,D], got {state.shape}")
    if k.shape != state.shape[:3]:
        raise ValueError(f"k must have shape {state.shape[:3]}, got {k.shape}")
    if v.shape != state.shape[:3]:
        raise ValueError(f"v must have shape {state.shape[:3]}, got {v.shape}")
    if decay.shape != state.shape[:3]:
        raise ValueError(f"decay must have shape {state.shape[:3]}, got {decay.shape}")
