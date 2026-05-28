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
P85_FUSED_SCAN_SCAFFOLD = "P86 fused/scan Pallas WKV kernel scaffold"
P85_SEQUENCE_PARITY_FIX = "P86 targeted Pallas sequence parity fix"
P85_SEQUENCE_SHAPE_FIX = "P86 targeted Pallas sequence shape/layout fix"
P85_SEQUENCE_DTYPE_FIX = "P86 targeted Pallas sequence dtype fix"
P85_DEPENDENCY_FIX = "P86 targeted Pallas dependency/backend availability fix"
P85_RUNTIME_HARDENING = "P86 kernel runtime selection hardening"


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


@dataclass(frozen=True)
class PallasWKVSequenceParityCase:
    case_id: str
    batch: int
    heads: int
    dim: int
    seq_len: int
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


def reference_wkv_sequence_update(
    initial_state: jax.Array,
    k_seq: jax.Array,
    v_seq: jax.Array,
    decay_seq: jax.Array,
) -> dict[str, jax.Array]:
    state = jnp.asarray(initial_state)
    k_seq = jnp.asarray(k_seq)
    v_seq = jnp.asarray(v_seq)
    decay_seq = jnp.asarray(decay_seq)
    _validate_sequence_shapes(state, k_seq, v_seq, decay_seq)

    states = []
    for token in range(k_seq.shape[0]):
        state = reference_wkv_update(
            state,
            k_seq[token],
            v_seq[token],
            decay_seq[token],
        )
        states.append(state)
    return {"final_state": state, "per_step_states": jnp.stack(states, axis=0)}


def pallas_wkv_sequence_update_repeated(
    initial_state: jax.Array,
    k_seq: jax.Array,
    v_seq: jax.Array,
    decay_seq: jax.Array,
) -> dict[str, jax.Array]:
    state = jnp.asarray(initial_state)
    k_seq = jnp.asarray(k_seq)
    v_seq = jnp.asarray(v_seq)
    decay_seq = jnp.asarray(decay_seq)
    _validate_sequence_shapes(state, k_seq, v_seq, decay_seq)

    states = []
    for token in range(k_seq.shape[0]):
        state = pallas_wkv_update(
            state,
            k_seq[token],
            v_seq[token],
            decay_seq[token],
        )
        states.append(state)
    per_step_states = jnp.stack(states, axis=0)
    per_step_states.block_until_ready()
    return {"final_state": state, "per_step_states": per_step_states}


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


def pallas_wkv_sequence_parity_cases() -> tuple[PallasWKVSequenceParityCase, ...]:
    return (
        PallasWKVSequenceParityCase(
            "float32_b1_h1_d2_t2",
            batch=1,
            heads=1,
            dim=2,
            seq_len=2,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVSequenceParityCase(
            "float32_b1_h1_d2_t4",
            batch=1,
            heads=1,
            dim=2,
            seq_len=4,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVSequenceParityCase(
            "float32_b1_h2_d2_t4",
            batch=1,
            heads=2,
            dim=2,
            seq_len=4,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVSequenceParityCase(
            "float32_b2_h2_d4_t4",
            batch=2,
            heads=2,
            dim=4,
            seq_len=4,
            dtype="float32",
            atol=1e-6,
            rtol=1e-6,
        ),
        PallasWKVSequenceParityCase(
            "bfloat16_b1_h1_d2_t4",
            batch=1,
            heads=1,
            dim=2,
            seq_len=4,
            dtype="bfloat16",
            atol=5e-2,
            rtol=5e-2,
            required=False,
        ),
        PallasWKVSequenceParityCase(
            "bfloat16_b2_h2_d4_t4",
            batch=2,
            heads=2,
            dim=4,
            seq_len=4,
            dtype="bfloat16",
            atol=5e-2,
            rtol=5e-2,
            required=False,
        ),
    )


def run_pallas_wkv_sequence_parity_matrix() -> dict[str, Any]:
    cases = pallas_wkv_sequence_parity_cases()
    available, reason = pallas_available()
    if not available:
        unavailable_cases = [
            {
                **asdict(case),
                "initial_state_shape": [case.batch, case.heads, case.dim, case.dim],
                "k_seq_shape": [case.seq_len, case.batch, case.heads, case.dim],
                "v_seq_shape": [case.seq_len, case.batch, case.heads, case.dim],
                "decay_seq_shape": [case.seq_len, case.batch, case.heads, case.dim],
                "final_state_shape": None,
                "per_step_states_shape": None,
                "final_shape_match": False,
                "per_step_shape_match": False,
                "final_state_finite": False,
                "per_step_finite": False,
                "final_max_abs_error": None,
                "final_max_rel_error": None,
                "worst_step_max_abs_error": None,
                "worst_step_max_rel_error": None,
                "parity_status": "unavailable",
                "reason": reason or "missing_pallas_dependency_or_backend",
            }
            for case in cases
        ]
        return _p85_matrix_result(
            cases=unavailable_cases,
            pallas_available=False,
            wkv_runtime_effective="unavailable",
            recommended_next_phase=P85_DEPENDENCY_FIX,
            reason=reason or "missing_pallas_dependency_or_backend",
        )

    rows = [_run_pallas_wkv_sequence_case(case) for case in cases]
    return _p85_matrix_result(
        cases=rows,
        pallas_available=True,
        wkv_runtime_effective="pallas",
        recommended_next_phase=_p85_recommendation(rows),
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


def _run_pallas_wkv_sequence_case(
    case: PallasWKVSequenceParityCase,
) -> dict[str, Any]:
    row = {
        **asdict(case),
        "initial_state_shape": [case.batch, case.heads, case.dim, case.dim],
        "k_seq_shape": [case.seq_len, case.batch, case.heads, case.dim],
        "v_seq_shape": [case.seq_len, case.batch, case.heads, case.dim],
        "decay_seq_shape": [case.seq_len, case.batch, case.heads, case.dim],
        "final_state_shape": None,
        "per_step_states_shape": None,
        "final_shape_match": False,
        "per_step_shape_match": False,
        "final_state_finite": False,
        "per_step_finite": False,
        "final_max_abs_error": None,
        "final_max_rel_error": None,
        "worst_step_max_abs_error": None,
        "worst_step_max_rel_error": None,
        "parity_status": "unavailable",
        "reason": None,
    }
    try:
        initial_state, k_seq, v_seq, decay_seq = _sequence_case_inputs(case)
        expected = reference_wkv_sequence_update(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )
        output = pallas_wkv_sequence_update_repeated(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )
        final_state = output["final_state"]
        per_step_states = output["per_step_states"]
        expected_final = expected["final_state"]
        expected_steps = expected["per_step_states"]
        final_shape_match = final_state.shape == expected_final.shape
        per_step_shape_match = per_step_states.shape == expected_steps.shape
        final_state_finite = bool(jnp.all(jnp.isfinite(final_state)))
        per_step_finite = bool(jnp.all(jnp.isfinite(per_step_states)))
        final_abs_error = jnp.abs(
            final_state.astype(jnp.float32) - expected_final.astype(jnp.float32)
        )
        step_abs_error = jnp.abs(
            per_step_states.astype(jnp.float32) - expected_steps.astype(jnp.float32)
        )
        final_max_abs_error = float(jnp.max(final_abs_error))
        final_denom = jnp.maximum(
            jnp.abs(expected_final.astype(jnp.float32)),
            jnp.asarray(1e-12),
        )
        final_max_rel_error = float(jnp.max(final_abs_error / final_denom))
        worst_step_max_abs_error = float(jnp.max(step_abs_error))
        step_denom = jnp.maximum(
            jnp.abs(expected_steps.astype(jnp.float32)),
            jnp.asarray(1e-12),
        )
        worst_step_max_rel_error = float(jnp.max(step_abs_error / step_denom))
        parity_pass = bool(
            final_shape_match
            and per_step_shape_match
            and final_state_finite
            and per_step_finite
            and final_max_abs_error <= case.atol
            and final_max_rel_error <= case.rtol
            and worst_step_max_abs_error <= case.atol
            and worst_step_max_rel_error <= case.rtol
        )
        row.update(
            {
                "final_state_shape": list(final_state.shape),
                "per_step_states_shape": list(per_step_states.shape),
                "final_shape_match": final_shape_match,
                "per_step_shape_match": per_step_shape_match,
                "final_state_finite": final_state_finite,
                "per_step_finite": per_step_finite,
                "final_max_abs_error": final_max_abs_error,
                "final_max_rel_error": final_max_rel_error,
                "worst_step_max_abs_error": worst_step_max_abs_error,
                "worst_step_max_rel_error": worst_step_max_rel_error,
                "parity_status": "pass" if parity_pass else "fail",
                "reason": None
                if parity_pass
                else "sequence_case_exceeded_tolerance_or_shape",
            }
        )
    except Exception as exc:
        row.update(
            {
                "parity_status": "unavailable" if case.dtype == "bfloat16" else "fail",
                "reason": f"pallas_wkv_sequence_case_failed:{type(exc).__name__}:{exc}",
            }
        )
    return row


def _sequence_case_inputs(
    case: PallasWKVSequenceParityCase,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    dtype = _dtype_for_case(case)
    state_size = case.batch * case.heads * case.dim * case.dim
    vector_size = case.seq_len * case.batch * case.heads * case.dim
    initial_state = jnp.arange(state_size, dtype=jnp.float32).reshape(
        case.batch, case.heads, case.dim, case.dim
    ) / jnp.asarray(case.dim + 1, dtype=jnp.float32)
    base = jnp.arange(vector_size, dtype=jnp.float32).reshape(
        case.seq_len, case.batch, case.heads, case.dim
    )
    input_scale = jnp.asarray(max(vector_size, 1), dtype=jnp.float32)
    k_seq = (base + 1.0) / (input_scale * jnp.asarray(case.dim + 2, dtype=jnp.float32))
    v_seq = (base + 2.0) / (input_scale * jnp.asarray(case.dim + 3, dtype=jnp.float32))
    decay_seq = jnp.linspace(
        0.20,
        0.80,
        vector_size,
        dtype=jnp.float32,
    ).reshape(case.seq_len, case.batch, case.heads, case.dim)
    return (
        initial_state.astype(dtype),
        k_seq.astype(dtype),
        v_seq.astype(dtype),
        decay_seq.astype(dtype),
    )


def _dtype_for_case(
    case: PallasWKVParityCase | PallasWKVSequenceParityCase,
) -> jnp.dtype:
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


def _p85_matrix_result(
    *,
    cases: list[dict[str, Any]],
    pallas_available: bool,
    wkv_runtime_effective: str,
    recommended_next_phase: str,
    reason: str | None,
) -> dict[str, Any]:
    summary = _p85_summary(cases)
    return {
        "schema": "qrwkv_xla.p85_pallas_sequence_parity_matrix.v1",
        "phase": "P85",
        "parity_scope": "short_sequence_repeated_one_step_wkv",
        "one_step_formula": WKV_UPDATE_FORMULA,
        "sequence_method": "repeated_one_step_pallas",
        "default_runtime": "reference",
        "wkv_runtime_requested": "pallas",
        "wkv_runtime_effective": wkv_runtime_effective,
        "pallas_available": pallas_available,
        "backend": "pallas_call_interpret" if pallas_available else None,
        "cases": cases,
        "summary": summary,
        "kernel_parity_claimed": summary["kernel_parity_claimed"],
        "fused_sequence_kernel_status": "not_implemented",
        "reason": reason,
        "recommended_next_phase": recommended_next_phase,
    }


def _p85_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
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
        "worst_final_max_abs_error": _max_present(cases, "final_max_abs_error"),
        "worst_final_max_rel_error": _max_present(cases, "final_max_rel_error"),
        "worst_step_max_abs_error": _max_present(cases, "worst_step_max_abs_error"),
        "worst_step_max_rel_error": _max_present(cases, "worst_step_max_rel_error"),
    }


def _p85_recommendation(cases: list[dict[str, Any]]) -> str:
    required_cases = [case for case in cases if case.get("required", True)]
    if required_cases and all(
        case.get("parity_status") == "pass" for case in required_cases
    ):
        return P85_FUSED_SCAN_SCAFFOLD
    failing = [case for case in required_cases if case.get("parity_status") != "pass"]
    if any(case.get("parity_status") == "unavailable" for case in failing):
        return P85_DEPENDENCY_FIX
    if any(case.get("dtype") != "float32" for case in failing):
        return P85_SEQUENCE_DTYPE_FIX
    if any(
        not case.get("final_shape_match") or not case.get("per_step_shape_match")
        for case in failing
    ):
        return P85_SEQUENCE_SHAPE_FIX
    if failing:
        return P85_SEQUENCE_PARITY_FIX
    return P85_RUNTIME_HARDENING


def _max_present(cases: list[dict[str, Any]], key: str) -> float | None:
    values = [case.get(key) for case in cases if case.get(key) is not None]
    return max(values) if values else None


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


def _validate_sequence_shapes(
    initial_state: jax.Array,
    k_seq: jax.Array,
    v_seq: jax.Array,
    decay_seq: jax.Array,
) -> None:
    if k_seq.ndim != 4:
        raise ValueError(f"k_seq must have shape [T,B,H,D], got {k_seq.shape}")
    if v_seq.shape != k_seq.shape:
        raise ValueError(f"v_seq must have shape {k_seq.shape}, got {v_seq.shape}")
    if decay_seq.shape != k_seq.shape:
        raise ValueError(
            f"decay_seq must have shape {k_seq.shape}, got {decay_seq.shape}"
        )
    if initial_state.ndim != 4:
        raise ValueError(
            f"initial_state must have shape [B,H,D,D], got {initial_state.shape}"
        )
    if initial_state.shape[:3] != k_seq.shape[1:]:
        raise ValueError(
            "initial_state leading shape must match sequence [B,H,D], got "
            f"{initial_state.shape[:3]} and {k_seq.shape[1:]}"
        )
    if initial_state.shape[-1] != initial_state.shape[-2]:
        raise ValueError(
            f"initial_state must be square in D, got {initial_state.shape}"
        )
