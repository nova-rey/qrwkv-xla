from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


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
        name="p83_tiny_wkv_parity_probe",
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
