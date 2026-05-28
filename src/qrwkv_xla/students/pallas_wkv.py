from __future__ import annotations

from typing import Any


def pallas_available() -> tuple[bool, str | None]:
    try:
        import jax.experimental.pallas  # noqa: F401
    except Exception as exc:
        return False, f"missing_pallas_dependency_or_backend:{type(exc).__name__}"
    return True, None


def run_minimal_pallas_wkv_probe() -> dict[str, Any]:
    available, reason = pallas_available()
    if not available:
        return {
            "prototype_status": "unavailable",
            "pallas_available": False,
            "reason": reason or "missing_pallas_dependency_or_backend",
            "recommended_next_phase": (
                "P83 targeted Pallas dependency/backend availability fix"
            ),
        }

    try:
        import jax
        import jax.experimental.pallas as pl
        import jax.numpy as jnp

        state = jnp.arange(4, dtype=jnp.float32).reshape(1, 1, 2, 2)
        k = jnp.asarray([[[2.0, 3.0]]], dtype=jnp.float32)
        v = jnp.asarray([[[5.0, 7.0]]], dtype=jnp.float32)
        decay = jnp.asarray([[[0.5, 0.25]]], dtype=jnp.float32)

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
            name="p82_minimal_wkv_probe",
        )
        output = run_kernel(state, k, v, decay)
        output.block_until_ready()
        expected = state * decay[..., None, :] + k[..., :, None] * v[..., None, :]
        max_abs_error = float(jnp.max(jnp.abs(output - expected)))
        finite = bool(jnp.all(jnp.isfinite(output)))
    except Exception as exc:
        return {
            "prototype_status": "failed",
            "pallas_available": False,
            "reason": f"minimal_pallas_wkv_probe_failed:{type(exc).__name__}:{exc}",
            "recommended_next_phase": (
                "P83 targeted Pallas WKV shape/layout contract fix"
            ),
        }

    return {
        "prototype_status": "pass",
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
        "finite": finite,
        "max_abs_error_vs_formula": max_abs_error,
        "output_abs_max": float(jnp.max(jnp.abs(output))),
        "recommended_next_phase": "P83 reference-vs-Pallas parity gate",
    }
