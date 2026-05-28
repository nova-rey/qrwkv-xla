from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("artifacts/p88_tpu_compile_smoke/pallas_tpu_compile_smoke.json")
DEFAULT_MARKDOWN = Path(
    "artifacts/p88_tpu_compile_smoke/P88_TPU_COMPILE_PERFORMANCE_SMOKE_REPORT.md"
)
CLAIMS_NOT_MADE = [
    "production_pallas_ready",
    "training_ready",
    "throughput_proven",
    "full_model_quality_proven",
    "pallas_default_ready",
]
RECOMMENDED_NEXT = (
    "P89 cleanup/hardening only if P88 exposes issues; otherwise post-Pallas "
    "Radjax extraction planning"
)


def inspect_jax_environment() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:
        return {
            "jax_import_ok": False,
            "jax_version": "unknown",
            "jaxlib_version": "unknown",
            "platform": "unknown",
            "default_backend": "unknown",
            "devices": [],
            "tpu_devices_detected": False,
            "error_type": type(exc).__name__,
            "error_message": _bounded(str(exc)),
        }

    try:
        import jaxlib

        jaxlib_version = str(jaxlib.__version__)
    except Exception:
        jaxlib_version = "unknown"

    devices = [_device_info(device) for device in jax.devices()]
    platforms = sorted({device["platform"] for device in devices})
    return {
        "jax_import_ok": True,
        "jax_version": str(jax.__version__),
        "jaxlib_version": jaxlib_version,
        "platform": ",".join(platforms) if platforms else "unknown",
        "default_backend": str(jax.default_backend()),
        "devices": devices,
        "tpu_devices_detected": any(device["platform"] == "tpu" for device in devices),
        "error_type": None,
        "error_message": None,
    }


def run_smoke(*, require_tpu: bool = False) -> dict[str, Any]:
    env = inspect_jax_environment()
    base = _base_report(env)
    pallas_import_ok = _pallas_import_ok()
    pallas_runtime_import_ok = _pallas_runtime_import_ok()
    base.update(
        {
            "pallas_import_ok": pallas_import_ok,
            "pallas_runtime_import_ok": pallas_runtime_import_ok,
        }
    )

    if not env["jax_import_ok"]:
        return {
            **base,
            "status": "fail",
            "reason": "jax_import_failed",
            "error_type": env["error_type"],
            "error_message": env["error_message"],
        }
    if not env["tpu_devices_detected"]:
        return {
            **base,
            "status": "unavailable",
            "reason": "no_tpu_devices_detected",
            "require_tpu": require_tpu,
        }
    if not pallas_import_ok or not pallas_runtime_import_ok:
        return {
            **base,
            "status": "fail",
            "reason": "pallas_import_failed",
        }

    try:
        result = _run_tpu_compile_execution()
    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "reason": "tpu_compile_or_execution_failed",
            "error_type": type(exc).__name__,
            "error_message": _bounded(str(exc)),
            "jit_lowering_attempted": True,
            "execution_attempted": True,
        }

    return {
        **base,
        **result,
        "status": "pass" if result["numeric_check_ok"] else "fail",
        "reason": "pass" if result["numeric_check_ok"] else "numeric_check_failed",
    }


def write_reports(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = output.parent / DEFAULT_MARKDOWN.name
    markdown.write_text(_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P88 opt-in Pallas TPU compile/execution smoke."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-tpu", action="store_true")
    args = parser.parse_args()

    report = run_smoke(require_tpu=args.require_tpu)
    write_reports(report, args.output)
    print(f"P88 Pallas TPU smoke status={report['status']} reason={report['reason']}")
    if report["status"] == "fail" or (
        args.require_tpu and report["status"] == "unavailable"
    ):
        raise SystemExit(1)


def _run_tpu_compile_execution() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from qrwkv_xla.students.pallas_wkv import (
        pallas_wkv_update,
        reference_wkv_sequence_update,
    )

    initial_state = jnp.arange(4, dtype=jnp.float32).reshape(1, 1, 2, 2) / 3.0
    k_seq = jnp.asarray([[[[0.1, 0.2]]], [[[0.3, 0.4]]]], dtype=jnp.float32)
    v_seq = jnp.asarray([[[[0.2, 0.3]]], [[[0.4, 0.5]]]], dtype=jnp.float32)
    decay_seq = jnp.asarray([[[[0.5, 0.25]]], [[[0.75, 0.5]]]], dtype=jnp.float32)

    def smoke_fn(state, k, v, decay):
        def step(carry, item):
            token_k, token_v, token_decay = item
            next_state = pallas_wkv_update(carry, token_k, token_v, token_decay)
            return next_state, next_state

        return jax.lax.scan(step, state, (k, v, decay))

    lowered = jax.jit(smoke_fn).lower(initial_state, k_seq, v_seq, decay_seq)
    compiled = lowered.compile()
    final_state, per_step_states = compiled(initial_state, k_seq, v_seq, decay_seq)
    final_state, per_step_states = jax.block_until_ready((final_state, per_step_states))
    expected = reference_wkv_sequence_update(initial_state, k_seq, v_seq, decay_seq)
    final_error = jnp.max(jnp.abs(final_state - expected["final_state"]))
    step_error = jnp.max(jnp.abs(per_step_states - expected["per_step_states"]))
    max_abs_error = float(jnp.maximum(final_error, step_error))
    return {
        "pallas_interpret_mode": True,
        "jit_lowering_attempted": True,
        "jit_lowering_ok": True,
        "execution_attempted": True,
        "execution_ok": True,
        "numeric_check_attempted": True,
        "numeric_check_ok": bool(max_abs_error <= 1e-6),
        "max_abs_error": max_abs_error,
    }


def _base_report(env: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "P88",
        "status": "unavailable",
        "reason": "not_run",
        "jax_version": env["jax_version"],
        "jaxlib_version": env["jaxlib_version"],
        "platform": env["platform"],
        "default_backend": env["default_backend"],
        "devices": env["devices"],
        "tpu_devices_detected": env["tpu_devices_detected"],
        "pallas_import_ok": False,
        "pallas_runtime_import_ok": False,
        "pallas_interpret_mode": True,
        "jit_lowering_attempted": False,
        "jit_lowering_ok": False,
        "execution_attempted": False,
        "execution_ok": False,
        "numeric_check_attempted": False,
        "numeric_check_ok": False,
        "max_abs_error": None,
        "scope": "tiny_pallas_wkv_compile_execution_smoke",
        "claims_not_made": CLAIMS_NOT_MADE,
        "recommended_next_phase": RECOMMENDED_NEXT,
        "error_type": None,
        "error_message": None,
    }


def _pallas_import_ok() -> bool:
    try:
        import jax.experimental.pallas  # noqa: F401
    except Exception:
        return False
    return True


def _pallas_runtime_import_ok() -> bool:
    try:
        from qrwkv_xla.students import (
            pallas_wkv_sequence_update_fused_or_scan,  # noqa: F401
        )
    except Exception:
        return False
    return True


def _device_info(device: object) -> dict[str, str]:
    return {
        "id": str(getattr(device, "id", "?")),
        "platform": str(getattr(device, "platform", "unknown")),
        "device_kind": str(getattr(device, "device_kind", type(device).__name__)),
    }


def _bounded(message: str, limit: int = 500) -> str:
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P88 TPU Compile / Performance Smoke Report",
            "",
            f"- status: `{report['status']}`",
            f"- reason: `{report['reason']}`",
            f"- default_backend: `{report['default_backend']}`",
            f"- tpu_devices_detected: `{report['tpu_devices_detected']}`",
            f"- pallas_import_ok: `{report['pallas_import_ok']}`",
            f"- pallas_runtime_import_ok: `{report['pallas_runtime_import_ok']}`",
            f"- jit_lowering_attempted: `{report['jit_lowering_attempted']}`",
            f"- execution_attempted: `{report['execution_attempted']}`",
            f"- numeric_check_ok: `{report['numeric_check_ok']}`",
            f"- max_abs_error: `{report['max_abs_error']}`",
            f"- recommended_next_phase: `{report['recommended_next_phase']}`",
            "",
            "## Claims Not Made",
            *[f"- {claim}" for claim in report["claims_not_made"]],
            "",
        ]
    )


if __name__ == "__main__":
    main()
