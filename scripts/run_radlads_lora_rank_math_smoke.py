from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from qrwkv_xla.parity import write_parameter_surface_map_reports
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

DEFAULT_OUT = Path("artifacts/p48_radlads_lora_rank_math")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P48 RADLADS low-rank math smoke."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = run_smoke(args.out_dir)
    print(
        f"wrote P48 RADLADS low-rank math smoke to {args.out_dir} "
        f"status={report['status']}"
    )


def run_smoke(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = RWKV7QwenReferenceConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
        emit_mixer_outputs=True,
        radlads_compatible_math=True,
        lora_rank_decay=4,
        lora_rank_iclr=4,
        lora_rank_value_residual_mix=4,
    )
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(48))
    input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 0]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=jnp.int32)
    output, state = student.apply_with_state(params, input_ids, attention_mask)

    assert output.logits is not None
    assert output.mixer_outputs is not None
    finite = bool(
        jnp.all(jnp.isfinite(output.hidden_states))
        and jnp.all(jnp.isfinite(output.logits))
        and jnp.all(jnp.isfinite(output.mixer_outputs))
        and jnp.all(jnp.isfinite(state.wkv_matrix_state))
    )
    report: dict[str, object] = {
        "schema": "qrwkv_xla.p48_radlads_lora_rank_math_report.v1",
        "status": "pass" if finite else "fail",
        "config": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.effective_num_kv_heads,
            "radlads_compatible_math": config.radlads_compatible_math,
            "lora_rank_decay": config.effective_lora_rank_decay,
            "lora_rank_iclr": config.effective_lora_rank_iclr,
            "lora_rank_value_residual_mix": (
                config.effective_lora_rank_value_residual_mix
            ),
        },
        "output_shapes": {
            "hidden_states": list(output.hidden_states.shape),
            "logits": list(output.logits.shape),
            "mixer_outputs": list(output.mixer_outputs.shape),
            "wkv_matrix_state": list(state.wkv_matrix_state.shape),
            "shift_state": list(state.shift_state.shape),
        },
        "finite": finite,
        "notes": [
            "CPU/offline slow-reference smoke only.",
            (
                "r_k is represented but inactive because the source residual "
                "line is commented out."
            ),
        ],
    }
    (out_dir / "lora_rank_math_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P48_RESULTS.md").write_text(_markdown(report), encoding="utf-8")
    write_parameter_surface_map_reports(out_dir)
    p40_map = out_dir / "P40_PARAMETER_SURFACE_MAP.md"
    if p40_map.exists():
        (out_dir / "P48_PARAMETER_SURFACE_MAP.md").write_text(
            p40_map.read_text(encoding="utf-8").replace(
                "P40 RADLADS Parameter Surface Map",
                "P48 RADLADS Parameter Surface Map",
                1,
            ),
            encoding="utf-8",
        )
    return report


def _markdown(report: dict[str, object]) -> str:
    shapes = report["output_shapes"]
    assert isinstance(shapes, dict)
    lines = [
        "# P48 RADLADS LoRA Rank Math Smoke",
        "",
        f"- Status: `{report['status']}`",
        f"- Finite outputs/state: `{report['finite']}`",
        (
            "- Scope: slow-reference math-surface smoke only; no optimized "
            "kernel parity claim."
        ),
        (
            "- r_k: parameter surface represented; source residual use remains "
            "inactive/commented."
        ),
        "",
        "| Tensor | Shape |",
        "| --- | --- |",
    ]
    for name, shape in shapes.items():
        lines.append(f"| {name} | `{shape}` |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
