from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.checkpointing import save_checkpoint
from qrwkv_xla.export import (
    export_checkpoint_to_hf_safetensors,
    load_hf_safetensors_export,
)
from qrwkv_xla.students import create_student

DEFAULT_OUTPUT_DIR = Path("artifacts/p41_hf_safetensors_export_smoke")
REPORT_JSON = "export_smoke_report.json"
REPORT_MD = "P41_EXPORT_SMOKE_REPORT.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny local QRWKV-XLA checkpoint to HF/safetensors smoke"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report = run_export_smoke(output_dir, overwrite=args.overwrite)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"export_smoke_report: {output_dir / REPORT_JSON}")
        print(f"passed: {report['passed']}")


def run_export_smoke(output_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"smoke output already exists at {output_dir}; pass --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_config = {
        "architecture": "tiny_student",
        "vocab_size": 32,
        "hidden_size": 8,
        "num_layers": 2,
        "emit_logits": True,
        "tie_embeddings": False,
        "emit_mixer_outputs": False,
    }
    student = create_student(
        "tiny_student",
        vocab_size=student_config["vocab_size"],
        hidden_size=student_config["hidden_size"],
        num_layers=student_config["num_layers"],
        emit_logits=student_config["emit_logits"],
        tie_embeddings=student_config["tie_embeddings"],
    )
    params = student.init_params(jax.random.PRNGKey(41))
    checkpoint_dir = output_dir / "checkpoints" / "source"
    save_checkpoint(
        checkpoint_dir,
        params,
        student_architecture="tiny_student",
        student_config=student_config,
        step=1,
        learning_rate=0.001,
        loss_config={"export_smoke": {"enabled": True, "weight": 1.0}},
        target_manifest={"schema_version": "p41_export_smoke", "source": "local_tiny"},
        notes=["P41 tiny local HF/safetensors export smoke checkpoint"],
        overwrite=True,
    )

    export = export_checkpoint_to_hf_safetensors(
        checkpoint_dir,
        output_dir,
        overwrite=True,
    )
    reloaded = load_hf_safetensors_export(output_dir)
    input_ids = jnp.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    original_output = student.apply(params, input_ids, attention_mask)
    reloaded_output = reloaded.student.apply(reloaded.params, input_ids, attention_mask)
    if original_output.logits is None or reloaded_output.logits is None:
        raise ValueError("export smoke requires logits-capable student outputs")

    hidden_max_abs_diff = _max_abs_diff(
        original_output.hidden_states,
        reloaded_output.hidden_states,
    )
    logits_max_abs_diff = _max_abs_diff(original_output.logits, reloaded_output.logits)
    passed = hidden_max_abs_diff == 0.0 and logits_max_abs_diff == 0.0
    report = {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "P41",
        "passed": passed,
        "source_checkpoint": str(checkpoint_dir),
        "export_dir": str(output_dir),
        "files": {
            "config": str(export.config_path),
            "model": str(export.model_path),
            "metadata": str(export.metadata_path),
            "weight_map": str(export.weight_map_path),
            "json_report": str(output_dir / REPORT_JSON),
            "markdown_report": str(output_dir / REPORT_MD),
        },
        "student_architecture": "tiny_student",
        "student_config": student_config,
        "input_shape": list(input_ids.shape),
        "tensor_count": export.metadata["tensor_count"],
        "hidden_max_abs_diff": hidden_max_abs_diff,
        "logits_max_abs_diff": logits_max_abs_diff,
        "limitations": [
            "tiny CPU/local checkpoint export and reload parity only",
            "no production Hugging Face model class",
            "no sharded/pjit export",
            "no Qwen-scale export",
            "no model quality claim",
        ],
    }
    (output_dir / REPORT_JSON).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(_markdown_report(report), encoding="utf-8")
    if not passed:
        raise ValueError(
            "export smoke parity failed: "
            f"hidden_max_abs_diff={hidden_max_abs_diff}, "
            f"logits_max_abs_diff={logits_max_abs_diff}"
        )
    return report


def _max_abs_diff(left: jax.Array, right: jax.Array) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _markdown_report(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    return (
        "# P41 HF Safetensors Export Smoke Report\n\n"
        f"- status: {status}\n"
        f"- source checkpoint: `{report['source_checkpoint']}`\n"
        f"- export dir: `{report['export_dir']}`\n"
        f"- student architecture: `{report['student_architecture']}`\n"
        f"- tensor count: {report['tensor_count']}\n"
        f"- hidden max abs diff: {report['hidden_max_abs_diff']}\n"
        f"- logits max abs diff: {report['logits_max_abs_diff']}\n\n"
        "This proves only tiny local checkpoint export/reload parity for the "
        "QRWKV-XLA helper loader. It does not provide a production Hugging Face "
        "model class, Qwen-scale export, sharded export, or any model quality "
        "claim.\n"
    )


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(f"P41 export smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (FileExistsError, FileNotFoundError, ValueError, TypeError) as exc:
        print(f"P41 export smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
