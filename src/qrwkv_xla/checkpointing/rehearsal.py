from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.checkpointing.simple import load_checkpoint, save_checkpoint
from qrwkv_xla.students import create_student

CHECKPOINT_RESUME_EXPORT_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "production_checkpointing_ready",
    "production_hf_export_ready",
    "distributed_checkpointing_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
)


@dataclass(frozen=True)
class CheckpointResumeExportRehearsalResult:
    status: str
    checkpoint_roundtrip: bool
    export_roundtrip: bool
    output_match: bool
    checkpoint_path: str
    export_path: str | None
    checkpoint_step: int
    student_architecture: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    tensor_count: int
    reason: str
    claims_not_made: tuple[str, ...] = CHECKPOINT_RESUME_EXPORT_CLAIMS_NOT_MADE
    phase: str = "P108"
    scope: str = "checkpoint_resume_export_rehearsal"

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def run_checkpoint_resume_export_rehearsal(
    *,
    output_dir: str | Path,
    overwrite: bool = True,
    vocab_size: int = 17,
    hidden_size: int = 4,
    num_layers: int = 1,
    step: int = 2,
    learning_rate: float = 0.001,
) -> CheckpointResumeExportRehearsalResult:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    root = Path(output_dir)
    checkpoint_dir = root / "checkpoints" / "p108_tiny_student"
    export_dir = root / "exports" / "p108_tiny_student_hf_safetensors"
    student_config = {
        "architecture": "tiny_student",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "emit_logits": True,
        "tie_embeddings": False,
    }
    student = create_student(
        "tiny_student",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        emit_logits=True,
        tie_embeddings=False,
    )
    params = student.init_params(jax.random.PRNGKey(108))
    save_checkpoint(
        checkpoint_dir,
        params,
        student_architecture="tiny_student",
        student_config=student_config,
        step=step,
        learning_rate=learning_rate,
        loss_config={"logits_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={
            "schema_version": "p108.synthetic_rehearsal",
            "source": "checkpoint_resume_export_rehearsal",
        },
        notes=["P108 checkpoint/resume/export rehearsal"],
        overwrite=overwrite,
    )

    loaded_checkpoint = load_checkpoint(checkpoint_dir)
    checkpoint_roundtrip = loaded_checkpoint.manifest.step == step and _outputs_match(
        student,
        params,
        loaded_checkpoint.params,
        vocab_size=vocab_size,
    )
    if not checkpoint_roundtrip:
        return _result(
            status="fail",
            checkpoint_roundtrip=False,
            export_roundtrip=False,
            output_match=False,
            checkpoint_dir=checkpoint_dir,
            export=None,
            loaded_export_tensor_count=0,
            step=step,
            student_config=student_config,
            reason="checkpoint resume output mismatch",
        )

    try:
        from qrwkv_xla.export import (
            export_checkpoint_to_hf_safetensors,
            load_hf_safetensors_export,
        )

        export = export_checkpoint_to_hf_safetensors(
            checkpoint_dir,
            export_dir,
            overwrite=overwrite,
        )
        loaded_export = load_hf_safetensors_export(export_dir)
    except ImportError as exc:
        return _result(
            status="unavailable",
            checkpoint_roundtrip=True,
            export_roundtrip=False,
            output_match=True,
            checkpoint_dir=checkpoint_dir,
            export=None,
            loaded_export_tensor_count=0,
            step=step,
            student_config=student_config,
            reason=str(exc),
        )

    export_output_match = _outputs_match(
        student,
        loaded_checkpoint.params,
        loaded_export.params,
        vocab_size=vocab_size,
    )
    export_roundtrip = loaded_export.metadata["checkpoint_step"] == step
    passed = export_roundtrip and export_output_match
    return _result(
        status="pass" if passed else "fail",
        checkpoint_roundtrip=True,
        export_roundtrip=export_roundtrip,
        output_match=export_output_match,
        checkpoint_dir=checkpoint_dir,
        export=export,
        loaded_export_tensor_count=len(loaded_export.weight_map),
        step=step,
        student_config=student_config,
        reason=(
            "checkpoint, resume, export, and export reload outputs match"
            if passed
            else "export reload output mismatch"
        ),
    )


def _outputs_match(
    student: Any,
    left_params: dict[str, Any],
    right_params: dict[str, Any],
    *,
    vocab_size: int,
) -> bool:
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32) % vocab_size
    attention_mask = jnp.ones_like(input_ids)
    left = student.apply(left_params, input_ids, attention_mask)
    right = student.apply(right_params, input_ids, attention_mask)
    return bool(
        jnp.array_equal(left.hidden_states, right.hidden_states)
        and left.logits is not None
        and right.logits is not None
        and jnp.array_equal(left.logits, right.logits)
    )


def _result(
    *,
    status: str,
    checkpoint_roundtrip: bool,
    export_roundtrip: bool,
    output_match: bool,
    checkpoint_dir: Path,
    export: Any | None,
    loaded_export_tensor_count: int,
    step: int,
    student_config: dict[str, Any],
    reason: str,
) -> CheckpointResumeExportRehearsalResult:
    return CheckpointResumeExportRehearsalResult(
        status=status,
        checkpoint_roundtrip=checkpoint_roundtrip,
        export_roundtrip=export_roundtrip,
        output_match=output_match,
        checkpoint_path=str(checkpoint_dir),
        export_path=None if export is None else str(export.export_dir),
        checkpoint_step=step,
        student_architecture=str(student_config["architecture"]),
        vocab_size=int(student_config["vocab_size"]),
        hidden_size=int(student_config["hidden_size"]),
        num_layers=int(student_config["num_layers"]),
        tensor_count=loaded_export_tensor_count,
        reason=reason,
    )
