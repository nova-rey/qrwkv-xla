from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
CHECKPOINT_RESUME_UPDATE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "production_checkpointing_ready",
    "distributed_training_ready",
    "hf_export_ready",
    "training_ready",
    "qwen_specific_support",
    "large_scale_performance_proven",
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


@dataclass(frozen=True)
class CheckpointResumeUpdateRehearsalResult:
    status: str
    initial_loss: float
    checkpoint_loss: float
    resumed_loss: float
    checkpoint_step: int
    final_step: int
    steps_after_resume: int
    restored_matches: bool
    resumed_loss_finite: bool
    params_changed_after_resume: bool
    checkpoint_path: str
    path_used: str
    claims_not_made: tuple[str, ...] = CHECKPOINT_RESUME_UPDATE_CLAIMS_NOT_MADE
    phase: str = "P108.1"
    scope: str = "resume_update_closure"

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


def run_checkpoint_resume_update_rehearsal(
    *,
    output_dir: str | Path,
    overwrite: bool = True,
    steps_before_checkpoint: int = 2,
    steps_after_resume: int = 1,
    learning_rate: float = 0.2,
) -> CheckpointResumeUpdateRehearsalResult:
    if steps_before_checkpoint <= 0:
        raise ValueError(
            f"steps_before_checkpoint must be > 0, got {steps_before_checkpoint}"
        )
    if steps_after_resume <= 0:
        raise ValueError(f"steps_after_resume must be > 0, got {steps_after_resume}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    checkpoint_dir = Path(output_dir) / "checkpoints" / "p108_1_resume_update"
    params = _resume_update_initial_params()
    target = _resume_update_target()
    initial_loss = _resume_update_loss(params, target)

    checkpoint_params = _resume_update_steps(
        params,
        target,
        steps=steps_before_checkpoint,
        learning_rate=learning_rate,
    )
    checkpoint_loss = _resume_update_loss(checkpoint_params, target)
    save_checkpoint(
        checkpoint_dir,
        checkpoint_params,
        student_architecture="tiny_resume_update_state",
        student_config={"state_size": int(checkpoint_params["weights"].shape[0])},
        step=steps_before_checkpoint,
        learning_rate=learning_rate,
        loss_config={
            "mse": {
                "enabled": True,
                "initial_loss": float(initial_loss),
                "checkpoint_loss": float(checkpoint_loss),
            }
        },
        target_manifest={
            "schema_version": "p108_1.resume_update_closure",
            "target": target.tolist(),
        },
        notes=["P108.1 resume update closure"],
        overwrite=overwrite,
    )

    loaded = load_checkpoint(checkpoint_dir)
    restored_matches = (
        loaded.manifest.step == steps_before_checkpoint
        and float(loaded.manifest.loss_config["mse"]["checkpoint_loss"])
        == float(checkpoint_loss)
        and _trees_equal(checkpoint_params, loaded.params)
    )
    resumed_params = _resume_update_steps(
        loaded.params,
        target,
        steps=steps_after_resume,
        learning_rate=learning_rate,
    )
    resumed_loss = _resume_update_loss(resumed_params, target)
    final_step = loaded.manifest.step + steps_after_resume
    resumed_loss_finite = bool(np.isfinite(resumed_loss))
    params_changed_after_resume = not _trees_equal(loaded.params, resumed_params)
    passed = (
        restored_matches
        and resumed_loss_finite
        and params_changed_after_resume
        and final_step == steps_before_checkpoint + steps_after_resume
        and resumed_loss <= checkpoint_loss
    )
    return CheckpointResumeUpdateRehearsalResult(
        status="pass" if passed else "fail",
        initial_loss=float(initial_loss),
        checkpoint_loss=float(checkpoint_loss),
        resumed_loss=float(resumed_loss),
        checkpoint_step=steps_before_checkpoint,
        final_step=final_step,
        steps_after_resume=steps_after_resume,
        restored_matches=restored_matches,
        resumed_loss_finite=resumed_loss_finite,
        params_changed_after_resume=params_changed_after_resume,
        checkpoint_path=str(checkpoint_dir),
        path_used="tiny_deterministic_mse_update",
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


def _resume_update_initial_params() -> dict[str, np.ndarray]:
    return {
        "weights": np.asarray([0.25, -0.5, 0.75, 1.25], dtype=np.float32),
    }


def _resume_update_target() -> np.ndarray:
    return np.asarray([1.0, -1.0, 0.5, 0.0], dtype=np.float32)


def _resume_update_loss(
    params: dict[str, Any],
    target: np.ndarray,
) -> float:
    weights = np.asarray(params["weights"], dtype=np.float32)
    return float(np.mean(np.square(weights - target)))


def _resume_update_steps(
    params: dict[str, Any],
    target: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, np.ndarray]:
    weights = np.asarray(params["weights"], dtype=np.float32)
    for _ in range(steps):
        grad = (2.0 / weights.size) * (weights - target)
        weights = weights - learning_rate * grad
    return {"weights": np.asarray(weights, dtype=np.float32)}


def _trees_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _trees_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _trees_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _trees_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


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
