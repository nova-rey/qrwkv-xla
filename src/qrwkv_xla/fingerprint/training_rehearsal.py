from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qrwkv_xla.artifacts import (
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillCheckpointConfig,
    DistillFingerprintConfig,
    DistillStageConfig,
    run_distill_stage,
)
from qrwkv_xla.fingerprint.real_teacher import (
    DEFAULT_TINY_REAL_TEACHER,
    TinyRealTeacherFingerprintCaptureConfig,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.teachers import HFTeacherBackend

DEFAULT_TINY_TEXTS = Path(
    "tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl"
)


@dataclass(frozen=True)
class RealTeacherFingerprintTrainingRehearsalConfig:
    output_dir: Path
    fingerprint_artifact: Path | None = None
    build_real_teacher_artifact: bool = False
    texts_path: Path = DEFAULT_TINY_TEXTS
    teacher_model: str = DEFAULT_TINY_REAL_TEACHER
    tokenizer: str | None = None
    sequence_length: int = 32
    max_examples: int = 4
    max_target_positions: int = 64
    max_exemplars: int = 16
    bounds_method: str = "quantile"
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    exemplar_selection_policy: str = "stratified_interestingness_v0"
    per_mode_min: int = 1
    local_files_only: bool = True
    allow_downloads: bool = False
    training_steps: int = 3
    batch_size: int = 2
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    seed: int = 0
    student_backend: str = "current_qrwkv"
    resume_from: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class RealTeacherFingerprintTrainingRehearsalResult:
    status: str
    output_dir: Path
    report_path: Path
    summary_path: Path
    artifact_dir: Path
    artifact_source: str
    teacher_real: bool
    teacher_model_name_or_path: str
    teacher_required_during_training: bool
    optimizer_steps_completed: int
    params_changed: bool
    param_delta_norm: float
    final_loss: float
    checkpoint_dir: Path | None


def run_real_teacher_fingerprint_training_rehearsal(
    config: RealTeacherFingerprintTrainingRehearsalConfig,
    *,
    backend: HFTeacherBackend | None = None,
) -> RealTeacherFingerprintTrainingRehearsalResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir, artifact_source, capture_result = _resolve_artifact(
        config,
        backend=backend,
    )

    validation = validate_fingerprint_artifact(artifact_dir)
    if not validation.ok:
        report_path = config.output_dir / "p146_rehearsal_report.json"
        summary_path = config.output_dir / "p146_rehearsal_summary.md"
        report = _failure_report(
            config=config,
            artifact_dir=artifact_dir,
            artifact_source=artifact_source,
            validation=validation,
        )
        write_json(report_path, report)
        summary_path.write_text(_render_summary(report), encoding="utf-8")
        return RealTeacherFingerprintTrainingRehearsalResult(
            status="fail",
            output_dir=config.output_dir,
            report_path=report_path,
            summary_path=summary_path,
            artifact_dir=artifact_dir,
            artifact_source=artifact_source,
            teacher_real=False,
            teacher_model_name_or_path="",
            teacher_required_during_training=False,
            optimizer_steps_completed=0,
            params_changed=False,
            param_delta_norm=0.0,
            final_loss=float("nan"),
            checkpoint_dir=None,
        )

    artifact = summarize_fingerprint_artifact(artifact_dir)
    capture_summary = _read_optional_json(artifact_dir / "capture_summary.json")
    training_output_dir = config.output_dir / "training"
    distill_config = DistillStageConfig(
        stage=146,
        mode=DISTILL_MODE_FINGERPRINT_CORRIDOR,
        training=replace(
            DistillStageConfig().training,
            max_steps=config.training_steps,
            seed=config.seed,
        ),
        optimizer=replace(
            DistillStageConfig().optimizer,
            type=config.optimizer,
            learning_rate=config.learning_rate,
        ),
        checkpoint=DistillCheckpointConfig(
            checkpoint_out=training_output_dir / "checkpoints" / "final",
            resume_from=config.resume_from,
            overwrite=True,
        ),
        fingerprint=DistillFingerprintConfig(
            artifact_dir=artifact_dir,
            batch_size=config.batch_size,
            student_backend=config.student_backend,
            output_dir=training_output_dir,
            input_conditioned_rehearsal=True,
        ),
    )
    training = run_distill_stage(distill_config)
    training_report = (
        _read_optional_json(training.report_path)
        if training.report_path is not None
        else {}
    )
    checkpoint_loadable = _checkpoint_loadable(training.checkpoint_out)
    training_payload = _training_payload(
        training=training,
        training_report=training_report,
        checkpoint_loadable=checkpoint_loadable,
    )
    teacher_real = bool(
        capture_summary.get(
            "teacher_real",
            capture_result.teacher_real if capture_result is not None else True,
        )
    )
    report = {
        "phase": "P146",
        "run_kind": "real_teacher_artifact_student_training_rehearsal",
        "status": _overall_status(
            training_status=training.status,
            training_payload=training_payload,
            artifact_validated=validation.ok,
        ),
        "artifact_source": artifact_source,
        "teacher_real": teacher_real,
        "teacher_model_name_or_path": str(
            capture_summary.get(
                "teacher_model_name_or_path",
                capture_result.teacher_model_name_or_path
                if capture_result is not None
                else artifact.teacher_model_name,
            )
        ),
        "local_files_only": bool(
            capture_summary.get(
                "local_files_only",
                capture_result.local_files_only
                if capture_result is not None
                else config.local_files_only,
            )
        ),
        "capture": _capture_payload(
            artifact=artifact,
            artifact_dir=artifact_dir,
            validation=validation,
            capture_summary=capture_summary,
        ),
        "training": training_payload,
        "limitations": [
            "This is a tiny real-teacher artifact to real student rehearsal.",
            "The teacher is used only to build the fingerprint artifact.",
            "Student training uses fingerprint_corridor mode without a teacher.",
            "No quality, baseline, or model-improvement claim is made.",
        ],
    }
    report_path = config.output_dir / "p146_rehearsal_report.json"
    summary_path = config.output_dir / "p146_rehearsal_summary.md"
    write_json(report_path, report)
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    return RealTeacherFingerprintTrainingRehearsalResult(
        status=str(report["status"]),
        output_dir=config.output_dir,
        report_path=report_path,
        summary_path=summary_path,
        artifact_dir=artifact_dir,
        artifact_source=artifact_source,
        teacher_real=teacher_real,
        teacher_model_name_or_path=str(report["teacher_model_name_or_path"]),
        teacher_required_during_training=False,
        optimizer_steps_completed=int(training_payload["optimizer_steps_completed"]),
        params_changed=bool(training_payload["params_changed"]),
        param_delta_norm=float(training_payload["param_delta_norm"]),
        final_loss=float(training_payload["final_loss"]),
        checkpoint_dir=training.checkpoint_out,
    )


def _validate_config(config: RealTeacherFingerprintTrainingRehearsalConfig) -> None:
    if config.fingerprint_artifact is None and not config.build_real_teacher_artifact:
        raise ValueError(
            "set fingerprint_artifact or enable build_real_teacher_artifact"
        )
    if config.fingerprint_artifact is not None and config.build_real_teacher_artifact:
        raise ValueError(
            "fingerprint_artifact and build_real_teacher_artifact are mutually "
            "exclusive"
        )
    if config.training_steps <= 0:
        raise ValueError("training_steps must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if config.optimizer not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer must be one of {'sgd', 'adam', 'adamw'}")


def _resolve_artifact(
    config: RealTeacherFingerprintTrainingRehearsalConfig,
    *,
    backend: HFTeacherBackend | None,
) -> tuple[Path, str, object | None]:
    if config.fingerprint_artifact is not None:
        return config.fingerprint_artifact, "existing_artifact", None

    artifact_dir = config.output_dir / "p145_artifact"
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=artifact_dir,
            texts_path=config.texts_path,
            teacher_model=config.teacher_model,
            tokenizer=config.tokenizer,
            sequence_length=config.sequence_length,
            max_examples=config.max_examples,
            max_target_positions=config.max_target_positions,
            local_files_only=config.local_files_only,
            allow_downloads=config.allow_downloads,
            overwrite=config.overwrite,
            max_exemplars=config.max_exemplars,
            bounds_method=config.bounds_method,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            exemplar_selection_policy=config.exemplar_selection_policy,
            per_mode_min=config.per_mode_min,
        ),
        backend=backend,
    )
    return artifact_dir, "built_from_tiny_real_teacher", result


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return read_json_object(path)


def _checkpoint_loadable(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        load_checkpoint(path)
    except (OSError, ValueError):
        return False
    return True


def _capture_payload(
    *,
    artifact: object,
    artifact_dir: Path,
    validation: object,
    capture_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_dir": str(artifact_dir),
        "capture_engine": str(
            capture_summary.get(
                "capture_engine",
                "teacher_side_capture_skeleton_v0",
            )
        ),
        "artifact_validated": bool(validation.ok),
        "validation_status": validation.status,
        "validation_blockers": list(validation.blockers),
        "modes_discovered": int(artifact.num_modes),
        "target_positions_processed": int(artifact.num_corridor_records),
        "exemplars_retained": int(artifact.num_exemplar_records),
        "vocab_size": int(artifact.vocab_size),
        "max_seq_len": int(artifact.max_seq_len),
        "tracked_stats": list(artifact.tracked_stats),
    }


def _training_payload(
    *,
    training: object,
    training_report: dict[str, Any],
    checkpoint_loadable: bool,
) -> dict[str, Any]:
    final_metrics = training.final_metrics or {}
    params_changed = bool(
        final_metrics.get("fingerprint/rehearsal/params_changed", 0.0) == 1.0
    )
    initial_loss = float(
        final_metrics.get("fingerprint/rehearsal/initial_loss", training.initial_loss)
    )
    final_loss = float(
        final_metrics.get("fingerprint/rehearsal/final_loss", training.final_loss)
    )
    loss_delta = float(
        final_metrics.get("fingerprint/rehearsal/loss_delta", final_loss - initial_loss)
    )
    return {
        "distill_mode": training.distill_mode,
        "training_path_kind": training.training_path_kind,
        "main_runner_integrated": bool(training.main_runner_integrated),
        "real_student_backend_integrated": bool(
            training.real_student_backend_integrated
        ),
        "student_backend": training.student_backend,
        "teacher_required_during_training": bool(training.teacher_required),
        "optimizer_steps_completed": int(training.optimizer_steps_completed),
        "batches_consumed": int(training.batches_consumed),
        "params_changed": params_changed,
        "param_delta_norm": float(
            final_metrics.get("fingerprint/rehearsal/param_delta_norm", 0.0)
        ),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "loss_non_increasing": bool(
            final_metrics.get("fingerprint/rehearsal/loss_non_increasing", 0.0) == 1.0
        ),
        "loss_non_increasing_required": bool(
            training_report.get("loss_non_increasing_required", False)
        ),
        "metrics_finite": all(
            math.isfinite(float(value)) for value in final_metrics.values()
        ),
        "checkpoint_written": training.checkpoint_out is not None
        and (training.checkpoint_out / "checkpoint.json").is_file()
        and (training.checkpoint_out / "params.npz").is_file(),
        "checkpoint_loadable": checkpoint_loadable,
        "checkpoint_dir": (
            None if training.checkpoint_out is None else str(training.checkpoint_out)
        ),
        "runner_report_path": (
            None if training.report_path is None else str(training.report_path)
        ),
        "student": training_report.get("student", {}),
        "artifact": training_report.get("artifact", {}),
    }


def _overall_status(
    *,
    training_status: str,
    training_payload: dict[str, Any],
    artifact_validated: bool,
) -> str:
    required = (
        artifact_validated,
        training_status == "pass",
        training_payload["main_runner_integrated"],
        training_payload["real_student_backend_integrated"],
        training_payload["teacher_required_during_training"] is False,
        training_payload["optimizer_steps_completed"] > 0,
        training_payload["batches_consumed"] > 0,
        training_payload["params_changed"],
        training_payload["param_delta_norm"] > 0.0,
        training_payload["metrics_finite"],
        training_payload["checkpoint_written"],
        training_payload["checkpoint_loadable"],
    )
    return "pass" if all(required) else "fail"


def _failure_report(
    *,
    config: RealTeacherFingerprintTrainingRehearsalConfig,
    artifact_dir: Path,
    artifact_source: str,
    validation: object,
) -> dict[str, Any]:
    return {
        "phase": "P146",
        "run_kind": "real_teacher_artifact_student_training_rehearsal",
        "status": "fail",
        "artifact_source": artifact_source,
        "teacher_real": False,
        "teacher_model_name_or_path": config.teacher_model,
        "local_files_only": config.local_files_only,
        "capture": {
            "artifact_dir": str(artifact_dir),
            "artifact_validated": False,
            "validation_status": validation.status,
            "validation_blockers": list(validation.blockers),
        },
        "training": {
            "distill_mode": DISTILL_MODE_FINGERPRINT_CORRIDOR,
            "teacher_required_during_training": False,
            "optimizer_steps_completed": 0,
            "params_changed": False,
            "checkpoint_written": False,
        },
    }


def _render_summary(report: dict[str, Any]) -> str:
    capture = report["capture"]
    training = report["training"]
    return "\n".join(
        (
            "# P146 Real Teacher Fingerprint Training Rehearsal",
            "",
            f"Status: {report['status']}",
            f"Artifact source: {report['artifact_source']}",
            f"Teacher real: {report['teacher_real']}",
            "Teacher required during training: false",
            "",
            "This is a tiny real-teacher artifact to real student training "
            "rehearsal. The teacher was used only to build the fingerprint "
            "artifact. Student training used fingerprint_corridor mode and did "
            "not require the teacher. No quality or baseline claim is made.",
            "",
            "## Capture",
            f"- Artifact: {capture['artifact_dir']}",
            f"- Artifact validated: {capture['artifact_validated']}",
            f"- Modes discovered: {capture.get('modes_discovered', 0)}",
            f"- Target positions: {capture.get('target_positions_processed', 0)}",
            f"- Exemplars retained: {capture.get('exemplars_retained', 0)}",
            "",
            "## Training",
            f"- Mode: {training['distill_mode']}",
            f"- Path: {training.get('training_path_kind')}",
            f"- Optimizer steps: {training['optimizer_steps_completed']}",
            f"- Params changed: {training['params_changed']}",
            f"- Param delta norm: {training.get('param_delta_norm', 0.0)}",
            f"- Initial loss: {training.get('initial_loss')}",
            f"- Final loss: {training.get('final_loss')}",
            f"- Loss delta: {training.get('loss_delta')}",
            f"- Checkpoint: {training.get('checkpoint_dir')}",
            "",
        )
    )
