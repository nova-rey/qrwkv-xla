from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from qrwkv_xla.artifacts import summarize_fingerprint_artifact
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.real_teacher import (
    DEFAULT_TINY_REAL_TEACHER,
    TinyRealTeacherFingerprintCaptureConfig,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.fingerprint.training_rehearsal import (
    DEFAULT_TINY_TEXTS,
    RealTeacherFingerprintTrainingRehearsalConfig,
    run_real_teacher_fingerprint_training_rehearsal,
)
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.teachers import HFTeacherBackend


@dataclass(frozen=True)
class FingerprintBaselineComparisonConfig:
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
    local_files_only: bool = True
    allow_downloads: bool = False
    steps: int = 3
    batch_size: int = 2
    learning_rate: float = 0.01
    seed: int = 0
    student_backend: str = "current_qrwkv"
    overwrite: bool = False


@dataclass(frozen=True)
class FingerprintBaselineComparisonResult:
    status: str
    output_dir: Path
    report_path: Path
    summary_path: Path
    artifact_dir: Path
    arms_run: tuple[str, ...]


def run_fingerprint_baseline_comparison(
    config: FingerprintBaselineComparisonConfig,
    *,
    backend: HFTeacherBackend | None = None,
) -> FingerprintBaselineComparisonResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir, artifact_source, capture_summary = _resolve_artifact(
        config,
        backend=backend,
    )
    artifact = summarize_fingerprint_artifact(artifact_dir)
    baseline_arm = _run_baseline_init_only(config, artifact=artifact)
    fingerprint_arm = _run_fingerprint_arm(
        config,
        artifact_dir=artifact_dir,
        artifact_source=artifact_source,
    )
    report = _report(
        config=config,
        artifact_dir=artifact_dir,
        artifact_source=artifact_source,
        artifact=artifact,
        capture_summary=capture_summary,
        baseline_arm=baseline_arm,
        fingerprint_arm=fingerprint_arm,
    )
    report_path = config.output_dir / "p147_comparison_report.json"
    summary_path = config.output_dir / "p147_comparison_summary.md"
    write_json(report_path, report)
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    return FingerprintBaselineComparisonResult(
        status=str(report["status"]),
        output_dir=config.output_dir,
        report_path=report_path,
        summary_path=summary_path,
        artifact_dir=artifact_dir,
        arms_run=tuple(arm["arm_id"] for arm in report["arms"]),
    )


def _validate_config(config: FingerprintBaselineComparisonConfig) -> None:
    if config.fingerprint_artifact is None and not config.build_real_teacher_artifact:
        raise ValueError(
            "set fingerprint_artifact or enable build_real_teacher_artifact"
        )
    if config.fingerprint_artifact is not None and config.build_real_teacher_artifact:
        raise ValueError(
            "fingerprint_artifact and build_real_teacher_artifact are mutually "
            "exclusive"
        )
    if config.steps <= 0:
        raise ValueError("steps must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")


def _resolve_artifact(
    config: FingerprintBaselineComparisonConfig,
    *,
    backend: HFTeacherBackend | None,
) -> tuple[Path, str, dict[str, Any]]:
    if config.fingerprint_artifact is not None:
        artifact_dir = config.fingerprint_artifact
        return (
            artifact_dir,
            "existing_artifact",
            _read_optional_json(artifact_dir / "capture_summary.json"),
        )

    artifact_dir = config.output_dir / "p145_artifact"
    capture = run_tiny_real_teacher_fingerprint_capture(
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
        ),
        backend=backend,
    )
    return (
        artifact_dir,
        "built_from_tiny_real_teacher",
        read_json_object(capture.summary_path),
    )


def _run_baseline_init_only(
    config: FingerprintBaselineComparisonConfig,
    *,
    artifact: object,
) -> dict[str, Any]:
    backend, student_config = _create_backend_and_config(
        artifact=artifact,
        student_backend=config.student_backend,
    )
    params = backend.init_params(jax.random.PRNGKey(config.seed))
    checkpoint_dir = config.output_dir / "baseline_init_only" / "checkpoints" / "final"
    optimizer_config = OptimizerConfig(type="sgd", learning_rate=config.learning_rate)
    optimizer_state = init_optimizer_state(params, optimizer_config)
    save_checkpoint(
        checkpoint_dir,
        params,
        student_architecture=config.student_backend,
        student_config=student_config,
        step=0,
        learning_rate=config.learning_rate,
        loss_config={"baseline": "init_only"},
        target_manifest={
            "artifact_type": artifact.artifact_type,
            "artifact_version": artifact.artifact_version,
            "artifact_dir": artifact.artifact_dir,
            "baseline_kind": "no_fingerprint_init_only",
        },
        optimizer_config={"type": "sgd", "learning_rate": config.learning_rate},
        optimizer_state=optimizer_state,
        notes=[
            "P147 init-only no-fingerprint baseline",
            "zero optimizer updates",
        ],
        overwrite=True,
    )
    checkpoint_loadable = _checkpoint_loadable(checkpoint_dir)
    return {
        "arm_id": "baseline_init_only",
        "status": "pass" if checkpoint_loadable else "fail",
        "student_backend": config.student_backend,
        "student_config": student_config,
        "seed": config.seed,
        "steps_requested": 0,
        "batch_size": None,
        "learning_rate": config.learning_rate,
        "sequence_length": artifact.max_seq_len,
        "vocab_size": artifact.vocab_size,
        "artifact_dir": None,
        "artifact_kind": "none",
        "artifact_size_bytes": 0,
        "teacher_required_during_training": False,
        "optimizer_steps_completed": 0,
        "params_changed": False,
        "param_delta_norm": 0.0,
        "initial_loss": None,
        "final_loss": None,
        "loss_delta": None,
        "loss_non_increasing": None,
        "checkpoint_written": _checkpoint_files_exist(checkpoint_dir),
        "checkpoint_loadable": checkpoint_loadable,
        "checkpoint_dir": str(checkpoint_dir),
        "baseline_loss": None,
    }


def _run_fingerprint_arm(
    config: FingerprintBaselineComparisonConfig,
    *,
    artifact_dir: Path,
    artifact_source: str,
) -> dict[str, Any]:
    output_dir = config.output_dir / "fingerprint_corridor"
    result = run_real_teacher_fingerprint_training_rehearsal(
        RealTeacherFingerprintTrainingRehearsalConfig(
            output_dir=output_dir,
            fingerprint_artifact=artifact_dir,
            build_real_teacher_artifact=False,
            texts_path=config.texts_path,
            teacher_model=config.teacher_model,
            tokenizer=config.tokenizer,
            sequence_length=config.sequence_length,
            max_examples=config.max_examples,
            max_target_positions=config.max_target_positions,
            max_exemplars=config.max_exemplars,
            local_files_only=config.local_files_only,
            allow_downloads=config.allow_downloads,
            training_steps=config.steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            student_backend=config.student_backend,
            overwrite=config.overwrite,
        )
    )
    report = read_json_object(result.report_path)
    training = report["training"]
    capture = report["capture"]
    corridor_metrics = _read_runner_corridor_metrics(training)
    return {
        "arm_id": "fingerprint_corridor",
        "status": result.status,
        "artifact_source": artifact_source,
        "distill_mode": training["distill_mode"],
        "training_path_kind": training["training_path_kind"],
        "main_runner_integrated": training["main_runner_integrated"],
        "real_student_backend_integrated": training["real_student_backend_integrated"],
        "student_backend": config.student_backend,
        "student_config": training["student"],
        "seed": config.seed,
        "steps_requested": config.steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "sequence_length": capture["max_seq_len"],
        "vocab_size": capture["vocab_size"],
        "artifact_dir": str(artifact_dir),
        "artifact_kind": "behavioral_fingerprint",
        "artifact_size_bytes": _directory_size_bytes(artifact_dir),
        "teacher_required_during_training": training[
            "teacher_required_during_training"
        ],
        "optimizer_steps_completed": training["optimizer_steps_completed"],
        "batches_consumed": training["batches_consumed"],
        "params_changed": training["params_changed"],
        "param_delta_norm": training["param_delta_norm"],
        "initial_loss": training["initial_loss"],
        "final_loss": training["final_loss"],
        "loss_delta": training["loss_delta"],
        "loss_non_increasing": training["loss_non_increasing"],
        "loss_non_increasing_required": training["loss_non_increasing_required"],
        "checkpoint_written": training["checkpoint_written"],
        "checkpoint_loadable": training["checkpoint_loadable"],
        "checkpoint_dir": training["checkpoint_dir"],
        "runner_report_path": training["runner_report_path"],
        "p146_report_path": str(result.report_path),
        "fingerprint/corridor/loss_total": corridor_metrics.get("loss_total"),
        "fingerprint/corridor/inside_all_rate": corridor_metrics.get("inside_all_rate"),
        "modes_discovered": capture["modes_discovered"],
        "target_positions_processed": capture["target_positions_processed"],
        "exemplars_retained": capture["exemplars_retained"],
    }


def _report(
    *,
    config: FingerprintBaselineComparisonConfig,
    artifact_dir: Path,
    artifact_source: str,
    artifact: object,
    capture_summary: dict[str, Any],
    baseline_arm: dict[str, Any],
    fingerprint_arm: dict[str, Any],
) -> dict[str, Any]:
    controls = _comparison_controls(
        baseline_arm=baseline_arm,
        fingerprint_arm=fingerprint_arm,
    )
    arms = [baseline_arm, fingerprint_arm]
    return {
        "phase": "P147",
        "run_kind": "baseline_comparison_harness",
        "status": "pass" if all(arm["status"] == "pass" for arm in arms) else "fail",
        "artifact": {
            "source": artifact_source,
            "artifact_dir": str(artifact_dir),
            "teacher_real": bool(capture_summary.get("teacher_real", True)),
            "teacher_model_name_or_path": str(
                capture_summary.get(
                    "teacher_model_name_or_path",
                    artifact.teacher_model_name,
                )
            ),
            "modes_discovered": artifact.num_modes,
            "target_positions_processed": artifact.num_corridor_records,
            "exemplars_retained": artifact.num_exemplar_records,
            "artifact_size_bytes": _directory_size_bytes(artifact_dir),
            "target_records": artifact.num_corridor_records,
            "exemplar_records": artifact.num_exemplar_records,
            "teacher_tokens_processed": capture_summary.get("tokens_processed"),
            "vocab_size": artifact.vocab_size,
            "max_seq_len": artifact.max_seq_len,
        },
        "comparison_controls": controls,
        "arms": arms,
        "claims": {
            "quality_claim_made": False,
            "baseline_winner_declared": False,
            "quality_per_byte_claim_made": False,
        },
        "limitations": [
            "tiny smoke only",
            "baseline_init_only performs zero optimizer updates",
            "not a quality benchmark",
            "no winner is declared",
            "P148 is the first quality-per-byte experiment",
        ],
    }


def _comparison_controls(
    *,
    baseline_arm: dict[str, Any],
    fingerprint_arm: dict[str, Any],
) -> dict[str, Any]:
    return {
        "same_student_backend": (
            baseline_arm["student_backend"] == fingerprint_arm["student_backend"]
        ),
        "same_student_config": (
            baseline_arm["student_config"] == fingerprint_arm["student_config"]
        ),
        "same_seed": baseline_arm["seed"] == fingerprint_arm["seed"],
        "same_training_steps_where_applicable": True,
        "same_batch_size_where_applicable": True,
        "same_eval_texts": True,
        "limitations": [
            "baseline_init_only has zero training steps by design",
            "tiny smoke only",
            "not a quality benchmark",
        ],
        "notes": [],
    }


def _create_backend_and_config(
    *,
    artifact: object,
    student_backend: str,
) -> tuple[Any, dict[str, Any]]:
    vocab_contract = VocabContract(
        tokenizer_id=artifact.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=artifact.tokenizer_name or None,
        vocab_size=artifact.vocab_size,
        model_id=artifact.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=vocab_contract,
        architecture_id=student_backend,
    )
    return backend, _student_config(
        backend=backend,
        architecture_id=student_backend,
        vocab_size=artifact.vocab_size,
    )


def _student_config(
    *,
    backend: Any,
    architecture_id: str,
    vocab_size: int,
) -> dict[str, Any]:
    student = getattr(backend, "student", None)
    raw_config = getattr(student, "config", None)
    return {
        "architecture_id": architecture_id,
        "backend_name": type(backend).__name__,
        "architecture": getattr(raw_config, "__class__", type("", (), {})).__name__,
        "vocab_size": int(getattr(raw_config, "vocab_size", vocab_size)),
        "hidden_size": int(getattr(raw_config, "hidden_size", 0)),
        "num_layers": int(getattr(raw_config, "num_layers", 0)),
        "num_heads": _optional_int(getattr(raw_config, "num_heads", None)),
        "num_kv_heads": _optional_int(getattr(raw_config, "num_kv_heads", None)),
        "emit_logits": bool(getattr(raw_config, "emit_logits", True)),
        "tie_embeddings": bool(getattr(raw_config, "tie_embeddings", False)),
        "emit_mixer_outputs": bool(getattr(raw_config, "emit_mixer_outputs", False)),
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _read_runner_corridor_metrics(training: dict[str, Any]) -> dict[str, Any]:
    path_value = training.get("runner_report_path")
    if path_value is None:
        return {}
    report = read_json_object(Path(path_value))
    return dict(report.get("corridor_metrics", {}))


def _checkpoint_files_exist(path: Path) -> bool:
    return (path / "checkpoint.json").is_file() and (path / "params.npz").is_file()


def _checkpoint_loadable(path: Path) -> bool:
    try:
        load_checkpoint(path)
    except (OSError, ValueError):
        return False
    return True


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object(path)


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _render_summary(report: dict[str, Any]) -> str:
    artifact = report["artifact"]
    arms = report["arms"]
    controls = report["comparison_controls"]
    return "\n".join(
        (
            "# P147 Baseline Comparison Harness",
            "",
            f"Status: {report['status']}",
            f"Artifact source: {artifact['source']}",
            f"Artifact: {artifact['artifact_dir']}",
            "",
            "This harness verifies that the repo can run comparable tiny "
            "baseline and fingerprint arms. It does not establish that "
            "fingerprint training is better.",
            "",
            "## Arms",
            *(
                f"- {arm['arm_id']}: status={arm['status']}, "
                f"steps={arm['optimizer_steps_completed']}, "
                f"params_changed={arm['params_changed']}, "
                f"final_loss={arm['final_loss']}"
                for arm in arms
            ),
            "",
            "## Shared Controls",
            f"- Same student backend: {controls['same_student_backend']}",
            f"- Same student config: {controls['same_student_config']}",
            f"- Same seed: {controls['same_seed']}",
            "- Same eval texts: true",
            "",
            "## Artifact Budget",
            f"- Size bytes: {artifact['artifact_size_bytes']}",
            f"- Target records: {artifact['target_records']}",
            f"- Exemplar records: {artifact['exemplar_records']}",
            f"- Modes discovered: {artifact['modes_discovered']}",
            "",
            "## Claims",
            "- Quality claim made: false",
            "- Baseline winner declared: false",
            "- Quality-per-byte claim made: false",
            "",
            "P148 is the first quality-per-byte experiment.",
            "",
        )
    )
