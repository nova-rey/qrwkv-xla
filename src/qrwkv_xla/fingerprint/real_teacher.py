from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.distill import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillFingerprintConfig,
    DistillStageConfig,
    run_distill_stage,
)
from qrwkv_xla.fingerprint.capture import (
    FingerprintCaptureBudgetConfig,
    FingerprintCaptureConfig,
    FingerprintCaptureExample,
    FingerprintCorridorBoundsConfig,
    FingerprintExemplarReservoirCaptureConfig,
    FingerprintModeDiscoveryConfig,
    capture_fingerprint_artifact,
)
from qrwkv_xla.teachers import HFTeacherBackend, HFTeacherUnavailable
from qrwkv_xla.training import (
    RealStudentFingerprintForwardConfig,
    run_real_student_fingerprint_forward_smoke,
)

DEFAULT_TINY_REAL_TEACHER = "sshleifer/tiny-gpt2"
DEFAULT_CONSUMER_VOCAB_LIMIT = 4096


@dataclass(frozen=True)
class TinyRealTeacherFingerprintCaptureConfig:
    output_dir: Path
    texts_path: Path
    teacher_model: str = DEFAULT_TINY_REAL_TEACHER
    tokenizer: str | None = None
    sequence_length: int = 32
    max_examples: int = 4
    max_target_positions: int = 64
    local_files_only: bool = True
    allow_downloads: bool = False
    overwrite: bool = False
    max_exemplars: int = 16
    bounds_method: str = "quantile"
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    exemplar_selection_policy: str = "stratified_interestingness_v0"
    per_mode_min: int = 1
    consumer_vocab_limit: int = DEFAULT_CONSUMER_VOCAB_LIMIT
    example_id_prefix: str = "p145-real-teacher"


@dataclass(frozen=True)
class TinyRealTeacherFingerprintCaptureResult:
    status: str
    output_dir: Path
    artifact_validated: bool
    summary_path: Path
    manifest_path: Path
    teacher_real: bool
    teacher_backend: str
    teacher_model_name_or_path: str
    tokenizer_name_or_path: str
    local_files_only: bool
    examples_processed: int
    tokens_processed: int
    target_positions_processed: int
    modes_discovered: int
    exemplars_retained: int
    consumer_sanity: dict[str, Any]
    reason: str | None = None


def run_tiny_real_teacher_fingerprint_capture(
    config: TinyRealTeacherFingerprintCaptureConfig,
    *,
    backend: HFTeacherBackend | None = None,
) -> TinyRealTeacherFingerprintCaptureResult:
    _validate_config(config)
    texts = load_text_fixture(config.texts_path)
    prompts = tuple(texts[: config.max_examples])
    if not prompts:
        raise ValueError("tiny real teacher capture requires at least one text")

    effective_local_files_only = (
        False if config.allow_downloads else config.local_files_only
    )
    teacher = backend or HFTeacherBackend(
        config.teacher_model,
        local_files_only=effective_local_files_only,
        prompts=prompts,
    )
    teacher.prompts = prompts
    try:
        emitted = teacher.emit_targets(
            num_examples=len(prompts),
            sequence_length=config.sequence_length,
        )
    except HFTeacherUnavailable:
        raise

    input_ids = np.asarray(emitted["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(emitted["attention_mask"], dtype=np.int32)
    logits = np.asarray(emitted["logits"], dtype=np.float32)
    _validate_emitted_shapes(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits=logits,
        examples=len(prompts),
        sequence_length=config.sequence_length,
    )

    vocab_size = int(logits.shape[-1])
    examples = tuple(
        FingerprintCaptureExample(
            example_id=f"{config.example_id_prefix}-{index:06d}",
            input_ids=tuple(int(token) for token in input_ids[index]),
            logits=logits[index],
        )
        for index in range(input_ids.shape[0])
    )
    capture_config = FingerprintCaptureConfig(
        output_dir=config.output_dir,
        overwrite=config.overwrite,
        teacher_model_name=config.teacher_model,
        tokenizer_name=config.tokenizer or config.teacher_model,
        capture_budget=FingerprintCaptureBudgetConfig(
            max_examples=config.max_examples,
            max_target_positions=config.max_target_positions,
        ),
        mode_discovery=FingerprintModeDiscoveryConfig(),
        corridor_bounds=FingerprintCorridorBoundsConfig(
            method=config.bounds_method,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        ),
        exemplar_reservoir=FingerprintExemplarReservoirCaptureConfig(
            enabled=True,
            max_exemplars=config.max_exemplars,
            selection_policy=config.exemplar_selection_policy,
            per_mode_min=config.per_mode_min,
        ),
    )
    capture = capture_fingerprint_artifact(capture_config, examples)
    _rewrite_manifest_for_p145(
        capture.manifest_path,
        config=config,
        teacher=teacher,
        vocab_size=vocab_size,
        effective_local_files_only=effective_local_files_only,
    )
    validation = validate_fingerprint_artifact(config.output_dir)
    targets_loadable = False
    exemplars_loadable = False
    target_records = 0
    exemplar_records = 0
    if validation.ok:
        targets = load_fingerprint_targets(config.output_dir, batch_size=1)
        target_records = targets.num_records
        targets_loadable = True
        exemplars = load_fingerprint_exemplars(config.output_dir, batch_size=1)
        exemplar_records = exemplars.num_records
        exemplars_loadable = True
    artifact_summary = summarize_fingerprint_artifact(config.output_dir)
    base_summary = read_json_object(capture.capture_summary_path)
    consumer_sanity = _run_consumer_sanity(
        config,
        vocab_size=vocab_size,
        validation_ok=validation.ok,
    )
    summary = _p145_summary(
        config=config,
        base_summary=base_summary,
        teacher=teacher,
        effective_local_files_only=effective_local_files_only,
        examples_processed=len(prompts),
        tokens_processed=int(np.sum(attention_mask)),
        target_positions_processed=target_records,
        vocab_size=vocab_size,
        artifact_validated=validation.ok,
        validation_blockers=validation.blockers,
        targets_loadable=targets_loadable,
        exemplars_loadable=exemplars_loadable,
        exemplar_records=exemplar_records,
        artifact_summary=artifact_summary.to_dict(),
        consumer_sanity=consumer_sanity,
    )
    write_json(capture.capture_summary_path, summary)
    status = "pass" if validation.ok and targets_loadable else "fail"
    return TinyRealTeacherFingerprintCaptureResult(
        status=status,
        output_dir=config.output_dir,
        artifact_validated=validation.ok,
        summary_path=capture.capture_summary_path,
        manifest_path=capture.manifest_path,
        teacher_real=True,
        teacher_backend="hf_causal_lm",
        teacher_model_name_or_path=teacher.model_id,
        tokenizer_name_or_path=config.tokenizer or config.teacher_model,
        local_files_only=effective_local_files_only,
        examples_processed=len(prompts),
        tokens_processed=int(np.sum(attention_mask)),
        target_positions_processed=target_records,
        modes_discovered=int(summary["modes_discovered"]),
        exemplars_retained=exemplar_records,
        consumer_sanity=consumer_sanity,
    )


def load_text_fixture(path: str | Path) -> tuple[str, ...]:
    rows: list[str] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(
                f"text fixture line {line_number} must contain string text"
            )
        text = payload["text"].strip()
        if text:
            rows.append(text)
    return tuple(rows)


def _validate_config(config: TinyRealTeacherFingerprintCaptureConfig) -> None:
    if not config.example_id_prefix.strip():
        raise ValueError("example_id_prefix must be non-empty")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if config.max_examples <= 0:
        raise ValueError("max_examples must be > 0")
    if config.max_target_positions <= 0:
        raise ValueError("max_target_positions must be > 0")
    if config.max_exemplars < 0:
        raise ValueError("max_exemplars must be >= 0")
    if config.consumer_vocab_limit < 0:
        raise ValueError("consumer_vocab_limit must be >= 0")
    if config.local_files_only and config.allow_downloads:
        raise ValueError("local_files_only and allow_downloads cannot both be true")


def _validate_emitted_shapes(
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    logits: np.ndarray,
    examples: int,
    sequence_length: int,
) -> None:
    expected = (examples, sequence_length)
    if input_ids.shape != expected:
        raise ValueError(f"input_ids shape {input_ids.shape} != {expected}")
    if attention_mask.shape != expected:
        raise ValueError(f"attention_mask shape {attention_mask.shape} != {expected}")
    if logits.shape[:2] != expected or logits.ndim != 3:
        raise ValueError(f"logits shape {logits.shape} incompatible with {expected}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("teacher logits must be finite")


def _rewrite_manifest_for_p145(
    path: Path,
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    vocab_size: int,
    effective_local_files_only: bool,
) -> None:
    manifest = read_json_object(path)
    manifest["teacher"] = {
        **manifest["teacher"],
        "backend": "hf_causal_lm",
        "model_name": teacher.model_id,
        "model_name_or_path": teacher.model_id,
        "tokenizer_name": config.tokenizer or _tokenizer_name(teacher),
        "tokenizer_name_or_path": config.tokenizer or _tokenizer_name(teacher),
        "local_files_only": effective_local_files_only,
        "vocab_size": vocab_size,
        "dtype": teacher.dtype,
        "device": "cpu",
        "teacher_real": True,
    }
    manifest["capture"] = {
        **manifest.get("capture", {}),
        "phase": "P145",
        "run_kind": "tiny_real_teacher_capture",
        "capture_engine": "teacher_side_capture_skeleton_v0",
    }
    write_json(path, manifest)


def _run_consumer_sanity(
    config: TinyRealTeacherFingerprintCaptureConfig,
    *,
    vocab_size: int,
    validation_ok: bool,
) -> dict[str, Any]:
    if not validation_ok:
        return {
            "kind": "loader_only",
            "status": "fail",
            "reason": "artifact validation failed",
        }
    if vocab_size > config.consumer_vocab_limit:
        return {
            "kind": "loader_only",
            "status": "pass",
            "reason": "teacher vocab too large for cheap CPU smoke",
            "vocab_size": vocab_size,
            "consumer_vocab_limit": config.consumer_vocab_limit,
        }
    try:
        result = run_distill_stage(
            DistillStageConfig(
                mode=DISTILL_MODE_FINGERPRINT_CORRIDOR,
                training=replace(DistillStageConfig().training, max_steps=1),
                optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
                fingerprint=DistillFingerprintConfig(
                    artifact_dir=config.output_dir,
                    batch_size=1,
                    student_backend="current_qrwkv",
                    output_dir=config.output_dir / "consumer_sanity",
                ),
            )
        )
        return {
            "kind": "p141_one_step",
            "status": result.status,
            "reason": None if result.status == "pass" else "p141 returned non-pass",
            "final_loss": result.final_loss,
        }
    except Exception as p141_error:
        try:
            result = run_real_student_fingerprint_forward_smoke(
                RealStudentFingerprintForwardConfig(
                    artifact_dir=config.output_dir,
                    output_dir=config.output_dir / "p140_consumer_sanity",
                    batch_size=1,
                )
            )
            return {
                "kind": "p140_forward",
                "status": result.status,
                "reason": None if result.status == "pass" else "p140 returned non-pass",
            }
        except Exception as p140_error:
            return {
                "kind": "loader_only",
                "status": "pass",
                "reason": (
                    "student consumer smoke unavailable; "
                    f"p141={type(p141_error).__name__}: {p141_error}; "
                    f"p140={type(p140_error).__name__}: {p140_error}"
                ),
            }


def _p145_summary(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    base_summary: dict[str, Any],
    teacher: HFTeacherBackend,
    effective_local_files_only: bool,
    examples_processed: int,
    tokens_processed: int,
    target_positions_processed: int,
    vocab_size: int,
    artifact_validated: bool,
    validation_blockers: tuple[str, ...],
    targets_loadable: bool,
    exemplars_loadable: bool,
    exemplar_records: int,
    artifact_summary: dict[str, Any],
    consumer_sanity: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base_summary,
        "phase": "P145",
        "capture_engine": "teacher_side_capture_skeleton_v0",
        "run_kind": "tiny_real_teacher_capture",
        "teacher_real": True,
        "teacher_backend": "hf_causal_lm",
        "teacher_model_name_or_path": teacher.model_id,
        "tokenizer_name_or_path": config.tokenizer or _tokenizer_name(teacher),
        "local_files_only": effective_local_files_only,
        "teacher": {
            "backend": "hf_causal_lm",
            "model_name_or_path": teacher.model_id,
            "tokenizer_name_or_path": config.tokenizer or _tokenizer_name(teacher),
            "local_files_only": effective_local_files_only,
            "vocab_size": vocab_size,
            "dtype": teacher.dtype,
            "device": "cpu",
        },
        "examples_processed": examples_processed,
        "tokens_processed": tokens_processed,
        "target_positions_processed": target_positions_processed,
        "positions_policy": "fixed_all_positions",
        "mode_discovery_method": base_summary["mode_discovery_method"],
        "modes_discovered": base_summary["modes_discovered"],
        "records_per_mode": base_summary["records_per_mode"],
        "corridor_bounds_method": base_summary["corridor_bounds_method"],
        "exemplar_selection_policy": config.exemplar_selection_policy,
        "max_exemplars": config.max_exemplars,
        "exemplars_retained": exemplar_records,
        "artifact_validated": artifact_validated,
        "validation_blockers": list(validation_blockers),
        "targets_loadable": targets_loadable,
        "exemplars_loadable": exemplars_loadable,
        "artifact_summary": artifact_summary,
        "consumer_sanity": consumer_sanity,
        "claims_not_made": (
            "real_scale_teacher_capture",
            "tome_textbook_integration",
            "student_quality_improvement",
            "baseline_comparison",
            "quality_per_byte_gain",
            "production_capture_performance",
        ),
        "capture_config": _json_safe(asdict(config)),
    }


def _tokenizer_name(teacher: HFTeacherBackend) -> str:
    tokenizer = teacher.tokenizer
    value = getattr(tokenizer, "name_or_path", None)
    return str(value or teacher.model_id)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
