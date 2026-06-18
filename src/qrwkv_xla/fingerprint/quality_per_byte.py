from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    FingerprintBatch,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.baseline_comparison import (
    FingerprintBaselineComparisonConfig,
    run_fingerprint_baseline_comparison,
)
from qrwkv_xla.fingerprint.real_teacher import DEFAULT_TINY_REAL_TEACHER
from qrwkv_xla.fingerprint.training_rehearsal import DEFAULT_TINY_TEXTS
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.teachers import HFTeacherBackend
from qrwkv_xla.training.fingerprint_loss import (
    FingerprintCorridorLossConfig,
    compute_fingerprint_corridor_loss,
)
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats_at_positions,
)

EPS = 1e-8


@dataclass(frozen=True)
class FingerprintQualityPerByteExperimentConfig:
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
    eval_split: str = "train_artifact_reuse"
    overwrite: bool = False


@dataclass(frozen=True)
class FingerprintQualityPerByteExperimentResult:
    status: str
    output_dir: Path
    report_path: Path
    summary_path: Path
    comparison_report_path: Path
    artifact_dir: Path


def run_fingerprint_quality_per_byte_experiment(
    config: FingerprintQualityPerByteExperimentConfig,
    *,
    backend: HFTeacherBackend | None = None,
) -> FingerprintQualityPerByteExperimentResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = run_fingerprint_baseline_comparison(
        FingerprintBaselineComparisonConfig(
            output_dir=config.output_dir / "p147_comparison",
            fingerprint_artifact=config.fingerprint_artifact,
            build_real_teacher_artifact=config.build_real_teacher_artifact,
            texts_path=config.texts_path,
            teacher_model=config.teacher_model,
            tokenizer=config.tokenizer,
            sequence_length=config.sequence_length,
            max_examples=config.max_examples,
            max_target_positions=config.max_target_positions,
            max_exemplars=config.max_exemplars,
            local_files_only=config.local_files_only,
            allow_downloads=config.allow_downloads,
            steps=config.steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            student_backend=config.student_backend,
            overwrite=config.overwrite,
        ),
        backend=backend,
    )
    comparison_report = read_json_object(comparison.report_path)
    artifact_dir = comparison.artifact_dir
    artifact = summarize_fingerprint_artifact(artifact_dir)
    baseline_arm = _arm(comparison_report, "baseline_init_only")
    fingerprint_arm = _arm(comparison_report, "fingerprint_corridor")
    baseline_eval = evaluate_student_corridor_adherence(
        checkpoint_dir=Path(str(baseline_arm["checkpoint_dir"])),
        artifact_dir=artifact_dir,
        batch_size=config.batch_size,
        student_backend=config.student_backend,
    )
    fingerprint_eval = evaluate_student_corridor_adherence(
        checkpoint_dir=Path(str(fingerprint_arm["checkpoint_dir"])),
        artifact_dir=artifact_dir,
        batch_size=config.batch_size,
        student_backend=config.student_backend,
    )
    arms = [
        _p148_arm(
            source=baseline_arm,
            arm_kind="reference_baseline",
            trained=False,
            eval_metrics=baseline_eval,
        ),
        _p148_arm(
            source=fingerprint_arm,
            arm_kind="fingerprint_method",
            trained=True,
            eval_metrics=fingerprint_eval,
        ),
    ]
    artifact_budget = _artifact_budget(artifact_dir, artifact, comparison_report)
    report = {
        "phase": "P148",
        "run_kind": "first_quality_per_byte_experiment",
        "status": "pass"
        if comparison.status == "pass"
        and all(arm["eval"]["metrics_finite"] for arm in arms)
        else "fail",
        "experiment": {
            "seed": config.seed,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "student_backend": config.student_backend,
            "eval_split": config.eval_split,
            "quality_proxy": "corridor_adherence",
            "comparison_report_path": str(comparison.report_path),
        },
        "artifact_budget": artifact_budget,
        "arms": arms,
        "quality_per_byte": _quality_per_byte(
            baseline_eval=baseline_eval,
            fingerprint_eval=fingerprint_eval,
            fingerprint_artifact_size_bytes=artifact_budget[
                "fingerprint_artifact_size_bytes"
            ],
        ),
        "fairness": {
            "trained_baseline_available": False,
            "comparison_fairness": "reference_only",
            "baseline_init_only_is_competitive": False,
            "eval_split": config.eval_split,
            "generalization_claim_made": False,
        },
        "claims": {
            "winner_declared": False,
            "general_quality_claim_made": False,
            "radlads_parity_claim_made": False,
            "scale_claim_made": False,
            "quality_per_byte_claim_scope": "tiny_smoke_only",
        },
        "limitations": [
            "The baseline is init-only, not a competitive trained baseline.",
            "Eval split reuses the training artifact.",
            "Metrics are corridor-adherence proxies, not perplexity or human quality.",
            "No winner or general quality claim is made.",
        ],
    }
    report_path = config.output_dir / "p148_quality_per_byte_report.json"
    summary_path = config.output_dir / "p148_quality_per_byte_summary.md"
    write_json(report_path, report)
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    return FingerprintQualityPerByteExperimentResult(
        status=str(report["status"]),
        output_dir=config.output_dir,
        report_path=report_path,
        summary_path=summary_path,
        comparison_report_path=comparison.report_path,
        artifact_dir=artifact_dir,
    )


def evaluate_student_corridor_adherence(
    *,
    checkpoint_dir: Path,
    artifact_dir: Path,
    batch_size: int,
    student_backend: str = "current_qrwkv",
) -> dict[str, float | bool | int]:
    artifact = summarize_fingerprint_artifact(artifact_dir)
    checkpoint = load_checkpoint(checkpoint_dir)
    backend = _create_backend(artifact=artifact, student_backend=student_backend)
    dataset = load_fingerprint_targets(artifact_dir, batch_size=batch_size)
    weighted: dict[str, float] = {
        "corridor_loss_total": 0.0,
        "corridor_inside_all_rate": 0.0,
        "corridor_inside_entropy_rate": 0.0,
        "corridor_inside_top1_margin_rate": 0.0,
        "corridor_inside_top8_mass_rate": 0.0,
        "corridor_inside_top32_mass_rate": 0.0,
        "corridor_inside_tail_mass_rate": 0.0,
    }
    records = 0
    for batch in dataset.iter_batches():
        size = int(batch.input_ids.shape[0])
        metrics = _evaluate_batch(
            backend=backend,
            params=checkpoint.params,
            batch=batch,
        )
        records += size
        for key in weighted:
            weighted[key] += float(metrics[key]) * size
    if records == 0:
        raise ValueError("fingerprint corridor eval requires at least one record")
    averaged = {key: value / records for key, value in weighted.items()}
    return {
        **averaged,
        "metrics_finite": all(np.isfinite(value) for value in averaged.values()),
        "records_evaluated": records,
    }


def _validate_config(config: FingerprintQualityPerByteExperimentConfig) -> None:
    if config.eval_split != "train_artifact_reuse":
        raise ValueError("P148 currently supports eval_split='train_artifact_reuse'")
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


def _evaluate_batch(
    *,
    backend: Any,
    params: Any,
    batch: FingerprintBatch,
) -> dict[str, float]:
    input_ids = jnp.asarray(batch.input_ids, dtype=jnp.int32)
    positions = jnp.asarray(batch.position, dtype=jnp.int32)
    output, _state = backend.forward_full(params, input_ids)
    logits = backend.logits(output)
    stats = compute_fingerprint_distribution_stats_at_positions(logits, positions)
    corridor = compute_fingerprint_corridor_loss(
        stats,
        FingerprintBatch(
            input_ids=input_ids,
            position=positions,
            mode_id=jnp.asarray(batch.mode_id, dtype=jnp.int32),
            entropy_min=jnp.asarray(batch.entropy_min, dtype=jnp.float32),
            entropy_max=jnp.asarray(batch.entropy_max, dtype=jnp.float32),
            top1_margin_min=jnp.asarray(batch.top1_margin_min, dtype=jnp.float32),
            top1_margin_max=jnp.asarray(batch.top1_margin_max, dtype=jnp.float32),
            top8_mass_min=jnp.asarray(batch.top8_mass_min, dtype=jnp.float32),
            top8_mass_max=jnp.asarray(batch.top8_mass_max, dtype=jnp.float32),
            top32_mass_min=jnp.asarray(batch.top32_mass_min, dtype=jnp.float32),
            top32_mass_max=jnp.asarray(batch.top32_mass_max, dtype=jnp.float32),
            tail_mass_min=jnp.asarray(batch.tail_mass_min, dtype=jnp.float32),
            tail_mass_max=jnp.asarray(batch.tail_mass_max, dtype=jnp.float32),
            weight=jnp.asarray(batch.weight, dtype=jnp.float32),
        ),
        FingerprintCorridorLossConfig(),
    )
    return {
        "corridor_loss_total": float(corridor.loss),
        "corridor_inside_all_rate": float(corridor.all_inside_rate),
        "corridor_inside_entropy_rate": float(corridor.entropy_inside_rate),
        "corridor_inside_top1_margin_rate": float(corridor.top1_margin_inside_rate),
        "corridor_inside_top8_mass_rate": float(corridor.top8_mass_inside_rate),
        "corridor_inside_top32_mass_rate": float(corridor.top32_mass_inside_rate),
        "corridor_inside_tail_mass_rate": float(corridor.tail_mass_inside_rate),
    }


def _create_backend(*, artifact: object, student_backend: str) -> Any:
    vocab_contract = VocabContract(
        tokenizer_id=artifact.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=artifact.tokenizer_name or None,
        vocab_size=artifact.vocab_size,
        model_id=artifact.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    return create_student_backend(
        vocab_contract=vocab_contract,
        architecture_id=student_backend,
    )


def _p148_arm(
    *,
    source: dict[str, Any],
    arm_kind: str,
    trained: bool,
    eval_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "arm_id": source["arm_id"],
        "arm_kind": arm_kind,
        "trained": trained,
        "status": source["status"],
        "optimizer_steps_completed": source["optimizer_steps_completed"],
        "params_changed": source["params_changed"],
        "checkpoint_dir": source["checkpoint_dir"],
        "eval": {
            "corridor_loss_total": eval_metrics["corridor_loss_total"],
            "inside_all_rate": eval_metrics["corridor_inside_all_rate"],
            "inside_entropy_rate": eval_metrics["corridor_inside_entropy_rate"],
            "inside_top1_margin_rate": eval_metrics["corridor_inside_top1_margin_rate"],
            "inside_top8_mass_rate": eval_metrics["corridor_inside_top8_mass_rate"],
            "inside_top32_mass_rate": eval_metrics["corridor_inside_top32_mass_rate"],
            "inside_tail_mass_rate": eval_metrics["corridor_inside_tail_mass_rate"],
            "metrics_finite": eval_metrics["metrics_finite"],
            "records_evaluated": eval_metrics["records_evaluated"],
        },
    }


def _quality_per_byte(
    *,
    baseline_eval: dict[str, Any],
    fingerprint_eval: dict[str, Any],
    fingerprint_artifact_size_bytes: int,
) -> dict[str, Any]:
    artifact_size_mb = max(fingerprint_artifact_size_bytes / 1_000_000.0, EPS)
    baseline_loss = float(baseline_eval["corridor_loss_total"])
    fingerprint_loss = float(fingerprint_eval["corridor_loss_total"])
    absolute_delta = baseline_loss - fingerprint_loss
    relative_delta = absolute_delta / max(abs(baseline_loss), EPS)
    inside_delta = float(fingerprint_eval["corridor_inside_all_rate"]) - float(
        baseline_eval["corridor_inside_all_rate"]
    )
    return {
        "quality_proxy": "corridor_adherence",
        "artifact_byte_denominator": "fingerprint_artifact_size_bytes",
        "reference_delta_vs_init_only": {
            "absolute_corridor_loss_delta": absolute_delta,
            "relative_corridor_loss_delta": relative_delta,
            "corridor_loss_delta_per_mb": absolute_delta / artifact_size_mb,
            "inside_all_rate_delta": inside_delta,
            "inside_all_rate_delta_per_mb": inside_delta / artifact_size_mb,
        },
        "trained_baseline_available": False,
        "delta_vs_trained_baseline": None,
    }


def _artifact_budget(
    artifact_dir: Path,
    artifact: object,
    comparison_report: dict[str, Any],
) -> dict[str, Any]:
    capture_summary = _optional_size(artifact_dir / "capture_summary.json")
    return {
        "artifact_kind": "behavioral_fingerprint",
        "artifact_dir": str(artifact_dir),
        "fingerprint_artifact_size_bytes": _directory_size_bytes(artifact_dir),
        "manifest_size_bytes": _optional_size(artifact_dir / "manifest.json"),
        "targets_size_bytes": _glob_size(artifact_dir, "targets/*.jsonl"),
        "exemplars_size_bytes": _glob_size(artifact_dir, "exemplars/*.jsonl"),
        "modes_size_bytes": _optional_size(artifact_dir / "modes.json"),
        "capture_summary_size_bytes": capture_summary,
        "target_records": artifact.num_corridor_records,
        "exemplar_records": artifact.num_exemplar_records,
        "modes_discovered": artifact.num_modes,
        "target_positions_processed": artifact.num_corridor_records,
        "teacher_tokens_processed": comparison_report["artifact"].get(
            "teacher_tokens_processed"
        ),
    }


def _arm(report: dict[str, Any], arm_id: str) -> dict[str, Any]:
    return next(arm for arm in report["arms"] if arm["arm_id"] == arm_id)


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _glob_size(path: Path, pattern: str) -> int:
    return sum(item.stat().st_size for item in path.glob(pattern) if item.is_file())


def _optional_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _render_summary(report: dict[str, Any]) -> str:
    baseline = _arm(report, "baseline_init_only")
    fingerprint = _arm(report, "fingerprint_corridor")
    qpb = report["quality_per_byte"]["reference_delta_vs_init_only"]
    budget = report["artifact_budget"]
    return "\n".join(
        (
            "# P148 First Quality-Per-Byte Experiment",
            "",
            f"Status: {report['status']}",
            f"Eval split: {report['experiment']['eval_split']}",
            "Quality proxy: corridor_adherence",
            "",
            "This P148 run measures corridor-adherence change per fingerprint "
            "artifact byte in a tiny smoke setting. Because the baseline is "
            "init-only, the result is a reference delta, not a method-vs-method "
            "win.",
            "",
            "## Arms",
            f"- baseline_init_only: loss={baseline['eval']['corridor_loss_total']}, "
            f"inside_all={baseline['eval']['inside_all_rate']}",
            "- fingerprint_corridor: "
            f"loss={fingerprint['eval']['corridor_loss_total']}, "
            f"inside_all={fingerprint['eval']['inside_all_rate']}",
            "",
            "## Artifact Budget",
            "- Fingerprint artifact bytes: "
            f"{budget['fingerprint_artifact_size_bytes']}",
            f"- Target records: {budget['target_records']}",
            f"- Exemplar records: {budget['exemplar_records']}",
            f"- Modes discovered: {budget['modes_discovered']}",
            "",
            "## Quality Per Byte Proxy",
            f"- Absolute corridor loss delta: {qpb['absolute_corridor_loss_delta']}",
            f"- Relative corridor loss delta: {qpb['relative_corridor_loss_delta']}",
            f"- Corridor loss delta per MB: {qpb['corridor_loss_delta_per_mb']}",
            f"- Inside-all rate delta: {qpb['inside_all_rate_delta']}",
            f"- Inside-all rate delta per MB: {qpb['inside_all_rate_delta_per_mb']}",
            "",
            "## Limits",
            "- The baseline is an init-only reference, not a competitive "
            "trained baseline.",
            "- No winner/general quality/RADLADS parity/scale claim is made.",
            "",
        )
    )
