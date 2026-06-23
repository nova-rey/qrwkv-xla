from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
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
from qrwkv_xla.fingerprint.budgeted_artifact import (
    BudgetedArtifactConfig,
    materialize_budgeted_artifact,
    validate_budgeted_artifact,
)
from qrwkv_xla.fingerprint.held_out_evaluation import paired_bootstrap_interval
from qrwkv_xla.fingerprint.provenance import file_sha256, stable_hash
from qrwkv_xla.fingerprint.real_teacher import DEFAULT_TINY_REAL_TEACHER
from qrwkv_xla.fingerprint.training_rehearsal import DEFAULT_TINY_TEXTS
from qrwkv_xla.fingerprint.two_cycle_experiment import (
    ARM_NAMES,
    TwoCycleExperimentConfig,
    run_two_cycle_experiment,
)
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.teachers import HFTeacherBackend
from qrwkv_xla.tracking import get_git_metadata
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


@dataclass(frozen=True)
class QualityBudgetPoint:
    name: str
    teacher_artifact_bytes: int
    total_steps: int
    wall_clock_seconds: float


@dataclass(frozen=True)
class ControlledQualityPerByteConfig:
    training_fingerprint_artifact: Path
    calibration_fingerprint_artifact: Path
    final_test_fingerprint_artifact: Path
    source_texts: Path
    selected_profile_receipt: Path
    output_dir: Path
    budget_points: tuple[QualityBudgetPoint, ...]
    seeds: tuple[int, ...]
    student_backend: str = "current_qrwkv"
    student_architecture: str | None = None
    batch_size: int = 1
    optimizer: str = "adamw"
    baseline_learning_rate: float = 1e-4
    exemplar_learning_rate: float = 5e-5
    exemplar_max_grad_norm: float | None = 1.0
    target_quality_threshold: float = 1.0
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    tie_tolerance: float = 1e-12
    require_backend: str | None = "cpu"
    resume: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ControlledQualityPerByteResult:
    status: str
    output_dir: Path
    report_path: Path
    matrix_state_path: Path


def validate_backend_requirement(
    requested_backend: str | None,
    *,
    observed_backend: str | None = None,
    device_count: int | None = None,
    process_count: int | None = None,
) -> dict[str, Any]:
    observed = observed_backend or jax.default_backend()
    devices = jax.device_count() if device_count is None else int(device_count)
    processes = jax.process_count() if process_count is None else int(process_count)
    met = requested_backend is None or observed == requested_backend
    if not met:
        raise ValueError(
            f"backend requirement not met: requested={requested_backend}, "
            f"observed={observed}"
        )
    if processes != 1:
        raise ValueError("P156 requires single-host, non-distributed execution")
    return {
        "requested_backend": requested_backend,
        "observed_backend": observed,
        "distributed": False,
        "device_count": devices,
        "process_count": processes,
        "backend_requirement_met": True,
    }


def validate_budget_allocation(
    *,
    total_bytes: int,
    corridor_bytes: int,
    exemplar_bytes: int,
    total_steps: int,
    corridor_steps: int,
    exemplar_steps: int,
) -> dict[str, bool]:
    values = (
        total_bytes,
        corridor_bytes,
        exemplar_bytes,
        total_steps,
        corridor_steps,
        exemplar_steps,
    )
    if any(value < 0 for value in values):
        raise ValueError("budget allocations must be non-negative")
    byte_match = corridor_bytes + exemplar_bytes == total_bytes
    step_match = corridor_steps + exemplar_steps == total_steps
    if not byte_match:
        raise ValueError("teacher artifact byte budget overflow or under-allocation")
    if not step_match:
        raise ValueError("optimizer step budget overflow or under-allocation")
    return {"budget_match_valid": True, "step_budget_match_valid": True}


def efficiency_ratios(
    *,
    reference_score: float,
    final_score: float,
    artifact_bytes: int,
    optimizer_steps: int,
    wall_clock_seconds: float,
) -> dict[str, float]:
    denominators = (artifact_bytes, optimizer_steps, wall_clock_seconds)
    if any(value <= 0 for value in denominators):
        raise ValueError("efficiency ratio denominators must be positive")
    improvement = reference_score - final_score
    return {
        "primary_score_improvement": improvement,
        "primary_score_improvement_per_megabyte": improvement
        / (artifact_bytes / 1_000_000),
        "primary_score_improvement_per_100_steps": improvement
        / (optimizer_steps / 100),
        "primary_score_improvement_per_minute": improvement / (wall_clock_seconds / 60),
    }


def observed_target_quality_cost(
    points: list[dict[str, Any]], *, threshold: float
) -> dict[str, Any]:
    reached = [
        row
        for row in sorted(points, key=lambda row: row["optimizer_steps"])
        if float(row["final_test_primary_score"]) <= threshold
    ]
    if not reached:
        return {
            "target_quality_threshold": threshold,
            "target_reached": False,
            "steps_to_target": None,
            "bytes_to_target": None,
            "seconds_to_target": None,
            "records_to_target": None,
        }
    first = reached[0]
    return {
        "target_quality_threshold": threshold,
        "target_reached": True,
        "steps_to_target": first["optimizer_steps"],
        "bytes_to_target": first["teacher_artifact_bytes"],
        "seconds_to_target": first["total_wall_clock_seconds"],
        "records_to_target": first["records_consumed"],
        "observed_budget_point": first["budget_point"],
    }


def trapezoidal_auc(points: list[tuple[float, float]]) -> float:
    ordered = sorted((float(x), float(y)) for x, y in points)
    if len(ordered) < 2 or len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("AUC requires at least two unique x-axis points")
    return float(
        sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for (left_x, left_y), (right_x, right_y) in zip(
                ordered[:-1], ordered[1:], strict=True
            )
        )
    )


def controlled_matrix_config_hash(config: ControlledQualityPerByteConfig) -> str:
    payload = {
        "artifacts": {
            name: file_sha256(path / "manifest.json")
            for name, path in (
                ("training", config.training_fingerprint_artifact),
                ("calibration", config.calibration_fingerprint_artifact),
                ("final_test", config.final_test_fingerprint_artifact),
            )
        },
        "source_texts_sha256": file_sha256(config.source_texts),
        "selected_profile_receipt_sha256": file_sha256(config.selected_profile_receipt),
        "student_backend": config.student_backend,
        "student_architecture": config.student_architecture,
        "budget_points": [point.__dict__ for point in config.budget_points],
        "seeds": list(config.seeds),
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "learning_rates": [
            config.baseline_learning_rate,
            config.exemplar_learning_rate,
        ],
        "target_quality_threshold": config.target_quality_threshold,
        "bootstrap": [config.bootstrap_samples, config.bootstrap_seed],
        "software_commit": get_git_metadata(Path(__file__).resolve().parents[3]).get(
            "commit"
        ),
    }
    return stable_hash(payload)


def paired_seed_record_comparison(
    left: dict[str, float],
    right: dict[str, float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    if not left or set(left) != set(right):
        raise ValueError("paired comparison requires aligned seed-record keys")
    keys = sorted(left)
    deltas = np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)
    low, high = paired_bootstrap_interval(
        deltas, samples=bootstrap_samples, seed=bootstrap_seed
    )
    mean = float(np.mean(deltas))
    if low <= 0.0 <= high or abs(mean) <= tie_tolerance:
        result = "inconclusive"
    elif mean < 0.0:
        result = "left_better"
    else:
        result = "right_better"
    return {
        "mean_paired_delta": mean,
        "median_paired_delta": float(np.median(deltas)),
        "ci95": [low, high],
        "fraction_won": float(np.mean(deltas < -tie_tolerance)),
        "fraction_tied": float(np.mean(np.abs(deltas) <= tie_tolerance)),
        "fraction_lost": float(np.mean(deltas > tie_tolerance)),
        "paired_observation_count": len(keys),
        "result": result,
    }


def quality_per_byte_claims(*, gates: dict[str, bool]) -> dict[str, Any]:
    allowed = bool(gates) and all(gates.values())
    return {
        "quality_per_byte_claim_allowed": allowed,
        "claim_scope": "tiny_cpu_controlled_experiment",
        "scale_claim_made": False,
        "gpt2_claim_made": False,
        "radlads_parity_claim_made": False,
        "winner_declared": False,
        "gates": gates,
    }


def run_controlled_quality_per_byte_experiment(
    config: ControlledQualityPerByteConfig,
) -> ControlledQualityPerByteResult:
    _validate_controlled_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    backend_receipt = validate_backend_requirement(config.require_backend)
    write_json(config.output_dir / "cpu_backend_receipt.json", backend_receipt)
    config_hash = controlled_matrix_config_hash(config)
    state_path = config.output_dir / "experiment_matrix_state.json"
    previous = read_json_object(state_path) if state_path.is_file() else None
    if previous and previous.get("config_hash") != config_hash:
        raise ValueError("resume configuration hash mismatch")

    expected_runs = len(config.budget_points) * len(config.seeds)
    expected_arm_cells = expected_runs * len(ARM_NAMES)
    completed: list[str] = []
    failed: dict[str, str] = {}
    cell_reports: list[dict[str, Any]] = []
    for point in config.budget_points:
        for seed in config.seeds:
            cell_id = f"{point.name}/seed_{seed}"
            cell_dir = (
                config.output_dir / "runs" / "controlled" / point.name / f"seed_{seed}"
            )
            cell_path = cell_dir / "cell_report.json"
            if config.resume and cell_path.is_file():
                cell = read_json_object(cell_path)
                if cell.get("config_hash") != config_hash:
                    raise ValueError(f"completed cell config hash mismatch: {cell_id}")
                if cell.get("status") == "pass":
                    completed.append(cell_id)
                    cell_reports.append(cell)
                    _write_matrix_state(
                        state_path,
                        config_hash=config_hash,
                        expected_runs=expected_runs,
                        expected_arm_cells=expected_arm_cells,
                        completed=completed,
                        failed=failed,
                    )
                    continue
            try:
                cell = _run_controlled_cell(config, point=point, seed=seed)
                completed.append(cell_id)
                cell_reports.append(cell)
            except Exception as exc:  # preserve partial matrix for resume
                failed[cell_id] = f"{type(exc).__name__}: {exc}"
            _write_matrix_state(
                state_path,
                config_hash=config_hash,
                expected_runs=expected_runs,
                expected_arm_cells=expected_arm_cells,
                completed=completed,
                failed=failed,
            )

    complete = len(completed) == expected_runs and not failed
    outputs = _build_controlled_outputs(config, cell_reports, complete=complete)
    report_path = config.output_dir / "quality_per_byte_report.json"
    write_json(report_path, outputs["report"])
    write_json(config.output_dir / "budget_fairness_contract.json", outputs["fairness"])
    write_json(config.output_dir / "artifact_byte_accounting.json", outputs["bytes"])
    write_json(
        config.output_dir / "publication_grade_receipt.json",
        outputs["publication"],
    )
    _write_rows(config.output_dir / "quality_budget_curve.jsonl", outputs["curves"])
    _write_rows(
        config.output_dir / "paired_budget_comparisons.jsonl",
        outputs["comparisons"],
    )
    (config.output_dir / "quality_per_byte_summary.md").write_text(
        _controlled_summary(outputs["report"]), encoding="utf-8"
    )
    return ControlledQualityPerByteResult(
        status=outputs["report"]["status"],
        output_dir=config.output_dir,
        report_path=report_path,
        matrix_state_path=state_path,
    )


def _run_controlled_cell(
    config: ControlledQualityPerByteConfig, *, point: QualityBudgetPoint, seed: int
) -> dict[str, Any]:
    corridor_steps = max(1, point.total_steps // 2)
    exemplar_steps = point.total_steps - corridor_steps
    if exemplar_steps < 1:
        raise ValueError("two-cycle budget requires at least two total steps")
    corridor_bytes = point.teacher_artifact_bytes // 2
    exemplar_bytes = point.teacher_artifact_bytes - corridor_bytes
    validation = validate_budget_allocation(
        total_bytes=point.teacher_artifact_bytes,
        corridor_bytes=corridor_bytes,
        exemplar_bytes=exemplar_bytes,
        total_steps=point.total_steps,
        corridor_steps=corridor_steps,
        exemplar_steps=exemplar_steps,
    )
    cell_dir = config.output_dir / "runs" / "controlled" / point.name / f"seed_{seed}"
    p155_dir = cell_dir / "p155"
    result = run_two_cycle_experiment(
        TwoCycleExperimentConfig(
            training_fingerprint_artifact=config.training_fingerprint_artifact,
            calibration_fingerprint_artifact=config.calibration_fingerprint_artifact,
            final_test_fingerprint_artifact=config.final_test_fingerprint_artifact,
            source_texts=config.source_texts,
            selected_profile_receipt=config.selected_profile_receipt,
            output_dir=p155_dir,
            student_backend=config.student_backend,
            student_architecture=config.student_architecture,
            baseline_steps=point.total_steps,
            corridor_steps=corridor_steps,
            exemplar_steps=exemplar_steps,
            corridor_only_steps=point.total_steps,
            exemplar_only_steps=point.total_steps,
            budget_comparison_mode="equal_total_budget",
            batch_size=config.batch_size,
            optimizer=config.optimizer,
            baseline_learning_rate=config.baseline_learning_rate,
            exemplar_learning_rate=config.exemplar_learning_rate,
            exemplar_max_grad_norm=config.exemplar_max_grad_norm,
            checkpoint_every=max(1, point.total_steps),
            seed=seed,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
            tie_tolerance=config.tie_tolerance,
            overwrite=config.overwrite,
        )
    )
    p155 = read_json_object(result.report_path)
    if result.status != "pass" or not p155["split_integrity"]["three_way_split_valid"]:
        raise ValueError("P155.1 cell failed split or experiment validation")
    arms = _cell_arms(config, point=point, p155=p155)
    if any(arm["wall_clock_budget_exceeded"] for arm in arms.values()):
        raise ValueError("declared wall-clock budget exceeded")
    cell = {
        "phase": "P156",
        "status": "pass",
        "config_hash": controlled_matrix_config_hash(config),
        "budget_point": point.name,
        "seed": seed,
        "p155_report": str(result.report_path),
        "split_integrity": p155["split_integrity"],
        "arms": arms,
        "per_record_metrics": str(p155_dir / "per_record_arm_metrics.jsonl"),
        "budget_validation": validation,
    }
    write_json(cell_dir / "cell_report.json", cell)
    return cell


def _cell_arms(config, *, point, p155):
    evaluation_names = {
        "conventional_baseline": "conventional_baseline",
        "corridor_only": "corridor_only",
        "exemplar_only": "exemplar_only",
        "two_cycle": "two_cycle_final",
    }
    resources = {
        "conventional_baseline": p155["resources"]["conventional_baseline"],
        "corridor_only": p155["resources"]["corridor_only"],
        "exemplar_only": p155["resources"]["exemplar_only"],
        "two_cycle": p155["resources"]["two_cycle"]["combined"],
    }
    init_score = p155["evaluation_metrics"]["shared_initialization"]["teacher"][
        "teacher_student_kl"
    ]
    source_bytes = config.source_texts.stat().st_size
    output = {}
    for arm, evaluation_name in evaluation_names.items():
        metrics = p155["evaluation_metrics"][evaluation_name]
        resource = resources[arm]
        teacher_bytes = (
            0 if arm == "conventional_baseline" else point.teacher_artifact_bytes
        )
        if arm == "two_cycle":
            corridor_bytes = point.teacher_artifact_bytes // 2
            exemplar_bytes = point.teacher_artifact_bytes - corridor_bytes
            corridor_steps = point.total_steps // 2
            exemplar_steps = point.total_steps - corridor_steps
        elif arm == "corridor_only":
            corridor_bytes, exemplar_bytes = point.teacher_artifact_bytes, 0
            corridor_steps, exemplar_steps = point.total_steps, 0
        elif arm == "exemplar_only":
            corridor_bytes, exemplar_bytes = 0, point.teacher_artifact_bytes
            corridor_steps, exemplar_steps = 0, point.total_steps
        else:
            corridor_bytes = exemplar_bytes = corridor_steps = exemplar_steps = 0
        score = metrics["teacher"]["teacher_student_kl"]
        ratios = None
        if teacher_bytes > 0:
            ratios = efficiency_ratios(
                reference_score=init_score,
                final_score=score,
                artifact_bytes=teacher_bytes,
                optimizer_steps=resource["optimizer_steps"],
                wall_clock_seconds=resource["total_wall_clock_seconds"],
            )
        quality = {
            "teacher_student_kl": score,
            "held_out_exemplar_loss": metrics["teacher"].get(
                "teacher_student_cross_entropy"
            ),
            "top1_agreement": metrics["teacher"].get("top1_agreement"),
            "topk_overlap": metrics["teacher"].get("topk_overlap"),
            "teacher_entropy": metrics["teacher"].get("teacher_entropy"),
            "student_entropy": metrics["teacher"].get("student_entropy"),
            "entropy_absolute_error": metrics["teacher"].get("entropy_absolute_error"),
            "held_out_corridor_loss": metrics["corridor"].get("corridor_loss_total"),
            "inside_all_rate": metrics["corridor"].get("inside_all_rate"),
            "mean_distance_outside_corridor": metrics["corridor"].get(
                "mean_distance_outside_corridor"
            ),
        }
        output[arm] = {
            "quality": quality,
            "final_test_primary_score": score,
            "teacher_artifact_byte_budget": point.teacher_artifact_bytes,
            "teacher_artifact_bytes": teacher_bytes,
            "corridor_bytes_allocated": corridor_bytes,
            "exemplar_bytes_allocated": exemplar_bytes,
            "teacher_artifact_bytes_consumed": teacher_bytes,
            "source_text_bytes_consumed": source_bytes
            if arm == "conventional_baseline"
            else 0,
            "physical_artifact_bytes_on_disk": _directory_size_bytes(
                config.training_fingerprint_artifact
            ),
            "logical_artifact_bytes_consumed": resource[
                "artifact_bytes_logically_consumed"
            ],
            "budget_match_valid": corridor_bytes + exemplar_bytes == teacher_bytes,
            "total_step_budget": point.total_steps,
            "corridor_steps_allocated": corridor_steps,
            "exemplar_steps_allocated": exemplar_steps,
            "completed_total_steps": resource["optimizer_steps"],
            "step_budget_match_valid": resource["optimizer_steps"] == point.total_steps,
            "wall_clock_budget_seconds": point.wall_clock_seconds,
            "actual_training_seconds": resource["training_seconds"],
            "actual_evaluation_seconds": resource["evaluation_seconds"],
            "actual_checkpoint_seconds": resource["checkpoint_seconds"],
            "actual_total_seconds": resource["total_wall_clock_seconds"],
            "wall_clock_budget_exceeded": resource["total_wall_clock_seconds"]
            > point.wall_clock_seconds,
            "optimizer_steps": resource["optimizer_steps"],
            "records_consumed": resource["records_consumed"],
            "tokens_consumed": resource["tokens_consumed"],
            "training_seconds": resource["training_seconds"],
            "total_wall_clock_seconds": resource["total_wall_clock_seconds"],
            "efficiency_reference_arm": "shared_initialization",
            "efficiency_reference_score": init_score,
            "efficiency": ratios,
        }
    return output


def _build_controlled_outputs(config, cells, *, complete):
    curves = _curve_rows(cells)
    comparisons = _budget_comparisons(config, cells)
    target_costs = _target_costs(config, cells)
    auc = _curve_auc(curves) if len(config.budget_points) >= 2 and complete else []
    budget_valid = complete and all(
        arm["budget_match_valid"]
        and arm["step_budget_match_valid"]
        and not arm["wall_clock_budget_exceeded"]
        for cell in cells
        for arm in cell["arms"].values()
    )
    metrics_finite = complete and all(
        value is None or (isinstance(value, (int, float)) and math.isfinite(value))
        for cell in cells
        for arm in cell["arms"].values()
        for value in arm["quality"].values()
    )
    split_valid = complete and all(
        cell["split_integrity"]["three_way_split_valid"]
        and cell["split_integrity"]["final_test_independent"]
        and cell["split_integrity"]["configuration_frozen_before_final_test_access"]
        for cell in cells
    )
    publication_grade = False
    gates = {
        "three_way_split_valid": split_valid,
        "final_test_independent": split_valid,
        "configuration_frozen": split_valid,
        "all_required_arms_complete": complete,
        "all_required_seeds_complete": complete and len(config.seeds) >= 3,
        "budget_matching_valid": budget_valid,
        "all_required_metrics_finite": metrics_finite,
        "paired_statistics_complete": bool(comparisons),
        "artifact_byte_accounting_valid": budget_valid,
        "lineage_valid": split_valid,
    }
    claims = {
        **quality_per_byte_claims(gates={**gates, "independent_budget_views": False}),
        "quality_per_step_claim_allowed": False,
        "quality_per_time_claim_allowed": False,
    }
    report = {
        "phase": "P156",
        "status": "pass" if complete and budget_valid and split_valid else "fail",
        "experiment_kind": "controlled_quality_per_byte",
        "result_classification": "confounded_resource_matrix",
        "actual_byte_budget_enforced": False,
        "independent_budget_views": False,
        "execution_backend": jax.default_backend(),
        "required_arms_complete": complete,
        "required_seeds_complete": complete and len(config.seeds) >= 3,
        "budget_views": ["artifact_bytes", "optimizer_steps", "wall_clock"],
        "budget_points": [point.__dict__ for point in config.budget_points],
        "seeds": list(config.seeds),
        "primary_metric": "independent_final_test_teacher_student_kl",
        "primary_metric_direction": "lower_is_better",
        "metric_selection_predeclared": True,
        "fallback_metric_used": False,
        "primary_comparison": "two_cycle_vs_exemplar_only",
        "cells": cells,
        "target_quality": target_costs,
        "curve_auc": auc,
        "auc_method": "deterministic_trapezoidal_integration",
        "comparisons": comparisons,
        **claims,
    }
    fairness = {
        "status": "pass" if budget_valid and split_valid else "fail",
        "same_final_test_records": split_valid,
        "aligned_seeds": list(config.seeds),
        "equal_teacher_artifact_budget": budget_valid,
        "equal_total_optimizer_steps": budget_valid,
        "same_declared_wall_clock_maximum": budget_valid,
        "three_way_split_valid": split_valid,
        "configuration_frozen_before_final_test": split_valid,
    }
    byte_rows = [
        {
            "budget_point": cell["budget_point"],
            "seed": cell["seed"],
            "arm": arm,
            "teacher_artifact_byte_budget": values["teacher_artifact_byte_budget"],
            "physical_artifact_bytes_on_disk": values[
                "physical_artifact_bytes_on_disk"
            ],
            "logical_artifact_bytes_consumed": values[
                "logical_artifact_bytes_consumed"
            ],
            "physical_file_counted_once": True,
            "budget_match_valid": values["budget_match_valid"],
        }
        for cell in cells
        for arm, values in cell["arms"].items()
    ]
    return {
        "report": report,
        "fairness": fairness,
        "bytes": {
            "status": "pass" if budget_valid else "fail",
            "physical_and_logical_bytes_distinct": True,
            "rows": byte_rows,
        },
        "publication": {
            "phase": "P156",
            "status": "pass" if publication_grade else "fail",
            "publication_grade": publication_grade,
            "scope": "tiny_cpu_controlled_experiment",
            "expected_arm_cells": len(config.budget_points)
            * len(config.seeds)
            * len(ARM_NAMES),
            "completed_arm_cells": len(cells) * len(ARM_NAMES),
            "quality_per_byte_claim_allowed": claims["quality_per_byte_claim_allowed"],
            "scale_claim_made": False,
            "gpt2_claim_made": False,
            "radlads_parity_claim_made": False,
        },
        "curves": curves,
        "comparisons": comparisons,
    }


def _curve_rows(cells):
    rows = []
    views = (
        ("artifact_bytes", "teacher_artifact_bytes"),
        ("optimizer_steps", "optimizer_steps"),
        ("wall_clock", "total_wall_clock_seconds"),
    )
    for cell in cells:
        for arm, values in cell["arms"].items():
            for view, field in views:
                rows.append(
                    {
                        "arm": arm,
                        "seed": cell["seed"],
                        "budget_view": view,
                        "budget_point": cell["budget_point"],
                        "x": values[field],
                        "final_test_primary_score": values["final_test_primary_score"],
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["budget_view"],
            row["arm"],
            row["seed"],
            row["x"],
            row["budget_point"],
        ),
    )


def _curve_auc(rows):
    output = []
    groups = {(row["budget_view"], row["arm"], row["seed"]) for row in rows}
    for view, arm, seed in sorted(groups):
        selected = [
            (row["x"], row["final_test_primary_score"])
            for row in rows
            if (row["budget_view"], row["arm"], row["seed"]) == (view, arm, seed)
        ]
        x_values = {x for x, _ in selected}
        row = {
            "budget_view": view,
            "arm": arm,
            "seed": seed,
            "x_range": [min(x_values), max(x_values)],
        }
        if len(x_values) < 2:
            row.update(
                {
                    "area_under_quality_curve": None,
                    "available": False,
                    "unavailable_reason": "budget view has no nonzero x-axis range",
                }
            )
        else:
            row.update(
                {
                    "area_under_quality_curve": trapezoidal_auc(selected),
                    "available": True,
                    "unavailable_reason": None,
                }
            )
        output.append(row)
    return output


def _budget_comparisons(config, cells):
    pairs = (
        ("two_cycle", "exemplar_only"),
        ("two_cycle", "conventional_baseline"),
        ("corridor_only", "exemplar_only"),
        ("two_cycle", "corridor_only"),
    )
    output = []
    for point in config.budget_points:
        selected = [cell for cell in cells if cell["budget_point"] == point.name]
        if len(selected) != len(config.seeds):
            continue
        for index, (left_arm, right_arm) in enumerate(pairs):
            left: dict[str, float] = {}
            right: dict[str, float] = {}
            for cell in selected:
                rows = _read_rows(Path(cell["per_record_metrics"]))
                left_field = _per_record_field(left_arm)
                right_field = _per_record_field(right_arm)
                for row in rows:
                    key = f"{cell['seed']}:{row['record_key']}"
                    left[key] = float(row[left_field])
                    right[key] = float(row[right_field])
            stats = paired_seed_record_comparison(
                left,
                right,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed + index,
                tie_tolerance=config.tie_tolerance,
            )
            output.append(
                {
                    "budget_point": point.name,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "aligned_seeds": list(config.seeds),
                    **stats,
                }
            )
    return output


def _per_record_field(arm):
    return {
        "conventional_baseline": "baseline_score",
        "corridor_only": "corridor_only_score",
        "exemplar_only": "exemplar_only_score",
        "two_cycle": "two_cycle_final_score",
    }[arm]


def _target_costs(config, cells):
    output = []
    for seed in config.seeds:
        for arm in ARM_NAMES:
            points = []
            for cell in cells:
                if cell["seed"] != seed:
                    continue
                values = cell["arms"][arm]
                points.append({"budget_point": cell["budget_point"], **values})
            output.append(
                {
                    "seed": seed,
                    "arm": arm,
                    **observed_target_quality_cost(
                        points, threshold=config.target_quality_threshold
                    ),
                }
            )
    return output


def _write_matrix_state(
    path,
    *,
    config_hash,
    expected_runs,
    expected_arm_cells,
    completed,
    failed,
):
    pending_runs = expected_runs - len(completed) - len(failed)
    write_json(
        path,
        {
            "phase": "P156",
            "config_hash": config_hash,
            "expected_cells": expected_arm_cells,
            "completed_cells": len(completed) * len(ARM_NAMES),
            "failed_cells": len(failed) * len(ARM_NAMES),
            "pending_cells": pending_runs * len(ARM_NAMES),
            "expected_experiment_runs": expected_runs,
            "completed_experiment_runs": sorted(completed),
            "failed_experiment_runs": failed,
            "resume_supported": True,
        },
    )


def _validate_controlled_config(config):
    if not config.budget_points:
        raise ValueError("at least one budget point is required")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if len({point.name for point in config.budget_points}) != len(config.budget_points):
        raise ValueError("budget point names must be unique")
    for point in config.budget_points:
        if point.teacher_artifact_bytes <= 0:
            raise ValueError("teacher artifact byte budgets must be positive")
        if point.total_steps < 2:
            raise ValueError("budget points require at least two optimizer steps")
        if point.wall_clock_seconds <= 0 or not math.isfinite(point.wall_clock_seconds):
            raise ValueError("wall-clock budgets must be finite and positive")
    if config.output_dir.exists() and not (config.resume or config.overwrite):
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass resume=True or overwrite=True"
        )


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _controlled_summary(report):
    primary = [
        row
        for row in report["comparisons"]
        if row["left_arm"] == "two_cycle" and row["right_arm"] == "exemplar_only"
    ]
    return (
        "# P156 Controlled Quality-Per-Byte Experiment\n\n"
        f"- Status: {report['status']}\n"
        f"- Backend: {report['execution_backend']}\n"
        f"- Budget points: {len(report['budget_points'])}\n"
        f"- Seeds: {len(report['seeds'])}\n"
        f"- Quality-per-byte claim allowed: "
        f"{str(report['quality_per_byte_claim_allowed']).lower()}\n"
        f"- Primary comparison results: "
        f"{', '.join(row['result'] for row in primary) or 'incomplete'}\n"
        "- Scale, GPT-2, and RADLADS parity claims: false\n"
    )


@dataclass(frozen=True)
class UnconfoundedQualityExperimentConfig:
    training_fingerprint_artifact: Path
    calibration_fingerprint_artifact: Path
    final_test_fingerprint_artifact: Path
    source_texts: Path
    selected_profile_receipt: Path
    output_dir: Path
    experiment_family: str
    seeds: tuple[int, ...]
    byte_budgets: tuple[int, ...] = ()
    step_budgets: tuple[int, ...] = ()
    fixed_total_steps: int = 3
    fixed_artifact_budget: int | None = None
    wall_clock_safety_ceiling: float = 300.0
    corridor_byte_fraction: float = 0.5
    selection_seed: int = 0
    student_backend: str = "current_qrwkv"
    student_architecture: str | None = None
    batch_size: int = 1
    optimizer: str = "adamw"
    baseline_learning_rate: float = 1e-4
    exemplar_learning_rate: float = 5e-5
    exemplar_max_grad_norm: float | None = 1.0
    target_quality_threshold: float = 1.0
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    tie_tolerance: float = 1e-12
    require_backend: str | None = "cpu"
    resume: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class UnconfoundedQualityExperimentResult:
    status: str
    output_dir: Path
    report_path: Path
    integrity_path: Path


def run_unconfounded_quality_experiment(
    config: UnconfoundedQualityExperimentConfig,
) -> UnconfoundedQualityExperimentResult:
    _validate_unconfounded_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    backend = validate_backend_requirement(config.require_backend)
    write_json(config.output_dir / "cpu_backend_receipt.json", backend)
    if config.experiment_family == "time":
        return _write_deferred_time_family(config)

    points = _family_points(config)
    initialization_subset = _materialize_initialization_subset(
        config, byte_budget=max(point[1] for point in points)
    )
    config_hash = _unconfounded_config_hash(config)
    state_path = config.output_dir / "experiment_matrix_state.json"
    if state_path.is_file():
        prior = read_json_object(state_path)
        if prior.get("config_hash") != config_hash:
            raise ValueError("resume configuration hash mismatch")
    bundles: dict[str, dict[str, Any]] = {}
    cells = []
    completed = []
    failed = {}
    for point_name, artifact_budget, total_steps in points:
        bundle_key = (
            point_name if config.experiment_family == "bytes" else "fixed_artifact"
        )
        if bundle_key not in bundles:
            bundles[bundle_key] = _materialize_subset_bundle(
                config,
                name=bundle_key,
                byte_budget=artifact_budget,
            )
        bundle = bundles[bundle_key]
        bundle["paths"]["initialization_subset"] = initialization_subset["path"]
        bundle["subset_hashes"]["initialization_subset"] = initialization_subset[
            "subset_hash"
        ]
        bundle["artifact_manifest_hashes"]["initialization_subset"] = (
            initialization_subset["artifact_manifest_hash"]
        )
        for seed in config.seeds:
            cell_id = f"{point_name}/seed_{seed}"
            cell_dir = (
                config.output_dir
                / "runs"
                / config.experiment_family
                / point_name
                / f"seed_{seed}"
            )
            cell_path = cell_dir / "cell_report.json"
            cell_hash = stable_hash(
                {
                    "matrix_config_hash": config_hash,
                    "family": config.experiment_family,
                    "point": point_name,
                    "artifact_budget": artifact_budget,
                    "total_steps": total_steps,
                    "seed": seed,
                    "subset_hashes": bundle["subset_hashes"],
                }
            )
            if config.resume and cell_path.is_file():
                cell = read_json_object(cell_path)
                if cell.get("cell_config_hash") != cell_hash:
                    raise ValueError(f"completed cell config hash mismatch: {cell_id}")
                if cell.get("status") == "pass":
                    cells.append(cell)
                    completed.append(cell_id)
                    _write_family_matrix_state(
                        state_path, config_hash, points, config.seeds, completed, failed
                    )
                    continue
            try:
                cell = _run_unconfounded_cell(
                    config,
                    point_name=point_name,
                    artifact_budget=artifact_budget,
                    total_steps=total_steps,
                    seed=seed,
                    bundle=bundle,
                    cell_hash=cell_hash,
                )
                cells.append(cell)
                completed.append(cell_id)
            except Exception as exc:  # preserve completed cells for resume
                failed[cell_id] = f"{type(exc).__name__}: {exc}"
            _write_family_matrix_state(
                state_path, config_hash, points, config.seeds, completed, failed
            )
    complete = len(completed) == len(points) * len(config.seeds) and not failed
    subset_index = {
        "phase": "P156.1",
        "source_artifact": str(config.training_fingerprint_artifact),
        "bundles": bundles,
        "fixed_initialization_subset": initialization_subset,
    }
    write_json(config.output_dir / "budget_subset_index.json", subset_index)
    outputs = _unconfounded_outputs(config, points, cells, complete=complete)
    report_name = (
        "byte_controlled_report.json"
        if config.experiment_family == "bytes"
        else "step_controlled_report.json"
    )
    report_path = config.output_dir / report_name
    integrity_path = config.output_dir / "control_family_integrity.json"
    write_json(report_path, outputs["report"])
    write_json(config.output_dir / "quality_per_byte_report.json", outputs["report"])
    other_report = (
        "step_controlled_report.json"
        if config.experiment_family == "bytes"
        else "byte_controlled_report.json"
    )
    write_json(
        config.output_dir / other_report,
        {
            "phase": "P156.1",
            "status": "not_run",
            "quality_per_byte_claim_allowed": False,
            "quality_per_step_claim_allowed": False,
            "winner_declared": False,
        },
    )
    write_json(
        config.output_dir / "time_controlled_report.json",
        {
            "phase": "P156.1",
            "status": "deferred",
            "deadline_enforced_during_training": False,
            "quality_per_time_claim_allowed": False,
            "winner_declared": False,
        },
    )
    write_json(integrity_path, outputs["integrity"])
    write_json(
        config.output_dir / "publication_grade_receipt.json",
        outputs["publication"],
    )
    write_json(
        config.output_dir / "artifact_byte_accounting.json", outputs["accounting"]
    )
    _write_rows(
        config.output_dir / "paired_budget_comparisons.jsonl",
        outputs["comparisons"],
    )
    _write_rows(config.output_dir / "quality_budget_curve.jsonl", outputs["curves"])
    (config.output_dir / "quality_per_byte_summary.md").write_text(
        _unconfounded_summary(outputs["report"]), encoding="utf-8"
    )
    return UnconfoundedQualityExperimentResult(
        status=outputs["report"]["status"],
        output_dir=config.output_dir,
        report_path=report_path,
        integrity_path=integrity_path,
    )


def _materialize_subset_bundle(config, *, name, byte_budget):
    root = config.output_dir / "subsets" / name
    paths = {}
    receipts = {}
    for role in (
        "corridor_subset",
        "exemplar_subset",
        "combined_two_cycle_subset",
    ):
        result = materialize_budgeted_artifact(
            BudgetedArtifactConfig(
                source_artifact=config.training_fingerprint_artifact,
                source_texts=config.source_texts,
                output_dir=root / role,
                subset_role=role,
                declared_byte_budget=byte_budget,
                selection_seed=config.selection_seed,
                corridor_byte_fraction=config.corridor_byte_fraction,
                overwrite=config.overwrite,
            )
        )
        validation = validate_budgeted_artifact(result.output_dir)
        if not validation["valid"]:
            raise ValueError("materialized subset cache validation failed")
        paths[role] = str(result.output_dir)
        receipts[role] = read_json_object(result.manifest_path)
    return {
        "byte_budget": byte_budget,
        "paths": paths,
        "subset_hashes": {
            role: receipt["subset_manifest_sha256"]
            for role, receipt in receipts.items()
        },
        "artifact_manifest_hashes": {
            role: receipt["artifact_manifest_sha256"]
            for role, receipt in receipts.items()
        },
        "cache_reusable_across_seeds": True,
    }


def _materialize_initialization_subset(config, *, byte_budget):
    result = materialize_budgeted_artifact(
        BudgetedArtifactConfig(
            source_artifact=config.training_fingerprint_artifact,
            source_texts=config.source_texts,
            output_dir=config.output_dir
            / "subsets"
            / "fixed_initialization"
            / "corridor_subset",
            subset_role="corridor_subset",
            declared_byte_budget=byte_budget,
            selection_seed=config.selection_seed,
            corridor_byte_fraction=config.corridor_byte_fraction,
            overwrite=config.overwrite,
        )
    )
    receipt = read_json_object(result.manifest_path)
    return {
        "path": str(result.output_dir),
        "byte_budget": byte_budget,
        "subset_hash": receipt["subset_manifest_sha256"],
        "artifact_manifest_hash": receipt["artifact_manifest_sha256"],
        "fixed_across_points_and_seeds": True,
    }


def _run_unconfounded_cell(
    config,
    *,
    point_name,
    artifact_budget,
    total_steps,
    seed,
    bundle,
    cell_hash,
):
    corridor_steps = max(1, total_steps // 2)
    exemplar_steps = total_steps - corridor_steps
    if exemplar_steps < 1:
        raise ValueError("two-cycle budget requires at least two total steps")
    paths = {name: Path(path) for name, path in bundle["paths"].items()}
    if any(path == config.training_fingerprint_artifact for path in paths.values()):
        raise ValueError("full_artifact_used_in_byte_controlled_cell")
    cell_dir = (
        config.output_dir
        / "runs"
        / config.experiment_family
        / point_name
        / f"seed_{seed}"
    )
    p155 = run_two_cycle_experiment(
        TwoCycleExperimentConfig(
            training_fingerprint_artifact=config.training_fingerprint_artifact,
            calibration_fingerprint_artifact=config.calibration_fingerprint_artifact,
            final_test_fingerprint_artifact=config.final_test_fingerprint_artifact,
            source_texts=config.source_texts,
            selected_profile_receipt=config.selected_profile_receipt,
            output_dir=cell_dir / "p155",
            initialization_training_artifact=paths["initialization_subset"],
            corridor_training_artifact=paths["corridor_subset"],
            exemplar_training_artifact=paths["exemplar_subset"],
            two_cycle_corridor_training_artifact=paths["combined_two_cycle_subset"],
            two_cycle_exemplar_training_artifact=paths["combined_two_cycle_subset"],
            allow_payload_specific_artifacts=True,
            student_backend=config.student_backend,
            student_architecture=config.student_architecture,
            baseline_steps=total_steps,
            corridor_steps=corridor_steps,
            exemplar_steps=exemplar_steps,
            corridor_only_steps=total_steps,
            exemplar_only_steps=total_steps,
            budget_comparison_mode="equal_total_budget",
            batch_size=config.batch_size,
            optimizer=config.optimizer,
            baseline_learning_rate=config.baseline_learning_rate,
            exemplar_learning_rate=config.exemplar_learning_rate,
            exemplar_max_grad_norm=config.exemplar_max_grad_norm,
            checkpoint_every=max(1, total_steps),
            seed=seed,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
            tie_tolerance=config.tie_tolerance,
            overwrite=config.overwrite,
        )
    )
    report = read_json_object(p155.report_path)
    if p155.status != "pass":
        raise ValueError("P155.1 subset-backed cell failed")
    arms = _unconfounded_cell_arms(
        config,
        artifact_budget=artifact_budget,
        total_steps=total_steps,
        bundle=bundle,
        p155=report,
    )
    integrity = bool(
        all(
            arm["budget_ceiling_respected"]
            and arm["step_budget_match_valid"]
            and arm["loader_observed_subset_hash_matches"]
            and not arm["wall_clock_budget_exceeded"]
            for arm in arms.values()
        )
        and report["split_integrity"]["three_way_split_valid"]
    )
    if not integrity:
        raise ValueError("cell budget or subset integrity failed")
    cell = {
        "phase": "P156.1",
        "status": "pass",
        "cell_config_hash": cell_hash,
        "control_family": config.experiment_family,
        "budget_point": point_name,
        "seed": seed,
        "varied_resource": "artifact_bytes"
        if config.experiment_family == "bytes"
        else "optimizer_steps",
        "fixed_resource_receipt": {
            "total_steps": total_steps,
            "artifact_budget": artifact_budget,
            "wall_clock_safety_ceiling": config.wall_clock_safety_ceiling,
            "batch_size": config.batch_size,
            "final_test_artifact_sha256": file_sha256(
                config.final_test_fingerprint_artifact / "manifest.json"
            ),
        },
        "subset_hashes": bundle["subset_hashes"],
        "arms": arms,
        "actual_consumption": {
            arm: {
                key: values[key]
                for key in (
                    "records_consumed",
                    "logical_payload_bytes_consumed",
                    "optimizer_steps",
                    "total_wall_clock_seconds",
                )
            }
            for arm, values in arms.items()
        },
        "budget_integrity_valid": True,
        "split_integrity": report["split_integrity"],
        "per_record_metrics": str(cell_dir / "p155" / "per_record_arm_metrics.jsonl"),
        "p155_report": str(p155.report_path),
    }
    write_json(cell_dir / "cell_report.json", cell)
    return cell


def _unconfounded_cell_arms(config, *, artifact_budget, total_steps, bundle, p155):
    resources = {
        "conventional_baseline": p155["resources"]["conventional_baseline"],
        "corridor_only": p155["resources"]["corridor_only"],
        "exemplar_only": p155["resources"]["exemplar_only"],
        "two_cycle": p155["resources"]["two_cycle"]["combined"],
    }
    evaluations = {
        "conventional_baseline": "conventional_baseline",
        "corridor_only": "corridor_only",
        "exemplar_only": "exemplar_only",
        "two_cycle": "two_cycle_final",
    }
    subset_roles = {
        "corridor_only": "corridor_subset",
        "exemplar_only": "exemplar_subset",
        "two_cycle": "combined_two_cycle_subset",
    }
    init_score = p155["evaluation_metrics"]["shared_initialization"]["teacher"][
        "teacher_student_kl"
    ]
    output = {}
    for arm, evaluation_name in evaluations.items():
        resource = resources[arm]
        metrics = p155["evaluation_metrics"][evaluation_name]
        if arm == "conventional_baseline":
            subset_role = None
            accounting = {
                "arm_charged_bytes": 0,
                "logical_payload_bytes_selected": 0,
                "physical_subset_bytes": 0,
                "shared_metadata_bytes": 0,
                "unused_budget_bytes": artifact_budget,
            }
            selected_count = 0
            configured_path = None
            subset_hash = None
            artifact_hash = None
        else:
            subset_role = subset_roles[arm]
            configured_path = Path(bundle["paths"][subset_role])
            accounting = read_json_object(
                configured_path / "artifact_byte_accounting.json"
            )
            selection = read_json_object(
                configured_path / "record_selection_receipt.json"
            )
            selected_count = (
                selection["selected_target_count"]
                if arm == "corridor_only"
                else selection["selected_exemplar_count"]
                if arm == "exemplar_only"
                else selection["selected_target_count"]
                + selection["selected_exemplar_count"]
            )
            subset_hash = bundle["subset_hashes"][subset_role]
            artifact_hash = bundle["artifact_manifest_hashes"][subset_role]
        charged = accounting["arm_charged_bytes"]
        consumed_fraction = min(resource["records_consumed"], selected_count) / max(
            selected_count, 1
        )
        logical_consumed = round(charged * consumed_fraction)
        score = metrics["teacher"]["teacher_student_kl"]
        output[arm] = {
            "quality": {
                "teacher_student_kl": score,
                "held_out_exemplar_loss": metrics["teacher"].get(
                    "teacher_student_cross_entropy"
                ),
                "top1_agreement": metrics["teacher"].get("top1_agreement"),
                "topk_overlap": metrics["teacher"].get("topk_overlap"),
                "teacher_entropy": metrics["teacher"].get("teacher_entropy"),
                "student_entropy": metrics["teacher"].get("student_entropy"),
                "entropy_absolute_error": metrics["teacher"].get(
                    "entropy_absolute_error"
                ),
                "held_out_corridor_loss": metrics["corridor"].get(
                    "corridor_loss_total"
                ),
                "inside_all_rate": metrics["corridor"].get("inside_all_rate"),
                "mean_distance_outside_corridor": metrics["corridor"].get(
                    "mean_distance_outside_corridor"
                ),
            },
            "final_test_primary_score": score,
            "configured_training_artifact_path": str(configured_path)
            if configured_path
            else None,
            "configured_subset_manifest_sha256": subset_hash,
            "loader_observed_artifact_sha256": artifact_hash,
            "loader_observed_subset_hash_matches": arm == "conventional_baseline"
            or file_sha256(configured_path / "manifest.json") == artifact_hash,
            "records_available": selected_count,
            "records_consumed": resource["records_consumed"],
            "logical_bytes_available": charged,
            "teacher_artifact_bytes": charged,
            "logical_payload_bytes_consumed": logical_consumed,
            "declared_byte_budget": artifact_budget,
            "physical_subset_bytes": accounting["physical_subset_bytes"],
            "shared_metadata_bytes": accounting["shared_metadata_bytes"],
            "arm_charged_bytes": charged,
            "unused_budget_bytes": accounting["unused_budget_bytes"],
            "budget_ceiling_respected": charged <= artifact_budget,
            "total_step_budget": total_steps,
            "optimizer_steps": resource["optimizer_steps"],
            "step_budget_match_valid": resource["optimizer_steps"] == total_steps,
            "training_seconds": resource["training_seconds"],
            "evaluation_seconds": resource["evaluation_seconds"],
            "checkpoint_seconds": resource["checkpoint_seconds"],
            "total_wall_clock_seconds": resource["total_wall_clock_seconds"],
            "wall_clock_budget_seconds": config.wall_clock_safety_ceiling,
            "wall_clock_budget_exceeded": resource["total_wall_clock_seconds"]
            > config.wall_clock_safety_ceiling,
            "efficiency_reference_arm": "shared_initialization",
            "efficiency_reference_score": init_score,
            "efficiency": efficiency_ratios(
                reference_score=init_score,
                final_score=score,
                artifact_bytes=charged,
                optimizer_steps=resource["optimizer_steps"],
                wall_clock_seconds=resource["total_wall_clock_seconds"],
            )
            if charged > 0
            else None,
        }
    return output


def _unconfounded_outputs(config, points, cells, *, complete):
    point_names = [point[0] for point in points]
    comparisons = _unconfounded_comparisons(config, point_names, cells)
    curves = _curve_rows(cells)
    all_arms = [arm for cell in cells for arm in cell["arms"].values()]
    wall_nonbinding = bool(all_arms) and all(
        not arm["wall_clock_budget_exceeded"] for arm in all_arms
    )
    actual_subsets = bool(cells) and all(
        arm["configured_training_artifact_path"]
        != str(config.training_fingerprint_artifact)
        for cell in cells
        for name, arm in cell["arms"].items()
        if name != "conventional_baseline"
    )
    split_valid = complete and all(
        cell["split_integrity"]["three_way_split_valid"]
        and cell["split_integrity"]["final_test_independent"]
        for cell in cells
    )
    if config.experiment_family == "bytes":
        family_integrity = {
            "bytes_vary": len(points) >= 2
            and len({point[1] for point in points}) == len(points),
            "steps_fixed": len({point[2] for point in points}) == 1,
            "wall_clock_nonbinding": wall_nonbinding,
            "actual_subsets_consumed": actual_subsets,
        }
    else:
        hashes_by_arm = {
            arm: {
                cell["subset_hashes"][role]
                for cell in cells
                for role in (
                    "corridor_subset",
                    "exemplar_subset",
                    "combined_two_cycle_subset",
                )
                if (
                    (arm == "corridor_only" and role == "corridor_subset")
                    or (arm == "exemplar_only" and role == "exemplar_subset")
                    or (arm == "two_cycle" and role == "combined_two_cycle_subset")
                )
            }
            for arm in ("corridor_only", "exemplar_only", "two_cycle")
        }
        family_integrity = {
            "steps_vary": len(points) >= 2
            and len({point[2] for point in points}) == len(points),
            "artifact_subset_fixed": all(
                len(values) == 1 for values in hashes_by_arm.values()
            ),
            "byte_availability_fixed": len({point[1] for point in points}) == 1,
            "wall_clock_nonbinding": wall_nonbinding,
            "actual_subsets_consumed": actual_subsets,
        }
    family_integrity["valid"] = bool(
        complete and split_valid and all(family_integrity.values())
    )
    execution_valid = bool(
        complete and split_valid and wall_nonbinding and actual_subsets
    )
    byte_integrity = (
        family_integrity
        if config.experiment_family == "bytes"
        else {"valid": False, "status": "not_run"}
    )
    step_integrity = (
        family_integrity
        if config.experiment_family == "steps"
        else {"valid": False, "status": "not_run"}
    )
    integrity = {
        "phase": "P156.1",
        "byte_controlled": byte_integrity,
        "step_controlled": step_integrity,
        "time_controlled": {
            "time_varies": False,
            "artifact_subset_fixed": False,
            "step_ceiling_nonbinding": False,
            "deadline_enforced_during_training": False,
            "valid": False,
            "status": "deferred",
        },
    }
    claim_allowed = bool(
        family_integrity["valid"]
        and len(points) >= 2
        and len(config.seeds) >= 3
        and comparisons
    )
    byte_claim = claim_allowed and config.experiment_family == "bytes"
    step_claim = claim_allowed and config.experiment_family == "steps"
    primary_comparisons = [
        row
        for row in comparisons
        if row["left_arm"] == "two_cycle" and row["right_arm"] == "exemplar_only"
    ]
    normalized_results = [
        "two_cycle_better"
        if row["result"] == "left_better"
        else "exemplar_only_better"
        if row["result"] == "right_better"
        else "inconclusive"
        for row in primary_comparisons
    ]
    primary_result = (
        normalized_results[0]
        if normalized_results and len(set(normalized_results)) == 1
        else "mixed"
        if normalized_results
        else None
    )
    report = {
        "phase": "P156.1",
        "status": "pass" if execution_valid else "fail",
        "experiment_kind": "unconfounded_controlled_efficiency",
        "experiment_family": f"{config.experiment_family}_controlled",
        "execution_backend": jax.default_backend(),
        "actual_subsets_consumed": actual_subsets,
        "required_arms_complete": complete,
        "required_seeds_complete": complete and len(config.seeds) >= 3,
        "independent_final_test": split_valid,
        "primary_metric": "independent_final_test_teacher_student_kl",
        "primary_metric_direction": "lower_is_better",
        "primary_comparison": "two_cycle_vs_exemplar_only",
        "primary_result": primary_result,
        "primary_result_by_point": {
            row["budget_point"]: normalized
            for row, normalized in zip(
                primary_comparisons, normalized_results, strict=True
            )
        },
        "points": [
            {
                "name": name,
                "artifact_byte_ceiling": byte_budget,
                "total_step_budget": steps,
            }
            for name, byte_budget, steps in points
        ],
        "seeds": list(config.seeds),
        "cells": cells,
        "comparisons": comparisons,
        "target_quality": _target_costs(config, cells),
        "curve_auc": _curve_auc(curves) if complete and len(points) >= 2 else [],
        "auc_method": "deterministic_trapezoidal_integration",
        "quality_per_byte_claim_allowed": byte_claim,
        "quality_per_step_claim_allowed": step_claim,
        "quality_per_time_claim_allowed": False,
        "winner_declared": bool(
            claim_allowed and primary_result not in {None, "inconclusive", "mixed"}
        ),
        "claim_scope": "tiny_cpu_controlled_experiment",
        "scale_claim_made": False,
        "gpt2_claim_made": False,
        "radlads_parity_claim_made": False,
    }
    accounting_rows = [
        {
            "budget_point": cell["budget_point"],
            "seed": cell["seed"],
            "arm": arm,
            **{
                key: values[key]
                for key in (
                    "declared_byte_budget",
                    "physical_subset_bytes",
                    "logical_bytes_available",
                    "logical_payload_bytes_consumed",
                    "shared_metadata_bytes",
                    "arm_charged_bytes",
                    "unused_budget_bytes",
                    "budget_ceiling_respected",
                )
            },
        }
        for cell in cells
        for arm, values in cell["arms"].items()
    ]
    return {
        "report": report,
        "integrity": integrity,
        "publication": {
            "phase": "P156.1",
            "status": "pass" if claim_allowed else "fail",
            "publication_grade": claim_allowed,
            "experiment_family": config.experiment_family,
            "completed_arm_cells": len(cells) * len(ARM_NAMES),
            "expected_arm_cells": len(points) * len(config.seeds) * len(ARM_NAMES),
            "quality_per_byte_claim_allowed": byte_claim,
            "quality_per_step_claim_allowed": step_claim,
            "quality_per_time_claim_allowed": False,
        },
        "accounting": {
            "phase": "P156.1",
            "byte_accounting_policy": "arm_charged_logical_payload_bytes_v1",
            "physical_and_logical_bytes_distinct": True,
            "rows": accounting_rows,
        },
        "comparisons": comparisons,
        "curves": curves,
    }


def _unconfounded_comparisons(config, point_names, cells):
    pairs = (
        ("two_cycle", "exemplar_only"),
        ("two_cycle", "conventional_baseline"),
        ("corridor_only", "exemplar_only"),
        ("two_cycle", "corridor_only"),
    )
    output = []
    for point_name in point_names:
        selected = [cell for cell in cells if cell["budget_point"] == point_name]
        if len(selected) != len(config.seeds):
            continue
        for index, (left_arm, right_arm) in enumerate(pairs):
            left = {}
            right = {}
            for cell in selected:
                for row in _read_rows(Path(cell["per_record_metrics"])):
                    key = f"{cell['seed']}:{row['record_key']}"
                    left[key] = float(row[_per_record_field(left_arm)])
                    right[key] = float(row[_per_record_field(right_arm)])
            output.append(
                {
                    "budget_point": point_name,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "aligned_seeds": list(config.seeds),
                    **paired_seed_record_comparison(
                        left,
                        right,
                        bootstrap_samples=config.bootstrap_samples,
                        bootstrap_seed=config.bootstrap_seed + index,
                        tie_tolerance=config.tie_tolerance,
                    ),
                }
            )
    return output


def _family_points(config):
    if config.experiment_family == "bytes":
        return [
            (f"bytes_{budget}", budget, config.fixed_total_steps)
            for budget in config.byte_budgets
        ]
    return [
        (f"steps_{steps}", int(config.fixed_artifact_budget), steps)
        for steps in config.step_budgets
    ]


def _unconfounded_config_hash(config):
    return stable_hash(
        {
            "phase": "P156.1",
            "experiment_family": config.experiment_family,
            "training_artifact_sha256": file_sha256(
                config.training_fingerprint_artifact / "manifest.json"
            ),
            "calibration_artifact_sha256": file_sha256(
                config.calibration_fingerprint_artifact / "manifest.json"
            ),
            "final_test_artifact_sha256": file_sha256(
                config.final_test_fingerprint_artifact / "manifest.json"
            ),
            "selected_profile_receipt_sha256": file_sha256(
                config.selected_profile_receipt
            ),
            "source_texts_sha256": file_sha256(config.source_texts),
            "byte_budgets": list(config.byte_budgets),
            "step_budgets": list(config.step_budgets),
            "fixed_total_steps": config.fixed_total_steps,
            "fixed_artifact_budget": config.fixed_artifact_budget,
            "wall_clock_safety_ceiling": config.wall_clock_safety_ceiling,
            "corridor_byte_fraction": config.corridor_byte_fraction,
            "selection_seed": config.selection_seed,
            "seeds": list(config.seeds),
            "student_backend": config.student_backend,
            "student_architecture": config.student_architecture,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer,
            "software_commit": get_git_metadata(
                Path(__file__).resolve().parents[3]
            ).get("commit"),
        }
    )


def _write_family_matrix_state(path, config_hash, points, seeds, completed, failed):
    expected_runs = len(points) * len(seeds)
    pending = expected_runs - len(completed) - len(failed)
    write_json(
        path,
        {
            "phase": "P156.1",
            "config_hash": config_hash,
            "expected_cells": expected_runs * len(ARM_NAMES),
            "completed_cells": len(completed) * len(ARM_NAMES),
            "failed_cells": len(failed) * len(ARM_NAMES),
            "pending_cells": pending * len(ARM_NAMES),
            "completed_experiment_runs": sorted(completed),
            "failed_experiment_runs": failed,
            "resume_supported": True,
        },
    )


def _write_deferred_time_family(config):
    report = {
        "phase": "P156.1",
        "status": "deferred",
        "experiment_family": "time_controlled",
        "deadline_enforced_during_training": False,
        "declared_training_deadline_seconds": None,
        "deadline_start_timestamp": None,
        "deadline_stop_timestamp": None,
        "completed_steps_before_deadline": None,
        "deadline_triggered": False,
        "deadline_overshoot_seconds": None,
        "post_hoc_only_enforcement": False,
        "quality_per_time_claim_allowed": False,
        "winner_declared": False,
        "reason": "true optimizer-step-boundary deadline enforcement is deferred",
    }
    report_path = config.output_dir / "time_controlled_report.json"
    integrity_path = config.output_dir / "control_family_integrity.json"
    write_json(report_path, report)
    write_json(
        integrity_path,
        {
            "phase": "P156.1",
            "byte_controlled": {"valid": False, "status": "not_run"},
            "step_controlled": {"valid": False, "status": "not_run"},
            "time_controlled": {
                "deadline_enforced_during_training": False,
                "valid": False,
                "status": "deferred",
            },
        },
    )
    return UnconfoundedQualityExperimentResult(
        status="deferred",
        output_dir=config.output_dir,
        report_path=report_path,
        integrity_path=integrity_path,
    )


def _validate_unconfounded_config(config):
    if config.experiment_family not in {"bytes", "steps", "time"}:
        raise ValueError("experiment_family must be bytes, steps, or time")
    if not config.seeds or len(config.seeds) != len(set(config.seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if config.experiment_family == "bytes":
        if not config.byte_budgets or any(value <= 0 for value in config.byte_budgets):
            raise ValueError("byte-controlled family requires positive byte budgets")
        if config.fixed_total_steps < 2:
            raise ValueError("fixed_total_steps must be at least two")
    if config.experiment_family == "steps":
        if not config.step_budgets or any(value < 2 for value in config.step_budgets):
            raise ValueError("step-controlled family requires step budgets >= 2")
        if config.fixed_artifact_budget is None or config.fixed_artifact_budget <= 0:
            raise ValueError("step family requires fixed_artifact_budget")
    if config.wall_clock_safety_ceiling <= 0:
        raise ValueError("wall_clock_safety_ceiling must be positive")


def _unconfounded_summary(report):
    primary = [
        row
        for row in report.get("comparisons", [])
        if row["left_arm"] == "two_cycle" and row["right_arm"] == "exemplar_only"
    ]
    return (
        "# P156.1 Unconfounded Controlled Efficiency\n\n"
        f"- Status: {report['status']}\n"
        f"- Family: {report['experiment_family']}\n"
        f"- Actual subsets consumed: "
        f"{str(report['actual_subsets_consumed']).lower()}\n"
        f"- Primary results: "
        f"{', '.join(row['result'] for row in primary) or 'incomplete'}\n"
        f"- Quality-per-byte claim allowed: "
        f"{str(report['quality_per_byte_claim_allowed']).lower()}\n"
        f"- Quality-per-step claim allowed: "
        f"{str(report['quality_per_step_claim_allowed']).lower()}\n"
        "- Quality-per-time claim allowed: false\n"
    )
