from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    FingerprintBatch,
    FingerprintTargetRecord,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.held_out_evaluation import (
    validate_fingerprint_provenance,
)
from qrwkv_xla.fingerprint.provenance import (
    build_artifact_source_lineage,
    hash_checkpoint_bundle,
    parameter_fingerprint,
)
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats_at_positions,
)
from qrwkv_xla.training.gradients import clip_gradients_by_global_norm

STAT_NAMES = ("entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass")
STOP_REASONS = {
    "requested_steps_completed",
    "stable_corridor_entry",
    "non_finite_loss",
    "non_finite_gradient",
    "checkpoint_failure",
    "evaluation_failure",
    "stability_abort",
    "user_abort",
}


@dataclass(frozen=True)
class CorridorMeasurementConfig:
    fingerprint_artifact: Path
    held_out_fingerprint_artifact: Path
    source_texts: Path
    output_dir: Path
    steps: int = 25
    eval_every: int | None = None
    checkpoint_every: int = 25
    batch_size: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    max_grad_norm: float | None = 1.0
    corridor_loss_weight: float = 1.0
    penalty_kind: str = "powered_hinge"
    penalty_power: float = 2.0
    entropy_weight: float = 1.0
    top1_margin_weight: float = 1.0
    top8_mass_weight: float = 1.0
    top32_mass_weight: float = 1.0
    tail_mass_weight: float = 1.0
    worst_stat_boost: float = 1.0
    distance_normalization: str = "none"
    stability_abort_enabled: bool = True
    parameter_norm_limit: float = 1e6
    gradient_norm_hard_limit: float = 1e6
    held_out_loss_abort_multiple: float = 100.0
    seed: int = 0
    initial_checkpoint: Path | None = None
    student_backend: str = "current_qrwkv"
    corridor_entry_threshold: float = 0.95
    stable_entry_evals: int = 3
    stop_on_stable_entry: bool = False
    p151_report: Path | None = None
    selected_aggressiveness_profile: str | None = None
    selected_profile_config_sha256: str | None = None
    held_out_artifact_role: str = "held_out_evaluation"
    overwrite: bool = False


@dataclass(frozen=True)
class CorridorMeasurementResult:
    status: str
    output_dir: Path
    report_path: Path
    summary_path: Path
    trajectory_path: Path
    checkpoint_dir: Path
    completed_steps: int
    stable_entry_achieved: bool


def corridor_distance(
    value: float,
    lower: float,
    upper: float,
    *,
    epsilon: float = 1e-8,
) -> tuple[float, float]:
    if lower > upper:
        raise ValueError("corridor lower bound must be <= upper bound")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    raw = max(lower - value, 0.0, value - upper)
    normalized = raw / max(upper - lower, epsilon)
    return float(raw), float(normalized)


def detect_corridor_entries(
    trajectory: list[dict[str, Any]],
    *,
    threshold: float,
    stable_entry_evals: int,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("corridor_entry_threshold must be within [0, 1]")
    if stable_entry_evals < 1:
        raise ValueError("stable_entry_evals must be >= 1")
    strict = None
    threshold_step = None
    stable = None
    consecutive = 0
    for point in trajectory:
        rate = float(point["inside_all_rate"])
        step = int(point["optimizer_step"])
        if strict is None and rate == 1.0:
            strict = step
        if rate >= threshold:
            if threshold_step is None:
                threshold_step = step
            consecutive += 1
            if stable is None and consecutive >= stable_entry_evals:
                stable = step
        else:
            consecutive = 0
    return {
        "first_strict_entry_step": strict,
        "first_threshold_entry_step": threshold_step,
        "first_stable_entry_step": stable,
        "stable_entry_achieved": stable is not None,
    }


def run_corridor_measurement(
    config: CorridorMeasurementConfig,
) -> CorridorMeasurementResult:
    _validate_config(config)
    started_perf = time.perf_counter()
    run_started_at = _utc_now()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_interval = config.eval_every or max(1, config.steps // 20)

    train_provenance = validate_fingerprint_provenance(
        config.fingerprint_artifact,
        expected_role="training",
    )
    held_out_provenance = validate_fingerprint_provenance(
        config.held_out_fingerprint_artifact,
        expected_role=config.held_out_artifact_role,
    )
    artifact_lineage = build_artifact_source_lineage(
        config.fingerprint_artifact,
        config.source_texts,
    )
    lineage_receipt = _lineage_receipt(
        config,
        train_provenance=train_provenance,
        held_out_provenance=held_out_provenance,
        artifact_lineage=artifact_lineage,
    )
    write_json(
        config.output_dir / "checkpoint_lineage_validation.json",
        lineage_receipt,
    )
    if not lineage_receipt["publication_grade_lineage"]:
        raise ValueError(
            "P153 lineage validation failed: " + "; ".join(lineage_receipt["blockers"])
        )

    train_summary = summarize_fingerprint_artifact(config.fingerprint_artifact)
    train_dataset = load_fingerprint_targets(
        config.fingerprint_artifact,
        batch_size=config.batch_size,
    )
    train_batches = tuple(train_dataset.iter_batches())
    if not train_batches:
        raise ValueError("training fingerprint artifact yielded zero batches")
    held_out_records = tuple(
        load_fingerprint_targets(
            config.held_out_fingerprint_artifact,
            batch_size=1,
        ).iter_records()
    )
    if not held_out_records:
        raise ValueError("held-out fingerprint artifact yielded zero records")

    backend, student_config = _create_backend(config, train_summary)
    checkpoint_load_started = time.perf_counter()
    if config.initial_checkpoint is None:
        initial_params = backend.init_params(jax.random.PRNGKey(config.seed))
        optimizer_config = OptimizerConfig(
            type=config.optimizer,
            learning_rate=config.learning_rate,
        )
        optimizer_state = init_optimizer_state(initial_params, optimizer_config)
        start_step = 0
    else:
        loaded = load_checkpoint(config.initial_checkpoint)
        initial_params = loaded.params
        optimizer_config = OptimizerConfig(
            type=config.optimizer,
            learning_rate=config.learning_rate,
        )
        optimizer_state = loaded.optimizer_state or init_optimizer_state(
            initial_params,
            optimizer_config,
        )
        start_step = loaded.manifest.step
        if loaded.manifest.student_architecture != config.student_backend:
            raise ValueError("initial checkpoint student backend mismatch")
        if loaded.manifest.student_config != student_config:
            raise ValueError("initial checkpoint student config mismatch")
    checkpoint_load_seconds = time.perf_counter() - checkpoint_load_started
    state = TrainState(
        params=initial_params,
        step=start_step,
        learning_rate=config.learning_rate,
        optimizer_state=optimizer_state,
    )
    train_step = _make_train_step(
        backend,
        optimizer_config=optimizer_config,
        max_grad_norm=config.max_grad_norm,
        config=config,
    )
    jax_batches = tuple(_batch_to_jax(batch) for batch in train_batches)
    record_sizes = _target_record_sizes(config.fingerprint_artifact)
    batch_record_indices = _batch_indices(len(record_sizes), config.batch_size)

    startup_seconds = time.perf_counter() - started_perf
    training_started_at = _utc_now()
    training_seconds = 0.0
    held_out_evaluation_seconds = 0.0
    checkpoint_write_seconds = 0.0
    total_record_visits = 0
    tokens_consumed = 0
    artifact_bytes_logically_consumed = 0
    visited_record_indices: set[int] = set()
    trajectory: list[dict[str, Any]] = []
    last_training_metrics = _evaluate_training_batch(
        backend,
        state.params,
        train_batches[0],
        config=config,
    )

    checkpoint_started = time.perf_counter()
    step_zero_checkpoint = config.output_dir / "checkpoints" / "step_000000"
    _save_measurement_checkpoint(
        config,
        state,
        student_config=student_config,
        path=step_zero_checkpoint,
        artifact_lineage=artifact_lineage,
    )
    checkpoint_write_seconds += time.perf_counter() - checkpoint_started
    lineage_receipt["initialization"].update(
        {
            "parameter_fingerprint": parameter_fingerprint(initial_params),
            "measurement_step_zero_checkpoint": str(step_zero_checkpoint),
            **{
                f"measurement_{key}": value
                for key, value in hash_checkpoint_bundle(step_zero_checkpoint).items()
            },
        }
    )
    if config.p151_report is not None:
        p151 = read_json_object(config.p151_report)
        shared = p151.get("lineage", {}).get("shared_initialization", {})
        lineage_receipt["p151_initialization_match"] = bool(
            shared.get("seed") == config.seed
            and shared.get("parameter_fingerprint")
            == parameter_fingerprint(initial_params)
        )
        if not lineage_receipt["p151_initialization_match"]:
            lineage_receipt["blockers"].append(
                "P151 shared initialization lineage mismatch"
            )
            lineage_receipt["publication_grade_lineage"] = False
    write_json(
        config.output_dir / "checkpoint_lineage_validation.json",
        lineage_receipt,
    )
    if not lineage_receipt["publication_grade_lineage"]:
        raise ValueError(
            "P153 initialization lineage validation failed: "
            + "; ".join(lineage_receipt["blockers"])
        )

    eval_started = time.perf_counter()
    trajectory.append(
        _trajectory_point(
            optimizer_step=0,
            elapsed=time.perf_counter() - started_perf,
            records_consumed=0,
            tokens_consumed=0,
            artifact_bytes_read=0,
            training_metrics=last_training_metrics,
            held_out=_evaluate_held_out(
                backend,
                state.params,
                held_out_records,
            ),
            grad_metrics=None,
            initial_params=initial_params,
            params=state.params,
            learning_rate=config.learning_rate,
        )
    )
    held_out_evaluation_seconds += time.perf_counter() - eval_started

    initial_entries = detect_corridor_entries(
        trajectory,
        threshold=config.corridor_entry_threshold,
        stable_entry_evals=config.stable_entry_evals,
    )
    stop_reason = (
        "stable_corridor_entry"
        if config.stop_on_stable_entry and initial_entries["stable_entry_achieved"]
        else "requested_steps_completed"
    )
    completed_steps = 0
    grad_metrics: dict[str, float] | None = None
    steps_to_run = 0 if stop_reason == "stable_corridor_entry" else config.steps
    for local_step in range(1, steps_to_run + 1):
        batch_index = (local_step - 1) % len(jax_batches)
        train_started = time.perf_counter()
        state, raw_metrics = train_step(state, jax_batches[batch_index])
        jax.block_until_ready(state.params)
        training_seconds += time.perf_counter() - train_started
        grad_metrics = {key: float(value) for key, value in raw_metrics.items()}
        completed_steps = local_step
        batch = train_batches[batch_index]
        size = int(batch.input_ids.shape[0])
        total_record_visits += size
        tokens_consumed += int(batch.input_ids.size)
        indices = batch_record_indices[batch_index]
        visited_record_indices.update(indices)
        artifact_bytes_logically_consumed += sum(
            record_sizes[index] for index in indices
        )
        if not math.isfinite(grad_metrics["loss"]):
            stop_reason = "non_finite_loss"
            break
        if not math.isfinite(grad_metrics["grad_global_norm"]):
            stop_reason = "non_finite_gradient"
            break
        if config.stability_abort_enabled and (
            grad_metrics["grad_global_norm"] > config.gradient_norm_hard_limit
            or _tree_norm(state.params) > config.parameter_norm_limit
        ):
            stop_reason = "stability_abort"
            break
        last_training_metrics = {
            "corridor_loss": grad_metrics["loss"],
            "inside_all_rate": grad_metrics["inside_all_rate"],
        }

        should_eval = (
            local_step % evaluation_interval == 0 or local_step == config.steps
        )
        if should_eval:
            eval_started = time.perf_counter()
            held_out = _evaluate_held_out(backend, state.params, held_out_records)
            trajectory.append(
                _trajectory_point(
                    optimizer_step=local_step,
                    elapsed=time.perf_counter() - started_perf,
                    records_consumed=total_record_visits,
                    tokens_consumed=tokens_consumed,
                    artifact_bytes_read=artifact_bytes_logically_consumed,
                    training_metrics=last_training_metrics,
                    held_out=held_out,
                    grad_metrics=grad_metrics,
                    initial_params=initial_params,
                    params=state.params,
                    learning_rate=config.learning_rate,
                )
            )
            held_out_evaluation_seconds += time.perf_counter() - eval_started
            if (
                config.stability_abort_enabled
                and held_out["corridor_loss"]
                > max(
                    trajectory[0]["held_out_corridor_loss"],
                    1e-12,
                )
                * config.held_out_loss_abort_multiple
            ):
                stop_reason = "stability_abort"
            entries = detect_corridor_entries(
                trajectory,
                threshold=config.corridor_entry_threshold,
                stable_entry_evals=config.stable_entry_evals,
            )
            if (
                stop_reason != "stability_abort"
                and config.stop_on_stable_entry
                and entries["stable_entry_achieved"]
            ):
                stop_reason = "stable_corridor_entry"

        should_checkpoint = (
            local_step % config.checkpoint_every == 0
            or local_step == config.steps
            or stop_reason == "stable_corridor_entry"
            or stop_reason == "stability_abort"
        )
        if should_checkpoint:
            checkpoint_started = time.perf_counter()
            _save_measurement_checkpoint(
                config,
                state,
                student_config=student_config,
                path=config.output_dir / "checkpoints" / f"step_{local_step:06d}",
                artifact_lineage=artifact_lineage,
            )
            checkpoint_write_seconds += time.perf_counter() - checkpoint_started
        if stop_reason in {"stable_corridor_entry", "stability_abort"}:
            break

    training_finished_at = _utc_now()
    if trajectory[-1]["optimizer_step"] != completed_steps:
        eval_started = time.perf_counter()
        trajectory.append(
            _trajectory_point(
                optimizer_step=completed_steps,
                elapsed=time.perf_counter() - started_perf,
                records_consumed=total_record_visits,
                tokens_consumed=tokens_consumed,
                artifact_bytes_read=artifact_bytes_logically_consumed,
                training_metrics=last_training_metrics,
                held_out=_evaluate_held_out(backend, state.params, held_out_records),
                grad_metrics=grad_metrics,
                initial_params=initial_params,
                params=state.params,
                learning_rate=config.learning_rate,
            )
        )
        held_out_evaluation_seconds += time.perf_counter() - eval_started

    final_checkpoint = config.output_dir / "checkpoints" / "final"
    checkpoint_started = time.perf_counter()
    _save_measurement_checkpoint(
        config,
        state,
        student_config=student_config,
        path=final_checkpoint,
        artifact_lineage=artifact_lineage,
    )
    checkpoint_write_seconds += time.perf_counter() - checkpoint_started

    entries = detect_corridor_entries(
        trajectory,
        threshold=config.corridor_entry_threshold,
        stable_entry_evals=config.stable_entry_evals,
    )
    efficiency = _efficiency_metrics(trajectory, entries)
    resource = _resource_accounting(
        config,
        record_sizes=record_sizes,
        total_record_visits=total_record_visits,
        unique_records_consumed=len(visited_record_indices),
        tokens_consumed=tokens_consumed,
        logical_bytes=artifact_bytes_logically_consumed,
    )
    run_finished_at = _utc_now()
    total_wall_clock_seconds = time.perf_counter() - started_perf
    timing = {
        "startup_seconds": startup_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "compile_seconds": None,
        "compile_seconds_available": False,
        "training_seconds": training_seconds,
        "held_out_evaluation_seconds": held_out_evaluation_seconds,
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "total_wall_clock_seconds": total_wall_clock_seconds,
        "run_started_at": run_started_at,
        "training_started_at": training_started_at,
        "training_finished_at": training_finished_at,
        "run_finished_at": run_finished_at,
    }
    _validate_trajectory(trajectory)
    status = (
        "pass"
        if stop_reason in {"requested_steps_completed", "stable_corridor_entry"}
        else "fail"
    )
    best = min(
        trajectory,
        key=lambda point: (
            point["mean_distance_outside_corridor"],
            -point["inside_all_rate"],
            point["optimizer_step"],
        ),
    )
    report = {
        "phase": "P153",
        "status": status,
        "measurement_kind": "corridor_pass_trajectory",
        "training_cycle": "corridor_only",
        "exemplar_training_enabled": False,
        "held_out_evaluation_enabled": True,
        "requested_steps": config.steps,
        "completed_steps": completed_steps,
        "evaluation_interval_steps": evaluation_interval,
        "evaluation_steps": [point["optimizer_step"] for point in trajectory],
        "num_held_out_evaluations": len(trajectory),
        "corridor_entry_threshold": config.corridor_entry_threshold,
        "stable_entry_evals": config.stable_entry_evals,
        **entries,
        "entry_not_reached": entries["first_threshold_entry_step"] is None,
        "best_inside_all_rate": max(point["inside_all_rate"] for point in trajectory),
        "best_mean_distance_outside_corridor": best["mean_distance_outside_corridor"],
        "best_step": best["optimizer_step"],
        "initial_held_out_corridor_loss": trajectory[0]["held_out_corridor_loss"],
        "final_held_out_corridor_loss": trajectory[-1]["held_out_corridor_loss"],
        "initial_inside_all_rate": trajectory[0]["inside_all_rate"],
        "final_inside_all_rate": trajectory[-1]["inside_all_rate"],
        "initial_mean_distance_outside_corridor": trajectory[0][
            "mean_distance_outside_corridor"
        ],
        "final_mean_distance_outside_corridor": trajectory[-1][
            "mean_distance_outside_corridor"
        ],
        "stopping_policy": (
            "stable_entry" if config.stop_on_stable_entry else "fixed_step"
        ),
        "stop_reason": stop_reason,
        "stop_step": completed_steps,
        "stop_trigger_metric": "inside_all_rate",
        "efficiency": efficiency,
        "resource_accounting": resource,
        "wall_clock": timing,
        "corridor_aggressiveness": {
            "corridor_loss_weight": config.corridor_loss_weight,
            "penalty_kind": config.penalty_kind,
            "penalty_power": config.penalty_power,
            "per_stat_weights": {
                "entropy": config.entropy_weight,
                "top1_margin": config.top1_margin_weight,
                "top8_mass": config.top8_mass_weight,
                "top32_mass": config.top32_mass_weight,
                "tail_mass": config.tail_mass_weight,
            },
            "worst_stat_boost": config.worst_stat_boost,
            "adaptive_weighting_enabled": False,
            "distance_normalization": config.distance_normalization,
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "stability_abort_enabled": config.stability_abort_enabled,
        },
        "abort_guard_config": {
            "parameter_norm_limit": config.parameter_norm_limit,
            "gradient_norm_hard_limit": config.gradient_norm_hard_limit,
            "held_out_loss_abort_multiple": config.held_out_loss_abort_multiple,
        },
        "abort_triggered": stop_reason
        in {"stability_abort", "non_finite_loss", "non_finite_gradient"},
        "abort_reason": (
            stop_reason
            if stop_reason
            in {"stability_abort", "non_finite_loss", "non_finite_gradient"}
            else None
        ),
        "abort_step": (
            completed_steps
            if stop_reason
            in {"stability_abort", "non_finite_loss", "non_finite_gradient"}
            else None
        ),
        "lineage": lineage_receipt,
        "checkpoint": {
            "final_dir": str(final_checkpoint),
            "final_parameter_fingerprint": parameter_fingerprint(state.params),
            **hash_checkpoint_bundle(final_checkpoint),
        },
        "baseline_comparison": {
            "available": False,
            "reason": (
                "P153 does not synthesize a baseline trajectory from a final "
                "baseline checkpoint; matched baseline trajectory input was "
                "not supplied"
            ),
        },
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "scale_claim_made": False,
        "radlads_parity_claim_made": False,
    }
    paths = _write_outputs(
        config,
        report=report,
        trajectory=trajectory,
        efficiency=efficiency,
        resource=resource,
        entries=entries,
    )
    return CorridorMeasurementResult(
        status=status,
        output_dir=config.output_dir,
        report_path=paths["report"],
        summary_path=paths["summary"],
        trajectory_path=paths["trajectory"],
        checkpoint_dir=final_checkpoint,
        completed_steps=completed_steps,
        stable_entry_achieved=entries["stable_entry_achieved"],
    )


def _validate_config(config: CorridorMeasurementConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be > 0")
    if config.eval_every is not None and config.eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.optimizer not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer must be one of {'sgd', 'adam', 'adamw'}")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0 when set")
    if config.corridor_loss_weight <= 0:
        raise ValueError("corridor_loss_weight must be > 0")
    if config.penalty_kind != "powered_hinge":
        raise ValueError("penalty_kind must be 'powered_hinge'")
    if config.penalty_power < 1.0:
        raise ValueError("penalty_power must be >= 1")
    if any(
        weight < 0
        for weight in (
            config.entropy_weight,
            config.top1_margin_weight,
            config.top8_mass_weight,
            config.top32_mass_weight,
            config.tail_mass_weight,
        )
    ):
        raise ValueError("per-stat weights must be >= 0")
    if config.worst_stat_boost < 1.0:
        raise ValueError("worst_stat_boost must be >= 1")
    if config.distance_normalization not in {"none", "corridor_width"}:
        raise ValueError("distance_normalization must be 'none' or 'corridor_width'")
    if config.parameter_norm_limit <= 0:
        raise ValueError("parameter_norm_limit must be > 0")
    if config.gradient_norm_hard_limit <= 0:
        raise ValueError("gradient_norm_hard_limit must be > 0")
    if config.held_out_loss_abort_multiple <= 1.0:
        raise ValueError("held_out_loss_abort_multiple must be > 1")
    if not 0.0 <= config.corridor_entry_threshold <= 1.0:
        raise ValueError("corridor_entry_threshold must be within [0, 1]")
    if config.stable_entry_evals < 1:
        raise ValueError("stable_entry_evals must be >= 1")
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass overwrite=True"
        )


def _lineage_receipt(
    config: CorridorMeasurementConfig,
    *,
    train_provenance: dict[str, Any],
    held_out_provenance: dict[str, Any],
    artifact_lineage: dict[str, Any],
) -> dict[str, Any]:
    blockers = [
        *train_provenance["blockers"],
        *held_out_provenance["blockers"],
    ]
    train = train_provenance["provenance"]
    held_out = held_out_provenance["provenance"]
    id_overlap = set(train["ordered_example_ids"]) & set(
        held_out["ordered_example_ids"]
    )
    token_overlap = set(train["token_sequence_hashes"]) & set(
        held_out["token_sequence_hashes"]
    )
    if id_overlap:
        blockers.append("training and held-out example IDs overlap")
    if token_overlap:
        blockers.append("training and held-out tokenized inputs overlap")
    if (
        train["artifact_manifest_sha256"]
        != artifact_lineage["artifact_manifest_sha256"]
    ):
        blockers.append("training artifact lineage mismatch")
    if train["source_file_sha256"] != artifact_lineage["source_file_sha256"]:
        blockers.append("training source lineage mismatch")
    for field in ("capture_config_sha256", "teacher_identity_sha256"):
        if train[field] != held_out[field]:
            blockers.append(f"held-out {field} mismatch")
    publication_grade = bool(
        not blockers
        and train.get("publication_grade_lineage", False)
        and held_out.get("publication_grade_lineage", False)
    )
    initial = {
        "kind": "seed" if config.initial_checkpoint is None else "checkpoint",
        "seed": config.seed if config.initial_checkpoint is None else None,
        "checkpoint_dir": (
            None
            if config.initial_checkpoint is None
            else str(config.initial_checkpoint)
        ),
    }
    if config.initial_checkpoint is not None:
        initial.update(hash_checkpoint_bundle(config.initial_checkpoint))
        initial["parameter_fingerprint"] = parameter_fingerprint(
            load_checkpoint(config.initial_checkpoint).params
        )
        if config.p151_report is not None:
            p151 = read_json_object(config.p151_report)
            shared = p151.get("lineage", {}).get("shared_initialization", {})
            if initial["checkpoint_bundle_sha256"] != shared.get(
                "checkpoint_bundle_sha256"
            ):
                blockers.append("shared initialization checkpoint mismatch")
            if initial["parameter_fingerprint"] != shared.get("parameter_fingerprint"):
                blockers.append("shared initialization fingerprint mismatch")
    return {
        "phase": "P153",
        "status": "pass" if publication_grade else "fail",
        "publication_grade_lineage": publication_grade,
        "blockers": blockers,
        "training_artifact_valid": train_provenance["valid"],
        "held_out_artifact_valid": held_out_provenance["valid"],
        "id_overlap_count": len(id_overlap),
        "token_sequence_overlap_count": len(token_overlap),
        "source_join_kind": artifact_lineage["source_join_kind"],
        "source_join_complete": artifact_lineage["source_join_complete"],
        "artifact_manifest_sha256": artifact_lineage["artifact_manifest_sha256"],
        "initialization": initial,
    }


def _create_backend(
    config: CorridorMeasurementConfig, summary: Any
) -> tuple[Any, dict[str, Any]]:
    contract = VocabContract(
        tokenizer_id=summary.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=summary.tokenizer_name or None,
        vocab_size=summary.vocab_size,
        model_id=summary.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=contract,
        architecture_id=config.student_backend,
    )
    raw = getattr(getattr(backend, "student", None), "config", None)
    return backend, {
        "architecture_id": config.student_backend,
        "backend_name": type(backend).__name__,
        "architecture": type(raw).__name__,
        "vocab_size": int(getattr(raw, "vocab_size", summary.vocab_size)),
        "hidden_size": int(getattr(raw, "hidden_size", 0)),
        "num_layers": int(getattr(raw, "num_layers", 0)),
        "num_heads": _optional_int(getattr(raw, "num_heads", None)),
        "num_kv_heads": _optional_int(getattr(raw, "num_kv_heads", None)),
        "emit_logits": bool(getattr(raw, "emit_logits", True)),
        "tie_embeddings": bool(getattr(raw, "tie_embeddings", False)),
        "emit_mixer_outputs": bool(getattr(raw, "emit_mixer_outputs", False)),
    }


def _make_train_step(
    backend: Any,
    *,
    optimizer_config: OptimizerConfig,
    max_grad_norm: float | None,
    config: CorridorMeasurementConfig,
):
    def train_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any):
            output, _ = backend.forward_full(params, batch["input_ids"])
            stats = compute_fingerprint_distribution_stats_at_positions(
                backend.logits(output),
                batch["position"],
            )
            loss, inside_all_rate = _aggressive_corridor_loss(
                stats,
                _jax_fingerprint_batch(batch),
                config,
            )
            return loss, inside_all_rate

        (loss, inside_all_rate), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        clipped = clip_gradients_by_global_norm(
            grads,
            max_grad_norm=max_grad_norm,
        )
        update_config = OptimizerConfig(
            type=optimizer_config.type,
            learning_rate=state.learning_rate,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        params, optimizer_state, _ = optimizer_update(
            state.params,
            clipped.gradients,
            state.optimizer_state,
            update_config,
        )
        candidate_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(params))
        )
        update_is_safe = (
            jnp.isfinite(loss)
            & jnp.isfinite(clipped.global_norm)
            & (clipped.global_norm <= config.gradient_norm_hard_limit)
            & jnp.isfinite(candidate_norm)
            & (candidate_norm <= config.parameter_norm_limit)
        )
        params = jax.tree_util.tree_map(
            lambda candidate, previous: jnp.where(update_is_safe, candidate, previous),
            params,
            state.params,
        )
        optimizer_state = jax.tree_util.tree_map(
            lambda candidate, previous: jnp.where(update_is_safe, candidate, previous),
            optimizer_state,
            state.optimizer_state,
        )
        return (
            TrainState(
                params=params,
                step=state.step + 1,
                learning_rate=state.learning_rate,
                optimizer_state=optimizer_state,
            ),
            {
                "loss": loss,
                "inside_all_rate": inside_all_rate,
                "grad_global_norm": clipped.global_norm,
                "grad_clip_scale": clipped.clip_scale,
            },
        )

    return jax.jit(train_step)


def _evaluate_training_batch(
    backend: Any,
    params: Any,
    batch: FingerprintBatch,
    *,
    config: CorridorMeasurementConfig,
) -> dict[str, float]:
    output, _ = backend.forward_full(
        params,
        jnp.asarray(batch.input_ids, dtype=jnp.int32),
    )
    stats = compute_fingerprint_distribution_stats_at_positions(
        backend.logits(output),
        jnp.asarray(batch.position, dtype=jnp.int32),
    )
    loss, inside_all_rate = _aggressive_corridor_loss(
        stats,
        _jax_fingerprint_batch(_batch_to_jax(batch)),
        config,
    )
    return {
        "corridor_loss": float(loss),
        "inside_all_rate": float(inside_all_rate),
    }


def _aggressive_corridor_loss(
    stats: Any,
    batch: FingerprintBatch,
    config: CorridorMeasurementConfig,
) -> tuple[jax.Array, jax.Array]:
    weights = {
        "entropy": config.entropy_weight,
        "top1_margin": config.top1_margin_weight,
        "top8_mass": config.top8_mass_weight,
        "top32_mass": config.top32_mass_weight,
        "tail_mass": config.tail_mass_weight,
    }
    penalties = []
    inside = []
    for name in STAT_NAMES:
        values = jnp.asarray(getattr(stats, name))
        lower = jnp.asarray(getattr(batch, f"{name}_min"))
        upper = jnp.asarray(getattr(batch, f"{name}_max"))
        distance = jnp.maximum(lower - values, 0.0) + jnp.maximum(
            values - upper,
            0.0,
        )
        if config.distance_normalization == "corridor_width":
            distance = distance / jnp.maximum(upper - lower, 1e-8)
        penalties.append(jnp.power(distance, config.penalty_power) * weights[name])
        inside.append(distance == 0.0)
    stacked = jnp.stack(penalties, axis=-1)
    per_record = jnp.sum(stacked, axis=-1) + (config.worst_stat_boost - 1.0) * jnp.max(
        stacked, axis=-1
    )
    record_weights = jnp.asarray(batch.weight, dtype=jnp.float32)
    loss = config.corridor_loss_weight * (
        jnp.sum(per_record * record_weights)
        / jnp.maximum(jnp.sum(record_weights), 1e-8)
    )
    all_inside = jnp.all(jnp.stack(inside, axis=-1), axis=-1)
    return loss, jnp.mean(all_inside.astype(jnp.float32))


def _evaluate_held_out(
    backend: Any,
    params: Any,
    records: tuple[FingerprintTargetRecord, ...],
) -> dict[str, Any]:
    rows = []
    for record in records:
        output, _ = backend.forward_full(
            params,
            jnp.asarray([record.input_ids], dtype=jnp.int32),
        )
        stats = compute_fingerprint_distribution_stats_at_positions(
            backend.logits(output),
            jnp.asarray([record.position], dtype=jnp.int32),
        )
        values = {name: float(getattr(stats, name)[0]) for name in STAT_NAMES}
        bounds = _record_bounds(record)
        raw = {}
        normalized = {}
        for name in STAT_NAMES:
            raw[name], normalized[name] = corridor_distance(
                values[name],
                bounds[name][0],
                bounds[name][1],
            )
        rows.append(
            {
                "raw": raw,
                "normalized": normalized,
                "all_inside": all(value == 0.0 for value in raw.values()),
                "total_raw": sum(raw.values()),
                "total_normalized": sum(normalized.values()),
                "loss": sum(value * value for value in raw.values()),
            }
        )
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    distances = np.asarray([row["total_raw"] for row in rows], dtype=np.float64)
    normalized = np.asarray(
        [row["total_normalized"] for row in rows],
        dtype=np.float64,
    )
    result = {
        "corridor_loss": float(np.mean(losses)),
        "inside_all_rate": float(np.mean([row["all_inside"] for row in rows])),
        "mean_distance_outside_corridor": float(np.mean(distances)),
        "median_distance_outside_corridor": float(np.median(distances)),
        "max_distance_outside_corridor": float(np.max(distances)),
        "p90_distance_outside_corridor": float(np.quantile(distances, 0.90)),
        "p95_distance_outside_corridor": float(np.quantile(distances, 0.95)),
        "fraction_outside_corridor": float(np.mean(distances > 0.0)),
        "mean_normalized_distance_outside_corridor": float(np.mean(normalized)),
        "records_evaluated": len(rows),
    }
    for name in STAT_NAMES:
        values = np.asarray([row["raw"][name] for row in rows])
        normalized_values = np.asarray([row["normalized"][name] for row in rows])
        result[f"inside_{name}_rate"] = float(np.mean(values == 0.0))
        result[f"{name}_distance"] = float(np.mean(values))
        result[f"{name}_normalized_distance"] = float(np.mean(normalized_values))
    if not all(
        math.isfinite(float(value))
        for value in result.values()
        if isinstance(value, (int, float))
    ):
        raise ValueError("held-out evaluation produced non-finite metrics")
    return result


def _trajectory_point(
    *,
    optimizer_step: int,
    elapsed: float,
    records_consumed: int,
    tokens_consumed: int,
    artifact_bytes_read: int,
    training_metrics: dict[str, float],
    held_out: dict[str, Any],
    grad_metrics: dict[str, float] | None,
    initial_params: Any,
    params: Any,
    learning_rate: float,
) -> dict[str, Any]:
    point = {
        "optimizer_step": optimizer_step,
        "wall_clock_seconds": elapsed,
        "records_consumed": records_consumed,
        "tokens_consumed": tokens_consumed,
        "artifact_bytes_read": artifact_bytes_read,
        "training_corridor_loss": training_metrics["corridor_loss"],
        "held_out_corridor_loss": held_out["corridor_loss"],
        "grad_global_norm": (
            None
            if grad_metrics is None
            or not math.isfinite(grad_metrics["grad_global_norm"])
            else grad_metrics["grad_global_norm"]
        ),
        "grad_clip_scale": (
            None
            if grad_metrics is None
            or not math.isfinite(grad_metrics["grad_clip_scale"])
            else grad_metrics["grad_clip_scale"]
        ),
        "parameter_delta_from_initial": _tree_delta_norm(
            initial_params,
            params,
        ),
        "learning_rate": learning_rate,
        **{key: value for key, value in held_out.items() if key != "corridor_loss"},
    }
    return point


def _efficiency_metrics(
    trajectory: list[dict[str, Any]],
    entries: dict[str, Any],
) -> dict[str, Any]:
    first = trajectory[0]
    final = trajectory[-1]
    threshold = _point_for_step(
        trajectory,
        entries["first_threshold_entry_step"],
    )
    stable = _point_for_step(trajectory, entries["first_stable_entry_step"])
    steps = max(1, int(final["optimizer_step"]))
    seconds = max(1e-12, float(final["wall_clock_seconds"]))
    distance_reduction = (
        first["mean_distance_outside_corridor"]
        - final["mean_distance_outside_corridor"]
    )
    loss_reduction = first["held_out_corridor_loss"] - final["held_out_corridor_loss"]
    return {
        "steps_to_first_threshold_entry": entries["first_threshold_entry_step"],
        "steps_to_first_stable_entry": entries["first_stable_entry_step"],
        "seconds_to_first_threshold_entry": _value_or_none(
            threshold,
            "wall_clock_seconds",
        ),
        "seconds_to_first_stable_entry": _value_or_none(
            stable,
            "wall_clock_seconds",
        ),
        "records_to_first_threshold_entry": _value_or_none(
            threshold,
            "records_consumed",
        ),
        "records_to_first_stable_entry": _value_or_none(
            stable,
            "records_consumed",
        ),
        "tokens_to_first_threshold_entry": _value_or_none(
            threshold,
            "tokens_consumed",
        ),
        "tokens_to_first_stable_entry": _value_or_none(
            stable,
            "tokens_consumed",
        ),
        "artifact_bytes_to_first_threshold_entry": _value_or_none(
            threshold,
            "artifact_bytes_read",
        ),
        "artifact_bytes_to_first_stable_entry": _value_or_none(
            stable,
            "artifact_bytes_read",
        ),
        "held_out_loss_reduction_per_step": loss_reduction / steps,
        "held_out_loss_reduction_per_second": loss_reduction / seconds,
        "distance_reduction_per_step": distance_reduction / steps,
        "distance_reduction_per_1k_records": distance_reduction
        / max(final["records_consumed"] / 1000.0, 1e-12),
        "distance_reduction_per_megabyte": distance_reduction
        / max(final["artifact_bytes_read"] / 1_000_000.0, 1e-12),
        "inside_rate_gain_per_step": (
            final["inside_all_rate"] - first["inside_all_rate"]
        )
        / steps,
        "final_distance_to_corridor": final["mean_distance_outside_corridor"],
        "best_distance_to_corridor": min(
            point["mean_distance_outside_corridor"] for point in trajectory
        ),
        "best_inside_all_rate": max(point["inside_all_rate"] for point in trajectory),
    }


def _resource_accounting(
    config: CorridorMeasurementConfig,
    *,
    record_sizes: tuple[int, ...],
    total_record_visits: int,
    unique_records_consumed: int,
    tokens_consumed: int,
    logical_bytes: int,
) -> dict[str, Any]:
    manifest = config.fingerprint_artifact / "manifest.json"
    shard_paths = tuple((config.fingerprint_artifact / "targets").glob("*.jsonl"))
    return {
        "artifact_total_bytes_on_disk": _directory_size(config.fingerprint_artifact),
        "artifact_manifest_bytes": manifest.stat().st_size,
        "artifact_shard_bytes": sum(path.stat().st_size for path in shard_paths),
        "artifact_bytes_logically_consumed": logical_bytes,
        "artifact_records_available": len(record_sizes),
        "artifact_records_consumed": unique_records_consumed,
        "total_record_visits": total_record_visits,
        "artifact_reuse_count": max(0, total_record_visits - unique_records_consumed),
        "tokens_consumed": tokens_consumed,
    }


def _save_measurement_checkpoint(
    config: CorridorMeasurementConfig,
    state: TrainState,
    *,
    student_config: dict[str, Any],
    path: Path,
    artifact_lineage: dict[str, Any],
) -> None:
    save_checkpoint(
        path,
        state.params,
        student_architecture=config.student_backend,
        student_config=student_config,
        step=int(state.step),
        learning_rate=config.learning_rate,
        loss_config={"kind": "fingerprint_corridor", "cycle": 1},
        target_manifest={
            "artifact_dir": str(config.fingerprint_artifact),
            "artifact_manifest_sha256": artifact_lineage["artifact_manifest_sha256"],
            "selected_aggressiveness_profile": config.selected_aggressiveness_profile,
            "selected_profile_config_sha256": config.selected_profile_config_sha256,
        },
        optimizer_config={
            "type": config.optimizer,
            "learning_rate": config.learning_rate,
        },
        optimizer_state=state.optimizer_state,
        gradients={"max_grad_norm": config.max_grad_norm},
        notes=["P153 corridor-only measurement checkpoint"],
        overwrite=config.overwrite,
    )


def _write_outputs(
    config: CorridorMeasurementConfig,
    *,
    report: dict[str, Any],
    trajectory: list[dict[str, Any]],
    efficiency: dict[str, Any],
    resource: dict[str, Any],
    entries: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "report": config.output_dir / "corridor_measurement_report.json",
        "summary": config.output_dir / "corridor_measurement_summary.md",
        "trajectory": config.output_dir / "corridor_trajectory.jsonl",
    }
    write_json(paths["report"], report)
    write_json(
        config.output_dir / "corridor_efficiency_metrics.json",
        efficiency,
    )
    write_json(
        config.output_dir / "corridor_entry_receipt.json",
        {
            **entries,
            "corridor_entry_threshold": config.corridor_entry_threshold,
            "stable_entry_evals": config.stable_entry_evals,
            "stopping_policy": report["stopping_policy"],
            "stop_reason": report["stop_reason"],
            "stop_step": report["stop_step"],
        },
    )
    write_json(config.output_dir / "resource_accounting.json", resource)
    paths["trajectory"].write_text(
        "".join(json.dumps(point, sort_keys=True) + "\n" for point in trajectory),
        encoding="utf-8",
    )
    paths["summary"].write_text(_render_summary(report), encoding="utf-8")
    return paths


def _render_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# P153 Corridor-Pass Measurement",
            "",
            f"- Status: {report['status']}",
            f"- Completed steps: {report['completed_steps']}",
            f"- Best inside-all rate: {report['best_inside_all_rate']}",
            f"- Best mean distance: {report['best_mean_distance_outside_corridor']}",
            f"- First threshold entry: {report['first_threshold_entry_step']}",
            f"- First stable entry: {report['first_stable_entry_step']}",
            f"- Stop reason: {report['stop_reason']}",
            "- General quality claim: false",
            "",
        )
    )


def _batch_to_jax(batch: FingerprintBatch) -> dict[str, jax.Array]:
    return {
        "input_ids": jnp.asarray(batch.input_ids, dtype=jnp.int32),
        "position": jnp.asarray(batch.position, dtype=jnp.int32),
        "mode_id": jnp.asarray(batch.mode_id, dtype=jnp.int32),
        "entropy_min": jnp.asarray(batch.entropy_min, dtype=jnp.float32),
        "entropy_max": jnp.asarray(batch.entropy_max, dtype=jnp.float32),
        "top1_margin_min": jnp.asarray(batch.top1_margin_min, dtype=jnp.float32),
        "top1_margin_max": jnp.asarray(batch.top1_margin_max, dtype=jnp.float32),
        "top8_mass_min": jnp.asarray(batch.top8_mass_min, dtype=jnp.float32),
        "top8_mass_max": jnp.asarray(batch.top8_mass_max, dtype=jnp.float32),
        "top32_mass_min": jnp.asarray(batch.top32_mass_min, dtype=jnp.float32),
        "top32_mass_max": jnp.asarray(batch.top32_mass_max, dtype=jnp.float32),
        "tail_mass_min": jnp.asarray(batch.tail_mass_min, dtype=jnp.float32),
        "tail_mass_max": jnp.asarray(batch.tail_mass_max, dtype=jnp.float32),
        "weight": jnp.asarray(batch.weight, dtype=jnp.float32),
    }


def _jax_fingerprint_batch(batch: dict[str, jax.Array]) -> FingerprintBatch:
    return FingerprintBatch(**batch)


def _record_bounds(record: FingerprintTargetRecord) -> dict[str, tuple[float, float]]:
    return {
        "entropy": (record.entropy_min, record.entropy_max),
        "top1_margin": (record.top1_margin_min, record.top1_margin_max),
        "top8_mass": (record.top8_mass_min, record.top8_mass_max),
        "top32_mass": (record.top32_mass_min, record.top32_mass_max),
        "tail_mass": (record.tail_mass_min, record.tail_mass_max),
    }


def _target_record_sizes(artifact: Path) -> tuple[int, ...]:
    manifest = read_json_object(artifact / "manifest.json")
    sizes = []
    for shard in manifest["target_shards"]:
        with (artifact / shard["path"]).open("rb") as handle:
            sizes.extend(len(line) for line in handle if line.strip())
    return tuple(sizes)


def _batch_indices(num_records: int, batch_size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, min(start + batch_size, num_records)))
        for start in range(0, num_records, batch_size)
    )


def _tree_delta_norm(before: Any, after: Any) -> float:
    left = jax.tree_util.tree_leaves(before)
    right = jax.tree_util.tree_leaves(after)
    return float(
        np.sqrt(
            sum(
                float(np.sum(np.square(np.asarray(b) - np.asarray(a))))
                for a, b in zip(left, right, strict=True)
            )
        )
    )


def _tree_norm(tree: Any) -> float:
    return float(
        np.sqrt(
            sum(
                float(np.sum(np.square(np.asarray(leaf))))
                for leaf in jax.tree_util.tree_leaves(tree)
            )
        )
    )


def _validate_trajectory(trajectory: list[dict[str, Any]]) -> None:
    if not trajectory or trajectory[0]["optimizer_step"] != 0:
        raise ValueError("corridor trajectory must include step 0")
    steps = [int(point["optimizer_step"]) for point in trajectory]
    if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("corridor trajectory steps must be strictly increasing")
    for point in trajectory:
        for key, value in point.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite trajectory metric: {key}")


def _point_for_step(
    trajectory: list[dict[str, Any]],
    step: int | None,
) -> dict[str, Any] | None:
    if step is None:
        return None
    return next(point for point in trajectory if point["optimizer_step"] == step)


def _value_or_none(point: dict[str, Any] | None, key: str) -> Any:
    return None if point is None else point[key]


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
