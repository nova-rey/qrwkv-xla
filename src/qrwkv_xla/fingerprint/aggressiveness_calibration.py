from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.fingerprint.aggressiveness_profiles import (
    PROFILE_NAMES,
    CorridorAggressivenessProfile,
    resolve_aggressiveness_profile,
)
from qrwkv_xla.fingerprint.corridor_measurement import (
    CorridorMeasurementConfig,
    run_corridor_measurement,
)
from qrwkv_xla.fingerprint.provenance import (
    file_sha256,
    hash_checkpoint_bundle,
    parameter_fingerprint,
    stable_hash,
)

ALLOWED_PROFILE_DIFFERENCE_FIELDS = frozenset(
    {
        "corridor_loss_weight",
        "learning_rate",
        "max_grad_norm",
        "penalty_kind",
        "penalty_power",
        "per_stat_weights",
        "worst_stat_boost",
        "distance_normalization",
        "stability_abort_enabled",
        "parameter_norm_limit",
        "gradient_norm_hard_limit",
        "held_out_loss_abort_multiple",
    }
)


@dataclass(frozen=True)
class AggressivenessCalibrationConfig:
    fingerprint_artifact: Path
    held_out_fingerprint_artifact: Path
    source_texts: Path
    output_dir: Path
    profiles: tuple[str, ...] = PROFILE_NAMES
    seeds: tuple[int, ...] = (0, 1, 2)
    steps: int = 100
    eval_every: int = 5
    checkpoint_every: int = 25
    batch_size: int = 1
    student_backend: str = "current_qrwkv"
    optimizer: str = "adamw"
    corridor_entry_threshold: float = 0.95
    stable_entry_evals: int = 3
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 1531
    required_completed_run_rate: float = 1.0
    required_stable_entry_success_rate: float = 1.0
    maximum_abort_rate: float = 0.0
    maximum_non_finite_run_rate: float = 0.0
    maximum_entry_then_exit_rate: float = 1.0
    maximum_severe_rebound_rate: float = 1.0
    maximum_trajectory_variance: float = float("inf")
    minimum_seed_count: int = 3
    held_out_artifact_role: str = "held_out_evaluation"
    overrides: dict[str, float | str | bool | None] | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class AggressivenessCalibrationResult:
    status: str
    selected_profile: str | None
    report_path: Path
    output_dir: Path


def entry_exit_metrics(
    trajectory: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    inside = [float(row["inside_all_rate"]) >= threshold for row in trajectory]
    first_entry_index = next((i for i, value in enumerate(inside) if value), None)
    exits = [i for i in range(1, len(inside)) if inside[i - 1] and not inside[i]]
    longest = current = 0
    for value in inside:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "first_entry_step": (
            None
            if first_entry_index is None
            else trajectory[first_entry_index]["optimizer_step"]
        ),
        "first_exit_after_entry_step": (
            None if not exits else trajectory[exits[0]]["optimizer_step"]
        ),
        "entry_then_exit_count": len(exits),
        "max_consecutive_inside_evals": longest,
        "stable_after_first_entry": bool(
            first_entry_index is not None and all(inside[first_entry_index:])
        ),
    }


def bootstrap_ci95(
    values: list[float], *, samples: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def select_profile(
    summaries: list[dict[str, Any]],
    *,
    required_completed_run_rate: float = 1.0,
    required_stable_entry_success_rate: float = 1.0,
    maximum_abort_rate: float = 0.0,
    maximum_non_finite_run_rate: float = 0.0,
    maximum_entry_then_exit_rate: float = 1.0,
    maximum_severe_rebound_rate: float = 1.0,
    maximum_trajectory_variance: float = float("inf"),
    comparison_valid: bool = True,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 1531,
) -> tuple[str | None, list[dict[str, Any]]]:
    policy = {
        "required_completed_run_rate": required_completed_run_rate,
        "required_stable_entry_success_rate": required_stable_entry_success_rate,
        "maximum_abort_rate": maximum_abort_rate,
        "maximum_non_finite_run_rate": maximum_non_finite_run_rate,
        "maximum_entry_then_exit_rate": maximum_entry_then_exit_rate,
        "maximum_severe_rebound_rate": maximum_severe_rebound_rate,
        "maximum_trajectory_variance": maximum_trajectory_variance,
        "comparison_valid": comparison_valid,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    selected, ranked, _, _ = _select_profile_with_receipt(summaries, policy)
    return selected, ranked


def extract_exact_stable_entry_costs(
    report: dict[str, Any], trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    stable_step = report["first_stable_entry_step"]
    empty = {
        "steps_to_stable_entry": None,
        "seconds_to_stable_entry": None,
        "records_to_stable_entry": None,
        "tokens_to_stable_entry": None,
        "bytes_to_stable_entry": None,
        "entry_cost_source": None,
        "entry_cost_trajectory_step": None,
        "entry_cost_lookup_valid": True,
        "entry_cost_lookup_error": None,
    }
    if stable_step is None:
        return empty
    matches = [row for row in trajectory if row.get("optimizer_step") == stable_step]
    if len(matches) != 1:
        return {
            **empty,
            "steps_to_stable_entry": stable_step,
            "entry_cost_lookup_valid": False,
            "entry_cost_lookup_error": "exact_entry_row_count_not_one",
        }
    row = matches[0]
    canonical_fields = {
        "seconds_to_stable_entry": "wall_clock_seconds",
        "records_to_stable_entry": "records_consumed",
        "tokens_to_stable_entry": "tokens_consumed",
        "bytes_to_stable_entry": "artifact_bytes_read",
    }
    values: dict[str, float | int] = {}
    for output_name, trajectory_name in canonical_fields.items():
        value = row.get(trajectory_name)
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            return {
                **empty,
                "steps_to_stable_entry": stable_step,
                "entry_cost_trajectory_step": stable_step,
                "entry_cost_lookup_valid": False,
                "entry_cost_lookup_error": f"invalid_{trajectory_name}",
            }
        values[output_name] = value
    final_values = {
        "seconds_to_stable_entry": report["wall_clock"]["total_wall_clock_seconds"],
        "records_to_stable_entry": report["resource_accounting"]["total_record_visits"],
        "tokens_to_stable_entry": report["resource_accounting"]["tokens_consumed"],
        "bytes_to_stable_entry": report["resource_accounting"][
            "artifact_bytes_logically_consumed"
        ],
    }
    if any(float(values[key]) > float(final_values[key]) for key in values):
        return {
            **empty,
            "steps_to_stable_entry": stable_step,
            "entry_cost_trajectory_step": stable_step,
            "entry_cost_lookup_valid": False,
            "entry_cost_lookup_error": "entry_cost_exceeds_final_total",
        }
    return {
        "steps_to_stable_entry": stable_step,
        **values,
        "entry_cost_source": "trajectory_exact_step",
        "entry_cost_trajectory_step": stable_step,
        "entry_cost_lookup_valid": True,
        "entry_cost_lookup_error": None,
    }


def run_aggressiveness_calibration(
    config: AggressivenessCalibrationConfig,
) -> AggressivenessCalibrationResult:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = [
        resolve_aggressiveness_profile(name, config.overrides)
        for name in config.profiles
    ]
    write_json(
        config.output_dir / "resolved_profile_configs.json",
        [p.to_dict() for p in resolved],
    )
    rows: list[dict[str, Any]] = []
    aggregate_checks: list[dict[str, Any]] = []
    for profile in resolved:
        for seed in config.seeds:
            run_dir = (
                config.output_dir / "profiles" / profile.profile_name / f"seed_{seed}"
            )
            run_corridor_measurement(
                CorridorMeasurementConfig(
                    fingerprint_artifact=config.fingerprint_artifact,
                    held_out_fingerprint_artifact=config.held_out_fingerprint_artifact,
                    source_texts=config.source_texts,
                    output_dir=run_dir,
                    steps=config.steps,
                    eval_every=config.eval_every,
                    checkpoint_every=config.checkpoint_every,
                    batch_size=config.batch_size,
                    optimizer=config.optimizer,
                    learning_rate=profile.learning_rate,
                    max_grad_norm=profile.max_grad_norm,
                    seed=seed,
                    student_backend=config.student_backend,
                    corridor_entry_threshold=config.corridor_entry_threshold,
                    stable_entry_evals=config.stable_entry_evals,
                    corridor_loss_weight=profile.corridor_loss_weight,
                    penalty_kind=profile.penalty_kind,
                    penalty_power=profile.penalty_power,
                    entropy_weight=profile.per_stat_weights["entropy"],
                    top1_margin_weight=profile.per_stat_weights["top1_margin"],
                    top8_mass_weight=profile.per_stat_weights["top8_mass"],
                    top32_mass_weight=profile.per_stat_weights["top32_mass"],
                    tail_mass_weight=profile.per_stat_weights["tail_mass"],
                    worst_stat_boost=profile.worst_stat_boost,
                    distance_normalization=profile.distance_normalization,
                    stability_abort_enabled=profile.stability_abort_enabled,
                    parameter_norm_limit=profile.parameter_norm_limit,
                    gradient_norm_hard_limit=profile.gradient_norm_hard_limit,
                    held_out_loss_abort_multiple=profile.held_out_loss_abort_multiple,
                    stop_on_stable_entry=False,
                    selected_aggressiveness_profile=profile.profile_name,
                    selected_profile_config_sha256=stable_hash(profile.to_dict()),
                    held_out_artifact_role=config.held_out_artifact_role,
                    overwrite=config.overwrite,
                )
            )
            report = read_json_object(run_dir / "corridor_measurement_report.json")
            trajectory = _read_jsonl(run_dir / "corridor_trajectory.jsonl")
            seed_metrics = _seed_metrics(
                profile.to_dict(),
                seed,
                report,
                trajectory,
                config,
                run_dir=run_dir,
            )
            rows.append(seed_metrics)
            aggregate_checks.append(
                _validate_seed_aggregate(
                    profile_name=profile.profile_name,
                    seed=seed,
                    trajectory=trajectory,
                    report=report,
                    seed_metrics=seed_metrics,
                )
            )
    summaries = [
        _summarize_profile(name, rows, config, len(config.seeds))
        for name in config.profiles
    ]
    fairness = _fairness(config, resolved)
    aggregate_validation = _aggregate_validation(
        config=config,
        rows=rows,
        summaries=summaries,
        fairness=fairness,
        seed_checks=aggregate_checks,
    )
    selection_allowed, selection_block_reason = derive_selection_allowed(
        fairness=fairness,
        aggregate_validation=aggregate_validation,
        classification_complete=True,
        candidate_gating_complete=True,
    )
    selected, ranking, selection_receipt, pairwise_comparisons = (
        _select_profile_with_receipt(
            summaries,
            _selection_policy(config, fairness["comparison_valid"]),
            selection_allowed=selection_allowed,
            selection_block_reason=selection_block_reason,
        )
    )
    publication_receipt = _publication_grade_receipt(
        config=config,
        rows=rows,
        fairness=fairness,
        summaries=summaries,
        aggregate_validation=aggregate_validation,
        selection_receipt=selection_receipt,
    )
    status = derive_top_level_status(
        fairness_valid=fairness["comparison_valid"],
        aggregate_validation_status=aggregate_validation["status"],
        selection_logic_completed=selection_receipt["selection_status"]
        in {"selected", "no_eligible_profiles"},
    )
    report = {
        "phase": "P153.1.2",
        "status": status,
        "cleanup_kind": "aggressiveness_selection_safety",
        "profiles": list(config.profiles),
        "seed_count": len(config.seeds),
        "comparison_valid": fairness["comparison_valid"],
        "publication_grade": publication_receipt["publication_grade"],
        "candidate_count": len(selection_receipt["candidate_profiles"]),
        "selected_profile": selected,
        "selection_status": selection_receipt["selection_status"],
        "selection_allowed": selection_receipt["selection_allowed"],
        "winner_declared": selection_receipt["winner_declared"],
        "selection_rule": selection_receipt["selection_rule"],
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "scale_claim_made": False,
        "radlads_parity_claim_made": False,
        "primary_selection_metric": selection_receipt["primary_selection_metric"],
        "primary_selection_metric_direction": selection_receipt[
            "primary_selection_metric_direction"
        ],
        "paired_comparison_used": selection_receipt["paired_comparison_used"],
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
        "tie_break_rule": selection_receipt["tie_break_rule"],
    }
    write_json(config.output_dir / "profile_fairness_contract.json", fairness)
    write_json(config.output_dir / "profile_summary_metrics.json", summaries)
    write_json(config.output_dir / "profile_ranking.json", ranking)
    write_json(config.output_dir / "aggregate_validation.json", aggregate_validation)
    _write_jsonl(
        config.output_dir / "pairwise_selection_comparisons.jsonl",
        pairwise_comparisons,
    )
    write_json(
        config.output_dir / "publication_grade_receipt.json",
        publication_receipt,
    )
    _write_jsonl(config.output_dir / "profile_seed_metrics.jsonl", rows)
    _write_jsonl(
        config.output_dir / "paired_profile_deltas.jsonl",
        _paired(rows, config.profiles),
    )
    report_path = config.output_dir / "aggressiveness_calibration_report.json"
    write_json(report_path, report)
    if selected:
        selected_config = next(
            p.to_dict() for p in resolved if p.profile_name == selected
        )
        selected_seed = min(config.seeds)
        selected_checkpoint = (
            config.output_dir
            / "profiles"
            / selected
            / f"seed_{selected_seed}"
            / "checkpoints"
            / "final"
        )
        selection_receipt.update(
            {
                "selected_profile_config_sha256": stable_hash(selected_config),
                "selected_corridor_seed": selected_seed,
                "selected_corridor_checkpoint_bundle_sha256": hash_checkpoint_bundle(
                    selected_checkpoint
                )["checkpoint_bundle_sha256"],
                "selected_corridor_parameter_fingerprint": parameter_fingerprint(
                    load_checkpoint(selected_checkpoint).params
                ),
                "calibration_training_artifact_sha256": file_sha256(
                    config.fingerprint_artifact / "manifest.json"
                ),
                "calibration_validation_artifact_sha256": file_sha256(
                    config.held_out_fingerprint_artifact / "manifest.json"
                ),
                "calibration_student_config_sha256": stable_hash(
                    load_checkpoint(selected_checkpoint).manifest.student_config
                ),
                "calibration_report_sha256": file_sha256(report_path),
                "publication_grade_receipt_sha256": file_sha256(
                    config.output_dir / "publication_grade_receipt.json"
                ),
            }
        )
        write_json(config.output_dir / "selected_profile_config.json", selected_config)
    write_json(config.output_dir / "profile_selection_receipt.json", selection_receipt)
    (config.output_dir / "aggressiveness_calibration_summary.md").write_text(
        "# P153.1.2 Selection Integrity\n\n"
        f"- Status: {status}\n"
        f"- Selected profile: {selected or 'none'}\n"
        f"- Selection status: {selection_receipt['selection_status']}\n"
        f"- Selection reason: {selection_receipt.get('selection_reason', 'none')}\n"
        f"- Primary comparison statistically distinguishable: "
        f"{str(selection_receipt['primary_comparison_statistically_distinguishable']).lower()}\n"
        f"- Minimum-force tie-break used: "
        f"{str(selection_receipt['minimum_force_tie_break_used']).lower()}\n"
        f"- Aggregate validation passed: "
        f"{str(aggregate_validation['status'] == 'pass').lower()}\n"
        f"- Candidate count: {len(selection_receipt['candidate_profiles'])}\n"
        f"- Publication grade: "
        f"{str(publication_receipt['publication_grade']).lower()}\n"
        "- General quality claim: false\n",
        encoding="utf-8",
    )
    return AggressivenessCalibrationResult(
        status=status,
        selected_profile=selected,
        report_path=report_path,
        output_dir=config.output_dir,
    )


def _seed_metrics(
    profile: dict[str, Any],
    seed: int,
    report: dict[str, Any],
    trajectory: list[dict[str, Any]],
    config: AggressivenessCalibrationConfig,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    exits = entry_exit_metrics(trajectory, config.corridor_entry_threshold)
    losses = [float(x["held_out_corridor_loss"]) for x in trajectory]
    distances = [float(x["mean_distance_outside_corridor"]) for x in trajectory]
    grads = [
        float(x["grad_global_norm"])
        for x in trajectory
        if x["grad_global_norm"] is not None
    ]
    clips = [
        float(x["grad_clip_scale"]) < 1.0
        for x in trajectory
        if x["grad_clip_scale"] is not None
    ]
    resource, timing = report["resource_accounting"], report["wall_clock"]
    entry_costs = extract_exact_stable_entry_costs(report, trajectory)
    non_finite_run = bool(
        report["stop_reason"] in {"non_finite_loss", "non_finite_gradient"}
    )
    severe_rebound = bool(
        exits["entry_then_exit_count"] > 0
        or (losses[-1] - min(losses)) > 0.0
        or (distances[-1] - min(distances)) > 0.0
    )
    return {
        "profile_name": profile["profile_name"],
        "aggressiveness_rank": profile["aggressiveness_rank"],
        "seed": seed,
        "run_status": report["status"],
        "stable_entry_achieved": report["stable_entry_achieved"],
        "first_threshold_entry_step": report["first_threshold_entry_step"],
        "first_stable_entry_step": report["first_stable_entry_step"],
        **entry_costs,
        "initial_held_out_corridor_loss": losses[0],
        "final_held_out_corridor_loss": losses[-1],
        "best_held_out_corridor_loss": min(losses),
        "initial_mean_distance": distances[0],
        "final_mean_distance": distances[-1],
        "best_mean_distance": min(distances),
        "best_inside_all_rate": report["best_inside_all_rate"],
        "final_inside_all_rate": report["final_inside_all_rate"],
        "steps_completed": report["completed_steps"],
        "records_consumed": resource["total_record_visits"],
        "tokens_consumed": resource["tokens_consumed"],
        "artifact_bytes_logically_consumed": resource[
            "artifact_bytes_logically_consumed"
        ],
        "training_seconds": timing["training_seconds"],
        "evaluation_seconds": timing["held_out_evaluation_seconds"],
        "total_wall_clock_seconds": timing["total_wall_clock_seconds"],
        "final_steps_completed": report["completed_steps"],
        "final_total_seconds": timing["total_wall_clock_seconds"],
        "final_records_consumed": resource["total_record_visits"],
        "final_tokens_consumed": resource["tokens_consumed"],
        "final_bytes_consumed": resource["artifact_bytes_logically_consumed"],
        "parameter_delta_norm": trajectory[-1]["parameter_delta_from_initial"],
        "gradient_norms": grads,
        "gradient_spike_count": sum(
            g > profile["gradient_norm_hard_limit"] * 0.5 for g in grads
        ),
        "clip_event_count": sum(clips),
        "clip_event_rate": sum(clips) / max(len(clips), 1),
        "non_finite_loss": report["stop_reason"] == "non_finite_loss",
        "non_finite_gradient": report["stop_reason"] == "non_finite_gradient",
        "non_finite_run": non_finite_run,
        "abort_triggered": report["abort_triggered"],
        "abort_reason": report["abort_reason"],
        "stop_reason": report["stop_reason"],
        "held_out_loss_rebound": losses[-1] - min(losses),
        "distance_rebound": distances[-1] - min(distances),
        "trajectory_variance": float(np.var(distances)),
        "severe_rebound": severe_rebound,
        "lineage_valid": bool(
            read_json_object(run_dir / "checkpoint_lineage_validation.json")[
                "publication_grade_lineage"
            ]
        ),
        **exits,
    }


def _summarize_profile(
    name: str,
    rows: list[dict[str, Any]],
    config: AggressivenessCalibrationConfig,
    expected_seed_count: int,
) -> dict[str, Any]:
    selected = [r for r in rows if r["profile_name"] == name]
    stable_steps = [
        float(r["steps_to_stable_entry"])
        for r in selected
        if r["first_stable_entry_step"] is not None
    ]
    stable_seconds = [
        float(r["seconds_to_stable_entry"])
        for r in selected
        if r["first_stable_entry_step"] is not None
    ]
    stable_records = [
        float(r["records_to_stable_entry"])
        for r in selected
        if r["first_stable_entry_step"] is not None
    ]
    stable_bytes = [
        float(r["bytes_to_stable_entry"])
        for r in selected
        if r["first_stable_entry_step"] is not None
    ]
    rank = selected[0]["aggressiveness_rank"]

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in selected]))

    completed = [r["run_status"] == "pass" for r in selected]
    stable = [bool(r["stable_entry_achieved"]) for r in selected]
    aborted = [bool(r["abort_triggered"]) for r in selected]
    non_finite = [bool(r["non_finite_run"]) for r in selected]
    failed = [r["run_status"] != "pass" for r in selected]
    missing_run_count = max(0, expected_seed_count - len(selected))
    summary = {
        "profile_name": name,
        "aggressiveness_rank": rank,
        "expected_run_count": expected_seed_count,
        "observed_run_count": len(selected),
        "completed_run_count": int(sum(completed)),
        "failed_run_count": int(sum(failed)),
        "aborted_run_count": int(sum(aborted)),
        "missing_run_count": missing_run_count,
        "run_count": len(selected),
        "completed_run_rate": (float(sum(completed)) / expected_seed_count),
        "stable_entry_success_rate": (float(sum(stable)) / expected_seed_count),
        "median_steps_to_stable_entry": float(np.median(stable_steps))
        if stable_steps
        else None,
        "median_seconds_to_stable_entry": float(np.median(stable_seconds))
        if stable_seconds
        else None,
        "median_records_to_stable_entry": float(np.median(stable_records))
        if stable_records
        else None,
        "median_bytes_to_stable_entry": float(np.median(stable_bytes))
        if stable_bytes
        else None,
        "mean_steps_to_stable_entry": float(np.mean(stable_steps))
        if stable_steps
        else None,
        "steps_to_stable_entry_ci95": bootstrap_ci95(
            stable_steps,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + rank,
        ),
        "mean_best_distance": mean("best_mean_distance"),
        "mean_final_distance": mean("final_mean_distance"),
        "mean_trajectory_variance": mean("trajectory_variance"),
        "mean_entry_then_exit_count": mean("entry_then_exit_count"),
        "entry_then_exit_rate": float(
            np.mean([r["entry_then_exit_count"] > 0 for r in selected])
        ),
        "abort_rate": float(np.mean(aborted)),
        "non_finite_run_rate": float(np.mean(non_finite)),
        "severe_rebound_rate": float(
            np.mean([bool(r["severe_rebound"]) for r in selected])
        ),
        "median_parameter_delta": float(
            np.median([r["parameter_delta_norm"] for r in selected])
        ),
        "mean_distance_rebound": mean("distance_rebound"),
        "mean_held_out_loss_rebound": mean("held_out_loss_rebound"),
        "lineage_valid_for_all_runs": all(bool(r["lineage_valid"]) for r in selected),
        "seed_metrics": [
            {
                "seed": int(r["seed"]),
                "steps_to_stable_entry": r["steps_to_stable_entry"],
                "seconds_to_stable_entry": r["seconds_to_stable_entry"],
                "records_to_stable_entry": r["records_to_stable_entry"],
                "tokens_to_stable_entry": r["tokens_to_stable_entry"],
                "bytes_to_stable_entry": r["bytes_to_stable_entry"],
            }
            for r in sorted(selected, key=lambda row: row["seed"])
        ],
        "selection_eligible": False,
        "profile_status": "unclassified",
        "selection_exclusion_reasons": [],
    }
    summary["failed_run_rate"] = summary["failed_run_count"] / expected_seed_count
    return summary


def _selection_policy(
    config: AggressivenessCalibrationConfig, comparison_valid: bool
) -> dict[str, Any]:
    return {
        "required_completed_run_rate": config.required_completed_run_rate,
        "required_stable_entry_success_rate": config.required_stable_entry_success_rate,
        "maximum_abort_rate": config.maximum_abort_rate,
        "maximum_non_finite_run_rate": config.maximum_non_finite_run_rate,
        "maximum_entry_then_exit_rate": config.maximum_entry_then_exit_rate,
        "maximum_severe_rebound_rate": config.maximum_severe_rebound_rate,
        "maximum_trajectory_variance": config.maximum_trajectory_variance,
        "comparison_valid": comparison_valid,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
    }


def _select_profile_with_receipt(
    summaries: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    selection_allowed: bool = True,
    selection_block_reason: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    ranked = [_classify_profile(summary, policy) for summary in summaries]
    candidates = [row for row in ranked if row["selection_eligible"]]
    selected, comparisons = compare_eligible_profiles(candidates, policy)
    if not selection_allowed:
        selected = None
    candidate_names = [row["profile_name"] for row in candidates]
    excluded = {
        row["profile_name"]: row["selection_exclusion_reasons"]
        for row in ranked
        if not row["selection_eligible"]
    }
    ranking = sorted(
        ranked,
        key=lambda row: (
            not row["selection_eligible"],
            row["profile_name"] != selected,
            row["aggressiveness_rank"],
            row["profile_name"],
        ),
    )
    for index, row in enumerate(ranking, 1):
        row["profile_rank"] = index
        if row["profile_name"] == selected:
            row["selection_status"] = "selected"
            row["selection_reason"] = "minimum_sufficient_force"
        else:
            row["selection_status"] = row["profile_status"]
            row["selection_reason"] = (
                "candidate_gating"
                if not row["selection_eligible"]
                else "deterministic_efficiency_order"
            )
    selection_status = (
        "selection_blocked"
        if not selection_allowed
        else "selected"
        if selected
        else "no_eligible_profiles"
    )
    tie_used = any(
        row["comparison_result"] == "tie" and row["preferred_profile"] == selected
        for row in comparisons
    )
    receipt = {
        "status": "pass" if selection_allowed else "fail",
        "selection_allowed": selection_allowed,
        "selection_rule": "minimum_sufficient_force",
        "candidate_profiles": candidate_names,
        "excluded_profiles": excluded,
        "selected_profile": selected,
        "selection_status": selection_status,
        "selection_block_reason": selection_block_reason,
        "winner_declared": selected is not None,
        "selection_reason": "minimum_sufficient_force" if selected else None,
        "primary_metric": "steps_to_stable_entry",
        "primary_metric_direction": "lower_is_better",
        "primary_selection_metric": "steps_to_stable_entry",
        "primary_selection_metric_direction": "lower_is_better",
        "statistical_method": "paired_bootstrap_ci",
        "paired_comparison_used": True,
        "primary_comparison_statistically_distinguishable": bool(comparisons)
        and all(row["statistically_distinguishable"] for row in comparisons),
        "minimum_force_tie_break_used": tie_used,
        "bootstrap_samples": policy["bootstrap_samples"],
        "bootstrap_seed": policy["bootstrap_seed"],
        "tie_break_rule": (
            "if paired bootstrap interval includes zero, choose lower "
            "aggressiveness rank"
        ),
        "selection_order": ["paired_steps_to_stable_entry", "aggressiveness_rank"],
        "thresholds": {
            key: value
            for key, value in policy.items()
            if key.startswith("required_") or key.startswith("maximum_")
        },
    }
    return selected, ranking, receipt, comparisons


def _classify_profile(
    summary: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    row = dict(summary)
    reasons: list[str] = []
    status = "acceptable"
    if not policy["comparison_valid"]:
        status = "invalid"
        reasons.append("comparison_invalid")
    elif row["completed_run_rate"] < policy["required_completed_run_rate"]:
        status = "inconclusive"
        if row["missing_run_count"] > 0:
            reasons.append("missing_runs_present")
        if row["failed_run_count"] > 0:
            reasons.append("failed_runs_present")
        reasons.append("completed_run_rate_below_required")
    elif row["abort_rate"] > policy["maximum_abort_rate"]:
        status = "destructive"
        reasons.append("abort_rate_exceeded")
    elif row["non_finite_run_rate"] > policy["maximum_non_finite_run_rate"]:
        status = "destructive"
        reasons.append("non_finite_run_rate_exceeded")
    elif (
        row["stable_entry_success_rate"] < policy["required_stable_entry_success_rate"]
    ):
        status = "too_weak"
        reasons.append("stable_entry_success_rate_below_required")
    elif row["entry_then_exit_rate"] > policy["maximum_entry_then_exit_rate"]:
        status = "unstable"
        reasons.append("entry_then_exit_rate_exceeded")
    elif row["severe_rebound_rate"] > policy["maximum_severe_rebound_rate"]:
        status = "unstable"
        reasons.append("severe_rebound_rate_exceeded")
    elif row["mean_trajectory_variance"] > policy["maximum_trajectory_variance"]:
        status = "unstable"
        reasons.append("trajectory_variance_exceeded")
    row["profile_status"] = status
    row["selection_eligible"] = status == "acceptable"
    row["selection_exclusion_reasons"] = reasons
    return row


def paired_bootstrap_compare(
    left: dict[str, Any], right: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    left_map = {
        int(row["seed"]): row["steps_to_stable_entry"] for row in left["seed_metrics"]
    }
    right_map = {
        int(row["seed"]): row["steps_to_stable_entry"] for row in right["seed_metrics"]
    }
    if set(left_map) != set(right_map) or not left_map:
        raise ValueError("paired comparison requires identical aligned seed sets")
    seeds = sorted(left_map)
    if any(left_map[s] is None or right_map[s] is None for s in seeds):
        raise ValueError("paired comparison requires non-null aligned observations")
    deltas = [float(left_map[s]) - float(right_map[s]) for s in seeds]
    interval = bootstrap_ci95(
        deltas,
        samples=policy["bootstrap_samples"],
        seed=policy["bootstrap_seed"]
        + left["aggressiveness_rank"] * 31
        + right["aggressiveness_rank"],
    )
    assert interval is not None
    result = (
        "left_better"
        if interval[1] < 0
        else "right_better"
        if interval[0] > 0
        else "tie"
    )
    preferred = (
        left
        if result == "left_better"
        else right
        if result == "right_better"
        else min((left, right), key=lambda row: row["aggressiveness_rank"])
    )
    return {
        "left_profile": left["profile_name"],
        "right_profile": right["profile_name"],
        "metric": "steps_to_stable_entry",
        "direction": "lower_is_better",
        "aligned_seeds": seeds,
        "paired_deltas": deltas,
        "bootstrap_samples": policy["bootstrap_samples"],
        "bootstrap_seed": policy["bootstrap_seed"]
        + left["aggressiveness_rank"] * 31
        + right["aggressiveness_rank"],
        "ci95": interval,
        "statistically_distinguishable": result != "tie",
        "comparison_result": result,
        "tie_break_rule": "lower_aggressiveness_rank",
        "preferred_profile": preferred["profile_name"],
    }


def compare_eligible_profiles(
    candidates: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[str | None, list[dict[str, Any]]]:
    if not candidates:
        return None, []
    ordered = sorted(candidates, key=lambda row: row["aggressiveness_rank"])
    winner = ordered[0]
    comparisons: list[dict[str, Any]] = []
    for challenger in ordered[1:]:
        comparison = paired_bootstrap_compare(winner, challenger, policy)
        comparisons.append(comparison)
        winner = next(
            row
            for row in (winner, challenger)
            if row["profile_name"] == comparison["preferred_profile"]
        )
    return winner["profile_name"], comparisons


def _compare_optional_numeric(left: Any, right: Any) -> int:
    left_value = math.inf if left is None else float(left)
    right_value = math.inf if right is None else float(right)
    if left_value < right_value:
        return -1
    if left_value > right_value:
        return 1
    return 0


def _metric_array(summary: dict[str, Any], metric_name: str) -> list[float]:
    key = {
        "steps_to_stable_entry": "steps_to_stable_entry",
        "seconds_to_stable_entry": "seconds_to_stable_entry",
        "records_to_stable_entry": "records_to_stable_entry",
        "bytes_to_stable_entry": "bytes_to_stable_entry",
    }[metric_name]
    values = [row[key] for row in summary["seed_metrics"] if row[key] is not None]
    return [float(value) for value in values]


def _paired_bootstrap_delta_ci(
    left: list[float], right: list[float], *, samples: int, seed: int
) -> list[float] | None:
    if not left or not right or len(left) != len(right):
        return None
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(delta, size=(samples, len(delta)), replace=True), axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _fairness(
    config: AggressivenessCalibrationConfig,
    profiles: list[CorridorAggressivenessProfile],
) -> dict[str, Any]:
    reference = profiles[0]
    undeclared: list[str] = []
    for other in profiles[1:]:
        undeclared.extend(_undeclared_profile_differences(reference, other))
    only_declared = not undeclared
    required = {
        "same_student_architecture": True,
        "same_student_backend": True,
        "same_training_artifact": True,
        "same_held_out_artifact": True,
        "same_source_example_set": True,
        "same_initialization_procedure": True,
        "same_seed_set": True,
        "same_requested_steps": True,
        "same_batch_size": True,
        "same_eval_every": True,
        "same_checkpoint_every": True,
        "same_optimizer_family": True,
        "same_sequence_length": True,
        "same_stopping_policy": True,
        "only_declared_aggressiveness_fields_differ": only_declared,
    }
    return {
        **required,
        "declared_profile_difference_fields": sorted(ALLOWED_PROFILE_DIFFERENCE_FIELDS),
        "undeclared_profile_differences": sorted(set(undeclared)),
        "comparison_valid": len(set(config.profiles)) == len(config.profiles)
        and bool(config.seeds)
        and all(required.values()),
    }


def _undeclared_profile_differences(
    left: CorridorAggressivenessProfile, right: CorridorAggressivenessProfile
) -> list[str]:
    left_dict = left.to_dict()
    right_dict = right.to_dict()
    differences: list[str] = []
    for key in sorted(left_dict):
        if key in {
            "profile_name",
            "aggressiveness_rank",
            "profile_overrides_applied",
        }:
            continue
        if (
            left_dict[key] != right_dict[key]
            and key not in ALLOWED_PROFILE_DIFFERENCE_FIELDS
        ):
            differences.append(key)
    return differences


def _validate_seed_aggregate(
    *,
    profile_name: str,
    seed: int,
    trajectory: list[dict[str, Any]],
    report: dict[str, Any],
    seed_metrics: dict[str, Any],
) -> dict[str, Any]:
    all_metrics_finite = True
    all_resource_metrics_non_negative = True
    all_trajectories_ordered = True
    all_step_zero_points_present = bool(
        trajectory and trajectory[0]["optimizer_step"] == 0
    )
    final_step = int(report["completed_steps"])
    all_final_points_present = bool(
        trajectory and trajectory[-1]["optimizer_step"] == final_step
    )
    last_step = -1
    for point in trajectory:
        step = int(point["optimizer_step"])
        if step <= last_step:
            all_trajectories_ordered = False
        last_step = step
        for key in (
            "held_out_corridor_loss",
            "mean_distance_outside_corridor",
            "inside_all_rate",
            "parameter_delta_from_initial",
        ):
            value = point.get(key)
            if value is None or not math.isfinite(float(value)):
                all_metrics_finite = False
        for key in (
            "records_consumed",
            "tokens_consumed",
            "artifact_bytes_read",
            "wall_clock_seconds",
        ):
            value = point.get(key)
            if value is None or float(value) < 0.0:
                all_resource_metrics_non_negative = False
    all_entry_cost_lookups_valid = bool(seed_metrics["entry_cost_lookup_valid"])
    all_entry_costs_within_final_totals = all_entry_cost_lookups_valid
    status = (
        "pass"
        if all(
            (
                all_metrics_finite,
                all_resource_metrics_non_negative,
                all_trajectories_ordered,
                all_step_zero_points_present,
                all_final_points_present,
                all_entry_cost_lookups_valid,
                all_entry_costs_within_final_totals,
            )
        )
        else "fail"
    )
    return {
        "profile_name": profile_name,
        "seed": seed,
        "status": status,
        "all_metrics_finite": all_metrics_finite,
        "all_trajectories_ordered": all_trajectories_ordered,
        "all_step_zero_points_present": all_step_zero_points_present,
        "all_final_points_present": all_final_points_present,
        "all_resource_metrics_non_negative": all_resource_metrics_non_negative,
        "all_entry_cost_lookups_valid": all_entry_cost_lookups_valid,
        "all_entry_costs_within_final_totals": all_entry_costs_within_final_totals,
        "no_duplicate_trajectory_steps": len(
            {row["optimizer_step"] for row in trajectory}
        )
        == len(trajectory),
    }


def _aggregate_validation(
    *,
    config: AggressivenessCalibrationConfig,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    fairness: dict[str, Any],
    seed_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    all_metrics_finite = all(
        check["all_metrics_finite"] for check in seed_checks
    ) and all(_all_summary_metrics_finite(summary) for summary in summaries)
    all_trajectories_ordered = all(
        check["all_trajectories_ordered"] for check in seed_checks
    )
    all_step_zero_points_present = all(
        check["all_step_zero_points_present"] for check in seed_checks
    )
    all_final_points_present = all(
        check["all_final_points_present"] for check in seed_checks
    )
    all_resource_metrics_non_negative = all(
        check["all_resource_metrics_non_negative"] for check in seed_checks
    )
    all_entry_cost_lookups_valid = all(
        check.get("all_entry_cost_lookups_valid", True) for check in seed_checks
    )
    all_entry_costs_within_final_totals = all(
        check.get("all_entry_costs_within_final_totals", True) for check in seed_checks
    )
    no_duplicate_trajectory_steps = all(
        check.get("no_duplicate_trajectory_steps", True) for check in seed_checks
    )
    expected_run_count = len(config.profiles) * len(config.seeds)
    observed_run_count = len(rows)
    missing_run_count = max(0, expected_run_count - observed_run_count)
    expected_pairs = {
        (profile, seed) for profile in config.profiles for seed in config.seeds
    }
    observed_pairs = [(row["profile_name"], row["seed"]) for row in rows]
    all_required_profiles_present = {row["profile_name"] for row in summaries} == set(
        config.profiles
    )
    all_required_seed_rows_present = set(observed_pairs) == expected_pairs
    no_duplicate_seed_rows = len(observed_pairs) == len(set(observed_pairs))
    aligned_seed_sets_valid = all(
        {row["seed"] for row in rows if row["profile_name"] == profile}
        == set(config.seeds)
        for profile in config.profiles
    )
    bootstrap_inputs_valid = aligned_seed_sets_valid and all(
        row.get("steps_to_stable_entry") is not None
        for row in rows
        if any(
            summary["profile_name"] == row["profile_name"]
            and summary["stable_entry_success_rate"] == 1.0
            for summary in summaries
        )
    )
    required = (
        all_required_profiles_present,
        all_required_seed_rows_present,
        all_metrics_finite,
        all_trajectories_ordered,
        all_step_zero_points_present,
        all_final_points_present,
        all_entry_cost_lookups_valid,
        all_resource_metrics_non_negative,
        all_entry_costs_within_final_totals,
        no_duplicate_seed_rows,
        no_duplicate_trajectory_steps,
        aligned_seed_sets_valid,
        bootstrap_inputs_valid,
    )
    return {
        "status": "pass" if all(required) else "fail",
        "all_required_profiles_present": all_required_profiles_present,
        "all_required_seed_rows_present": all_required_seed_rows_present,
        "all_required_metrics_valid": all_metrics_finite,
        "all_metrics_finite": all_metrics_finite,
        "all_trajectories_ordered": all_trajectories_ordered,
        "all_step_zero_points_present": all_step_zero_points_present,
        "all_final_points_present": all_final_points_present,
        "all_resource_metrics_non_negative": all_resource_metrics_non_negative,
        "all_entry_cost_lookups_valid": all_entry_cost_lookups_valid,
        "all_entry_costs_within_final_totals": all_entry_costs_within_final_totals,
        "no_duplicate_seed_rows": no_duplicate_seed_rows,
        "no_duplicate_trajectory_steps": no_duplicate_trajectory_steps,
        "aligned_seed_sets_valid": aligned_seed_sets_valid,
        "bootstrap_inputs_valid": bootstrap_inputs_valid,
        "expected_run_count": expected_run_count,
        "observed_run_count": observed_run_count,
        "missing_run_count": missing_run_count,
        "comparison_valid": fairness["comparison_valid"],
        "seed_checks": seed_checks,
    }


def derive_selection_allowed(
    *,
    fairness: dict[str, Any],
    aggregate_validation: dict[str, Any],
    classification_complete: bool,
    candidate_gating_complete: bool,
) -> tuple[bool, str | None]:
    if not fairness["comparison_valid"]:
        return False, "fairness_validation_failed"
    if aggregate_validation["status"] != "pass":
        return False, "aggregate_validation_failed"
    if not classification_complete:
        return False, "classification_incomplete"
    if not candidate_gating_complete:
        return False, "candidate_gating_incomplete"
    return True, None


def derive_top_level_status(
    *,
    fairness_valid: bool,
    aggregate_validation_status: str,
    selection_logic_completed: bool,
) -> str:
    return (
        "pass"
        if fairness_valid
        and aggregate_validation_status == "pass"
        and selection_logic_completed
        else "fail"
    )


def _all_summary_metrics_finite(summary: dict[str, Any]) -> bool:
    for _key, value in summary.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return False
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                    return False
    return True


def _publication_grade_receipt(
    *,
    config: AggressivenessCalibrationConfig,
    rows: list[dict[str, Any]],
    fairness: dict[str, Any],
    summaries: list[dict[str, Any]],
    aggregate_validation: dict[str, Any],
    selection_receipt: dict[str, Any],
) -> dict[str, Any]:
    expected_profiles = set(config.profiles)
    seen_profiles = {row["profile_name"] for row in summaries}
    all_required_profiles_present = seen_profiles == expected_profiles
    all_required_seed_runs_present = all(
        row["observed_run_count"] == row["expected_run_count"] for row in summaries
    )
    all_required_runs_complete = all(
        row["completed_run_count"] == row["expected_run_count"] for row in summaries
    )
    minimum_seed_count_met = len(config.seeds) >= config.minimum_seed_count
    lineage_valid_for_all_runs = all(row["lineage_valid"] for row in rows)
    no_undeclared_profile_differences = fairness[
        "only_declared_aggressiveness_fields_differ"
    ]
    aggregate_metrics_finite = aggregate_validation["status"] == "pass"
    selection_logic_completed = bool(
        selection_receipt["selection_status"] in {"selected", "inconclusive"}
    )
    requirements = {
        "fairness_valid": fairness["comparison_valid"],
        "all_profiles_present": all_required_profiles_present,
        "all_seed_runs_present": all_required_seed_runs_present,
        "all_runs_complete": all_required_runs_complete,
        "minimum_seed_count_met": minimum_seed_count_met,
        "lineage_valid": lineage_valid_for_all_runs,
        "declared_differences_only": no_undeclared_profile_differences,
        "aggregate_metrics_finite": aggregate_metrics_finite,
        "selection_logic_completed": selection_logic_completed,
    }
    return {
        "publication_grade": all(requirements.values()),
        "publication_grade_requirements": requirements,
        "minimum_seed_count": config.minimum_seed_count,
    }


def _paired(
    rows: list[dict[str, Any]], profiles: tuple[str, ...]
) -> list[dict[str, Any]]:
    lookup = {(r["profile_name"], r["seed"]): r for r in rows}
    result = []
    for left, right in combinations(profiles, 2):
        seeds = sorted(
            {r["seed"] for r in rows if r["profile_name"] == left}
            & {r["seed"] for r in rows if r["profile_name"] == right}
        )
        for seed in seeds:
            a, b = lookup[(left, seed)], lookup[(right, seed)]
            result.append(
                {
                    "left_profile": left,
                    "right_profile": right,
                    "seed": seed,
                    "final_distance_delta_right_minus_left": b["final_mean_distance"]
                    - a["final_mean_distance"],
                    "parameter_delta_right_minus_left": b["parameter_delta_norm"]
                    - a["parameter_delta_norm"],
                    "steps_to_stable_entry_delta_right_minus_left": (
                        None
                        if a["first_stable_entry_step"] is None
                        or b["first_stable_entry_step"] is None
                        else b["first_stable_entry_step"] - a["first_stable_entry_step"]
                    ),
                }
            )
    return result


def _validate(config: AggressivenessCalibrationConfig) -> None:
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass overwrite=True"
        )
    if not config.profiles or not config.seeds:
        raise ValueError("profiles and seeds must be non-empty")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")
    if config.minimum_seed_count < 1:
        raise ValueError("minimum_seed_count must be >= 1")
    for name in config.profiles:
        resolve_aggressiveness_profile(name, config.overrides)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
