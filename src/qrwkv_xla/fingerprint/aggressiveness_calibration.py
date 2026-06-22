from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.fingerprint.aggressiveness_profiles import (
    PROFILE_NAMES,
    resolve_aggressiveness_profile,
)
from qrwkv_xla.fingerprint.corridor_measurement import (
    CorridorMeasurementConfig,
    run_corridor_measurement,
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
) -> tuple[str | None, list[dict[str, Any]]]:
    ranked = []
    candidates = []
    for row in summaries:
        if row["abort_rate"] > 0 or row["completed_run_rate"] < 1:
            status = "destructive" if row["abort_rate"] else "inconclusive"
        elif row["stable_entry_success_rate"] < 1:
            status = "too_weak"
        else:
            status = "acceptable"
            candidates.append(row)
        ranked.append({**row, "selection_status": status})
    candidates.sort(
        key=lambda row: (
            row["median_steps_to_stable_entry"]
            if row["median_steps_to_stable_entry"] is not None
            else float("inf"),
            row["mean_entry_then_exit_count"],
            row["mean_distance_rebound"],
            row["aggressiveness_rank"],
        )
    )
    selected = candidates[0]["profile_name"] if candidates else None
    for row in ranked:
        if row["profile_name"] == selected:
            row["selection_status"] = "selected"
            row["selection_reason"] = "minimum_sufficient_force"
        else:
            row["selection_reason"] = "validity_reliability_efficiency_order"
    ranked.sort(
        key=lambda row: (
            row["selection_status"] != "selected",
            row["aggressiveness_rank"],
        )
    )
    for index, row in enumerate(ranked, 1):
        row["profile_rank"] = index
    return selected, ranked


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
                    overwrite=config.overwrite,
                )
            )
            report = read_json_object(run_dir / "corridor_measurement_report.json")
            trajectory = _read_jsonl(run_dir / "corridor_trajectory.jsonl")
            rows.append(
                _seed_metrics(profile.to_dict(), seed, report, trajectory, config)
            )
    summaries = [_summarize_profile(name, rows, config) for name in config.profiles]
    selected, ranking = select_profile(summaries)
    fairness = _fairness(config, resolved)
    publication_grade = len(config.seeds) >= 3 and all(
        row["completed_run_rate"] == 1 for row in summaries
    )
    status = "pass" if fairness["comparison_valid"] else "fail"
    report = {
        "phase": "P153.1",
        "status": status,
        "calibration_kind": "corridor_aggressiveness_profiles",
        "profiles": list(config.profiles),
        "seed_count": len(config.seeds),
        "comparison_valid": fairness["comparison_valid"],
        "publication_grade": publication_grade,
        "publication_grade_blockers": (
            []
            if publication_grade
            else [
                "publication-grade calibration requires at least three "
                "complete seeds per profile"
            ]
        ),
        "selected_profile": selected,
        "selection_status": "selected" if selected else "inconclusive",
        "selection_rule": "minimum_sufficient_force",
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "scale_claim_made": False,
        "radlads_parity_claim_made": False,
    }
    write_json(config.output_dir / "profile_fairness_contract.json", fairness)
    write_json(config.output_dir / "profile_summary_metrics.json", summaries)
    write_json(config.output_dir / "profile_ranking.json", ranking)
    _write_jsonl(config.output_dir / "profile_seed_metrics.jsonl", rows)
    _write_jsonl(
        config.output_dir / "paired_profile_deltas.jsonl",
        _paired(rows, config.profiles),
    )
    if selected:
        selected_config = next(
            p.to_dict() for p in resolved if p.profile_name == selected
        )
        write_json(config.output_dir / "selected_profile_config.json", selected_config)
    report_path = config.output_dir / "aggressiveness_calibration_report.json"
    write_json(report_path, report)
    (config.output_dir / "aggressiveness_calibration_summary.md").write_text(
        "# P153.1 Aggressiveness Calibration\n\n"
        f"- Status: {status}\n- Selected profile: {selected or 'none'}\n"
        f"- Seeds per profile: {len(config.seeds)}\n"
        f"- Publication grade: {str(publication_grade).lower()}\n"
        "- General quality claim: false\n",
        encoding="utf-8",
    )
    return AggressivenessCalibrationResult(
        status, selected, report_path, config.output_dir
    )


def _seed_metrics(
    profile: dict[str, Any],
    seed: int,
    report: dict[str, Any],
    trajectory: list[dict[str, Any]],
    config: AggressivenessCalibrationConfig,
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
    return {
        "profile_name": profile["profile_name"],
        "aggressiveness_rank": profile["aggressiveness_rank"],
        "seed": seed,
        "run_status": report["status"],
        "stable_entry_achieved": report["stable_entry_achieved"],
        "first_threshold_entry_step": report["first_threshold_entry_step"],
        "first_stable_entry_step": report["first_stable_entry_step"],
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
        "parameter_delta_norm": trajectory[-1]["parameter_delta_from_initial"],
        "gradient_norms": grads,
        "gradient_spike_count": sum(
            g > profile["gradient_norm_hard_limit"] * 0.5 for g in grads
        ),
        "clip_event_count": sum(clips),
        "clip_event_rate": sum(clips) / max(len(clips), 1),
        "non_finite_loss": report["stop_reason"] == "non_finite_loss",
        "non_finite_gradient": report["stop_reason"] == "non_finite_gradient",
        "abort_triggered": report["abort_triggered"],
        "abort_reason": report["abort_reason"],
        "stop_reason": report["stop_reason"],
        "held_out_loss_rebound": losses[-1] - min(losses),
        "distance_rebound": distances[-1] - min(distances),
        "trajectory_variance": float(np.var(distances)),
        **exits,
    }


def _summarize_profile(
    name: str, rows: list[dict[str, Any]], config: AggressivenessCalibrationConfig
) -> dict[str, Any]:
    selected = [r for r in rows if r["profile_name"] == name]
    stable_steps = [
        float(r["first_stable_entry_step"])
        for r in selected
        if r["first_stable_entry_step"] is not None
    ]
    rank = selected[0]["aggressiveness_rank"]

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in selected]))

    return {
        "profile_name": name,
        "aggressiveness_rank": rank,
        "run_count": len(selected),
        "completed_run_rate": float(
            np.mean([r["run_status"] == "pass" for r in selected])
        ),
        "stable_entry_success_rate": mean("stable_entry_achieved"),
        "median_steps_to_stable_entry": float(np.median(stable_steps))
        if stable_steps
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
        "abort_rate": mean("abort_triggered"),
        "median_parameter_delta": float(
            np.median([r["parameter_delta_norm"] for r in selected])
        ),
        "mean_distance_rebound": mean("distance_rebound"),
    }


def _fairness(
    config: AggressivenessCalibrationConfig, profiles: list[Any]
) -> dict[str, Any]:
    return {
        "same_student_architecture": True,
        "same_student_backend": True,
        "same_training_artifact": True,
        "same_held_out_artifact": True,
        "same_source_example_set": True,
        "same_requested_steps": True,
        "same_batch_size": True,
        "same_eval_every": True,
        "same_checkpoint_every": True,
        "same_optimizer_family": True,
        "same_seed_set": True,
        "only_declared_aggressiveness_fields_differ": all(
            not p.adaptive_weighting_enabled for p in profiles
        ),
        "comparison_valid": len(set(config.profiles)) == len(config.profiles)
        and bool(config.seeds),
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
