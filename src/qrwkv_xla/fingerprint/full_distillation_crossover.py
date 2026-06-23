from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

ARMS = ("vanilla", "exemplar_only", "adaptive_two_cycle")
ARM_PAIRS = (
    ("adaptive_two_cycle", "exemplar_only"),
    ("adaptive_two_cycle", "vanilla"),
    ("exemplar_only", "vanilla"),
)
TEACHER_FORCED_METRICS = (
    "teacher_student_kl",
    "top1_agreement",
    "topk_overlap",
    "teacher_entropy",
    "student_entropy",
    "entropy_absolute_error",
    "held_out_exemplar_loss",
    "held_out_corridor_loss",
    "inside_all_rate",
    "mean_distance_outside_corridor",
    "worst_stat_violation",
)
STUDENT_PREFIX_METRICS = (
    "student_prefix_teacher_student_kl",
    "student_prefix_top1_agreement",
    "student_prefix_topk_overlap",
    "student_prefix_entropy_error",
    "trajectory_length",
)


@dataclass(frozen=True)
class FullDistillationCrossoverConfig:
    training_artifact: Path
    calibration_artifact: Path
    final_test_artifact: Path
    source_texts: Path
    student_config: Path
    selected_profile_receipt: Path
    output_dir: Path
    seeds: tuple[int, ...] = (0,)
    checkpoint_fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    target_quality_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {"teacher_student_kl": 1.0}
    )
    target_metric: str = "teacher_student_kl"
    target_direction: str = "lower"
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 1564
    require_backend: str = "cpu"
    maximum_steps: int | None = None
    convergence_metric: str = "teacher_student_kl"
    convergence_window: int = 3
    convergence_relative_improvement_threshold: float = 0.01
    convergence_absolute_improvement_threshold: float = 0.01
    convergence_patience: int = 2
    minimum_steps_before_convergence: int = 0
    convergence_enabled: bool = False
    max_new_training_runs: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        _validate_config(self)

    def semantic_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("output_dir")
        value.pop("resume")
        value["target_quality_thresholds"] = dict(
            sorted(self.target_quality_thresholds.items())
        )
        return _jsonable(value)


@dataclass(frozen=True)
class SharedInitialization:
    checkpoint: Path
    parameter_tree_hash: str
    student_config_hash: str
    checkpoint_hash: str


@dataclass(frozen=True)
class AdaptiveDiscovery:
    cycle_one_complete: bool
    optimizer_steps_completed: int | None
    completion_reason: str
    checkpoint: Path
    checkpoint_hash: str
    controller_config_hash: str
    scheduler_config_hash: str
    corridor_optimizer_state_hash: str
    mode_freeze_steps: Mapping[str, int | None]
    reactivation_steps: Mapping[str, Sequence[int]]
    confirmation_only_evaluations: int = 0


@dataclass(frozen=True)
class ArmCheckpoint:
    arm: str
    total_step: int
    checkpoint: Path
    checkpoint_hash: str
    parent_checkpoint_hash: str
    initial_parameter_hash: str
    corridor_steps: int
    exemplar_steps: int
    vanilla_steps: int
    optimizer_initial_state_hash: str
    optimizer_final_state_hash: str
    resource_accounting: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointEvaluation:
    teacher_forced: Mapping[str, Any]
    teacher_forced_records: Sequence[Mapping[str, Any]]
    student_prefix: Mapping[str, Any]
    free_running: Mapping[str, Any]
    final_test_record_order_hash: str
    evaluation_seconds: float = 0.0


@runtime_checkable
class CrossoverExecutionBackend(Protocol):
    def create_shared_initialization(
        self, *, seed: int, output_dir: Path
    ) -> SharedInitialization: ...

    def discover_adaptive_cycle_one(
        self,
        *,
        seed: int,
        shared_initialization: SharedInitialization,
        output_dir: Path,
    ) -> AdaptiveDiscovery: ...

    def train_arm(
        self,
        *,
        arm: str,
        seed: int,
        shared_initialization: SharedInitialization,
        adaptive_discovery: AdaptiveDiscovery,
        checkpoint_steps: tuple[int, ...],
        output_dir: Path,
    ) -> Sequence[ArmCheckpoint]: ...

    def evaluate_checkpoint(
        self,
        *,
        seed: int,
        checkpoint: ArmCheckpoint,
        final_test_artifact: Path,
        output_dir: Path,
    ) -> CheckpointEvaluation: ...


@dataclass(frozen=True)
class CrossoverPlan:
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    adaptive_discoveries_required: int
    checkpoint_schedule_formula: tuple[float, ...]
    estimated_maximum_training_runs: int
    estimated_maximum_evaluation_cells: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_crossover_plan(config: FullDistillationCrossoverConfig) -> CrossoverPlan:
    training_runs = len(config.seeds) * (1 + len(ARMS))
    evaluation_cells = len(config.seeds) * len(ARMS) * len(config.checkpoint_fractions)
    if (
        config.max_new_training_runs is not None
        and training_runs > config.max_new_training_runs
    ):
        raise ValueError(
            f"plan requires {training_runs} training runs, exceeding "
            f"max_new_training_runs={config.max_new_training_runs}"
        )
    return CrossoverPlan(
        seeds=config.seeds,
        arms=ARMS,
        adaptive_discoveries_required=len(config.seeds),
        checkpoint_schedule_formula=config.checkpoint_fractions,
        estimated_maximum_training_runs=training_runs,
        estimated_maximum_evaluation_cells=evaluation_cells,
    )


def derive_checkpoint_schedule(
    completion_step: int, fractions: Sequence[float]
) -> tuple[int, ...]:
    if completion_step <= 0:
        raise ValueError("adaptive completion step must be > 0")
    checkpoints = {
        completion_step + math.ceil(float(fraction) * completion_step)
        for fraction in fractions
    }
    result = tuple(sorted(checkpoints))
    if not result or result[0] != completion_step:
        raise ValueError("checkpoint fractions must include 0.0")
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError("derived checkpoints must be strictly increasing")
    return result


def paired_bootstrap_statistics(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    deltas = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(deltas)):
        raise ValueError("paired bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = np.mean(deltas[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "confidence_interval_95": [float(low), float(high)],
        "fraction_won": float(np.mean(deltas < 0)),
        "fraction_tied": float(np.mean(deltas == 0)),
        "fraction_lost": float(np.mean(deltas > 0)),
        "result": "inconclusive" if low <= 0 <= high else "conclusive",
    }


def first_observed_target_crossing(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    threshold: float,
    direction: str,
) -> Mapping[str, Any] | None:
    ordered = sorted(rows, key=lambda row: int(row["total_step"]))
    for row in ordered:
        value = row.get(metric)
        if value is None:
            continue
        reached = (
            float(value) <= threshold
            if direction == "lower"
            else float(value) >= threshold
        )
        if reached:
            return row
    return None


def run_full_distillation_crossover(
    config: FullDistillationCrossoverConfig,
    *,
    backend: CrossoverExecutionBackend,
) -> dict[str, Any]:
    plan = build_crossover_plan(config)
    config_hash = _stable_hash(config.semantic_dict())
    state_path = config.output_dir / "crossover_experiment_state.json"
    state = _load_or_create_state(config, state_path, config_hash)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "full_distillation_crossover_report.json"
    if config.resume and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    schedules: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        seed_key = str(seed)
        seed_dir = config.output_dir / f"seed_{seed}"
        if seed_key in state["shared_initializations"]:
            shared = _shared_from_dict(state["shared_initializations"][seed_key])
        else:
            shared = backend.create_shared_initialization(
                seed=seed, output_dir=seed_dir / "shared_init"
            )
            state["shared_initializations"][seed_key] = _jsonable(asdict(shared))
            _write_json(state_path, state)
        _write_json(
            seed_dir / "shared_initialization_receipt.json",
            {
                "phase": "P156.4",
                "seed": seed,
                "student_config_hash": shared.student_config_hash,
                "parameter_tree_hash": shared.parameter_tree_hash,
                "checkpoint_hash": shared.checkpoint_hash,
                "created_once": True,
                "used_by_arms": list(ARMS),
            },
        )
        if seed_key in state["adaptive_discoveries"]:
            discovery = _discovery_from_dict(state["adaptive_discoveries"][seed_key])
        else:
            discovery = backend.discover_adaptive_cycle_one(
                seed=seed,
                shared_initialization=shared,
                output_dir=seed_dir / "adaptive_discovery",
            )
            state["adaptive_discoveries"][seed_key] = _jsonable(asdict(discovery))
            _write_json(state_path, state)
        _write_json(
            seed_dir / "adaptive_completion_receipt.json",
            {
                "phase": "P156.4",
                "seed": seed,
                "cycle_one_complete": discovery.cycle_one_complete,
                "S": discovery.optimizer_steps_completed,
                "global_completion_reason": discovery.completion_reason,
                "mode_freeze_steps": dict(discovery.mode_freeze_steps),
                "reactivation_steps": {
                    key: list(value)
                    for key, value in discovery.reactivation_steps.items()
                },
                "confirmation_only_evaluations": (
                    discovery.confirmation_only_evaluations
                ),
                "controller_config_hash": discovery.controller_config_hash,
                "scheduler_config_hash": discovery.scheduler_config_hash,
                "corridor_optimizer_state_hash": (
                    discovery.corridor_optimizer_state_hash
                ),
                "adaptive_corridor_checkpoint_hash": discovery.checkpoint_hash,
            },
        )
        if (
            not discovery.cycle_one_complete
            or discovery.optimizer_steps_completed is None
        ):
            state["failed_items"].append(f"seed_{seed}:adaptive_discovery")
            _write_json(state_path, state)
            continue
        completion_step = discovery.optimizer_steps_completed
        schedule = derive_checkpoint_schedule(
            completion_step, config.checkpoint_fractions
        )
        if config.maximum_steps is not None and schedule[-1] > config.maximum_steps:
            raise ValueError("derived checkpoint schedule exceeds maximum_steps")
        schedule_row = {
            "seed": seed,
            "S": completion_step,
            "fractions": list(config.checkpoint_fractions),
            "total_step_checkpoints": list(schedule),
            "maximum_checkpoint": schedule[-1],
        }
        schedules.append(schedule_row)
        _write_json(seed_dir / "matched_checkpoint_schedule.json", schedule_row)
        state["completed_adaptive_discoveries"][str(seed)] = schedule_row
        arm_checkpoints: dict[str, Sequence[ArmCheckpoint]] = {}
        for arm in ARMS:
            arm_key = f"{seed}:{arm}"
            if arm_key in state["arm_checkpoint_results"]:
                checkpoints = tuple(
                    _arm_checkpoint_from_dict(value)
                    for value in state["arm_checkpoint_results"][arm_key]
                )
            else:
                checkpoints = backend.train_arm(
                    arm=arm,
                    seed=seed,
                    shared_initialization=shared,
                    adaptive_discovery=discovery,
                    checkpoint_steps=schedule,
                    output_dir=seed_dir / f"arm_{arm}",
                )
                state["arm_checkpoint_results"][arm_key] = [
                    _jsonable(asdict(checkpoint)) for checkpoint in checkpoints
                ]
                _write_json(state_path, state)
            _validate_arm_checkpoints(
                checkpoints,
                arm=arm,
                schedule=schedule,
                shared=shared,
                completion_step=completion_step,
            )
            arm_checkpoints[arm] = checkpoints
            for checkpoint in checkpoints:
                cell = f"{seed}:{arm}:{checkpoint.total_step}"
                if cell not in state["completed_arm_checkpoints"]:
                    state["completed_arm_checkpoints"].append(cell)
        _write_cycle_boundary_receipt(seed_dir, discovery, arm_checkpoints)

        evaluations: dict[tuple[str, int], CheckpointEvaluation] = {}
        for arm, checkpoints in arm_checkpoints.items():
            for checkpoint in checkpoints:
                evaluation_key = f"{seed}:{arm}:{checkpoint.total_step}"
                if evaluation_key in state["evaluation_results"]:
                    evaluation = _evaluation_from_dict(
                        state["evaluation_results"][evaluation_key]
                    )
                else:
                    evaluation = backend.evaluate_checkpoint(
                        seed=seed,
                        checkpoint=checkpoint,
                        final_test_artifact=config.final_test_artifact,
                        output_dir=(
                            seed_dir
                            / f"arm_{arm}"
                            / f"step_{checkpoint.total_step}"
                            / "evaluation"
                        ),
                    )
                    state["evaluation_results"][evaluation_key] = _jsonable(
                        asdict(evaluation)
                    )
                    _write_json(state_path, state)
                evaluations[(arm, checkpoint.total_step)] = evaluation
                common = {
                    "seed": seed,
                    "arm": arm,
                    "total_step": checkpoint.total_step,
                    "checkpoint_hash": checkpoint.checkpoint_hash,
                    "final_test_record_order_hash": (
                        evaluation.final_test_record_order_hash
                    ),
                }
                teacher_rows.append(
                    {
                        **common,
                        **_metrics_with_availability(
                            evaluation.teacher_forced, TEACHER_FORCED_METRICS
                        ),
                    }
                )
                prefix_rows.append(
                    {
                        **common,
                        **_metrics_with_availability(
                            evaluation.student_prefix, STUDENT_PREFIX_METRICS
                        ),
                    }
                )
                generation_rows.append({**common, **dict(evaluation.free_running)})
                resource_rows.append(
                    {
                        **common,
                        **dict(checkpoint.resource_accounting),
                        "evaluation_seconds": evaluation.evaluation_seconds,
                        "corridor_optimizer_steps": checkpoint.corridor_steps,
                        "exemplar_optimizer_steps": checkpoint.exemplar_steps,
                        "vanilla_optimizer_steps": checkpoint.vanilla_steps,
                    }
                )
                _write_json(
                    checkpoint.checkpoint / "checkpoint_lineage_receipt.json",
                    _checkpoint_lineage(checkpoint, seed, shared, config),
                )
                _write_json(
                    checkpoint.checkpoint / "evaluation_request_receipt.json",
                    {
                        "seed": seed,
                        "arm": arm,
                        "total_step": checkpoint.total_step,
                        "final_test_artifact": str(config.final_test_artifact),
                        "final_test_record_order_hash": (
                            evaluation.final_test_record_order_hash
                        ),
                    },
                )
                if evaluation_key not in state["completed_evaluations"]:
                    state["completed_evaluations"].append(evaluation_key)
        _validate_final_test_order(evaluations)
        comparison_rows.extend(
            _build_comparisons(
                seed,
                completion_step,
                schedule,
                evaluations,
                metric=config.target_metric,
                direction=config.target_direction,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
            )
        )
        for arm in ARMS:
            arm_rows = [
                row for row in teacher_rows if row["seed"] == seed and row["arm"] == arm
            ]
            threshold = config.target_quality_thresholds.get(config.target_metric)
            crossing = (
                None
                if threshold is None
                else first_observed_target_crossing(
                    arm_rows,
                    metric=config.target_metric,
                    threshold=threshold,
                    direction=config.target_direction,
                )
            )
            target_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "target_metric": config.target_metric,
                    "threshold": threshold,
                    "target_reached": crossing is not None,
                    "first_checkpoint_reaching_target": (
                        None if crossing is None else crossing["total_step"]
                    ),
                    **_target_costs(crossing, resource_rows, seed=seed, arm=arm),
                    "interpolation_used": False,
                    "extrapolation_used": False,
                }
            )
        _write_json(state_path, state)

    _write_jsonl(config.output_dir / "matched_checkpoint_schedule.jsonl", schedules)
    _write_jsonl(
        config.output_dir / "teacher_forced_checkpoint_metrics.jsonl", teacher_rows
    )
    _write_jsonl(
        config.output_dir / "student_prefix_checkpoint_metrics.jsonl", prefix_rows
    )
    _write_jsonl(
        config.output_dir / "free_running_checkpoint_metrics.jsonl", generation_rows
    )
    _write_jsonl(
        config.output_dir / "checkpoint_resource_accounting.jsonl", resource_rows
    )
    _write_jsonl(
        config.output_dir / "crossover_checkpoint_comparisons.jsonl",
        comparison_rows,
    )
    _write_json(config.output_dir / "target_quality_receipt.json", target_rows)
    status = "pass" if not state["failed_items"] else "fail"
    report = {
        "phase": "P156.4",
        "status": status,
        "implementation_complete": True,
        "implementation_smoke_complete": len(config.seeds) == 1,
        "run_kind": "implementation_smoke" if len(config.seeds) == 1 else "harness_run",
        "full_distillation_run_started": False,
        "publication_grade": False,
        "ready_for_P156_5": bool(
            status == "pass" and getattr(backend, "ready_for_p156_5", status == "pass")
        ),
        "checkpoint_execution_mode": getattr(
            backend, "checkpoint_execution_mode", "independent_replay"
        ),
        "strict_resource_accounting": getattr(
            backend, "strict_resource_accounting", False
        ),
        "config_sha256": config_hash,
        "plan": _jsonable(plan.to_dict()),
        "seeds_completed": len(state["completed_adaptive_discoveries"]),
        "checkpoint_cells_completed": len(state["completed_evaluations"]),
        "target_quality": target_rows,
        "claims": {
            "winner_declared": False,
            "economic_claim_allowed": False,
            "scale_claim_allowed": False,
        },
    }
    _write_json(report_path, report)
    _write_json(
        config.output_dir / "publication_claims_receipt.json",
        {"phase": "P156.4", "publication_grade": False, **report["claims"]},
    )
    (config.output_dir / "full_distillation_crossover_summary.md").write_text(
        _render_summary(report), encoding="utf-8"
    )
    return report


def _validate_config(config: FullDistillationCrossoverConfig) -> None:
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique and non-empty")
    if config.require_backend != "cpu":
        raise ValueError("P156.4 implementation supports CPU only")
    if not config.checkpoint_fractions or 0.0 not in config.checkpoint_fractions:
        raise ValueError("checkpoint fractions must include 0.0")
    if any(
        not math.isfinite(value) or value < 0 for value in config.checkpoint_fractions
    ):
        raise ValueError("checkpoint fractions must be finite and non-negative")
    if config.target_direction not in {"lower", "higher"}:
        raise ValueError("target_direction must be lower or higher")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")
    if config.convergence_window <= 0 or config.convergence_patience <= 0:
        raise ValueError("convergence window and patience must be > 0")
    roles = (
        config.training_artifact,
        config.calibration_artifact,
        config.final_test_artifact,
    )
    if len({path.resolve() for path in roles}) != 3:
        raise ValueError("training, calibration, and final-test artifacts must differ")
    required_paths = (
        *roles,
        config.source_texts,
        config.student_config,
        config.selected_profile_receipt,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(f"required crossover inputs do not exist: {missing}")


def _validate_arm_checkpoints(
    checkpoints: Sequence[ArmCheckpoint],
    *,
    arm: str,
    schedule: tuple[int, ...],
    shared: SharedInitialization,
    completion_step: int,
) -> None:
    if tuple(checkpoint.total_step for checkpoint in checkpoints) != schedule:
        raise ValueError(f"{arm} checkpoint schedule mismatch")
    for checkpoint in checkpoints:
        if checkpoint.arm != arm:
            raise ValueError("arm checkpoint identity mismatch")
        if checkpoint.initial_parameter_hash != shared.parameter_tree_hash:
            raise ValueError("arm initialization hash mismatch")
        if arm == "adaptive_two_cycle":
            if checkpoint.corridor_steps != completion_step:
                raise ValueError("adaptive checkpoint corridor step mismatch")
            if checkpoint.exemplar_steps != checkpoint.total_step - completion_step:
                raise ValueError("adaptive Cycle 2 step mismatch")
        elif arm == "vanilla" and checkpoint.vanilla_steps != checkpoint.total_step:
            raise ValueError("vanilla checkpoint step mismatch")
        elif (
            arm == "exemplar_only"
            and checkpoint.exemplar_steps != checkpoint.total_step
        ):
            raise ValueError("exemplar checkpoint step mismatch")
    if all(
        checkpoint.resource_accounting.get("continuous_trajectory_confirmed") is True
        for checkpoint in checkpoints
    ):
        for previous, current in zip(checkpoints, checkpoints[1:], strict=False):
            if current.parent_checkpoint_hash != previous.checkpoint_hash:
                raise ValueError(f"{arm} checkpoint parent lineage is discontinuous")


def _write_cycle_boundary_receipt(
    seed_dir: Path,
    discovery: AdaptiveDiscovery,
    arms: Mapping[str, Sequence[ArmCheckpoint]],
) -> None:
    adaptive = arms["adaptive_two_cycle"]
    boundary_step = int(discovery.optimizer_steps_completed or 0)
    boundary = next((row for row in adaptive if row.total_step == boundary_step), None)
    proof = next((row for row in adaptive if row.total_step > boundary_step), None)
    if boundary is None:
        raise ValueError("adaptive boundary checkpoint at S is missing")
    if proof is None:
        raise ValueError(
            "adaptive freshness proof requires a checkpoint greater than S"
        )
    boundary_resource = boundary.resource_accounting
    if (
        boundary_resource.get("cycle_two_optimizer_instantiated") is not False
        or boundary_resource.get("actual_initial_exemplar_optimizer_hash") is not None
        or boundary_resource.get("fresh_optimizer_proof_status") != "not_applicable"
    ):
        raise ValueError("checkpoint at S makes an invalid Cycle 2 optimizer claim")
    proof_resource = proof.resource_accounting
    expected_fresh = proof_resource.get("expected_fresh_exemplar_optimizer_hash")
    actual_fresh = proof_resource.get("actual_initial_exemplar_optimizer_hash")
    fresh = bool(
        proof_resource.get("cycle_two_optimizer_instantiated") is True
        and proof_resource.get("fresh_optimizer_proof_status") == "proven"
        and actual_fresh is not None
        and actual_fresh == expected_fresh
        and actual_fresh != discovery.corridor_optimizer_state_hash
    )
    _write_json(
        seed_dir / "cycle_boundary_receipt.json",
        {
            "phase": "P156.4.1.1",
            "boundary_checkpoint_hash": boundary.checkpoint_hash,
            "boundary_total_step": boundary_step,
            "freshness_proof_checkpoint_hash": proof.checkpoint_hash,
            "freshness_proof_total_step": proof.total_step,
            "corridor_final_optimizer_hash": discovery.corridor_optimizer_state_hash,
            "corridor_optimizer_state_hash": discovery.corridor_optimizer_state_hash,
            "expected_fresh_exemplar_optimizer_hash": expected_fresh,
            "actual_initial_exemplar_optimizer_hash": actual_fresh,
            "actual_cycle_two_initial_optimizer_hash": actual_fresh,
            "fresh_optimizer_exact_match": fresh,
            "fresh_optimizer_confirmed": fresh,
            "fresh_optimizer_proof_status": "proven" if fresh else "failed",
            "cycle_two_optimizer_instantiated": True,
        },
    )
    if not fresh:
        raise ValueError("adaptive Cycle 2 did not use a fresh optimizer")


def _build_comparisons(
    seed: int,
    completion_step: int,
    schedule: tuple[int, ...],
    evaluations: Mapping[tuple[str, int], CheckpointEvaluation],
    *,
    metric: str,
    direction: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for step in schedule:
        for left, right in ARM_PAIRS:
            left_eval = evaluations[(left, step)]
            right_eval = evaluations[(right, step)]
            left_values = [
                float(row[metric]) for row in left_eval.teacher_forced_records
            ]
            right_values = [
                float(row[metric]) for row in right_eval.teacher_forced_records
            ]
            stats = paired_bootstrap_statistics(
                left_values,
                right_values,
                samples=bootstrap_samples,
                seed=bootstrap_seed + seed + step,
            )
            left_metric = float(left_eval.teacher_forced[metric])
            right_metric = float(right_eval.teacher_forced[metric])
            if left_metric == right_metric:
                winner = "tie"
            elif stats["result"] == "inconclusive":
                winner = "inconclusive"
            elif (left_metric < right_metric) == (direction == "lower"):
                winner = left
            else:
                winner = right
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_total_step": step,
                    "checkpoint_fraction_of_S": (step - completion_step)
                    / completion_step,
                    "arm_pair": [left, right],
                    "primary_metric": metric,
                    "left_value": left_metric,
                    "right_value": right_metric,
                    "paired_delta": left_metric - right_metric,
                    "winner": winner,
                    "paired_statistics": stats,
                }
            )
    return rows


def _validate_final_test_order(
    evaluations: Mapping[tuple[str, int], CheckpointEvaluation],
) -> None:
    hashes = {
        evaluation.final_test_record_order_hash for evaluation in evaluations.values()
    }
    if len(hashes) != 1:
        raise ValueError("final-test record ordering differs across arms")


def _metrics_with_availability(
    values: Mapping[str, Any], required: Sequence[str]
) -> dict[str, Any]:
    result = {name: values.get(name) for name in required}
    result.update(values)
    result["metric_availability"] = {
        name: "available" if result[name] is not None else "unavailable"
        for name in required
    }
    return result


def _target_costs(
    crossing: Mapping[str, Any] | None,
    resources: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    if crossing is None:
        return {
            "steps_to_target": None,
            "bytes_to_target": None,
            "records_to_target": None,
            "tokens_to_target": None,
            "teacher_bytes_to_target": None,
            "training_wall_seconds_to_target": None,
            "total_wall_seconds_to_target": None,
            "wall_clock_to_target": None,
            "target_cost_provenance": "target_not_observed",
        }
    step = int(crossing["total_step"])
    match = next(
        row
        for row in resources
        if row["seed"] == seed and row["arm"] == arm and row["total_step"] == step
    )
    teacher_bytes = int(
        match.get(
            "artifact_bytes_logically_consumed",
            match.get("teacher_artifact_bytes_consumed", 0),
        )
    )
    corridor_bytes = int(match.get("corridor_artifact_bytes_consumed", 0))
    exemplar_bytes = int(match.get("exemplar_artifact_bytes_consumed", 0))
    if corridor_bytes + exemplar_bytes > teacher_bytes:
        raise ValueError("teacher artifact byte components exceed charged total")
    return {
        "steps_to_target": step,
        "bytes_to_target": teacher_bytes,
        "records_to_target": match.get("cumulative_training_records"),
        "tokens_to_target": match.get("cumulative_training_tokens"),
        "teacher_bytes_to_target": teacher_bytes,
        "training_wall_seconds_to_target": match.get(
            "cumulative_training_wall_seconds"
        ),
        "total_wall_seconds_to_target": match.get("cumulative_total_wall_seconds"),
        "wall_clock_to_target": float(match.get("total_seconds", 0.0)),
        "target_cost_provenance": "first_observed_checkpoint_cumulative_resources",
    }


def _checkpoint_lineage(
    checkpoint: ArmCheckpoint,
    seed: int,
    shared: SharedInitialization,
    config: FullDistillationCrossoverConfig,
) -> dict[str, Any]:
    return {
        "phase": "P156.4",
        "arm": checkpoint.arm,
        "seed": seed,
        "total_optimizer_step": checkpoint.total_step,
        "corridor_optimizer_steps": checkpoint.corridor_steps,
        "exemplar_optimizer_steps": checkpoint.exemplar_steps,
        "vanilla_optimizer_steps": checkpoint.vanilla_steps,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "parent_checkpoint_hash": checkpoint.parent_checkpoint_hash,
        "student_config_hash": shared.student_config_hash,
        "artifact_hashes": {
            "training": _path_hash(config.training_artifact),
            "calibration": _path_hash(config.calibration_artifact),
            "final_test": _path_hash(config.final_test_artifact),
        },
    }


def _load_or_create_state(
    config: FullDistillationCrossoverConfig, path: Path, config_hash: str
) -> dict[str, Any]:
    if path.exists():
        if not config.resume:
            raise FileExistsError("crossover state exists; pass resume=True")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["resume_config_hash"] != config_hash:
            raise ValueError("resume config hash mismatch")
        return state
    return {
        "phase": "P156.4",
        "expected_seeds": list(config.seeds),
        "completed_adaptive_discoveries": {},
        "shared_initializations": {},
        "adaptive_discoveries": {},
        "arm_checkpoint_results": {},
        "evaluation_results": {},
        "completed_arm_checkpoints": [],
        "completed_evaluations": [],
        "failed_items": [],
        "pending_items": [],
        "resume_config_hash": config_hash,
    }


def _shared_from_dict(value: Mapping[str, Any]) -> SharedInitialization:
    return SharedInitialization(
        checkpoint=Path(value["checkpoint"]),
        parameter_tree_hash=value["parameter_tree_hash"],
        student_config_hash=value["student_config_hash"],
        checkpoint_hash=value["checkpoint_hash"],
    )


def _discovery_from_dict(value: Mapping[str, Any]) -> AdaptiveDiscovery:
    data = dict(value)
    data["checkpoint"] = Path(data["checkpoint"])
    return AdaptiveDiscovery(**data)


def _arm_checkpoint_from_dict(value: Mapping[str, Any]) -> ArmCheckpoint:
    data = dict(value)
    data["checkpoint"] = Path(data["checkpoint"])
    return ArmCheckpoint(**data)


def _evaluation_from_dict(value: Mapping[str, Any]) -> CheckpointEvaluation:
    return CheckpointEvaluation(**value)


def _render_summary(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P156.4 Full Distillation Crossover Harness",
            "",
            f"- Status: {report['status']}",
            "- Run kind: implementation_smoke",
            "- Publication grade: false",
            "- Full distillation run started: false",
            "- Winner declared: false",
            "- Checkpoint execution mode: independent_replay",
            "- Matrix wall clock is not a single-trajectory cost.",
            "- P156.5 should use continuous trajectories unless explicitly overridden.",
            "",
        ]
    )


def _path_hash(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
