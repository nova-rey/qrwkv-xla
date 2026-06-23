from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

from qrwkv_xla.artifacts import (
    FingerprintBatch,
    FingerprintTargetRecord,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.artifacts.fingerprint_loader import _records_to_batch
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorScheduler,
    AdaptiveCorridorSchedulerConfig,
    adaptive_weighted_loss,
)
from qrwkv_xla.fingerprint.corridor_measurement import (
    STAT_NAMES,
    CorridorMeasurementConfig,
    _aggressive_corridor_loss,
    _batch_to_jax,
    _create_backend,
    _evaluate_held_out,
    _jax_fingerprint_batch,
)
from qrwkv_xla.fingerprint.provenance import file_sha256, hash_checkpoint_bundle
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats_at_positions,
)
from qrwkv_xla.training.gradients import clip_gradients_by_global_norm

CalibrationOverride = Callable[[int, str, Mapping[str, float]], Mapping[str, float]]


@dataclass(frozen=True)
class AdaptiveCorridorPassConfig:
    training_fingerprint_artifact: Path
    calibration_fingerprint_artifact: Path
    output_dir: Path
    scheduler: AdaptiveCorridorSchedulerConfig
    evaluation_interval_steps: int = 1
    checkpoint_interval_steps: int = 10
    optimizer: str = "sgd"
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 1.0
    student_backend: str = "tiny_debug"
    seed: int = 0
    corridor_loss_weight: float = 1.0
    penalty_power: float = 2.0
    entropy_weight: float = 1.0
    top1_margin_weight: float = 1.0
    top8_mass_weight: float = 1.0
    top32_mass_weight: float = 1.0
    tail_mass_weight: float = 1.0
    worst_stat_boost: float = 1.0
    distance_normalization: str = "none"
    parameter_norm_limit: float = 1e6
    gradient_norm_hard_limit: float = 1e6
    resume_checkpoint: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class AdaptiveCorridorPassResult:
    status: str
    cycle_one_complete: bool
    output_dir: Path
    report_path: Path
    final_checkpoint: Path
    interruption_checkpoint: Path | None = None


def run_adaptive_corridor_pass(
    config: AdaptiveCorridorPassConfig,
    *,
    calibration_override: CalibrationOverride | None = None,
    interrupt_after_optimizer_step: int | None = None,
) -> AdaptiveCorridorPassResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runner_config_sha256 = _stable_hash(_semantic_config(config))
    train_records = tuple(
        load_fingerprint_targets(
            config.training_fingerprint_artifact, batch_size=1
        ).iter_records()
    )
    calibration_records = tuple(
        load_fingerprint_targets(
            config.calibration_fingerprint_artifact, batch_size=1
        ).iter_records()
    )
    train_batches = _group_batches(train_records)
    calibration_groups = _group_records(calibration_records)
    expected_modes = set(config.scheduler.controller.all_modes)
    if set(train_batches) != expected_modes:
        raise ValueError("training artifact modes do not match scheduler modes")
    if set(calibration_groups) != expected_modes:
        raise ValueError("calibration artifact modes do not match scheduler modes")

    summary = summarize_fingerprint_artifact(config.training_fingerprint_artifact)
    measurement_config = _measurement_config(config)
    backend, student_config = _create_backend(measurement_config, summary)
    optimizer_config = OptimizerConfig(
        type=config.optimizer, learning_rate=config.learning_rate
    )
    resumed = config.resume_checkpoint is not None
    if resumed:
        loaded = load_checkpoint(config.resume_checkpoint)
        if loaded.manifest.student_architecture != config.student_backend:
            raise ValueError("adaptive resume student backend mismatch")
        if loaded.manifest.student_config != student_config:
            raise ValueError("adaptive resume student config mismatch")
        if loaded.optimizer_state is None:
            raise ValueError("adaptive resume checkpoint lacks optimizer state")
        sidecar = _read_json(config.resume_checkpoint / "adaptive_state.json")
        if sidecar["runner_config_sha256"] != runner_config_sha256:
            raise ValueError("adaptive runner config hash mismatch on resume")
        scheduler = AdaptiveCorridorScheduler.from_dict(sidecar["scheduler_state"])
        state = TrainState(
            params=loaded.params,
            step=loaded.manifest.step,
            learning_rate=config.learning_rate,
            optimizer_state=loaded.optimizer_state,
        )
    else:
        params = backend.init_params(jax.random.PRNGKey(config.seed))
        state = TrainState(
            params=params,
            step=0,
            learning_rate=config.learning_rate,
            optimizer_state=init_optimizer_state(params, optimizer_config),
        )
        scheduler = AdaptiveCorridorScheduler(config.scheduler)

    train_steps: dict[tuple[str, ...], Any] = {}
    interruption_checkpoint: Path | None = None
    if scheduler.calibration_evaluations_completed == 0:
        observations = _evaluate_all_modes(
            backend,
            state.params,
            calibration_groups,
            step=0,
            calibration_override=calibration_override,
        )
        events = scheduler.observe_calibration(step=0, observations=observations)
        if events:
            _save_adaptive_checkpoint(
                config,
                state,
                scheduler,
                student_config,
                runner_config_sha256,
                config.output_dir / "checkpoints" / "step_000000_event",
            )

    stop_reason: str | None = None
    while not scheduler.cycle_one_complete:
        if state.step >= config.scheduler.controller.maximum_corridor_steps:
            stop_reason = "maximum_step_cap"
            break
        active_mode_ids = tuple(scheduler.active_mode_ids)
        if active_mode_ids:
            next_step = int(state.step) + 1
            weights = scheduler.record_optimizer_step(next_step)
            if active_mode_ids not in train_steps:
                train_steps[active_mode_ids] = _make_adaptive_train_step(
                    backend,
                    {mode_id: train_batches[mode_id] for mode_id in active_mode_ids},
                    {mode_id: weights[mode_id] for mode_id in active_mode_ids},
                    optimizer_config=optimizer_config,
                    max_grad_norm=config.max_grad_norm,
                    measurement_config=measurement_config,
                )
            state, metrics = train_steps[active_mode_ids](state)
            jax.block_until_ready(state.params)
            loss = float(metrics["loss"])
            if not math.isfinite(loss):
                raise ValueError("adaptive corridor loss is non-finite")
            should_evaluate = int(state.step) % config.evaluation_interval_steps == 0
        else:
            should_evaluate = True

        events: list[dict[str, Any]] = []
        if should_evaluate:
            evaluation_step = (
                int(scheduler.controller.current_step or 0)
                + config.evaluation_interval_steps
            )
            observations = _evaluate_all_modes(
                backend,
                state.params,
                calibration_groups,
                step=evaluation_step,
                calibration_override=calibration_override,
            )
            events = scheduler.observe_calibration(
                step=evaluation_step, observations=observations
            )

        should_checkpoint = bool(events) or (
            active_mode_ids and int(state.step) % config.checkpoint_interval_steps == 0
        )
        if should_checkpoint:
            suffix = "event" if events else "periodic"
            _save_adaptive_checkpoint(
                config,
                state,
                scheduler,
                student_config,
                runner_config_sha256,
                config.output_dir
                / "checkpoints"
                / f"step_{int(state.step):06d}_{suffix}",
            )
        if (
            interrupt_after_optimizer_step is not None
            and int(state.step) >= interrupt_after_optimizer_step
        ):
            interruption_checkpoint = (
                config.output_dir
                / "checkpoints"
                / f"step_{int(state.step):06d}_interrupt"
            )
            _save_adaptive_checkpoint(
                config,
                state,
                scheduler,
                student_config,
                runner_config_sha256,
                interruption_checkpoint,
            )
            stop_reason = "interrupted"
            break

    if scheduler.cycle_one_complete:
        stop_reason = "all_required_modes_stably_frozen"
    final_checkpoint = (
        config.output_dir
        / "checkpoints"
        / (
            "adaptive_corridor_final_checkpoint"
            if scheduler.cycle_one_complete
            else "adaptive_corridor_latest_checkpoint"
        )
    )
    _save_adaptive_checkpoint(
        config,
        state,
        scheduler,
        student_config,
        runner_config_sha256,
        final_checkpoint,
    )
    report = _build_report(
        config,
        scheduler,
        final_checkpoint,
        stop_reason=stop_reason,
        resumed=resumed,
        runner_config_sha256=runner_config_sha256,
    )
    _write_outputs(config, scheduler, report, resumed=resumed)
    return AdaptiveCorridorPassResult(
        status=report["status"],
        cycle_one_complete=scheduler.cycle_one_complete,
        output_dir=config.output_dir,
        report_path=config.output_dir / "adaptive_corridor_report.json",
        final_checkpoint=final_checkpoint,
        interruption_checkpoint=interruption_checkpoint,
    )


def write_resume_equivalence_receipt(
    output_dir: Path,
    *,
    equivalent: bool,
    comparison: Mapping[str, Any],
) -> None:
    report_path = output_dir / "adaptive_corridor_report.json"
    report = _read_json(report_path)
    report["resume_equivalent"] = bool(equivalent)
    _write_json(report_path, report)
    _write_json(
        output_dir / "adaptive_corridor_resume_receipt.json",
        {
            "phase": "P156.3",
            "resume_equivalent": bool(equivalent),
            "comparison": dict(comparison),
        },
    )


def _make_adaptive_train_step(
    backend: Any,
    mode_batches: Mapping[str, FingerprintBatch],
    weights: Mapping[str, float],
    *,
    optimizer_config: OptimizerConfig,
    max_grad_norm: float | None,
    measurement_config: CorridorMeasurementConfig,
):
    jax_batches = {
        mode_id: _batch_to_jax(batch) for mode_id, batch in mode_batches.items()
    }

    def train_step(state: TrainState) -> tuple[TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any) -> jax.Array:
            losses = {}
            for mode_id, batch in jax_batches.items():
                output, _ = backend.forward_full(params, batch["input_ids"])
                stats = compute_fingerprint_distribution_stats_at_positions(
                    backend.logits(output), batch["position"]
                )
                mode_loss, _ = _aggressive_corridor_loss(
                    stats, _jax_fingerprint_batch(batch), measurement_config
                )
                losses[mode_id] = mode_loss
            return adaptive_weighted_loss(losses, weights)

        loss, gradients = jax.value_and_grad(loss_fn)(state.params)
        clipped = clip_gradients_by_global_norm(gradients, max_grad_norm=max_grad_norm)
        update_config = OptimizerConfig(
            type=optimizer_config.type,
            learning_rate=state.learning_rate,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        params, optimizer_state, _ = optimizer_update(
            state.params, clipped.gradients, state.optimizer_state, update_config
        )
        return (
            TrainState(
                params=params,
                step=state.step + 1,
                learning_rate=state.learning_rate,
                optimizer_state=optimizer_state,
            ),
            {"loss": loss, "grad_global_norm": clipped.global_norm},
        )

    return jax.jit(train_step)


def _evaluate_all_modes(
    backend: Any,
    params: Any,
    groups: Mapping[str, tuple[FingerprintTargetRecord, ...]],
    *,
    step: int,
    calibration_override: CalibrationOverride | None,
) -> dict[str, dict[str, float]]:
    observations = {}
    for mode_id, records in sorted(groups.items()):
        raw = _evaluate_held_out(backend, params, records)
        metrics = {
            "corridor_loss": float(raw["corridor_loss"]),
            "inside_corridor_rate": float(raw["inside_all_rate"]),
            "mean_distance_outside_corridor": float(
                raw["mean_distance_outside_corridor"]
            ),
            "worst_stat_violation": max(
                float(raw[f"{name}_distance"]) for name in STAT_NAMES
            ),
        }
        if calibration_override is not None:
            metrics = {
                name: float(value)
                for name, value in calibration_override(step, mode_id, metrics).items()
            }
        observations[mode_id] = metrics
    return observations


def _group_records(
    records: tuple[FingerprintTargetRecord, ...],
) -> dict[str, tuple[FingerprintTargetRecord, ...]]:
    grouped: dict[str, list[FingerprintTargetRecord]] = {}
    for record in records:
        grouped.setdefault(str(record.mode_id), []).append(record)
    return {mode_id: tuple(values) for mode_id, values in sorted(grouped.items())}


def _group_batches(
    records: tuple[FingerprintTargetRecord, ...],
) -> dict[str, FingerprintBatch]:
    if not records:
        raise ValueError("adaptive corridor artifact contains no records")
    max_seq_len = len(records[0].input_ids)
    return {
        mode_id: _records_to_batch(values, max_seq_len=max_seq_len)
        for mode_id, values in _group_records(records).items()
    }


def _measurement_config(
    config: AdaptiveCorridorPassConfig,
) -> CorridorMeasurementConfig:
    return CorridorMeasurementConfig(
        fingerprint_artifact=config.training_fingerprint_artifact,
        held_out_fingerprint_artifact=config.calibration_fingerprint_artifact,
        source_texts=config.training_fingerprint_artifact / "manifest.json",
        output_dir=config.output_dir,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        student_backend=config.student_backend,
        corridor_loss_weight=config.corridor_loss_weight,
        penalty_power=config.penalty_power,
        entropy_weight=config.entropy_weight,
        top1_margin_weight=config.top1_margin_weight,
        top8_mass_weight=config.top8_mass_weight,
        top32_mass_weight=config.top32_mass_weight,
        tail_mass_weight=config.tail_mass_weight,
        worst_stat_boost=config.worst_stat_boost,
        distance_normalization=config.distance_normalization,
        parameter_norm_limit=config.parameter_norm_limit,
        gradient_norm_hard_limit=config.gradient_norm_hard_limit,
        overwrite=config.overwrite,
    )


def _save_adaptive_checkpoint(
    config: AdaptiveCorridorPassConfig,
    state: TrainState,
    scheduler: AdaptiveCorridorScheduler,
    student_config: Mapping[str, Any],
    runner_config_sha256: str,
    path: Path,
) -> None:
    calibration_hash = _stable_hash(scheduler.calibration_trajectory)
    transition_hash = _stable_hash(scheduler.transition_events)
    save_checkpoint(
        path,
        state.params,
        student_architecture=config.student_backend,
        student_config=student_config,
        step=int(state.step),
        learning_rate=config.learning_rate,
        loss_config={
            "kind": "adaptive_fingerprint_corridor",
            "cycle": 1,
            "frozen_modes_have_zero_direct_loss_only": True,
        },
        target_manifest={
            "training_artifact_manifest_sha256": file_sha256(
                config.training_fingerprint_artifact / "manifest.json"
            ),
            "calibration_artifact_manifest_sha256": file_sha256(
                config.calibration_fingerprint_artifact / "manifest.json"
            ),
            "controller_config_sha256": scheduler.controller.config_sha256,
            "scheduler_config_sha256": scheduler.config_sha256,
            "runner_config_sha256": runner_config_sha256,
            "active_mode_ids": scheduler.active_mode_ids,
            "normalized_active_weights": scheduler.normalized_weights,
            "calibration_trajectory_sha256": calibration_hash,
            "transition_log_sha256": transition_hash,
        },
        optimizer_config=asdict(
            OptimizerConfig(type=config.optimizer, learning_rate=config.learning_rate)
        ),
        optimizer_state=state.optimizer_state,
        gradients={"max_grad_norm": config.max_grad_norm},
        notes=["P156.3 adaptive corridor checkpoint"],
        overwrite=config.overwrite,
    )
    _write_json(
        path / "adaptive_state.json",
        {
            "phase": "P156.3",
            "runner_config_sha256": runner_config_sha256,
            "scheduler_state": scheduler.to_dict(),
            "calibration_trajectory_sha256": calibration_hash,
            "transition_log_sha256": transition_hash,
        },
    )


def _build_report(
    config: AdaptiveCorridorPassConfig,
    scheduler: AdaptiveCorridorScheduler,
    final_checkpoint: Path,
    *,
    stop_reason: str | None,
    resumed: bool,
    runner_config_sha256: str,
) -> dict[str, Any]:
    checkpoint_hash = hash_checkpoint_bundle(final_checkpoint)[
        "checkpoint_bundle_sha256"
    ]
    return {
        "phase": "P156.3",
        "status": "pass" if scheduler.cycle_one_complete else "incomplete",
        "cycle_one_complete": scheduler.cycle_one_complete,
        "global_completion_step": scheduler.global_completion_step,
        "global_completion_reason": stop_reason,
        "required_mode_count": len(config.scheduler.controller.required_modes),
        "frozen_mode_count": len(scheduler.controller.frozen_mode_ids),
        "reactivation_count": scheduler.total_reactivations,
        "maximum_active_mode_count": scheduler.maximum_active_mode_count,
        "minimum_active_mode_count": scheduler.minimum_active_mode_count,
        "optimizer_steps_completed": scheduler.optimizer_steps_completed,
        "calibration_evaluations_completed": (
            scheduler.calibration_evaluations_completed
        ),
        "controller_config_sha256": scheduler.controller.config_sha256,
        "scheduler_config_sha256": scheduler.config_sha256,
        "runner_config_sha256": runner_config_sha256,
        "final_checkpoint_sha256": checkpoint_hash,
        "resumed": resumed,
        "resume_equivalent": None,
        "full_mode_step_equivalents": scheduler.full_mode_step_equivalents,
        "actual_active_mode_step_equivalents": (
            scheduler.actual_active_mode_step_equivalents
        ),
        "frozen_mode_step_equivalents_saved": (
            scheduler.frozen_mode_step_equivalents_saved
        ),
        "fraction_mode_work_saved": scheduler.fraction_mode_work_saved,
        "shared_parameter_freezing_semantics": (
            "frozen modes have zero direct loss; shared parameters remain trainable"
        ),
        "exemplar_training_launched": False,
        "modes": {
            mode_id: {
                "mode_id": mode_id,
                "required": state.required,
                "freeze_step": state.freeze_step,
                "reactivation_steps": scheduler.accounting[mode_id].reactivation_steps,
                "refreeze_steps": scheduler.accounting[mode_id].refreeze_steps,
                "final_state": state.state.value,
                "training_steps_while_active": scheduler.accounting[
                    mode_id
                ].training_steps_while_active,
                "training_steps_while_frozen": scheduler.accounting[
                    mode_id
                ].training_steps_while_frozen,
                "direct_loss_contribution_steps": scheduler.accounting[
                    mode_id
                ].direct_loss_contribution_steps,
            }
            for mode_id, state in sorted(scheduler.controller.modes.items())
        },
    }


def _write_outputs(
    config: AdaptiveCorridorPassConfig,
    scheduler: AdaptiveCorridorScheduler,
    report: Mapping[str, Any],
    *,
    resumed: bool,
) -> None:
    _write_json(config.output_dir / "adaptive_corridor_report.json", report)
    _write_jsonl(
        config.output_dir / "adaptive_corridor_transitions.jsonl",
        scheduler.transition_events,
    )
    _write_jsonl(
        config.output_dir / "adaptive_corridor_calibration_trajectory.jsonl",
        scheduler.calibration_trajectory,
    )
    _write_jsonl(
        config.output_dir / "adaptive_corridor_weight_trajectory.jsonl",
        scheduler.weight_trajectory,
    )
    _write_json(
        config.output_dir / "adaptive_corridor_checkpoint_lineage.json",
        {
            "phase": "P156.3",
            "controller_config_sha256": scheduler.controller.config_sha256,
            "scheduler_config_sha256": scheduler.config_sha256,
            "calibration_trajectory_sha256": _stable_hash(
                scheduler.calibration_trajectory
            ),
            "transition_log_sha256": _stable_hash(scheduler.transition_events),
            "training_artifact_manifest_sha256": file_sha256(
                config.training_fingerprint_artifact / "manifest.json"
            ),
            "calibration_artifact_manifest_sha256": file_sha256(
                config.calibration_fingerprint_artifact / "manifest.json"
            ),
        },
    )
    if not (config.output_dir / "adaptive_corridor_resume_receipt.json").exists():
        _write_json(
            config.output_dir / "adaptive_corridor_resume_receipt.json",
            {
                "phase": "P156.3",
                "resumed": resumed,
                "resume_equivalent": None,
            },
        )
    summary = "\n".join(
        [
            "# P156.3 Adaptive Corridor Summary",
            "",
            f"- Status: {report['status']}",
            f"- Cycle 1 complete: {report['cycle_one_complete']}",
            f"- Optimizer steps: {report['optimizer_steps_completed']}",
            f"- Reactivations: {report['reactivation_count']}",
            "- Direct mode-work fraction saved: "
            f"{report['fraction_mode_work_saved']:.6f}",
            "- Frozen modes remove direct loss only; parameters remain shared.",
            "- Exemplar training launched: false",
            "",
        ]
    )
    (config.output_dir / "adaptive_corridor_summary.md").write_text(
        summary, encoding="utf-8"
    )


def _semantic_config(config: AdaptiveCorridorPassConfig) -> dict[str, Any]:
    return {
        "training_fingerprint_artifact": str(
            config.training_fingerprint_artifact.resolve()
        ),
        "calibration_fingerprint_artifact": str(
            config.calibration_fingerprint_artifact.resolve()
        ),
        "scheduler": config.scheduler.to_dict(),
        "evaluation_interval_steps": config.evaluation_interval_steps,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "student_backend": config.student_backend,
        "seed": config.seed,
        "corridor_loss_weight": config.corridor_loss_weight,
        "penalty_power": config.penalty_power,
        "entropy_weight": config.entropy_weight,
        "top1_margin_weight": config.top1_margin_weight,
        "top8_mass_weight": config.top8_mass_weight,
        "top32_mass_weight": config.top32_mass_weight,
        "tail_mass_weight": config.tail_mass_weight,
        "worst_stat_boost": config.worst_stat_boost,
        "distance_normalization": config.distance_normalization,
        "parameter_norm_limit": config.parameter_norm_limit,
        "gradient_norm_hard_limit": config.gradient_norm_hard_limit,
    }


def _validate_config(config: AdaptiveCorridorPassConfig) -> None:
    if config.evaluation_interval_steps <= 0:
        raise ValueError("evaluation_interval_steps must be > 0")
    if (
        config.evaluation_interval_steps
        != config.scheduler.controller.evaluation_interval_steps
    ):
        raise ValueError("runner and controller evaluation intervals must match")
    if config.checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be > 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.resume_checkpoint is not None and not config.resume_checkpoint.is_dir():
        raise ValueError("resume checkpoint does not exist")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
