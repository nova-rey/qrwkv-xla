from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.distill.config import DistillStageConfig, validate_distill_stage_config
from qrwkv_xla.distill.losses import compute_distill_loss
from qrwkv_xla.distill.metrics import metrics_to_floats
from qrwkv_xla.optimizers import OptimizerState, init_optimizer_state
from qrwkv_xla.schedules import learning_rate_at_step
from qrwkv_xla.students import create_student
from qrwkv_xla.targets import read_manifest
from qrwkv_xla.targets.store import manifest_path
from qrwkv_xla.tracking import (
    MetricsLogger,
    create_run_context,
    get_environment_metadata,
    get_git_metadata,
    write_run_metadata,
    write_run_summary,
)
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.trainers.step import batch_to_jax, make_train_step


@dataclass(frozen=True)
class DistillStageResult:
    stage: int
    student_architecture: str
    steps: int
    initial_loss: float
    final_loss: float
    final_hidden_mse: float | None = None
    final_logits_kl: float | None = None
    final_attention_or_mixer: float | None = None
    target_bundle: Path | None = None
    checkpoint_out: Path | None = None
    resume_from: Path | None = None
    start_step: int = 0
    end_step: int = 0
    run_dir: Path | None = None
    metrics_path: Path | None = None
    summary_path: Path | None = None
    notes: tuple[str, ...] = ()
    lr_schedule_type: str = "constant"
    initial_learning_rate: float | None = None
    final_learning_rate: float | None = None
    final_grad_global_norm: float | None = None
    final_grad_clipped_global_norm: float | None = None
    final_grad_clip_scale: float | None = None


@dataclass(frozen=True)
class _TrainLoss:
    total: jax.Array
    components: dict[str, jax.Array]


def run_distill_stage(config: DistillStageConfig) -> DistillStageResult:
    validate_distill_stage_config(config)
    dataset = TargetBundleDataset.from_path(config.targets_dir)
    manifest = read_manifest(manifest_path(config.targets_dir))
    logits_kl_enabled = (
        config.losses.logits_kl.enabled and config.losses.logits_kl.weight > 0
    )
    attention_or_mixer_enabled = (
        config.losses.attention_or_mixer.enabled
        and config.losses.attention_or_mixer.weight > 0
    )
    if logits_kl_enabled and not manifest.targets.logits:
        raise ValueError(
            "logits_kl is enabled but teacher targets do not include logits"
        )
    if attention_or_mixer_enabled and not manifest.targets.attention_targets:
        raise ValueError(
            "attention_or_mixer is enabled but teacher targets do not include "
            "attention_targets"
        )

    hidden_size = config.student.hidden_size
    if hidden_size is None:
        hidden_size = manifest.hidden_size
    elif hidden_size != manifest.hidden_size:
        raise ValueError(
            "student.hidden_size "
            f"{hidden_size} does not match manifest hidden_size "
            f"{manifest.hidden_size}"
        )

    num_layers = config.student.num_layers
    if num_layers is None:
        num_layers = manifest.num_layers
    elif num_layers != manifest.num_layers:
        raise ValueError(
            "student.num_layers "
            f"{num_layers} does not match manifest num_layers "
            f"{manifest.num_layers}"
        )

    student = create_student(
        config.student.architecture,
        vocab_size=config.student.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        emit_logits=config.student.emit_logits,
        tie_embeddings=config.student.tie_embeddings,
        emit_mixer_outputs=(
            config.student.emit_mixer_outputs or attention_or_mixer_enabled
        ),
    )
    student_config = {
        "architecture": config.student.architecture,
        "vocab_size": config.student.vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "emit_logits": config.student.emit_logits,
        "tie_embeddings": config.student.tie_embeddings,
        "emit_mixer_outputs": (
            config.student.emit_mixer_outputs or attention_or_mixer_enabled
        ),
    }

    optimizer_config = config.optimizer.to_optimizer_config()
    train_step = make_train_step(
        student.apply,
        distillation_loss=_make_train_loss(config),
        optimizer_config=optimizer_config,
        max_grad_norm=config.gradients.max_grad_norm,
        clip_epsilon=config.gradients.clip_epsilon,
    )
    start_step = 0
    params = student.init_params(jax.random.PRNGKey(config.training.seed))
    optimizer_state = init_optimizer_state(params, optimizer_config)
    parent_checkpoint_manifest: dict[str, object] | None = None
    resume_notes: list[str] = []
    if config.checkpoint.resume_from is not None:
        loaded = load_checkpoint(config.checkpoint.resume_from)
        _validate_resume_checkpoint(
            loaded.manifest.student_architecture,
            loaded.manifest.student_config,
            expected_architecture=config.student.architecture,
            expected_student_config=student_config,
        )
        start_step = loaded.manifest.step
        params, merge_notes = _merge_checkpoint_params_for_resume(
            fresh_params=params,
            checkpoint_params=loaded.params,
            allow_missing_lm_head=config.student.emit_logits
            and not bool(loaded.manifest.student_config.get("emit_logits", False)),
        )
        resume_notes.extend(merge_notes)
        optimizer_state = _optimizer_state_for_resume(
            fresh_optimizer_state=init_optimizer_state(params, optimizer_config),
            checkpoint_optimizer_state=loaded.optimizer_state,
            optimizer_type=optimizer_config.type,
            checkpoint_step=start_step,
            allow_missing_lm_head=config.student.emit_logits
            and not bool(loaded.manifest.student_config.get("emit_logits", False)),
            notes=resume_notes,
        )
        parent_checkpoint_manifest = asdict(loaded.manifest)
        schedule_notes = _resume_schedule_notes(
            checkpoint_schedule=loaded.manifest.lr_schedule,
            current_schedule=_lr_schedule_metadata(
                config=config,
                step=start_step,
            ),
        )
        resume_notes.extend(schedule_notes)

    state = TrainState(
        params=params,
        step=start_step,
        learning_rate=config.optimizer.learning_rate,
        optimizer_state=optimizer_state,
    )

    shard_batches = [batch_to_jax(batch) for batch in dataset.iter_shards()]
    if not shard_batches:
        raise ValueError(f"Target bundle contains no shards: {dataset.bundle_dir}")

    checkpoint_out = config.checkpoint.checkpoint_out
    checkpoint_overwrite = config.checkpoint.overwrite
    run_context = None
    metrics_logger: MetricsLogger | None = None
    if config.tracking.enabled:
        command = list(sys.argv)
        git_metadata = get_git_metadata(Path(__file__).resolve().parents[3])
        environment_metadata = get_environment_metadata()
        checkpoint_metadata = {
            "checkpoint_out": checkpoint_out,
            "resume_from": config.checkpoint.resume_from,
            "overwrite": checkpoint_overwrite,
            "parent_checkpoint_manifest": parent_checkpoint_manifest,
        }
        teacher_target = {
            "targets_dir": config.targets_dir,
            "manifest": asdict(manifest),
        }
        if checkpoint_out is None:
            checkpoint_metadata["checkpoint_out"] = "runs/<run_id>/checkpoints/final"
        run_context = create_run_context(
            run_root=config.tracking.run_root or Path("runs"),
            stage=config.stage,
            student_architecture=config.student.architecture,
            run_name=config.tracking.run_name,
            command=command,
            git=git_metadata,
            environment=environment_metadata,
            distillation={
                "stage": config.stage,
                "targets_dir": config.targets_dir,
                "optimizer": asdict(config.optimizer),
                "lr_schedule": _lr_schedule_metadata(
                    config=config,
                    step=start_step,
                    include_step=False,
                ),
                "gradients": asdict(config.gradients),
                "training": asdict(config.training),
                "losses": asdict(config.losses),
            },
            teacher_target=teacher_target,
            student=student_config,
            checkpoint=checkpoint_metadata,
            tags=config.tracking.tags,
            notes=[*config.tracking.notes, *resume_notes],
            overwrite=config.tracking.overwrite,
        )
        write_run_metadata(run_context)
        metrics_logger = MetricsLogger(run_context.paths.metrics_jsonl)
        if checkpoint_out is None:
            checkpoint_out = run_context.paths.default_final_checkpoint
            checkpoint_overwrite = config.tracking.overwrite

    initial_loss: float | None = None
    initial_learning_rate: float | None = None
    final_learning_rate: float | None = None
    final_metrics: dict[str, float] | None = None
    with metrics_logger or _NullMetricsLogger() as active_logger:
        for step_index in range(config.training.max_steps):
            batch = shard_batches[step_index % len(shard_batches)]
            global_step = start_step + step_index
            scheduled_lr = learning_rate_at_step(
                step=global_step,
                base_learning_rate=config.optimizer.learning_rate,
                config=config.lr_schedule,
            )
            if initial_learning_rate is None:
                initial_learning_rate = scheduled_lr
            final_learning_rate = scheduled_lr
            state = state._replace(learning_rate=scheduled_lr)
            state, metrics = train_step(state, batch)
            float_metrics = metrics_to_floats(metrics)
            float_metrics["learning_rate"] = scheduled_lr
            float_metrics["base_learning_rate"] = config.optimizer.learning_rate
            float_metrics["global_step"] = float(global_step)
            float_metrics["local_step"] = float(step_index)
            if initial_loss is None:
                initial_loss = float_metrics["loss"]
            final_metrics = float_metrics
            if metrics_logger is not None:
                active_logger.log(
                    step=state.step,
                    values=float_metrics,
                    phase="train",
                    extra={
                        "local_step": step_index,
                        "global_step": global_step,
                        "start_step": start_step,
                        "shard_index": step_index % len(shard_batches),
                        "optimizer_type": config.optimizer.type,
                        "lr_schedule_type": config.lr_schedule.type,
                    },
                )

    assert initial_loss is not None
    assert initial_learning_rate is not None
    assert final_learning_rate is not None
    assert final_metrics is not None
    if checkpoint_out is not None:
        save_checkpoint(
            checkpoint_out,
            state.params,
            student_architecture=config.student.architecture,
            student_config=student_config,
            step=state.step,
            learning_rate=config.optimizer.learning_rate,
            loss_config=asdict(config.losses),
            target_manifest=manifest,
            optimizer_config=asdict(config.optimizer),
            optimizer_state=state.optimizer_state,
            lr_schedule=_lr_schedule_metadata(
                config=config,
                step=state.step,
            ),
            gradients=asdict(config.gradients),
            notes=[
                "simple JSON + NPZ checkpoint",
                f"distillation stage {config.stage}",
                f"optimizer {config.optimizer.type}",
                *resume_notes,
            ],
            overwrite=checkpoint_overwrite,
        )
    if run_context is not None:
        write_run_summary(
            context=run_context,
            summary={
                "status": "completed",
                "started_at_utc": run_context.metadata.created_at_utc,
                "finished_at_utc": _utc_now(),
                "run_id": run_context.metadata.run_id,
                "stage": config.stage,
                "student_architecture": config.student.architecture,
                "steps": config.training.max_steps,
                "start_step": start_step,
                "end_step": state.step,
                "initial_loss": initial_loss,
                "final_loss": final_metrics["loss"],
                "final_hidden_mse": final_metrics.get("hidden_mse"),
                "final_logits_kl": final_metrics.get("logits_kl"),
                "final_attention_or_mixer": final_metrics.get("attention_or_mixer"),
                "optimizer_type": config.optimizer.type,
                "learning_rate": final_learning_rate,
                "base_learning_rate": config.optimizer.learning_rate,
                "initial_learning_rate": initial_learning_rate,
                "final_learning_rate": final_learning_rate,
                "lr_schedule": _lr_schedule_metadata(
                    config=config,
                    step=state.step,
                ),
                "lr_schedule_type": config.lr_schedule.type,
                "gradients": asdict(config.gradients),
                "final_grad_global_norm": final_metrics.get("grad_global_norm"),
                "final_grad_clipped_global_norm": final_metrics.get(
                    "grad_clipped_global_norm"
                ),
                "final_grad_clip_scale": final_metrics.get("grad_clip_scale"),
                "checkpoint_out": checkpoint_out,
                "resume_from": config.checkpoint.resume_from,
                "target_bundle": dataset.bundle_dir,
                "notes": resume_notes,
            },
        )
    return DistillStageResult(
        stage=config.stage,
        student_architecture=config.student.architecture,
        steps=config.training.max_steps,
        initial_loss=initial_loss,
        final_loss=final_metrics["loss"],
        final_hidden_mse=final_metrics.get("hidden_mse"),
        final_logits_kl=final_metrics.get("logits_kl"),
        final_attention_or_mixer=final_metrics.get("attention_or_mixer"),
        target_bundle=dataset.bundle_dir,
        checkpoint_out=checkpoint_out,
        resume_from=config.checkpoint.resume_from,
        start_step=start_step,
        end_step=state.step,
        run_dir=run_context.paths.run_dir if run_context is not None else None,
        metrics_path=(
            run_context.paths.metrics_jsonl if run_context is not None else None
        ),
        summary_path=(
            run_context.paths.summary_json if run_context is not None else None
        ),
        notes=tuple(resume_notes),
        lr_schedule_type=config.lr_schedule.type,
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        final_grad_global_norm=final_metrics.get("grad_global_norm"),
        final_grad_clipped_global_norm=final_metrics.get("grad_clipped_global_norm"),
        final_grad_clip_scale=final_metrics.get("grad_clip_scale"),
    )


def _make_train_loss(config: DistillStageConfig):
    def loss_fn(student_output, batch):
        breakdown = compute_distill_loss(
            student_output=student_output,
            teacher_hidden_states=batch["hidden_states"],
            teacher_logits=batch.get("logits"),
            attention_mask=batch.get("attention_mask"),
            loss_mask=batch.get("loss_mask"),
            loss_config=config.losses,
            teacher_attention_targets=batch.get("attention_targets"),
        )
        components = {
            "loss": breakdown.total,
        }
        if breakdown.hidden_mse is not None:
            components["hidden_mse"] = breakdown.hidden_mse
        if breakdown.logits_kl is not None:
            components["logits_kl"] = breakdown.logits_kl
        if breakdown.attention_or_mixer is not None:
            components["attention_or_mixer"] = breakdown.attention_or_mixer
            components["attention_or_mixer_mse"] = breakdown.attention_or_mixer
        return _TrainLoss(total=breakdown.total, components=components)

    return loss_fn


def _validate_resume_checkpoint(
    checkpoint_architecture: str,
    checkpoint_student_config: dict[str, object],
    *,
    expected_architecture: str,
    expected_student_config: dict[str, bool | int | str],
) -> None:
    if checkpoint_architecture != expected_architecture:
        raise ValueError(
            "checkpoint student architecture mismatch: "
            f"{checkpoint_architecture!r} != {expected_architecture!r}"
        )
    for name in ("hidden_size", "num_layers", "vocab_size"):
        checkpoint_value = checkpoint_student_config.get(name)
        expected_value = expected_student_config[name]
        if checkpoint_value != expected_value:
            raise ValueError(
                f"checkpoint student {name} mismatch: "
                f"{checkpoint_value!r} != {expected_value!r}"
            )
    for name in ("emit_logits", "tie_embeddings", "emit_mixer_outputs"):
        checkpoint_value = bool(checkpoint_student_config.get(name, False))
        expected_value = bool(expected_student_config[name])
        if checkpoint_value == expected_value:
            continue
        if name in {"emit_logits", "emit_mixer_outputs"} and expected_value:
            continue
        raise ValueError(
            f"checkpoint student {name} mismatch: "
            f"{checkpoint_value!r} != {expected_value!r}"
        )


def _merge_checkpoint_params_for_resume(
    *,
    fresh_params: Any,
    checkpoint_params: Any,
    allow_missing_lm_head: bool,
) -> tuple[Any, list[str]]:
    notes: list[str] = []
    merged = _merge_param_tree(
        fresh_params=fresh_params,
        checkpoint_params=checkpoint_params,
        path=(),
        allow_missing_lm_head=allow_missing_lm_head,
        notes=notes,
    )
    return merged, notes


def _merge_param_tree(
    *,
    fresh_params: Any,
    checkpoint_params: Any,
    path: tuple[str, ...],
    allow_missing_lm_head: bool,
    notes: list[str],
) -> Any:
    if isinstance(fresh_params, dict):
        if not isinstance(checkpoint_params, dict):
            raise ValueError(
                f"checkpoint params at {_format_param_path(path)} must be a dict"
            )
        merged = {}
        for key, fresh_child in fresh_params.items():
            child_path = (*path, str(key))
            if key in checkpoint_params:
                merged[key] = _merge_param_tree(
                    fresh_params=fresh_child,
                    checkpoint_params=checkpoint_params[key],
                    path=child_path,
                    allow_missing_lm_head=allow_missing_lm_head,
                    notes=notes,
                )
            elif allow_missing_lm_head and key in {"lm_head", "lm_head_bias"}:
                merged[key] = fresh_child
                notes.append(
                    "initialized missing LM head params during hidden-only to "
                    "logits resume"
                )
            else:
                raise ValueError(
                    "checkpoint params are missing required path "
                    f"{_format_param_path(child_path)}"
                )
        extra_keys = set(checkpoint_params) - set(fresh_params)
        if extra_keys:
            extra = ", ".join(sorted(str(key) for key in extra_keys))
            raise ValueError(
                f"checkpoint params contain unexpected paths under "
                f"{_format_param_path(path)}: {extra}"
            )
        return merged

    fresh_array = np.asarray(fresh_params)
    checkpoint_array = np.asarray(checkpoint_params)
    if fresh_array.shape != checkpoint_array.shape:
        raise ValueError(
            f"checkpoint param {_format_param_path(path)} shape mismatch: "
            f"{checkpoint_array.shape} != {fresh_array.shape}"
        )
    return checkpoint_params


def _optimizer_state_for_resume(
    *,
    fresh_optimizer_state: OptimizerState,
    checkpoint_optimizer_state: OptimizerState | None,
    optimizer_type: str,
    checkpoint_step: int,
    allow_missing_lm_head: bool,
    notes: list[str],
) -> OptimizerState:
    if checkpoint_optimizer_state is None:
        notes.append("initialized optimizer state during resume from older checkpoint")
        return OptimizerState(
            type=optimizer_type,
            step=checkpoint_step,
            slots=fresh_optimizer_state.slots,
        )
    if checkpoint_optimizer_state.type != optimizer_type:
        raise ValueError(
            "checkpoint optimizer type mismatch: "
            f"{checkpoint_optimizer_state.type!r} != {optimizer_type!r}"
        )
    merged_slots = _merge_optimizer_slots_for_resume(
        fresh_slots=fresh_optimizer_state.slots,
        checkpoint_slots=checkpoint_optimizer_state.slots,
        allow_missing_lm_head=allow_missing_lm_head,
        notes=notes,
    )
    return OptimizerState(
        type=optimizer_type,
        step=checkpoint_optimizer_state.step,
        slots=merged_slots,
    )


def _merge_optimizer_slots_for_resume(
    *,
    fresh_slots: Any,
    checkpoint_slots: Any,
    allow_missing_lm_head: bool,
    notes: list[str],
) -> Any:
    if not fresh_slots:
        return fresh_slots
    return _merge_param_tree(
        fresh_params=fresh_slots,
        checkpoint_params=checkpoint_slots,
        path=("optimizer", "slots"),
        allow_missing_lm_head=allow_missing_lm_head,
        notes=notes,
    )


def _format_param_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _lr_schedule_metadata(
    *,
    config: DistillStageConfig,
    step: int,
    include_step: bool = True,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "type": config.lr_schedule.type,
        "warmup_steps": config.lr_schedule.warmup_steps,
        "total_steps": config.lr_schedule.total_steps,
        "min_learning_rate": config.lr_schedule.min_learning_rate,
        "base_learning_rate": config.optimizer.learning_rate,
    }
    if include_step:
        metadata["step"] = int(step)
    return metadata


def _resume_schedule_notes(
    *,
    checkpoint_schedule: dict[str, object],
    current_schedule: dict[str, object],
) -> list[str]:
    if not checkpoint_schedule:
        return ["resume checkpoint has no lr_schedule metadata"]
    comparable_checkpoint = {
        key: checkpoint_schedule.get(key) for key in current_schedule
    }
    if comparable_checkpoint == current_schedule:
        return []
    return ["resume lr_schedule metadata differs from current config"]


@dataclass(frozen=True)
class _NullMetricsLogger:
    def __enter__(self) -> _NullMetricsLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def log(self, *args, **kwargs) -> None:
        return None


DistillationStageResult = DistillStageResult
run_distillation_stage = run_distill_stage


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
