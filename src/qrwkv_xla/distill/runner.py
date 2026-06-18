from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
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
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.distill.config import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillFingerprintLossConfig,
    DistillStageConfig,
    validate_distill_stage_config,
)
from qrwkv_xla.distill.losses import compute_distill_loss
from qrwkv_xla.distill.metrics import metrics_to_floats
from qrwkv_xla.optimizers import (
    OptimizerConfig,
    OptimizerState,
    init_optimizer_state,
    optimizer_update,
)
from qrwkv_xla.schedules import learning_rate_at_step
from qrwkv_xla.students import create_student, create_student_backend
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
from qrwkv_xla.training.fingerprint_loss import (
    FingerprintCorridorLossConfig,
    FingerprintCorridorLossOutput,
    compute_fingerprint_corridor_loss,
)
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats_at_positions,
)
from qrwkv_xla.training.gradients import clip_gradients_by_global_norm


@dataclass(frozen=True)
class DistillStageResult:
    stage: int
    student_architecture: str
    steps: int
    initial_loss: float
    final_loss: float
    distill_mode: str = "teacher_targets"
    status: str = "completed"
    final_hidden_mse: float | None = None
    final_logits_kl: float | None = None
    final_attention_or_mixer: float | None = None
    target_bundle: Path | None = None
    fingerprint_artifact: Path | None = None
    checkpoint_out: Path | None = None
    resume_from: Path | None = None
    start_step: int = 0
    end_step: int = 0
    optimizer_steps_completed: int = 0
    batches_consumed: int = 0
    run_dir: Path | None = None
    metrics_path: Path | None = None
    report_path: Path | None = None
    summary_path: Path | None = None
    notes: tuple[str, ...] = ()
    lr_schedule_type: str = "constant"
    initial_learning_rate: float | None = None
    final_learning_rate: float | None = None
    final_grad_global_norm: float | None = None
    final_grad_clipped_global_norm: float | None = None
    final_grad_clip_scale: float | None = None
    final_metrics: dict[str, float] | None = None
    training_path_kind: str = "main_runner_teacher_targets"
    real_student_backend_integrated: bool = False
    main_runner_integrated: bool = True
    teacher_required: bool = True
    exemplar_reservoir_enabled: bool = False
    student_backend: str | None = None
    student_uses_input_ids: bool = True


@dataclass(frozen=True)
class _TrainLoss:
    total: jax.Array
    components: dict[str, jax.Array]


def run_distill_stage(config: DistillStageConfig) -> DistillStageResult:
    validate_distill_stage_config(config)
    if config.mode == DISTILL_MODE_FINGERPRINT_CORRIDOR:
        return _run_fingerprint_corridor_stage(config)

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
        num_heads=config.student.num_heads,
        num_kv_heads=config.student.num_kv_heads,
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
        "num_heads": config.student.num_heads,
        "num_kv_heads": config.student.num_kv_heads,
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
        distill_mode=config.mode,
        final_hidden_mse=final_metrics.get("hidden_mse"),
        final_logits_kl=final_metrics.get("logits_kl"),
        final_attention_or_mixer=final_metrics.get("attention_or_mixer"),
        target_bundle=dataset.bundle_dir,
        checkpoint_out=checkpoint_out,
        resume_from=config.checkpoint.resume_from,
        start_step=start_step,
        end_step=state.step,
        optimizer_steps_completed=state.step - start_step,
        batches_consumed=config.training.max_steps,
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
        final_metrics=final_metrics,
    )


def _run_fingerprint_corridor_stage(
    config: DistillStageConfig,
) -> DistillStageResult:
    assert config.fingerprint.artifact_dir is not None
    artifact_summary = summarize_fingerprint_artifact(config.fingerprint.artifact_dir)
    dataset = load_fingerprint_targets(
        config.fingerprint.artifact_dir,
        batch_size=config.fingerprint.batch_size,
        shuffle=config.fingerprint.shuffle,
        seed=config.fingerprint.seed,
        drop_remainder=config.fingerprint.drop_remainder,
        max_records=config.fingerprint.max_records,
    )
    if dataset.num_records == 0:
        raise ValueError("fingerprint_corridor mode requires target records")
    fingerprint_batches = tuple(dataset.iter_batches())
    if not fingerprint_batches:
        raise ValueError("fingerprint_corridor mode yielded zero batches")

    student_vocab_size = (
        config.fingerprint.student_vocab_size or artifact_summary.vocab_size
    )
    _validate_student_artifact_compatibility(
        artifact_vocab_size=artifact_summary.vocab_size,
        artifact_max_seq_len=artifact_summary.max_seq_len,
        student_vocab_size=student_vocab_size,
        student_max_seq_len=config.fingerprint.student_max_seq_len,
    )
    for batch in fingerprint_batches:
        _validate_batch_token_ids(batch.input_ids, vocab_size=student_vocab_size)
        _validate_positions_in_range(batch.position, seq_len=batch.input_ids.shape[1])

    vocab_contract = VocabContract(
        tokenizer_id=artifact_summary.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=artifact_summary.tokenizer_name or None,
        vocab_size=student_vocab_size,
        model_id=artifact_summary.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=vocab_contract,
        architecture_id=config.fingerprint.student_backend,
    )
    student_config = _fingerprint_student_config(
        backend=backend,
        architecture_id=config.fingerprint.student_backend,
        artifact_summary=artifact_summary,
    )

    optimizer_config = config.optimizer.to_optimizer_config()
    train_step = _make_fingerprint_train_step(
        backend=backend,
        loss_config=_fingerprint_loss_config(config.fingerprint_loss),
        optimizer_config=optimizer_config,
        max_grad_norm=config.gradients.max_grad_norm,
        clip_epsilon=config.gradients.clip_epsilon,
    )
    start_step = 0
    params = backend.init_params(jax.random.PRNGKey(config.training.seed))
    optimizer_state = init_optimizer_state(params, optimizer_config)
    resume_notes: list[str] = []
    parent_checkpoint_manifest: dict[str, object] | None = None
    if config.checkpoint.resume_from is not None:
        loaded = load_checkpoint(config.checkpoint.resume_from)
        _validate_resume_checkpoint(
            loaded.manifest.student_architecture,
            loaded.manifest.student_config,
            expected_architecture=config.fingerprint.student_backend,
            expected_student_config=student_config,
        )
        start_step = loaded.manifest.step
        params = loaded.params
        optimizer_state = _optimizer_state_for_resume(
            fresh_optimizer_state=init_optimizer_state(params, optimizer_config),
            checkpoint_optimizer_state=loaded.optimizer_state,
            optimizer_type=optimizer_config.type,
            checkpoint_step=start_step,
            allow_missing_lm_head=False,
            notes=resume_notes,
        )
        parent_checkpoint_manifest = asdict(loaded.manifest)

    state = TrainState(
        params=params,
        step=start_step,
        learning_rate=config.optimizer.learning_rate,
        optimizer_state=optimizer_state,
    )
    initial_params_for_diagnostics = params
    jax_batches = [_fingerprint_batch_to_jax(batch) for batch in fingerprint_batches]
    input_conditioning_detected, input_conditioning_delta_norm = (
        _detect_input_conditioning(
            backend=backend,
            params=params,
            batches=fingerprint_batches,
        )
    )

    checkpoint_out = config.checkpoint.checkpoint_out
    checkpoint_overwrite = config.checkpoint.overwrite
    output_dir = config.fingerprint.output_dir
    metrics_path = output_dir / "metrics.json" if output_dir is not None else None
    report_path = (
        output_dir / "fingerprint_corridor_report.json"
        if output_dir is not None
        else None
    )
    summary_path = (
        output_dir / "fingerprint_run_summary.md" if output_dir is not None else None
    )
    if checkpoint_out is None and output_dir is not None:
        checkpoint_out = output_dir / "checkpoints" / "final"
        checkpoint_overwrite = True

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
            "teacher_required": False,
            "fingerprint_artifact_dir": config.fingerprint.artifact_dir,
            "artifact": artifact_summary.to_dict(),
        }
        if checkpoint_out is None:
            checkpoint_metadata["checkpoint_out"] = "runs/<run_id>/checkpoints/final"
        run_context = create_run_context(
            run_root=config.tracking.run_root or Path("runs"),
            stage=config.stage,
            student_architecture=config.fingerprint.student_backend,
            run_name=config.tracking.run_name,
            command=command,
            git=git_metadata,
            environment=environment_metadata,
            distillation={
                "stage": config.stage,
                "mode": config.mode,
                "fingerprint": asdict(config.fingerprint),
                "fingerprint_loss": asdict(config.fingerprint_loss),
                "optimizer": asdict(config.optimizer),
                "lr_schedule": _lr_schedule_metadata(
                    config=config,
                    step=start_step,
                    include_step=False,
                ),
                "gradients": asdict(config.gradients),
                "training": asdict(config.training),
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
    metric_records: list[dict[str, float]] = []
    batches_consumed = 0
    with metrics_logger or _NullMetricsLogger() as active_logger:
        for step_index in range(config.training.max_steps):
            batch = jax_batches[step_index % len(jax_batches)]
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
            batches_consumed += 1
            float_metrics = metrics_to_floats(metrics)
            float_metrics["learning_rate"] = scheduled_lr
            float_metrics["base_learning_rate"] = config.optimizer.learning_rate
            float_metrics["global_step"] = float(global_step)
            float_metrics["local_step"] = float(step_index)
            float_metrics["step"] = float(state.step)
            float_metrics["fingerprint/runner/optimizer_steps_completed"] = float(
                state.step - start_step
            )
            float_metrics["fingerprint/runner/batches_consumed"] = float(
                batches_consumed
            )
            float_metrics["fingerprint/runner/artifact_num_records"] = float(
                dataset.num_records
            )
            if initial_loss is None:
                initial_loss = float_metrics["loss"]
            final_metrics = float_metrics
            metric_records.append(float_metrics)
            if metrics_logger is not None:
                active_logger.log(
                    step=state.step,
                    values=float_metrics,
                    phase="train",
                    extra={
                        "local_step": step_index,
                        "global_step": global_step,
                        "start_step": start_step,
                        "batch_index": step_index % len(jax_batches),
                        "optimizer_type": config.optimizer.type,
                        "lr_schedule_type": config.lr_schedule.type,
                        "distill_mode": config.mode,
                    },
                )

    assert initial_loss is not None
    assert initial_learning_rate is not None
    assert final_learning_rate is not None
    assert final_metrics is not None
    if batches_consumed == 0:
        raise ValueError("fingerprint_corridor mode consumed zero batches")
    loss_finite = bool(np.isfinite(initial_loss) and np.isfinite(final_metrics["loss"]))
    loss_non_negative = bool(initial_loss >= 0.0 and final_metrics["loss"] >= 0.0)
    loss_delta = final_metrics["loss"] - initial_loss
    loss_non_increasing = bool(final_metrics["loss"] <= initial_loss + 1e-6)
    param_delta_norm = _tree_delta_norm(initial_params_for_diagnostics, state.params)
    params_changed = bool(param_delta_norm > 1e-12)
    final_metrics.update(
        {
            "fingerprint/rehearsal/input_conditioning_detected": float(
                input_conditioning_detected
            ),
            "fingerprint/rehearsal/input_conditioning_delta_norm": float(
                input_conditioning_delta_norm
            ),
            "fingerprint/rehearsal/params_changed": float(params_changed),
            "fingerprint/rehearsal/param_delta_norm": float(param_delta_norm),
            "fingerprint/rehearsal/initial_loss": float(initial_loss),
            "fingerprint/rehearsal/final_loss": float(final_metrics["loss"]),
            "fingerprint/rehearsal/loss_delta": float(loss_delta),
            "fingerprint/rehearsal/loss_non_increasing": float(loss_non_increasing),
        }
    )
    for record in metric_records:
        record.setdefault(
            "fingerprint/rehearsal/input_conditioning_detected",
            float(input_conditioning_detected),
        )
        record.setdefault(
            "fingerprint/rehearsal/input_conditioning_delta_norm",
            float(input_conditioning_delta_norm),
        )
    metrics_finite = all(bool(np.isfinite(value)) for value in final_metrics.values())
    rehearsal_required = config.fingerprint.input_conditioned_rehearsal
    status = (
        "pass"
        if (
            state.step - start_step == config.training.max_steps
            and batches_consumed > 0
            and loss_finite
            and loss_non_negative
            and (input_conditioning_detected or not rehearsal_required)
            and (params_changed or not rehearsal_required)
            and metrics_finite
        )
        else "fail"
    )

    target_manifest = {
        "artifact_type": artifact_summary.artifact_type,
        "artifact_version": artifact_summary.artifact_version,
        "artifact_dir": str(config.fingerprint.artifact_dir),
        "teacher_model_name": artifact_summary.teacher_model_name,
        "tokenizer_name": artifact_summary.tokenizer_name,
        "vocab_size": artifact_summary.vocab_size,
        "max_seq_len": artifact_summary.max_seq_len,
        "tracked_stats": list(artifact_summary.tracked_stats),
        "num_corridor_records": dataset.num_records,
        "distill_mode": config.mode,
    }
    if checkpoint_out is not None:
        save_checkpoint(
            checkpoint_out,
            state.params,
            student_architecture=config.fingerprint.student_backend,
            student_config=student_config,
            step=state.step,
            learning_rate=config.optimizer.learning_rate,
            loss_config={
                "distill_mode": config.mode,
                "fingerprint_loss": asdict(config.fingerprint_loss),
            },
            target_manifest=target_manifest,
            optimizer_config=asdict(config.optimizer),
            optimizer_state=state.optimizer_state,
            lr_schedule=_lr_schedule_metadata(config=config, step=state.step),
            gradients=asdict(config.gradients),
            notes=[
                "simple JSON + NPZ checkpoint",
                "main runner fingerprint_corridor mode",
                f"optimizer {config.optimizer.type}",
                *resume_notes,
            ],
            overwrite=checkpoint_overwrite,
        )

    report = _fingerprint_corridor_report(
        config=config,
        artifact_summary=artifact_summary,
        dataset_num_records=dataset.num_records,
        state=state,
        start_step=start_step,
        batches_consumed=batches_consumed,
        initial_loss=initial_loss,
        final_metrics=final_metrics,
        checkpoint_out=checkpoint_out,
        metrics_path=metrics_path,
        summary_path=summary_path,
        student_config=student_config,
        status=status,
        input_conditioning_detected=input_conditioning_detected,
        input_conditioning_delta_norm=input_conditioning_delta_norm,
        params_changed=params_changed,
        param_delta_norm=param_delta_norm,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path is not None:
        metrics_path.write_text(
            json.dumps(
                {
                    "phase": (
                        "P142"
                        if config.fingerprint.input_conditioned_rehearsal
                        else "P141"
                    ),
                    "distill_mode": config.mode,
                    "final": final_metrics,
                    "steps": metric_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if report_path is not None:
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if summary_path is not None:
        summary_path.write_text(
            _render_fingerprint_corridor_summary(report),
            encoding="utf-8",
        )
    if run_context is not None:
        write_run_summary(
            context=run_context,
            summary={
                "status": status,
                "started_at_utc": run_context.metadata.created_at_utc,
                "finished_at_utc": _utc_now(),
                "run_id": run_context.metadata.run_id,
                "stage": config.stage,
                "distill_mode": config.mode,
                "training_path_kind": "main_runner_fingerprint_corridor",
                "student_architecture": config.fingerprint.student_backend,
                "steps": config.training.max_steps,
                "start_step": start_step,
                "end_step": state.step,
                "optimizer_steps_completed": state.step - start_step,
                "batches_consumed": batches_consumed,
                "initial_loss": initial_loss,
                "final_loss": final_metrics["loss"],
                "optimizer_type": config.optimizer.type,
                "learning_rate": final_learning_rate,
                "base_learning_rate": config.optimizer.learning_rate,
                "initial_learning_rate": initial_learning_rate,
                "final_learning_rate": final_learning_rate,
                "lr_schedule": _lr_schedule_metadata(config=config, step=state.step),
                "lr_schedule_type": config.lr_schedule.type,
                "gradients": asdict(config.gradients),
                "final_grad_global_norm": final_metrics.get("grad_global_norm"),
                "final_grad_clipped_global_norm": final_metrics.get(
                    "grad_clipped_global_norm"
                ),
                "final_grad_clip_scale": final_metrics.get("grad_clip_scale"),
                "checkpoint_out": checkpoint_out,
                "resume_from": config.checkpoint.resume_from,
                "fingerprint_artifact": config.fingerprint.artifact_dir,
                "teacher_required": False,
                "exemplar_reservoir_enabled": False,
                "notes": resume_notes,
            },
        )

    return DistillStageResult(
        stage=config.stage,
        student_architecture=config.fingerprint.student_backend,
        steps=config.training.max_steps,
        initial_loss=initial_loss,
        final_loss=final_metrics["loss"],
        distill_mode=config.mode,
        status=status,
        fingerprint_artifact=config.fingerprint.artifact_dir,
        checkpoint_out=checkpoint_out,
        resume_from=config.checkpoint.resume_from,
        start_step=start_step,
        end_step=state.step,
        optimizer_steps_completed=state.step - start_step,
        batches_consumed=batches_consumed,
        run_dir=run_context.paths.run_dir if run_context is not None else None,
        metrics_path=(
            metrics_path
            if metrics_path is not None
            else (run_context.paths.metrics_jsonl if run_context is not None else None)
        ),
        report_path=report_path,
        summary_path=(
            summary_path
            if summary_path is not None
            else (run_context.paths.summary_json if run_context is not None else None)
        ),
        notes=tuple(resume_notes),
        lr_schedule_type=config.lr_schedule.type,
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        final_grad_global_norm=final_metrics.get("grad_global_norm"),
        final_grad_clipped_global_norm=final_metrics.get("grad_clipped_global_norm"),
        final_grad_clip_scale=final_metrics.get("grad_clip_scale"),
        final_metrics=final_metrics,
        training_path_kind="main_runner_fingerprint_corridor",
        real_student_backend_integrated=True,
        main_runner_integrated=True,
        teacher_required=False,
        exemplar_reservoir_enabled=False,
        student_backend=config.fingerprint.student_backend,
        student_uses_input_ids=True,
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


def _make_fingerprint_train_step(
    *,
    backend: Any,
    loss_config: FingerprintCorridorLossConfig,
    optimizer_config: OptimizerConfig,
    max_grad_norm: float | None,
    clip_epsilon: float,
):
    def train_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any):
            output, _next_state = backend.forward_full(params, batch["input_ids"])
            logits = backend.logits(output)
            stats = compute_fingerprint_distribution_stats_at_positions(
                logits,
                batch["position"],
            )
            fingerprint_batch = FingerprintBatch(
                input_ids=batch["input_ids"],
                position=batch["position"],
                mode_id=batch["mode_id"],
                entropy_min=batch["entropy_min"],
                entropy_max=batch["entropy_max"],
                top1_margin_min=batch["top1_margin_min"],
                top1_margin_max=batch["top1_margin_max"],
                top8_mass_min=batch["top8_mass_min"],
                top8_mass_max=batch["top8_mass_max"],
                top32_mass_min=batch["top32_mass_min"],
                top32_mass_max=batch["top32_mass_max"],
                tail_mass_min=batch["tail_mass_min"],
                tail_mass_max=batch["tail_mass_max"],
                weight=batch["weight"],
            )
            corridor = compute_fingerprint_corridor_loss(
                stats,
                fingerprint_batch,
                loss_config,
            )
            return corridor.loss, _fingerprint_corridor_components(corridor)

        (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        clip_result = clip_gradients_by_global_norm(
            grads,
            max_grad_norm=max_grad_norm,
            epsilon=clip_epsilon,
        )
        update_config = OptimizerConfig(
            type=optimizer_config.type,
            learning_rate=state.learning_rate,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        optimizer_state = state.optimizer_state
        if optimizer_state is None:
            optimizer_state = init_optimizer_state(state.params, update_config)
        new_params, new_optimizer_state, optimizer_metrics = optimizer_update(
            state.params,
            clip_result.gradients,
            optimizer_state,
            update_config,
        )
        new_state = TrainState(
            params=new_params,
            step=state.step + 1,
            learning_rate=state.learning_rate,
            optimizer_state=new_optimizer_state,
        )
        return new_state, dict(
            components,
            loss=loss,
            **components,
            grad_global_norm=clip_result.global_norm,
            grad_clipped_global_norm=clip_result.clipped_global_norm,
            grad_clip_scale=clip_result.clip_scale,
            grad_was_clipped=clip_result.was_clipped,
            max_grad_norm=jnp.asarray(
                0.0 if max_grad_norm is None else max_grad_norm,
                dtype=jnp.float32,
            ),
            **optimizer_metrics,
        )

    return jax.jit(train_step)


def _fingerprint_corridor_components(
    output: FingerprintCorridorLossOutput,
) -> dict[str, jax.Array]:
    return {
        "train/loss": output.loss,
        "fingerprint/corridor/loss_total": output.loss,
        "fingerprint/corridor/loss_entropy": output.entropy_loss,
        "fingerprint/corridor/loss_top1_margin": output.top1_margin_loss,
        "fingerprint/corridor/loss_top8_mass": output.top8_mass_loss,
        "fingerprint/corridor/loss_top32_mass": output.top32_mass_loss,
        "fingerprint/corridor/loss_tail_mass": output.tail_mass_loss,
        "fingerprint/corridor/inside_entropy_rate": output.entropy_inside_rate,
        "fingerprint/corridor/inside_top1_margin_rate": (
            output.top1_margin_inside_rate
        ),
        "fingerprint/corridor/inside_top8_mass_rate": output.top8_mass_inside_rate,
        "fingerprint/corridor/inside_top32_mass_rate": output.top32_mass_inside_rate,
        "fingerprint/corridor/inside_tail_mass_rate": output.tail_mass_inside_rate,
        "fingerprint/corridor/inside_all_rate": output.all_inside_rate,
    }


def _fingerprint_batch_to_jax(batch: FingerprintBatch) -> dict[str, jax.Array]:
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


def _detect_input_conditioning(
    *,
    backend: Any,
    params: Any,
    batches: tuple[FingerprintBatch, ...],
) -> tuple[bool, float]:
    pair = _first_distinct_input_pair(batches)
    if pair is None:
        return False, 0.0
    inputs = jnp.asarray(np.stack(pair, axis=0), dtype=jnp.int32)
    output, _state = backend.forward_full(params, inputs)
    logits = backend.logits(output)
    if logits is None:
        return False, 0.0
    delta = jnp.linalg.norm(jnp.asarray(logits[0]) - jnp.asarray(logits[1]))
    delta_float = float(delta)
    return bool(np.isfinite(delta_float) and delta_float > 1e-7), delta_float


def _first_distinct_input_pair(
    batches: tuple[FingerprintBatch, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    seen: list[np.ndarray] = []
    for batch in batches:
        for row in batch.input_ids:
            candidate = np.asarray(row, dtype=np.int32)
            for prior in seen:
                if not np.array_equal(candidate, prior):
                    return prior, candidate
            seen.append(candidate)
    return None


def _tree_delta_norm(before: Any, after: Any) -> float:
    leaves_before, tree_def = jax.tree_util.tree_flatten(before)
    leaves_after, after_tree_def = jax.tree_util.tree_flatten(after)
    if tree_def != after_tree_def:
        raise ValueError(
            "parameter tree structure changed during fingerprint rehearsal"
        )
    total = 0.0
    for left, right in zip(leaves_before, leaves_after, strict=True):
        delta = np.asarray(right) - np.asarray(left)
        total += float(np.sum(np.square(delta)))
    return float(np.sqrt(total))


def _fingerprint_loss_config(
    config: DistillFingerprintLossConfig,
) -> FingerprintCorridorLossConfig:
    return FingerprintCorridorLossConfig(
        entropy_weight=config.entropy_weight,
        top1_margin_weight=config.top1_margin_weight,
        top8_mass_weight=config.top8_mass_weight,
        top32_mass_weight=config.top32_mass_weight,
        tail_mass_weight=config.tail_mass_weight,
        use_record_weights=config.use_record_weights,
        eps=config.eps,
    )


def _fingerprint_student_config(
    *,
    backend: Any,
    architecture_id: str,
    artifact_summary: Any,
) -> dict[str, Any]:
    student = getattr(backend, "student", None)
    raw_config = getattr(student, "config", None)
    return {
        "architecture_id": architecture_id,
        "backend_name": type(backend).__name__,
        "architecture": getattr(raw_config, "__class__", type("", (), {})).__name__,
        "vocab_size": int(
            getattr(raw_config, "vocab_size", artifact_summary.vocab_size)
        ),
        "hidden_size": int(getattr(raw_config, "hidden_size", 0)),
        "num_layers": int(getattr(raw_config, "num_layers", 0)),
        "num_heads": _optional_int_value(getattr(raw_config, "num_heads", None)),
        "num_kv_heads": _optional_int_value(getattr(raw_config, "num_kv_heads", None)),
        "emit_logits": bool(getattr(raw_config, "emit_logits", True)),
        "tie_embeddings": bool(getattr(raw_config, "tie_embeddings", False)),
        "emit_mixer_outputs": bool(getattr(raw_config, "emit_mixer_outputs", False)),
    }


def _optional_int_value(value: Any) -> int | None:
    return None if value is None else int(value)


def _validate_student_artifact_compatibility(
    *,
    artifact_vocab_size: int,
    artifact_max_seq_len: int,
    student_vocab_size: int,
    student_max_seq_len: int | None,
) -> None:
    if student_vocab_size != artifact_vocab_size:
        raise ValueError(
            "Fingerprint artifact vocab_size="
            f"{artifact_vocab_size} but student vocab_size={student_vocab_size}."
        )
    if student_max_seq_len is not None and artifact_max_seq_len > student_max_seq_len:
        raise ValueError(
            "Fingerprint artifact max_seq_len="
            f"{artifact_max_seq_len} exceeds student max_seq_len={student_max_seq_len}."
        )


def _validate_batch_token_ids(input_ids: np.ndarray, *, vocab_size: int) -> None:
    if input_ids.size == 0:
        raise ValueError("fingerprint_corridor input_ids must be non-empty")
    min_token = int(np.min(input_ids))
    max_token = int(np.max(input_ids))
    if min_token < 0 or max_token >= vocab_size:
        raise ValueError(
            "fingerprint_corridor input_ids outside student vocab: "
            f"min={min_token} max={max_token} vocab_size={vocab_size}"
        )


def _validate_positions_in_range(positions: np.ndarray, *, seq_len: int) -> None:
    if positions.size == 0:
        raise ValueError("fingerprint_corridor positions must be non-empty")
    min_position = int(np.min(positions))
    max_position = int(np.max(positions))
    if min_position < 0 or max_position >= seq_len:
        raise ValueError(
            "fingerprint_corridor target position outside logits sequence: "
            f"min={min_position} max={max_position} seq_len={seq_len}"
        )


def _fingerprint_corridor_report(
    *,
    config: DistillStageConfig,
    artifact_summary: Any,
    dataset_num_records: int,
    state: TrainState,
    start_step: int,
    batches_consumed: int,
    initial_loss: float,
    final_metrics: dict[str, float],
    checkpoint_out: Path | None,
    metrics_path: Path | None,
    summary_path: Path | None,
    student_config: dict[str, Any],
    status: str,
    input_conditioning_detected: bool,
    input_conditioning_delta_norm: float,
    params_changed: bool,
    param_delta_norm: float,
) -> dict[str, Any]:
    phase = "P142" if config.fingerprint.input_conditioned_rehearsal else "P141"
    loss_delta = final_metrics["loss"] - initial_loss
    loss_non_increasing = bool(final_metrics["loss"] <= initial_loss + 1e-6)
    return {
        "phase": phase,
        "status": status,
        "distill_mode": config.mode,
        "training_path_kind": "main_runner_fingerprint_corridor",
        "real_student_backend_integrated": True,
        "main_runner_integrated": True,
        "teacher_required": False,
        "hf_required": False,
        "accelerator_required": False,
        "exemplar_reservoir_enabled": False,
        "exemplar_forward_enabled": False,
        "student_uses_input_ids": True,
        "student_backend": config.fingerprint.student_backend,
        "input_conditioned_rehearsal": config.fingerprint.input_conditioned_rehearsal,
        "input_conditioning_detected": input_conditioning_detected,
        "input_conditioning_delta_norm": input_conditioning_delta_norm,
        "params_changed": params_changed,
        "param_delta_norm": param_delta_norm,
        "optimizer_steps_completed": int(state.step - start_step),
        "requested_steps": config.training.max_steps,
        "batches_consumed": batches_consumed,
        "initial_loss": initial_loss,
        "final_loss": final_metrics["loss"],
        "loss_delta": loss_delta,
        "loss_non_increasing": loss_non_increasing,
        "loss_non_increasing_required": False,
        "loss": {
            "initial_total": initial_loss,
            "final_total": final_metrics["loss"],
            "delta_total": loss_delta,
            "finite": bool(
                np.isfinite(initial_loss) and np.isfinite(final_metrics["loss"])
            ),
            "non_negative": bool(initial_loss >= 0.0 and final_metrics["loss"] >= 0.0),
            "non_increasing": loss_non_increasing,
            "non_increasing_required": False,
        },
        "student": student_config,
        "artifact": {
            "artifact_type": artifact_summary.artifact_type,
            "artifact_version": artifact_summary.artifact_version,
            "artifact_dir": str(config.fingerprint.artifact_dir),
            "vocab_size": artifact_summary.vocab_size,
            "max_seq_len": artifact_summary.max_seq_len,
            "tracked_stats": artifact_summary.tracked_stats,
        },
        "corridor_targets": {
            "num_records": dataset_num_records,
            "batch_size": config.fingerprint.batch_size,
            "batches_consumed": batches_consumed,
            "max_seq_len": artifact_summary.max_seq_len,
            "vocab_size": artifact_summary.vocab_size,
            "tracked_stats": artifact_summary.tracked_stats,
        },
        "corridor_metrics": {
            "loss_total": final_metrics["fingerprint/corridor/loss_total"],
            "loss_entropy": final_metrics["fingerprint/corridor/loss_entropy"],
            "loss_top1_margin": final_metrics["fingerprint/corridor/loss_top1_margin"],
            "loss_top8_mass": final_metrics["fingerprint/corridor/loss_top8_mass"],
            "loss_top32_mass": final_metrics["fingerprint/corridor/loss_top32_mass"],
            "loss_tail_mass": final_metrics["fingerprint/corridor/loss_tail_mass"],
            "inside_entropy_rate": final_metrics[
                "fingerprint/corridor/inside_entropy_rate"
            ],
            "inside_top1_margin_rate": final_metrics[
                "fingerprint/corridor/inside_top1_margin_rate"
            ],
            "inside_top8_mass_rate": final_metrics[
                "fingerprint/corridor/inside_top8_mass_rate"
            ],
            "inside_top32_mass_rate": final_metrics[
                "fingerprint/corridor/inside_top32_mass_rate"
            ],
            "inside_tail_mass_rate": final_metrics[
                "fingerprint/corridor/inside_tail_mass_rate"
            ],
            "inside_all_rate": final_metrics["fingerprint/corridor/inside_all_rate"],
        },
        "metrics": final_metrics,
        "checkpoint_out": None if checkpoint_out is None else str(checkpoint_out),
        "metrics_path": None if metrics_path is None else str(metrics_path),
        "summary_path": None if summary_path is None else str(summary_path),
        "limitations": (
            (
                "This is an input-conditioned tiny rehearsal."
                if phase == "P142"
                else "This is main-runner corridor-only fingerprint training."
            ),
            "It uses the main fingerprint_corridor runner mode.",
            "It trains a real registered student backend.",
            "No exemplar reservoir training is active.",
            "No teacher backend is required.",
            "Teacher-side fingerprint capture remains future work.",
            "No model-quality claim is made.",
        ),
    }


def _render_fingerprint_corridor_summary(report: dict[str, Any]) -> str:
    artifact = report["artifact"]
    metrics = report["corridor_metrics"]
    loss = report["loss"]
    heading = (
        "# Input-Conditioned Tiny Fingerprint Rehearsal Summary"
        if report.get("phase") == "P142"
        else "# Main Runner Fingerprint Corridor Summary"
    )
    return "\n".join(
        (
            heading,
            "",
            f"Phase: {report['phase']}",
            f"Status: {report['status']}",
            f"Mode: {report['distill_mode']}",
            f"Training path: {report['training_path_kind']}",
            f"Student backend: {report['student_backend']}",
            "Real student backend integrated: true",
            "Main runner integrated: true",
            "Teacher required: false",
            "Exemplar reservoir enabled: false",
            "",
            (
                "This is an input-conditioned tiny rehearsal."
                if report.get("phase") == "P142"
                else "This is main-runner corridor-only fingerprint training."
            ),
            "It uses the main fingerprint_corridor runner mode.",
            "It trains a real registered student backend.",
            "No exemplar reservoir training is active.",
            "No teacher backend is required.",
            "Teacher-side fingerprint capture remains future work.",
            "",
            "## Artifact",
            f"- Type: {artifact['artifact_type']} v{artifact['artifact_version']}",
            f"- Vocab size: {artifact['vocab_size']}",
            f"- Max sequence length: {artifact['max_seq_len']}",
            f"- Corridor records: {report['corridor_targets']['num_records']}",
            "",
            "## Run",
            f"- Requested steps: {report['requested_steps']}",
            f"- Optimizer steps completed: {report['optimizer_steps_completed']}",
            f"- Batches consumed: {report['batches_consumed']}",
            f"- Checkpoint: {report['checkpoint_out']}",
            f"- Input conditioning detected: {report['input_conditioning_detected']}",
            f"- Params changed: {report['params_changed']}",
            f"- Param delta norm: {report['param_delta_norm']}",
            "",
            "## Loss",
            f"- Initial: {loss['initial_total']}",
            f"- Final: {loss['final_total']}",
            f"- Delta: {loss['delta_total']}",
            f"- Non-increasing: {loss['non_increasing']} (diagnostic only)",
            f"- Non-increasing required: {loss['non_increasing_required']}",
            "",
            "## Corridor Metrics",
            f"- Loss total: {metrics['loss_total']}",
            f"- Inside all rate: {metrics['inside_all_rate']}",
            f"- Entropy inside rate: {metrics['inside_entropy_rate']}",
            "",
        )
    )


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
