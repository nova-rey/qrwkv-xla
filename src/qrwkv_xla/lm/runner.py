from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.distill.metrics import metrics_to_floats
from qrwkv_xla.generation import normalize_tokenizer_config
from qrwkv_xla.lm.config import LMStageConfig, validate_lm_stage_config
from qrwkv_xla.lm.data import (
    build_lm_batches,
    load_lm_token_sequences_with_tokenizer,
    load_lm_tokenizer,
)
from qrwkv_xla.lm.losses import masked_next_token_cross_entropy
from qrwkv_xla.lm.tokenized_corpus import load_tokenized_corpus
from qrwkv_xla.optimizers import OptimizerState, init_optimizer_state
from qrwkv_xla.prompting import (
    build_prompt_corpus_manifest,
    filter_prompt_corpus,
    read_prompt_corpus,
)
from qrwkv_xla.schedules import learning_rate_at_step
from qrwkv_xla.students import create_student
from qrwkv_xla.tracking import (
    MetricsLogger,
    create_run_context,
    get_environment_metadata,
    get_git_metadata,
    write_run_metadata,
    write_run_summary,
)
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.trainers.step import make_train_step


@dataclass(frozen=True)
class LMStageResult:
    stage: int
    student_architecture: str
    steps: int
    initial_loss: float
    final_loss: float
    final_ce_loss: float
    prompt_corpus: Path | None
    tokenized_corpus: Path | None = None
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


def run_lm_stage(config: LMStageConfig) -> LMStageResult:
    validate_lm_stage_config(config)
    tokenized_corpus = None
    if config.data.tokenized_corpus is not None:
        tokenized_corpus = load_tokenized_corpus(
            config.data.tokenized_corpus,
            expected_sequence_length=config.data.sequence_length,
            expected_tokenizer=normalize_tokenizer_config(config.data.tokenizer),
        )
        tokenizer_metadata = tokenized_corpus.manifest.tokenizer
        token_sequences = [
            list(sequence) for sequence in tokenized_corpus.token_sequences
        ]
    else:
        tokenizer = load_lm_tokenizer(config.data)
        tokenizer_metadata = tokenizer.metadata
        token_sequences = load_lm_token_sequences_with_tokenizer(config.data, tokenizer)
    if tokenizer_metadata.eos_token_id is None:
        raise ValueError("LM tokenizer must expose eos_token_id")
    if tokenizer_metadata.pad_token_id is None:
        raise ValueError("LM tokenizer must expose pad_token_id")
    if config.student.vocab_size != tokenizer_metadata.vocab_size:
        raise ValueError(
            "student.vocab_size must match tokenizer vocab_size: "
            f"{config.student.vocab_size} != {tokenizer_metadata.vocab_size}"
        )
    if config.data.shuffle:
        import random

        rng = random.Random(config.data.seed)
        rng.shuffle(token_sequences)
    batches = [
        {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "attention_mask": batch.attention_mask,
            "label_mask": batch.label_mask,
        }
        for batch in build_lm_batches(
            token_sequences,
            sequence_length=config.data.sequence_length,
            batch_size=config.data.batch_size,
            pad_token_id=tokenizer_metadata.pad_token_id,
            eos_token_id=tokenizer_metadata.eos_token_id,
        )
    ]

    student = create_student(
        config.student.architecture,
        vocab_size=config.student.vocab_size,
        hidden_size=config.student.hidden_size,
        num_layers=config.student.num_layers,
        emit_logits=config.student.emit_logits,
        tie_embeddings=config.student.tie_embeddings,
    )
    student_config = asdict(config.student)
    optimizer_config = config.optimizer.to_optimizer_config()
    train_step = make_train_step(
        student.apply,
        distillation_loss=_make_train_loss(),
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
        params = loaded.params
        optimizer_state = _optimizer_state_for_resume(
            fresh_optimizer_state=init_optimizer_state(params, optimizer_config),
            checkpoint_optimizer_state=loaded.optimizer_state,
            optimizer_type=optimizer_config.type,
            checkpoint_step=start_step,
            notes=resume_notes,
        )
        parent_checkpoint_manifest = asdict(loaded.manifest)
        resume_notes.extend(
            _resume_schedule_notes(
                checkpoint_schedule=loaded.manifest.lr_schedule,
                current_schedule=_lr_schedule_metadata(config=config, step=start_step),
            )
        )

    state = TrainState(
        params=params,
        step=start_step,
        learning_rate=config.optimizer.learning_rate,
        optimizer_state=optimizer_state,
    )

    data_manifest = _data_manifest_for_config(config, tokenized_corpus=tokenized_corpus)
    checkpoint_out = config.checkpoint.checkpoint_out
    checkpoint_overwrite = config.checkpoint.overwrite
    run_context = None
    metrics_logger: MetricsLogger | None = None
    if config.tracking.enabled:
        command = list(sys.argv)
        checkpoint_metadata: dict[str, object] = {
            "checkpoint_out": checkpoint_out,
            "resume_from": config.checkpoint.resume_from,
            "overwrite": checkpoint_overwrite,
            "parent_checkpoint_manifest": parent_checkpoint_manifest,
        }
        if checkpoint_out is None:
            checkpoint_metadata["checkpoint_out"] = "runs/<run_id>/checkpoints/final"
        run_context = create_run_context(
            run_root=config.tracking.run_root or Path("runs"),
            stage=config.training.stage,
            student_architecture=config.student.architecture,
            run_name=config.tracking.run_name,
            command=command,
            git=get_git_metadata(Path(__file__).resolve().parents[3]),
            environment=get_environment_metadata(),
            distillation={
                "mode": "lm_stage3_ce",
                "stage": config.training.stage,
                "optimizer": asdict(config.optimizer),
                "lr_schedule": _lr_schedule_metadata(
                    config=config,
                    step=start_step,
                    include_step=False,
                ),
                "gradients": asdict(config.gradients),
                "training": asdict(config.training),
                "losses": {"next_token_ce": {"enabled": True, "weight": 1.0}},
                "data": asdict(config.data),
                "data_manifest": data_manifest,
                "tokenizer": asdict(tokenizer_metadata),
            },
            teacher_target={},
            student=student_config,
            checkpoint=checkpoint_metadata,
            tags=config.tracking.tags,
            notes=[*config.tracking.notes, *resume_notes, "student-only stage 3 CE"],
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
            batch = batches[step_index % len(batches)]
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
                        "batch_index": step_index % len(batches),
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
            loss_config={"next_token_ce": {"enabled": True, "weight": 1.0}},
            target_manifest={
                "type": "tokenized_corpus"
                if config.data.tokenized_corpus is not None
                else "prompt_corpus",
                "stage": config.training.stage,
                "manifest": data_manifest,
                "data": asdict(config.data),
                "tokenizer": asdict(tokenizer_metadata),
            },
            optimizer_config=asdict(config.optimizer),
            optimizer_state=state.optimizer_state,
            lr_schedule=_lr_schedule_metadata(config=config, step=state.step),
            gradients=asdict(config.gradients),
            notes=[
                "simple JSON + NPZ checkpoint",
                "student-only stage 3 CE",
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
                "stage": config.training.stage,
                "mode": "lm_stage3_ce",
                "student_architecture": config.student.architecture,
                "steps": config.training.max_steps,
                "start_step": start_step,
                "end_step": state.step,
                "initial_loss": initial_loss,
                "final_loss": final_metrics["loss"],
                "final_ce_loss": final_metrics["ce_loss"],
                "optimizer_type": config.optimizer.type,
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
                "prompt_corpus": config.data.prompt_corpus,
                "tokenized_corpus": config.data.tokenized_corpus,
                "notes": resume_notes,
            },
        )

    return LMStageResult(
        stage=config.training.stage,
        student_architecture=config.student.architecture,
        steps=config.training.max_steps,
        initial_loss=initial_loss,
        final_loss=final_metrics["loss"],
        final_ce_loss=final_metrics["ce_loss"],
        prompt_corpus=config.data.prompt_corpus,
        tokenized_corpus=config.data.tokenized_corpus,
        checkpoint_out=checkpoint_out,
        resume_from=config.checkpoint.resume_from,
        start_step=start_step,
        end_step=state.step,
        run_dir=run_context.paths.run_dir if run_context is not None else None,
        metrics_path=run_context.paths.metrics_jsonl
        if run_context is not None
        else None,
        summary_path=run_context.paths.summary_json
        if run_context is not None
        else None,
        notes=tuple(resume_notes),
        lr_schedule_type=config.lr_schedule.type,
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        final_grad_global_norm=final_metrics.get("grad_global_norm"),
        final_grad_clipped_global_norm=final_metrics.get("grad_clipped_global_norm"),
        final_grad_clip_scale=final_metrics.get("grad_clip_scale"),
    )


def _make_train_loss():
    def loss_fn(student_output: Any, batch: dict[str, jax.Array]) -> _TrainLoss:
        if student_output.logits is None:
            raise ValueError("Stage 3 CE training requires student logits")
        ce_loss = masked_next_token_cross_entropy(
            logits=student_output.logits,
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )
        return _TrainLoss(
            total=ce_loss, components={"loss": ce_loss, "ce_loss": ce_loss}
        )

    return loss_fn


def _data_manifest_for_config(config: LMStageConfig, *, tokenized_corpus):
    if tokenized_corpus is not None:
        return asdict(tokenized_corpus.manifest)
    if config.data.prompt_corpus is None:
        raise ValueError("LM prompt_corpus is required when tokenized_corpus is absent")
    corpus = read_prompt_corpus(config.data.prompt_corpus)
    filtered = filter_prompt_corpus(
        corpus,
        split=config.data.prompt_split,
        tags=config.data.prompt_tags,
        limit=config.data.prompt_limit,
    )
    return asdict(
        build_prompt_corpus_manifest(
            filtered,
            description="Stage 3 CE prompt corpus selection.",
            notes=["student-only next-token CE"],
        )
    )


def _validate_resume_checkpoint(
    checkpoint_architecture: str,
    checkpoint_student_config: dict[str, object],
    *,
    expected_architecture: str,
    expected_student_config: dict[str, object],
) -> None:
    if checkpoint_architecture != expected_architecture:
        raise ValueError(
            "checkpoint student architecture mismatch: "
            f"{checkpoint_architecture!r} != {expected_architecture!r}"
        )
    for name in (
        "hidden_size",
        "num_layers",
        "vocab_size",
        "emit_logits",
        "tie_embeddings",
    ):
        checkpoint_value = checkpoint_student_config.get(name)
        expected_value = expected_student_config[name]
        if checkpoint_value != expected_value:
            raise ValueError(
                f"checkpoint student {name} mismatch: "
                f"{checkpoint_value!r} != {expected_value!r}"
            )


def _optimizer_state_for_resume(
    *,
    fresh_optimizer_state: OptimizerState,
    checkpoint_optimizer_state: OptimizerState | None,
    optimizer_type: str,
    checkpoint_step: int,
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
    return checkpoint_optimizer_state


def _lr_schedule_metadata(
    *,
    config: LMStageConfig,
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


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
