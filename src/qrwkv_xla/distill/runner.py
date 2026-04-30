from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax

from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.distill.config import DistillStageConfig, validate_distill_stage_config
from qrwkv_xla.distill.losses import compute_distill_loss
from qrwkv_xla.distill.metrics import metrics_to_floats
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
    target_bundle: Path | None = None
    checkpoint_out: Path | None = None
    resume_from: Path | None = None
    start_step: int = 0
    end_step: int = 0
    run_dir: Path | None = None
    metrics_path: Path | None = None
    summary_path: Path | None = None


@dataclass(frozen=True)
class _TrainLoss:
    total: jax.Array
    components: dict[str, jax.Array]


def run_distill_stage(config: DistillStageConfig) -> DistillStageResult:
    validate_distill_stage_config(config)
    dataset = TargetBundleDataset.from_path(config.targets_dir)
    manifest = read_manifest(manifest_path(config.targets_dir))

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
    )
    student_config = {
        "architecture": config.student.architecture,
        "vocab_size": config.student.vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }

    train_step = make_train_step(
        student.apply, distillation_loss=_make_train_loss(config)
    )
    start_step = 0
    params = student.init_params(jax.random.PRNGKey(config.training.seed))
    parent_checkpoint_manifest: dict[str, object] | None = None
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
        parent_checkpoint_manifest = asdict(loaded.manifest)

    state = TrainState(
        params=params,
        step=start_step,
        learning_rate=config.optimizer.learning_rate,
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
                "training": asdict(config.training),
                "losses": asdict(config.losses),
            },
            teacher_target=teacher_target,
            student=student_config,
            checkpoint=checkpoint_metadata,
            tags=config.tracking.tags,
            notes=config.tracking.notes,
            overwrite=config.tracking.overwrite,
        )
        write_run_metadata(run_context)
        metrics_logger = MetricsLogger(run_context.paths.metrics_jsonl)
        if checkpoint_out is None:
            checkpoint_out = run_context.paths.default_final_checkpoint
            checkpoint_overwrite = config.tracking.overwrite

    initial_loss: float | None = None
    final_metrics: dict[str, float] | None = None
    with metrics_logger or _NullMetricsLogger() as active_logger:
        for step_index in range(config.training.max_steps):
            batch = shard_batches[step_index % len(shard_batches)]
            state, metrics = train_step(state, batch)
            float_metrics = metrics_to_floats(metrics)
            if initial_loss is None:
                initial_loss = float_metrics["loss"]
            final_metrics = float_metrics
            if metrics_logger is not None:
                active_logger.log(
                    step=state.step,
                    values=float_metrics,
                    phase="train",
                    extra={
                        "local_step": step_index + 1,
                        "start_step": start_step,
                        "shard_index": step_index % len(shard_batches),
                    },
                )

    assert initial_loss is not None
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
            notes=[
                "simple JSON + NPZ checkpoint",
                f"distillation stage {config.stage}",
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
                "checkpoint_out": checkpoint_out,
                "resume_from": config.checkpoint.resume_from,
                "target_bundle": dataset.bundle_dir,
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
    )


def _make_train_loss(config: DistillStageConfig):
    def loss_fn(student_output, batch):
        breakdown = compute_distill_loss(
            student_output=student_output,
            teacher_hidden_states=batch["hidden_states"],
            teacher_logits=batch.get("logits"),
            attention_mask=batch.get("attention_mask"),
            loss_config=config.losses,
        )
        components = {
            "loss": breakdown.total,
        }
        if breakdown.hidden_mse is not None:
            components["hidden_mse"] = breakdown.hidden_mse
        if breakdown.logits_kl is not None:
            components["logits_kl"] = breakdown.logits_kl
        return _TrainLoss(total=breakdown.total, components=components)

    return loss_fn


def _validate_resume_checkpoint(
    checkpoint_architecture: str,
    checkpoint_student_config: dict[str, object],
    *,
    expected_architecture: str,
    expected_student_config: dict[str, int | str],
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
