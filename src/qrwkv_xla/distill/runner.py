from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax

from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.distill.config import DistillStageConfig, validate_distill_stage_config
from qrwkv_xla.distill.losses import compute_distill_loss
from qrwkv_xla.distill.metrics import metrics_to_floats
from qrwkv_xla.students import create_student
from qrwkv_xla.targets import read_manifest
from qrwkv_xla.targets.store import manifest_path
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

    train_step = make_train_step(
        student.apply, distillation_loss=_make_train_loss(config)
    )
    state = TrainState(
        params=student.init_params(jax.random.PRNGKey(config.training.seed)),
        step=0,
        learning_rate=config.optimizer.learning_rate,
    )

    shard_batches = [batch_to_jax(batch) for batch in dataset.iter_shards()]
    if not shard_batches:
        raise ValueError(f"Target bundle contains no shards: {dataset.bundle_dir}")

    initial_loss: float | None = None
    final_metrics: dict[str, float] | None = None
    for step_index in range(config.training.max_steps):
        batch = shard_batches[step_index % len(shard_batches)]
        state, metrics = train_step(state, batch)
        float_metrics = metrics_to_floats(metrics)
        if initial_loss is None:
            initial_loss = float_metrics["loss"]
        final_metrics = float_metrics

    assert initial_loss is not None
    assert final_metrics is not None
    return DistillStageResult(
        stage=config.stage,
        student_architecture=config.student.architecture,
        steps=config.training.max_steps,
        initial_loss=initial_loss,
        final_loss=final_metrics["loss"],
        final_hidden_mse=final_metrics.get("hidden_mse"),
        final_logits_kl=final_metrics.get("logits_kl"),
        target_bundle=dataset.bundle_dir,
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


DistillationStageResult = DistillStageResult
run_distillation_stage = run_distill_stage
