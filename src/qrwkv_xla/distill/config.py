from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qrwkv_xla.distributed.config import (
    DistributedConfig,
    load_distributed_config,
    validate_distributed_config,
)
from qrwkv_xla.optimizers import OptimizerConfig, validate_optimizer_config
from qrwkv_xla.schedules import (
    LearningRateScheduleConfig,
    validate_lr_schedule_config,
)
from qrwkv_xla.students.factory import STUDENT_ARCHITECTURES


@dataclass(frozen=True)
class DistillStudentConfig:
    architecture: str = "rwkv7_reference"
    vocab_size: int = 512
    hidden_size: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    emit_logits: bool = False
    tie_embeddings: bool = False
    emit_mixer_outputs: bool = False


@dataclass(frozen=True)
class DistillOptimizerConfig:
    type: str = "sgd"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0

    def to_optimizer_config(self) -> OptimizerConfig:
        return OptimizerConfig(
            type=self.type,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
        )


DistillLRScheduleConfig = LearningRateScheduleConfig


@dataclass(frozen=True)
class DistillGradientConfig:
    max_grad_norm: float | None = None
    clip_epsilon: float = 1e-6


@dataclass(frozen=True)
class DistillTrainingConfig:
    max_steps: int = 5
    seed: int = 0


@dataclass(frozen=True)
class DistillCheckpointConfig:
    checkpoint_out: Path | None = None
    resume_from: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class DistillTrackingConfig:
    run_root: Path | None = None
    run_name: str | None = None
    enabled: bool = False
    overwrite: bool = False
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LossWeightConfig:
    enabled: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class DistillLossConfig:
    hidden_mse: LossWeightConfig = field(default_factory=LossWeightConfig)
    logits_kl: LossWeightConfig = field(
        default_factory=lambda: LossWeightConfig(enabled=False, weight=0.0)
    )
    attention_or_mixer: LossWeightConfig = field(
        default_factory=lambda: LossWeightConfig(enabled=False, weight=0.0)
    )
    bucket_shape_loss_weight: float = 0.0
    bucket_shape_loss_type: str = "kl"


@dataclass(frozen=True)
class DistillStageConfig:
    stage: int = 0
    targets_dir: Path = Path("artifacts/teacher_targets/fake_export")
    student: DistillStudentConfig = field(default_factory=DistillStudentConfig)
    optimizer: DistillOptimizerConfig = field(default_factory=DistillOptimizerConfig)
    lr_schedule: DistillLRScheduleConfig = field(
        default_factory=DistillLRScheduleConfig
    )
    gradients: DistillGradientConfig = field(default_factory=DistillGradientConfig)
    training: DistillTrainingConfig = field(default_factory=DistillTrainingConfig)
    losses: DistillLossConfig = field(default_factory=DistillLossConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: DistillCheckpointConfig = field(default_factory=DistillCheckpointConfig)
    tracking: DistillTrackingConfig = field(default_factory=DistillTrackingConfig)


def load_distill_stage_config(path: str | Path) -> DistillStageConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    raw_stage = data.get("distillation")
    if raw_stage is None:
        raise ValueError("Config must contain a top-level distillation section")
    if not isinstance(raw_stage, dict):
        raise ValueError("distillation section must be a mapping")

    config = DistillStageConfig(
        stage=int(raw_stage.get("stage", 0)),
        targets_dir=Path(
            str(raw_stage.get("targets_dir", "artifacts/teacher_targets/fake_export"))
        ),
        student=_load_student(raw_stage.get("student", {})),
        optimizer=_load_optimizer(raw_stage.get("optimizer", {})),
        lr_schedule=_load_lr_schedule(raw_stage.get("lr_schedule", {})),
        gradients=_load_gradients(raw_stage.get("gradients", {})),
        training=_load_training(raw_stage.get("training", {})),
        losses=_load_losses(raw_stage.get("losses", {})),
        distributed=load_distributed_config(
            raw_stage.get("distributed"),
            section="distillation",
        ),
        checkpoint=_load_checkpoint(raw_stage.get("checkpoint", {})),
        tracking=_load_tracking(raw_stage.get("tracking", {})),
    )
    validate_distill_stage_config(config)
    return config


def validate_distill_stage_config(config: DistillStageConfig) -> None:
    if config.stage < 0:
        raise ValueError(f"stage must be >= 0, got {config.stage}")
    if not str(config.targets_dir):
        raise ValueError("targets_dir must be non-empty")
    if config.student.architecture not in STUDENT_ARCHITECTURES:
        raise ValueError(
            f"student.architecture must be one of {sorted(STUDENT_ARCHITECTURES)}"
        )
    if config.student.vocab_size <= 0:
        raise ValueError("student.vocab_size must be > 0")
    if config.student.hidden_size is not None and config.student.hidden_size <= 0:
        raise ValueError("student.hidden_size must be > 0 when provided")
    if config.student.num_layers is not None and config.student.num_layers <= 0:
        raise ValueError("student.num_layers must be > 0 when provided")
    if config.student.num_heads is not None and config.student.num_heads <= 0:
        raise ValueError("student.num_heads must be > 0 when provided")
    if config.student.num_kv_heads is not None and config.student.num_kv_heads <= 0:
        raise ValueError("student.num_kv_heads must be > 0 when provided")
    if (
        config.student.hidden_size is not None
        and config.student.num_heads is not None
        and config.student.hidden_size % config.student.num_heads != 0
    ):
        raise ValueError("student.hidden_size must be divisible by student.num_heads")
    if (
        config.student.num_heads is not None
        and config.student.num_kv_heads is not None
        and config.student.num_heads % config.student.num_kv_heads != 0
    ):
        raise ValueError("student.num_heads must be divisible by student.num_kv_heads")
    if config.student.architecture == "rwkv7_qwen_reference":
        if config.student.num_heads is None:
            raise ValueError("rwkv7_qwen_reference requires explicit student.num_heads")
        if config.student.num_kv_heads is None:
            raise ValueError(
                "rwkv7_qwen_reference requires explicit student.num_kv_heads"
            )
    if (
        config.losses.logits_kl.enabled
        and config.losses.logits_kl.weight > 0
        and not config.student.emit_logits
    ):
        raise ValueError("logits_kl requires student.emit_logits=true")
    validate_optimizer_config(config.optimizer.to_optimizer_config())
    validate_lr_schedule_config(
        config.lr_schedule,
        base_learning_rate=config.optimizer.learning_rate,
    )
    validate_distributed_config(config.distributed)
    if (
        config.gradients.max_grad_norm is not None
        and config.gradients.max_grad_norm <= 0
    ):
        raise ValueError("gradients.max_grad_norm must be > 0 when provided")
    if config.gradients.clip_epsilon <= 0:
        raise ValueError("gradients.clip_epsilon must be > 0")
    if config.training.max_steps <= 0:
        raise ValueError("training.max_steps must be > 0")
    if config.training.seed < 0:
        raise ValueError("training.seed must be >= 0")
    if (
        config.checkpoint.checkpoint_out is not None
        and config.checkpoint.resume_from is not None
        and config.checkpoint.checkpoint_out == config.checkpoint.resume_from
        and not config.checkpoint.overwrite
    ):
        raise ValueError(
            "checkpoint_out and resume_from cannot be the same path unless "
            "checkpoint overwrite is enabled"
        )
    for name, value in (
        ("checkpoint_out", config.checkpoint.checkpoint_out),
        ("resume_from", config.checkpoint.resume_from),
    ):
        if value is not None and "checkpoints" not in value.parts:
            raise ValueError(f"{name} must live under a checkpoints/ directory")
    if config.tracking.run_root is not None and (
        "runs" not in config.tracking.run_root.parts
        and config.tracking.run_root.name != "runs"
    ):
        raise ValueError("tracking.run_root must be runs/ or live under runs/")
    if not all(isinstance(tag, str) for tag in config.tracking.tags):
        raise ValueError("tracking.tags must be a list of strings")
    if not all(isinstance(note, str) for note in config.tracking.notes):
        raise ValueError("tracking.notes must be a list of strings")

    enabled_positive = False
    for name in ("hidden_mse", "logits_kl", "attention_or_mixer"):
        weight_config = getattr(config.losses, name)
        if weight_config.weight < 0:
            raise ValueError(f"loss weight for {name} must be >= 0")
        if weight_config.enabled and weight_config.weight > 0:
            enabled_positive = True
    if not enabled_positive:
        raise ValueError("at least one enabled loss must have weight > 0")
    if config.losses.bucket_shape_loss_weight < 0:
        raise ValueError("bucket_shape_loss_weight must be >= 0")
    if config.losses.bucket_shape_loss_type not in {"kl", "log_mse"}:
        raise ValueError("bucket_shape_loss_type must be 'kl' or 'log_mse'")


def _load_student(data: Any) -> DistillStudentConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.student must be a mapping")
    return DistillStudentConfig(
        architecture=str(data.get("architecture", "rwkv7_reference")),
        vocab_size=int(data.get("vocab_size", 512)),
        hidden_size=_optional_int(data.get("hidden_size")),
        num_layers=_optional_int(data.get("num_layers")),
        num_heads=_optional_int(data.get("num_heads")),
        num_kv_heads=_optional_int(data.get("num_kv_heads")),
        emit_logits=bool(data.get("emit_logits", False)),
        tie_embeddings=bool(data.get("tie_embeddings", False)),
        emit_mixer_outputs=bool(data.get("emit_mixer_outputs", False)),
    )


def _load_optimizer(data: Any) -> DistillOptimizerConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.optimizer must be a mapping")
    return DistillOptimizerConfig(
        type=str(data.get("type", "sgd")),
        learning_rate=float(data.get("learning_rate", 1e-3)),
        beta1=float(data.get("beta1", 0.9)),
        beta2=float(data.get("beta2", 0.999)),
        epsilon=float(data.get("epsilon", 1e-8)),
        weight_decay=float(data.get("weight_decay", 0.0)),
    )


def _load_lr_schedule(data: Any) -> DistillLRScheduleConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.lr_schedule must be a mapping")
    total_steps = data.get("total_steps")
    return DistillLRScheduleConfig(
        type=str(data.get("type", "constant")),
        warmup_steps=int(data.get("warmup_steps", 0)),
        total_steps=None if total_steps is None else int(total_steps),
        min_learning_rate=float(data.get("min_learning_rate", 0.0)),
    )


def _load_gradients(data: Any) -> DistillGradientConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.gradients must be a mapping")
    return DistillGradientConfig(
        max_grad_norm=_optional_float(data.get("max_grad_norm")),
        clip_epsilon=float(data.get("clip_epsilon", 1e-6)),
    )


def _load_training(data: Any) -> DistillTrainingConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.training must be a mapping")
    return DistillTrainingConfig(
        max_steps=int(data.get("max_steps", 5)),
        seed=int(data.get("seed", 0)),
    )


def _load_checkpoint(data: Any) -> DistillCheckpointConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.checkpoint must be a mapping")
    return DistillCheckpointConfig(
        checkpoint_out=_optional_path(data.get("checkpoint_out")),
        resume_from=_optional_path(data.get("resume_from")),
        overwrite=bool(data.get("overwrite", False)),
    )


def _load_tracking(data: Any) -> DistillTrackingConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.tracking must be a mapping")
    return DistillTrackingConfig(
        run_root=_optional_path(data.get("run_root")),
        run_name=_optional_str(data.get("run_name")),
        enabled=bool(data.get("enabled", False)),
        overwrite=bool(data.get("overwrite", False)),
        tags=_string_list(data.get("tags", []), "tracking.tags"),
        notes=_string_list(data.get("notes", []), "tracking.notes"),
    )


def _load_losses(data: Any) -> DistillLossConfig:
    if not isinstance(data, dict):
        raise ValueError("distillation.losses must be a mapping")
    return DistillLossConfig(
        hidden_mse=_load_loss_weight(data.get("hidden_mse"), enabled=True, weight=1.0),
        logits_kl=_load_loss_weight(data.get("logits_kl"), enabled=False, weight=0.0),
        attention_or_mixer=_load_loss_weight(
            data.get("attention_or_mixer"), enabled=False, weight=0.0
        ),
        bucket_shape_loss_weight=float(data.get("bucket_shape_loss_weight", 0.0)),
        bucket_shape_loss_type=str(data.get("bucket_shape_loss_type", "kl")),
    )


def _load_loss_weight(
    data: Any,
    *,
    enabled: bool,
    weight: float,
) -> LossWeightConfig:
    if data is None:
        return LossWeightConfig(enabled=enabled, weight=weight)
    if not isinstance(data, dict):
        raise ValueError("loss configuration sections must be mappings")
    return LossWeightConfig(
        enabled=bool(data.get("enabled", enabled)),
        weight=float(data.get("weight", weight)),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return list(value)


DistillationCheckpointConfig = DistillCheckpointConfig
DistillationTrackingConfig = DistillTrackingConfig
DistillationStudentConfig = DistillStudentConfig
DistillationOptimizerConfig = DistillOptimizerConfig
DistillationLRScheduleConfig = DistillLRScheduleConfig
DistillationGradientConfig = DistillGradientConfig
DistillationTrainingConfig = DistillTrainingConfig
DistillationLossConfig = DistillLossConfig
DistillationStageConfig = DistillStageConfig
load_distillation_config = load_distill_stage_config
