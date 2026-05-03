from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qrwkv_xla.distill.config import (
    DistillCheckpointConfig,
    DistillGradientConfig,
    DistillLRScheduleConfig,
    DistillOptimizerConfig,
    DistillTrackingConfig,
)
from qrwkv_xla.distributed.config import (
    DistributedConfig,
    load_distributed_config,
    validate_distributed_config,
)
from qrwkv_xla.generation.tokenizer import (
    TokenizerConfig,
    available_tokenizer_backends,
    normalize_tokenizer_config,
)
from qrwkv_xla.optimizers import validate_optimizer_config
from qrwkv_xla.schedules import validate_lr_schedule_config


@dataclass(frozen=True)
class LMDataConfig:
    prompt_corpus: Path
    prompt_split: str | None = "train"
    prompt_tags: tuple[str, ...] = ()
    prompt_limit: int | None = None
    sequence_length: int = 64
    batch_size: int = 2
    tokenizer: str | TokenizerConfig = "smoke"
    shuffle: bool = False
    seed: int = 0


@dataclass(frozen=True)
class LMStudentConfig:
    architecture: str = "rwkv7_reference"
    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    emit_logits: bool = True
    tie_embeddings: bool = False


@dataclass(frozen=True)
class LMTrainingConfig:
    stage: int = 3
    max_steps: int = 2
    seed: int = 0


@dataclass(frozen=True)
class LMStageConfig:
    data: LMDataConfig
    student: LMStudentConfig = field(default_factory=LMStudentConfig)
    optimizer: DistillOptimizerConfig = field(default_factory=DistillOptimizerConfig)
    lr_schedule: DistillLRScheduleConfig = field(
        default_factory=DistillLRScheduleConfig
    )
    gradients: DistillGradientConfig = field(default_factory=DistillGradientConfig)
    training: LMTrainingConfig = field(default_factory=LMTrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: DistillCheckpointConfig = field(default_factory=DistillCheckpointConfig)
    tracking: DistillTrackingConfig = field(default_factory=DistillTrackingConfig)


def load_lm_stage_config(path: str | Path) -> LMStageConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    raw_stage = data.get("lm") or data.get("lm_stage")
    if raw_stage is None:
        raise ValueError("Config must contain a top-level lm section")
    if not isinstance(raw_stage, dict):
        raise ValueError("lm section must be a mapping")

    config = LMStageConfig(
        data=_load_data(raw_stage.get("data", {})),
        student=_load_student(raw_stage.get("student", {})),
        optimizer=_load_optimizer(raw_stage.get("optimizer", {})),
        lr_schedule=_load_lr_schedule(raw_stage.get("lr_schedule", {})),
        gradients=_load_gradients(raw_stage.get("gradients", {})),
        training=_load_training(raw_stage.get("training", {})),
        distributed=load_distributed_config(raw_stage.get("distributed"), section="lm"),
        checkpoint=_load_checkpoint(raw_stage.get("checkpoint", {})),
        tracking=_load_tracking(raw_stage.get("tracking", {})),
    )
    validate_lm_stage_config(config)
    return config


def validate_lm_stage_config(config: LMStageConfig) -> None:
    if not config.data.prompt_corpus.exists():
        raise ValueError(
            f"data.prompt_corpus does not exist: {config.data.prompt_corpus}"
        )
    if config.data.sequence_length <= 1:
        raise ValueError("data.sequence_length must be > 1")
    if config.data.batch_size <= 0:
        raise ValueError("data.batch_size must be > 0")
    tokenizer_config = normalize_tokenizer_config(config.data.tokenizer)
    if tokenizer_config.backend not in available_tokenizer_backends():
        raise ValueError(
            f"data.tokenizer backend must be one of {available_tokenizer_backends()}"
        )
    if tokenizer_config.backend == "hf" and not tokenizer_config.tokenizer_id:
        raise ValueError("data.tokenizer.tokenizer_id is required for HF/Qwen")
    if tokenizer_config.backend == "smoke" and tokenizer_config.tokenizer_id not in (
        None,
        "smoke",
    ):
        raise ValueError("data.tokenizer.tokenizer_id is not used for smoke")
    if config.data.prompt_limit is not None and config.data.prompt_limit <= 0:
        raise ValueError("data.prompt_limit must be > 0 when provided")
    if config.data.seed < 0:
        raise ValueError("data.seed must be >= 0")
    if config.student.architecture not in {"tiny_student", "rwkv7_reference"}:
        raise ValueError(
            "student.architecture must be one of {'tiny_student', 'rwkv7_reference'}"
        )
    if config.student.vocab_size <= 0:
        raise ValueError("student.vocab_size must be > 0")
    if tokenizer_config.backend == "smoke" and config.student.vocab_size < 257:
        raise ValueError("student.vocab_size must be >= 257 for SmokeTokenizer")
    if (
        tokenizer_config.vocab_size is not None
        and config.student.vocab_size != tokenizer_config.vocab_size
    ):
        raise ValueError(
            "student.vocab_size must match tokenizer.vocab_size when provided"
        )
    if config.student.hidden_size <= 0:
        raise ValueError("student.hidden_size must be > 0")
    if config.student.num_layers <= 0:
        raise ValueError("student.num_layers must be > 0")
    if not config.student.emit_logits:
        raise ValueError("Stage 3 CE training requires student.emit_logits=true")
    if config.training.stage != 3:
        raise ValueError("training.stage must be 3")
    if config.training.max_steps <= 0:
        raise ValueError("training.max_steps must be > 0")
    if config.training.seed < 0:
        raise ValueError("training.seed must be >= 0")
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


def _load_data(data: Any) -> LMDataConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.data must be a mapping")
    if "prompt_corpus" not in data:
        raise ValueError("lm.data.prompt_corpus is required")
    return LMDataConfig(
        prompt_corpus=Path(str(data["prompt_corpus"])),
        prompt_split=_optional_str(data.get("prompt_split", "train")),
        prompt_tags=tuple(
            _string_list(data.get("prompt_tags", []), "data.prompt_tags")
        ),
        prompt_limit=_optional_int(data.get("prompt_limit")),
        sequence_length=int(data.get("sequence_length", 64)),
        batch_size=int(data.get("batch_size", 2)),
        tokenizer=normalize_tokenizer_config(data.get("tokenizer", "smoke")),
        shuffle=bool(data.get("shuffle", False)),
        seed=int(data.get("seed", 0)),
    )


def _load_student(data: Any) -> LMStudentConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.student must be a mapping")
    return LMStudentConfig(
        architecture=str(data.get("architecture", "rwkv7_reference")),
        vocab_size=int(data.get("vocab_size", 512)),
        hidden_size=int(data.get("hidden_size", 128)),
        num_layers=int(data.get("num_layers", 2)),
        emit_logits=bool(data.get("emit_logits", True)),
        tie_embeddings=bool(data.get("tie_embeddings", False)),
    )


def _load_optimizer(data: Any) -> DistillOptimizerConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.optimizer must be a mapping")
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
        raise ValueError("lm.lr_schedule must be a mapping")
    total_steps = data.get("total_steps")
    return DistillLRScheduleConfig(
        type=str(data.get("type", "constant")),
        warmup_steps=int(data.get("warmup_steps", 0)),
        total_steps=None if total_steps is None else int(total_steps),
        min_learning_rate=float(data.get("min_learning_rate", 0.0)),
    )


def _load_gradients(data: Any) -> DistillGradientConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.gradients must be a mapping")
    return DistillGradientConfig(
        max_grad_norm=_optional_float(data.get("max_grad_norm")),
        clip_epsilon=float(data.get("clip_epsilon", 1e-6)),
    )


def _load_training(data: Any) -> LMTrainingConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.training must be a mapping")
    return LMTrainingConfig(
        stage=int(data.get("stage", 3)),
        max_steps=int(data.get("max_steps", 2)),
        seed=int(data.get("seed", 0)),
    )


def _load_checkpoint(data: Any) -> DistillCheckpointConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.checkpoint must be a mapping")
    return DistillCheckpointConfig(
        checkpoint_out=_optional_path(data.get("checkpoint_out")),
        resume_from=_optional_path(data.get("resume_from")),
        overwrite=bool(data.get("overwrite", False)),
    )


def _load_tracking(data: Any) -> DistillTrackingConfig:
    if not isinstance(data, dict):
        raise ValueError("lm.tracking must be a mapping")
    return DistillTrackingConfig(
        run_root=_optional_path(data.get("run_root")),
        run_name=_optional_str(data.get("run_name")),
        enabled=bool(data.get("enabled", False)),
        overwrite=bool(data.get("overwrite", False)),
        tags=_string_list(data.get("tags", []), "tracking.tags"),
        notes=_string_list(data.get("notes", []), "tracking.notes"),
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
