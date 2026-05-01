from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningRateScheduleConfig:
    type: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_learning_rate: float = 0.0


def validate_lr_schedule_config(
    config: LearningRateScheduleConfig,
    *,
    base_learning_rate: float,
) -> None:
    if config.type not in {"constant", "warmup_cosine"}:
        raise ValueError(
            "lr_schedule.type must be one of {'constant', 'warmup_cosine'}"
        )
    if config.warmup_steps < 0:
        raise ValueError("lr_schedule.warmup_steps must be >= 0")
    if config.min_learning_rate < 0:
        raise ValueError("lr_schedule.min_learning_rate must be >= 0")
    if config.min_learning_rate > base_learning_rate:
        raise ValueError(
            "lr_schedule.min_learning_rate must be <= optimizer.learning_rate"
        )
    if config.type == "warmup_cosine":
        if config.total_steps is None:
            raise ValueError("lr_schedule.total_steps is required for warmup_cosine")
        if config.total_steps <= 0:
            raise ValueError("lr_schedule.total_steps must be > 0")
        if config.total_steps <= config.warmup_steps:
            raise ValueError(
                "lr_schedule.total_steps must be > lr_schedule.warmup_steps"
            )
