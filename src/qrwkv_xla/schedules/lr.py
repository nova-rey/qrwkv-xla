from __future__ import annotations

import math
from collections.abc import Callable

from qrwkv_xla.schedules.config import LearningRateScheduleConfig


def learning_rate_at_step(
    *,
    step: int,
    base_learning_rate: float,
    config: LearningRateScheduleConfig,
) -> float:
    if step < 0:
        raise ValueError("step must be >= 0")
    if config.type == "constant":
        return float(base_learning_rate)
    if config.type != "warmup_cosine":
        raise ValueError(f"unknown learning rate schedule type: {config.type!r}")

    if config.warmup_steps > 0 and step < config.warmup_steps:
        return float(base_learning_rate * (step + 1) / config.warmup_steps)

    if config.total_steps is None:
        raise ValueError("total_steps is required for warmup_cosine")
    decay_steps = max(1, config.total_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        config.min_learning_rate
        + (base_learning_rate - config.min_learning_rate) * cosine
    )


def build_lr_schedule(
    config: LearningRateScheduleConfig,
    *,
    base_learning_rate: float,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return learning_rate_at_step(
            step=step,
            base_learning_rate=base_learning_rate,
            config=config,
        )

    return schedule
