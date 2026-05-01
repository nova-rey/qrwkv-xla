from __future__ import annotations

import pytest

from qrwkv_xla.schedules import (
    LearningRateScheduleConfig,
    validate_lr_schedule_config,
)


def test_valid_constant_schedule() -> None:
    validate_lr_schedule_config(
        LearningRateScheduleConfig(),
        base_learning_rate=0.1,
    )


def test_valid_warmup_cosine_schedule() -> None:
    validate_lr_schedule_config(
        LearningRateScheduleConfig(
            type="warmup_cosine",
            warmup_steps=2,
            total_steps=10,
            min_learning_rate=0.01,
        ),
        base_learning_rate=0.1,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (LearningRateScheduleConfig(type="nope"), "lr_schedule.type"),
        (LearningRateScheduleConfig(warmup_steps=-1), "warmup_steps"),
        (
            LearningRateScheduleConfig(type="warmup_cosine"),
            "total_steps is required",
        ),
        (
            LearningRateScheduleConfig(
                type="warmup_cosine",
                warmup_steps=2,
                total_steps=2,
            ),
            "total_steps must be >",
        ),
        (
            LearningRateScheduleConfig(min_learning_rate=0.2),
            "min_learning_rate",
        ),
    ],
)
def test_invalid_schedule_config_raises(
    config: LearningRateScheduleConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_lr_schedule_config(config, base_learning_rate=0.1)
