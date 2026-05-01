from __future__ import annotations

import pytest

from qrwkv_xla.schedules import LearningRateScheduleConfig, learning_rate_at_step


def test_constant_returns_base_learning_rate() -> None:
    config = LearningRateScheduleConfig()

    assert learning_rate_at_step(step=0, base_learning_rate=0.1, config=config) == 0.1
    assert learning_rate_at_step(step=99, base_learning_rate=0.1, config=config) == 0.1


def test_warmup_cosine_warmup_and_decay() -> None:
    config = LearningRateScheduleConfig(
        type="warmup_cosine",
        warmup_steps=2,
        total_steps=6,
        min_learning_rate=0.01,
    )

    assert learning_rate_at_step(step=0, base_learning_rate=0.1, config=config) == 0.05
    assert learning_rate_at_step(step=1, base_learning_rate=0.1, config=config) == 0.1
    assert learning_rate_at_step(step=2, base_learning_rate=0.1, config=config) == 0.1
    assert learning_rate_at_step(
        step=4,
        base_learning_rate=0.1,
        config=config,
    ) == pytest.approx(0.055)
    assert learning_rate_at_step(
        step=6,
        base_learning_rate=0.1,
        config=config,
    ) == pytest.approx(0.01)
    assert learning_rate_at_step(
        step=99,
        base_learning_rate=0.1,
        config=config,
    ) == pytest.approx(0.01)


def test_warmup_cosine_without_warmup() -> None:
    config = LearningRateScheduleConfig(
        type="warmup_cosine",
        warmup_steps=0,
        total_steps=4,
        min_learning_rate=0.01,
    )

    assert learning_rate_at_step(step=0, base_learning_rate=0.1, config=config) == 0.1
    assert learning_rate_at_step(
        step=4,
        base_learning_rate=0.1,
        config=config,
    ) == pytest.approx(0.01)


def test_negative_step_raises() -> None:
    with pytest.raises(ValueError, match="step"):
        learning_rate_at_step(
            step=-1,
            base_learning_rate=0.1,
            config=LearningRateScheduleConfig(),
        )
