from __future__ import annotations

import jax.numpy as jnp
import pytest

from qrwkv_xla.distill import DistillLossConfig, LossWeightConfig
from qrwkv_xla.distill.losses import (
    compute_distill_loss,
    logits_kl_loss,
    masked_attention_mixer_mse,
)
from qrwkv_xla.students.base import StudentOutput


def test_hidden_only_loss_returns_finite_total_and_hidden_mse() -> None:
    output = StudentOutput(hidden_states=jnp.zeros((1, 2, 3, 4)))
    breakdown = compute_distill_loss(
        student_output=output,
        teacher_hidden_states=jnp.ones((1, 2, 3, 4)),
        teacher_logits=None,
        attention_mask=jnp.ones((1, 3)),
        loss_config=DistillLossConfig(),
    )
    assert jnp.isfinite(breakdown.total)
    assert breakdown.hidden_mse is not None


def test_hidden_weight_affects_total() -> None:
    output = StudentOutput(hidden_states=jnp.zeros((1, 1, 2, 2)))
    teacher_hidden = jnp.ones((1, 1, 2, 2))
    base = compute_distill_loss(
        student_output=output,
        teacher_hidden_states=teacher_hidden,
        teacher_logits=None,
        attention_mask=jnp.ones((1, 2)),
        loss_config=DistillLossConfig(
            hidden_mse=LossWeightConfig(enabled=True, weight=1.0)
        ),
    )
    doubled = compute_distill_loss(
        student_output=output,
        teacher_hidden_states=teacher_hidden,
        teacher_logits=None,
        attention_mask=jnp.ones((1, 2)),
        loss_config=DistillLossConfig(
            hidden_mse=LossWeightConfig(enabled=True, weight=2.0)
        ),
    )
    assert float(doubled.total) == pytest.approx(float(base.total) * 2)


def test_hidden_mse_uses_loss_mask_over_attention_mask() -> None:
    output = StudentOutput(hidden_states=jnp.zeros((1, 1, 2, 1)))
    teacher_hidden = jnp.array([[[[1.0], [1000.0]]]])

    breakdown = compute_distill_loss(
        student_output=output,
        teacher_hidden_states=teacher_hidden,
        teacher_logits=None,
        attention_mask=jnp.ones((1, 2), dtype=jnp.int32),
        loss_mask=jnp.array([[1, 0]], dtype=jnp.int32),
        loss_config=DistillLossConfig(
            hidden_mse=LossWeightConfig(enabled=True, weight=1.0)
        ),
    )

    assert float(breakdown.total) == pytest.approx(1.0)


def test_logits_kl_uses_loss_mask_over_attention_mask() -> None:
    student = jnp.array([[[0.0, 1.0], [1000.0, -1000.0]]])
    teacher = jnp.array([[[0.0, 1.0], [-1000.0, 1000.0]]])

    unmasked = logits_kl_loss(student[:, :1, :], teacher[:, :1, :], jnp.ones((1, 1)))
    breakdown = compute_distill_loss(
        student_output=StudentOutput(
            hidden_states=jnp.zeros((1, 1, 2, 1)),
            logits=student,
        ),
        teacher_hidden_states=jnp.zeros((1, 1, 2, 1)),
        teacher_logits=teacher,
        attention_mask=jnp.ones((1, 2), dtype=jnp.int32),
        loss_mask=jnp.array([[1, 0]], dtype=jnp.int32),
        loss_config=DistillLossConfig(
            hidden_mse=LossWeightConfig(enabled=False, weight=0.0),
            logits_kl=LossWeightConfig(enabled=True, weight=1.0),
        ),
    )

    assert float(breakdown.total) == pytest.approx(float(unmasked), abs=1e-6)


def test_logits_kl_near_zero_for_equal_logits() -> None:
    logits = jnp.array([[[0.0, 1.0], [2.0, -1.0]]])
    loss = logits_kl_loss(logits, logits, jnp.ones((1, 2)))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_logits_kl_positive_when_logits_differ() -> None:
    student = jnp.array([[[0.0, 1.0], [1.0, 0.0]]])
    teacher = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
    loss = logits_kl_loss(student, teacher, jnp.ones((1, 2)))
    assert float(loss) > 0.0


def test_logits_enabled_without_logits_raises() -> None:
    with pytest.raises(ValueError, match="student_output.logits"):
        compute_distill_loss(
            student_output=StudentOutput(hidden_states=jnp.zeros((1, 1, 1, 2))),
            teacher_hidden_states=jnp.zeros((1, 1, 1, 2)),
            teacher_logits=jnp.zeros((1, 1, 2)),
            attention_mask=jnp.ones((1, 1)),
            loss_config=DistillLossConfig(
                logits_kl=LossWeightConfig(enabled=True, weight=1.0),
            ),
        )


def test_attention_mixer_loss_zero_for_equal_values() -> None:
    loss = masked_attention_mixer_mse(
        jnp.ones((1, 2, 3, 4)),
        jnp.ones((1, 2, 3, 4)),
        jnp.ones((1, 3)),
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_attention_or_mixer_enabled_returns_component() -> None:
    breakdown = compute_distill_loss(
        student_output=StudentOutput(
            hidden_states=jnp.zeros((1, 1, 2, 2)),
            mixer_outputs=jnp.zeros((1, 1, 2, 2)),
        ),
        teacher_hidden_states=jnp.zeros((1, 1, 2, 2)),
        teacher_logits=None,
        teacher_attention_targets=jnp.ones((1, 1, 2, 2)),
        attention_mask=jnp.array([[1, 0]], dtype=jnp.int32),
        loss_config=DistillLossConfig(
            hidden_mse=LossWeightConfig(enabled=False, weight=0.0),
            attention_or_mixer=LossWeightConfig(enabled=True, weight=1.0),
        ),
    )
    assert breakdown.attention_or_mixer is not None
    assert float(breakdown.attention_or_mixer) > 0.0
