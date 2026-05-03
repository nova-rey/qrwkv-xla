from __future__ import annotations

import jax.numpy as jnp
import pytest

from qrwkv_xla.distill.config import DistillLossConfig, LossWeightConfig
from qrwkv_xla.distill.losses import compute_distill_loss, masked_attention_mixer_mse
from qrwkv_xla.students.base import StudentOutput


def test_masked_attention_mixer_mse_respects_mask() -> None:
    student = jnp.array([[[[0.0], [5.0]]]])
    teacher = jnp.array([[[[1.0], [100.0]]]])
    mask = jnp.array([[1, 0]])
    loss = masked_attention_mixer_mse(student, teacher, mask)
    assert float(loss) == pytest.approx(1.0, abs=1e-6)


def test_compute_distill_loss_requires_mixer_outputs() -> None:
    with pytest.raises(ValueError, match="mixer_outputs"):
        compute_distill_loss(
            student_output=StudentOutput(hidden_states=jnp.zeros((1, 1, 1, 1))),
            teacher_hidden_states=jnp.zeros((1, 1, 1, 1)),
            teacher_logits=None,
            teacher_attention_targets=jnp.zeros((1, 1, 1, 1)),
            attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
            loss_config=DistillLossConfig(
                hidden_mse=LossWeightConfig(enabled=False, weight=0.0),
                attention_or_mixer=LossWeightConfig(enabled=True, weight=1.0),
            ),
        )
