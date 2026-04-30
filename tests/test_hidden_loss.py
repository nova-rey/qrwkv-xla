from __future__ import annotations

import jax.numpy as jnp
import pytest

from qrwkv_xla.losses import hidden_mse_loss


def test_hidden_mse_loss_zero_for_equal_hidden_states() -> None:
    hidden = jnp.ones((2, 2, 3, 4))

    loss = hidden_mse_loss(hidden, hidden)

    assert float(loss) == pytest.approx(0.0)


def test_hidden_mse_loss_positive_for_different_hidden_states() -> None:
    student = jnp.zeros((1, 2, 3, 4))
    teacher = jnp.ones((1, 2, 3, 4))

    loss = hidden_mse_loss(student, teacher)

    assert float(loss) > 0.0


def test_hidden_mse_loss_ignores_masked_positions() -> None:
    student = jnp.array([[[[0.0], [100.0]]]])
    teacher = jnp.array([[[[0.0], [0.0]]]])
    attention_mask = jnp.array([[1, 0]])

    loss = hidden_mse_loss(student, teacher, attention_mask)

    assert float(loss) == pytest.approx(0.0)


def test_hidden_mse_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="student_hidden.shape must match"):
        hidden_mse_loss(jnp.zeros((1, 2, 3, 4)), jnp.zeros((1, 2, 3, 5)))
