from __future__ import annotations

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.students import StudentOutput, TinyStudent, TinyStudentConfig


def test_tiny_student_defaults() -> None:
    config = TinyStudentConfig()

    assert config.vocab_size == 512
    assert config.hidden_size == 128
    assert config.num_layers == 2


def test_tiny_student_init_params_and_output_shape() -> None:
    student = TinyStudent(TinyStudentConfig(vocab_size=32, hidden_size=8, num_layers=3))
    params = student.init_params(jax.random.PRNGKey(0))

    output = student.apply(params, jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]]))

    assert isinstance(output, StudentOutput)
    assert output.hidden_states.shape == (2, 3, 4, 8)
    assert output.logits is None
    with pytest.raises(FrozenInstanceError):
        output.logits = jnp.ones((2, 4, 32))  # type: ignore[misc]


def test_tiny_student_same_key_init_is_deterministic() -> None:
    student = TinyStudent(TinyStudentConfig(vocab_size=16, hidden_size=4, num_layers=2))

    first = student.init_params(jax.random.PRNGKey(123))
    second = student.init_params(jax.random.PRNGKey(123))

    for key in first:
        np.testing.assert_array_equal(np.asarray(first[key]), np.asarray(second[key]))


def test_invalid_tiny_student_config_raises() -> None:
    with pytest.raises(ValueError, match="hidden_size must be > 0"):
        TinyStudentConfig(hidden_size=0)
