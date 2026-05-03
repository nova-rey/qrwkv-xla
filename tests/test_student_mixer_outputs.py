from __future__ import annotations

import jax
import jax.numpy as jnp

from qrwkv_xla.students import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
    TinyStudent,
    TinyStudentConfig,
)


def test_tiny_student_can_emit_mixer_outputs() -> None:
    student = TinyStudent(
        TinyStudentConfig(
            vocab_size=16,
            hidden_size=4,
            num_layers=2,
            emit_mixer_outputs=True,
        )
    )
    params = student.init_params(jax.random.PRNGKey(0))
    output = student.apply(params, jnp.array([[1, 2, 3]], dtype=jnp.int32))
    assert output.hidden_states.shape == (1, 2, 3, 4)
    assert output.mixer_outputs is not None
    assert output.mixer_outputs.shape == (1, 2, 3, 4)


def test_rwkv7_reference_can_emit_mixer_outputs() -> None:
    student = RWKV7ReferenceStudent(
        RWKV7ReferenceConfig(
            vocab_size=16,
            hidden_size=4,
            num_layers=2,
            emit_mixer_outputs=True,
        )
    )
    params = student.init_params(jax.random.PRNGKey(0))
    output = student.apply(params, jnp.array([[1, 2, 3]], dtype=jnp.int32))
    assert output.hidden_states.shape == (1, 2, 3, 4)
    assert output.mixer_outputs is not None
    assert output.mixer_outputs.shape == (1, 2, 3, 4)
