from __future__ import annotations

import jax
import jax.numpy as jnp

from qrwkv_xla.students import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
    TinyStudent,
    TinyStudentConfig,
)


def test_tiny_student_logits_shape_and_hidden_shape() -> None:
    input_ids = jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32)
    hidden_only = TinyStudent(
        TinyStudentConfig(vocab_size=8, hidden_size=4, num_layers=2)
    )
    logits_student = TinyStudent(
        TinyStudentConfig(
            vocab_size=8,
            hidden_size=4,
            num_layers=2,
            emit_logits=True,
        )
    )

    hidden_output = hidden_only.apply(
        hidden_only.init_params(jax.random.PRNGKey(0)),
        input_ids,
    )
    logits_output = logits_student.apply(
        logits_student.init_params(jax.random.PRNGKey(0)),
        input_ids,
    )

    assert hidden_output.hidden_states.shape == (2, 2, 3, 4)
    assert hidden_output.logits is None
    assert logits_output.hidden_states.shape == (2, 2, 3, 4)
    assert logits_output.logits is not None
    assert logits_output.logits.shape == (2, 3, 8)


def test_rwkv7_reference_student_logits_shape_and_jit() -> None:
    input_ids = jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32)
    student = RWKV7ReferenceStudent(
        RWKV7ReferenceConfig(
            vocab_size=8,
            hidden_size=4,
            num_layers=2,
            emit_logits=True,
        )
    )
    params = student.init_params(jax.random.PRNGKey(0))

    output = jax.jit(student.apply)(params, input_ids)

    assert output.hidden_states.shape == (2, 2, 3, 4)
    assert output.logits is not None
    assert output.logits.shape == (2, 3, 8)


def test_tied_embedding_logits_shape() -> None:
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    student = TinyStudent(
        TinyStudentConfig(
            vocab_size=8,
            hidden_size=4,
            num_layers=2,
            emit_logits=True,
            tie_embeddings=True,
        )
    )
    params = student.init_params(jax.random.PRNGKey(0))
    output = student.apply(params, input_ids)

    assert "lm_head" not in params
    assert "lm_head_bias" in params
    assert output.logits is not None
    assert output.logits.shape == (1, 3, 8)
