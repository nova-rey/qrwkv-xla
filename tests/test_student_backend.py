from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qrwkv_xla.students import (
    CurrentQRWKVStudentBackend,
    RWKV7QwenReferenceConfig,
    RWKV7QwenReferenceState,
    RWKV7QwenReferenceStudent,
    WKVRuntime,
    create_current_qrwkv_student_backend,
)


def test_current_backend_init_state_matches_direct_student() -> None:
    student = _student()
    backend = CurrentQRWKVStudentBackend(student)

    direct = student.init_state(batch_size=2, next_position=3)
    wrapped = backend.init_state(batch_size=2, next_position=3)

    assert isinstance(wrapped, RWKV7QwenReferenceState)
    assert jnp.array_equal(wrapped.wkv_matrix_state, direct.wkv_matrix_state)
    assert jnp.array_equal(wrapped.shift_state, direct.shift_state)
    assert int(wrapped.next_position) == int(direct.next_position)


def test_current_backend_forward_full_matches_direct_student() -> None:
    student = _student(emit_logits=True)
    backend = CurrentQRWKVStudentBackend(student)
    params = student.init_params(jax.random.PRNGKey(0))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    direct_output, direct_state = student.apply_with_state(
        params,
        input_ids,
        attention_mask=attention_mask,
    )
    wrapped_output, wrapped_state = backend.forward_full(
        params,
        input_ids,
        attention_mask=attention_mask,
    )

    assert jnp.allclose(wrapped_output.hidden_states, direct_output.hidden_states)
    assert direct_output.logits is not None
    assert wrapped_output.logits is not None
    assert jnp.allclose(wrapped_output.logits, direct_output.logits)
    assert jnp.allclose(wrapped_state.wkv_matrix_state, direct_state.wkv_matrix_state)
    assert jnp.allclose(wrapped_state.shift_state, direct_state.shift_state)
    assert int(wrapped_state.next_position) == int(direct_state.next_position)


def test_current_backend_forward_step_matches_direct_student() -> None:
    student = _student(emit_logits=True)
    backend = CurrentQRWKVStudentBackend(student)
    params = student.init_params(jax.random.PRNGKey(1))
    state = student.init_state(batch_size=1)
    input_ids = jnp.array([[4]], dtype=jnp.int32)
    attention_mask = jnp.array([[1]], dtype=jnp.int32)

    direct_output, direct_state = student.step(
        params,
        input_ids,
        state,
        attention_mask=attention_mask,
    )
    wrapped_output, wrapped_state = backend.forward_step(
        params,
        input_ids,
        state,
        attention_mask=attention_mask,
    )

    assert jnp.allclose(wrapped_output.hidden_states, direct_output.hidden_states)
    assert direct_output.logits is not None
    assert wrapped_output.logits is not None
    assert jnp.allclose(wrapped_output.logits, direct_output.logits)
    assert jnp.allclose(wrapped_state.wkv_matrix_state, direct_state.wkv_matrix_state)
    assert jnp.allclose(wrapped_state.shift_state, direct_state.shift_state)
    assert int(wrapped_state.next_position) == int(direct_state.next_position)


def test_current_backend_export_import_state_delegates() -> None:
    student = _student()
    backend = CurrentQRWKVStudentBackend(student)
    state = student.init_state(batch_size=1, next_position=5)

    payload = backend.export_state(state)
    imported = backend.import_state(payload, template=state)

    assert payload["export_path"].endswith(".export_reference_state_object")
    assert isinstance(imported, RWKV7QwenReferenceState)
    assert jnp.array_equal(imported.wkv_matrix_state, state.wkv_matrix_state)
    assert jnp.array_equal(imported.shift_state, state.shift_state)
    assert int(imported.next_position) == int(state.next_position)


def test_current_backend_logits_extracts_existing_student_output_logits() -> None:
    student = _student(emit_logits=True)
    backend = CurrentQRWKVStudentBackend(student)
    params = student.init_params(jax.random.PRNGKey(2))
    output, _state = backend.forward_full(
        params,
        jnp.array([[1, 2]], dtype=jnp.int32),
    )

    logits = backend.logits(output)

    assert output.logits is not None
    assert jnp.array_equal(logits, output.logits)


def test_current_backend_logits_fails_when_logits_are_unavailable() -> None:
    student = _student(emit_logits=False)
    backend = CurrentQRWKVStudentBackend(student)
    params = student.init_params(jax.random.PRNGKey(3))
    output, _state = backend.forward_full(
        params,
        jnp.array([[1, 2]], dtype=jnp.int32),
    )

    with pytest.raises(ValueError, match="does not include logits"):
        backend.logits(output)


def test_current_backend_factory_preserves_reference_runtime_default() -> None:
    backend = create_current_qrwkv_student_backend(
        "rwkv7_qwen_reference",
        vocab_size=17,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
    )

    assert isinstance(backend, CurrentQRWKVStudentBackend)
    assert backend.student.config.effective_wkv_runtime is WKVRuntime.REFERENCE


def _student(*, emit_logits: bool = False) -> RWKV7QwenReferenceStudent:
    return RWKV7QwenReferenceStudent(
        RWKV7QwenReferenceConfig(
            vocab_size=17,
            hidden_size=4,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            emit_logits=emit_logits,
        )
    )
