from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qrwkv_xla.students import (
    CurrentQRWKVStudentBackend,
    PallasStudentRuntime,
    ReferenceJaxStudentRuntime,
    RWKV7QwenReferenceConfig,
    RWKV7QwenReferenceStudent,
    WKVRuntime,
    create_current_qrwkv_student_backend,
    create_student_runtime,
    reference_wkv_sequence_update,
    reference_wkv_update,
)


def test_student_runtime_default_is_reference_jax() -> None:
    runtime = create_student_runtime()

    assert isinstance(runtime, ReferenceJaxStudentRuntime)
    assert runtime.name == "reference_jax"
    assert runtime.wkv_runtime is WKVRuntime.REFERENCE


def test_student_runtime_explicit_reference_maps_to_reference_jax() -> None:
    runtime = create_student_runtime("reference")

    assert isinstance(runtime, ReferenceJaxStudentRuntime)
    assert runtime.wkv_runtime is WKVRuntime.REFERENCE


def test_student_runtime_explicit_pallas_maps_to_pallas_opt_in() -> None:
    runtime = create_student_runtime(WKVRuntime.PALLAS)

    assert isinstance(runtime, PallasStudentRuntime)
    assert runtime.name == "pallas"
    assert runtime.wkv_runtime is WKVRuntime.PALLAS


def test_student_runtime_unknown_value_fails_clearly() -> None:
    with pytest.raises(
        ValueError, match="wkv_runtime must be one of reference, pallas"
    ):
        create_student_runtime("bogus")


def test_reference_runtime_step_and_sequence_match_existing_reference_path() -> None:
    runtime = create_student_runtime("reference")
    initial_state = jnp.arange(4, dtype=jnp.float32).reshape(1, 1, 2, 2) / 3.0
    k = jnp.asarray([[[0.1, 0.2]]], dtype=jnp.float32)
    v = jnp.asarray([[[0.2, 0.3]]], dtype=jnp.float32)
    decay = jnp.asarray([[[0.5, 0.25]]], dtype=jnp.float32)
    k_seq = jnp.stack([k, k + 0.1], axis=0)
    v_seq = jnp.stack([v, v + 0.1], axis=0)
    decay_seq = jnp.stack([decay, decay], axis=0)

    assert jnp.allclose(
        runtime.step(initial_state, k, v, decay),
        reference_wkv_update(initial_state, k, v, decay),
    )
    runtime_sequence = runtime.sequence(initial_state, k_seq, v_seq, decay_seq)
    direct_sequence = reference_wkv_sequence_update(
        initial_state,
        k_seq,
        v_seq,
        decay_seq,
    )

    assert jnp.allclose(
        runtime_sequence["final_state"],
        direct_sequence["final_state"],
    )
    assert jnp.allclose(
        runtime_sequence["per_step_states"],
        direct_sequence["per_step_states"],
    )


def test_current_backend_default_runtime_preserves_direct_reference_behavior() -> None:
    student = _student(emit_logits=True)
    backend = CurrentQRWKVStudentBackend(student)
    params = student.init_params(jax.random.PRNGKey(0))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    direct_output, direct_state = student.apply_with_state(params, input_ids)
    wrapped_output, wrapped_state = backend.forward_full(params, input_ids)

    assert backend.runtime.wkv_runtime is WKVRuntime.REFERENCE
    assert jnp.allclose(wrapped_output.hidden_states, direct_output.hidden_states)
    assert wrapped_output.logits is not None
    assert direct_output.logits is not None
    assert jnp.allclose(wrapped_output.logits, direct_output.logits)
    assert jnp.allclose(wrapped_state.wkv_matrix_state, direct_state.wkv_matrix_state)


def test_current_backend_from_config_preserves_reference_default() -> None:
    backend = create_current_qrwkv_student_backend(
        "rwkv7_qwen_reference",
        vocab_size=17,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
    )

    assert isinstance(backend, CurrentQRWKVStudentBackend)
    assert backend.runtime.wkv_runtime is WKVRuntime.REFERENCE
    assert backend.student.config.effective_wkv_runtime is WKVRuntime.REFERENCE


def test_current_backend_from_config_keeps_pallas_explicit_opt_in() -> None:
    backend = create_current_qrwkv_student_backend(
        "rwkv7_qwen_reference",
        vocab_size=17,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        runtime="pallas",
    )

    assert isinstance(backend, CurrentQRWKVStudentBackend)
    assert backend.runtime.wkv_runtime is WKVRuntime.PALLAS
    assert backend.student.config.effective_wkv_runtime is WKVRuntime.PALLAS


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
