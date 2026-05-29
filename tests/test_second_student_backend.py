from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.contracts import (
    VocabContract,
    validate_direct_logit_eligibility,
    validate_store_for_student_config,
    vocab_contract_from_metadata,
)
from qrwkv_xla.students import (
    CURRENT_QRWKV_ARCHITECTURE_ID,
    TINY_DEBUG_ARCHITECTURE_ID,
    TinyDebugState,
    TinyDebugStudentBackend,
    WKVRuntime,
    available_student_architectures,
    create_student_backend,
)
from qrwkv_xla.targets import load_offline_target_batch, mse_logits_loss
from qrwkv_xla.teachers import SyntheticTeacherBackend, emit_teacher_target_store


def test_registry_lists_tiny_debug_with_current_qrwkv() -> None:
    architectures = available_student_architectures()

    assert CURRENT_QRWKV_ARCHITECTURE_ID in architectures
    assert TINY_DEBUG_ARCHITECTURE_ID in architectures


def test_create_tiny_debug_backend_from_registry() -> None:
    backend = create_student_backend(
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        vocab_contract=_contract("synthetic-a", 8),
    )

    assert isinstance(backend, TinyDebugStudentBackend)


def test_default_architecture_still_current_qrwkv() -> None:
    backend = create_student_backend(vocab_contract=_contract("synthetic-a", 8))

    assert not isinstance(backend, TinyDebugStudentBackend)
    assert backend.student.config.vocab_size == 8


def test_unknown_architecture_still_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown student architecture_id"):
        create_student_backend(
            architecture_id="missing",
            vocab_contract=_contract("synthetic-a", 8),
        )


def test_tiny_debug_vocab_size_8_controls_logits_dim() -> None:
    backend = _tiny_debug(vocab_size=8)

    logits = _logits(backend)

    assert logits.shape == (1, 3, 8)


def test_tiny_debug_vocab_size_16_controls_logits_dim() -> None:
    backend = _tiny_debug(vocab_size=16)

    logits = _logits(backend)

    assert logits.shape == (1, 3, 16)


def test_tiny_debug_outputs_are_deterministic() -> None:
    backend = _tiny_debug(vocab_size=8)
    params = backend.init_params(jax.random.PRNGKey(0))
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

    first, _ = backend.forward_full(params, input_ids)
    second, _ = backend.forward_full(params, input_ids)

    assert jnp.array_equal(backend.logits(first), backend.logits(second))


def test_tiny_debug_state_export_import_round_trips() -> None:
    backend = _tiny_debug(vocab_size=8)
    state = TinyDebugState(step=jnp.asarray(4, dtype=jnp.int32))

    payload = backend.export_state(state)
    imported = backend.import_state(payload)

    assert payload == {"architecture_id": TINY_DEBUG_ARCHITECTURE_ID, "step": 4}
    assert int(imported.step) == 4


def test_tiny_debug_finite_loss_against_compatible_artifact(tmp_path: Path) -> None:
    store = _emit_store(tmp_path, tokenizer_id="synthetic-a", vocab_size=8)
    contract = vocab_contract_from_metadata(store.metadata)
    backend = create_student_backend(
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        vocab_contract=contract,
    )
    compatibility = validate_direct_logit_eligibility(
        teacher_contract=contract,
        student_contract=contract,
        target_type=store.metadata.target_type,
    )
    batch = load_offline_target_batch(store)
    params = backend.init_params(jax.random.PRNGKey(0))

    output, _state = backend.forward_full(
        params,
        jnp.asarray(batch.input_ids),
        attention_mask=jnp.asarray(batch.attention_mask),
    )
    loss = mse_logits_loss(backend.logits(output), batch.teacher_logits)

    assert compatibility.compatible is True
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def test_artifact_a_and_tiny_debug_student_a_compatibility_passes(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, tokenizer_id="synthetic-a", vocab_size=8)
    backend = create_student_backend(
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        vocab_contract=vocab_contract_from_metadata(store.metadata),
    )

    result = validate_store_for_student_config(store, backend)

    assert result.compatible is True


def test_artifact_a_and_tiny_debug_student_b_compatibility_fails(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, tokenizer_id="synthetic-a", vocab_size=8)
    contract = vocab_contract_from_metadata(store.metadata)
    backend = create_student_backend(
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        vocab_contract=replace(contract, vocab_size=16),
    )

    result = validate_store_for_student_config(store, backend)

    assert result.compatible is False
    assert "vocab_size mismatch" in result.reason


def test_tiny_debug_rejects_pallas_runtime_clearly() -> None:
    with pytest.raises(ValueError, match="does not support pallas"):
        create_student_backend(
            architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
            vocab_contract=_contract("synthetic-a", 8),
            runtime=WKVRuntime.PALLAS,
        )


def test_tiny_debug_requires_no_hf_qwen_internet_or_accelerator() -> None:
    backend = _tiny_debug(vocab_size=8)

    logits = np.asarray(_logits(backend))

    assert logits.shape[-1] == 8
    assert np.isfinite(logits).all()


def _contract(tokenizer_id: str, vocab_size: int) -> VocabContract:
    return VocabContract(
        tokenizer_id=tokenizer_id,
        tokenizer_hash=tokenizer_id,
        vocab_size=vocab_size,
        model_id=f"{tokenizer_id}-model",
        model_family="synthetic",
    )


def _tiny_debug(*, vocab_size: int) -> TinyDebugStudentBackend:
    backend = create_student_backend(
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        vocab_contract=_contract("synthetic-a", vocab_size),
    )
    assert isinstance(backend, TinyDebugStudentBackend)
    return backend


def _logits(backend: TinyDebugStudentBackend) -> jax.Array:
    params = backend.init_params(jax.random.PRNGKey(0))
    output, _state = backend.forward_full(
        params,
        jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
    )
    return backend.logits(output)


def _emit_store(
    tmp_path: Path,
    *,
    tokenizer_id: str,
    vocab_size: int,
):
    return emit_teacher_target_store(
        SyntheticTeacherBackend(
            tokenizer_id=tokenizer_id,
            tokenizer_hash=tokenizer_id,
            vocab_size=vocab_size,
        ),
        tmp_path / tokenizer_id,
        num_examples=2,
        sequence_length=3,
    )
