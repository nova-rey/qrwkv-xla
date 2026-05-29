from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from qrwkv_xla.contracts import (
    VocabContract,
    validate_vocab_compatibility,
    vocab_contract_from_metadata,
)
from qrwkv_xla.students import (
    RWKV7QwenReferenceConfig,
    RWKV7QwenReferenceStudent,
    WKVRuntime,
    qrwkv_student_config_from_vocab_contract,
)
from qrwkv_xla.teachers import SyntheticTeacherBackend, emit_teacher_target_store


def test_contract_a_selects_student_config_vocab_size_8() -> None:
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-a", 8))

    assert selected.config.vocab_size == 8
    assert selected.vocab_contract.vocab_size == 8


def test_contract_b_selects_student_config_vocab_size_16() -> None:
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-b", 16))

    assert selected.config.vocab_size == 16
    assert selected.vocab_contract.vocab_size == 16


def test_student_config_selection_does_not_mutate_base_config() -> None:
    base = RWKV7QwenReferenceConfig(
        vocab_size=8,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
    )

    selected = qrwkv_student_config_from_vocab_contract(
        _contract("synthetic-b", 16),
        base_config=base,
    )

    assert base.vocab_size == 8
    assert selected.config.vocab_size == 16
    assert selected.config.hidden_size == base.hidden_size
    assert selected.config.num_layers == base.num_layers
    assert selected.config.num_heads == base.num_heads
    assert selected.config.num_kv_heads == base.num_kv_heads


def test_initialized_student_shapes_match_selected_contract_vocab_size() -> None:
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-b", 16))
    student = RWKV7QwenReferenceStudent(selected.config)
    params = student.init_params(jax.random.PRNGKey(0))

    output, _state = student.apply_with_state(
        params,
        jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
    )

    assert params["token_embedding"]["weight"].shape == (16, 4)
    assert params["lm_head"]["weight"].shape == (4, 16)
    assert params["lm_head"]["bias"].shape == (16,)
    assert output.logits is not None
    assert output.logits.shape[-1] == selected.vocab_contract.vocab_size


def test_target_artifact_and_matching_student_contract_are_compatible(
    tmp_path: Path,
) -> None:
    target_contract = _emit_target_contract(tmp_path, "synthetic-a", 8)
    selected = qrwkv_student_config_from_vocab_contract(target_contract)

    result = validate_vocab_compatibility(target_contract, selected.vocab_contract)

    assert result.compatible is True


def test_target_artifact_and_mismatched_student_contract_fail_clearly(
    tmp_path: Path,
) -> None:
    target_contract = _emit_target_contract(tmp_path, "synthetic-a", 8)
    selected = qrwkv_student_config_from_vocab_contract(
        replace(target_contract, vocab_size=16)
    )

    result = validate_vocab_compatibility(target_contract, selected.vocab_contract)

    assert result.compatible is False
    assert "vocab_size mismatch" in result.reason


def test_student_config_selection_preserves_reference_default() -> None:
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-a", 8))

    assert selected.runtime is WKVRuntime.REFERENCE
    assert selected.config.effective_wkv_runtime is WKVRuntime.REFERENCE


def test_student_config_selection_keeps_pallas_explicit_opt_in() -> None:
    selected = qrwkv_student_config_from_vocab_contract(
        _contract("synthetic-a", 8),
        runtime=WKVRuntime.PALLAS,
    )

    assert selected.runtime is WKVRuntime.PALLAS
    assert selected.config.effective_wkv_runtime is WKVRuntime.PALLAS


def _contract(tokenizer_id: str, vocab_size: int) -> VocabContract:
    return VocabContract(
        tokenizer_id=tokenizer_id,
        tokenizer_hash=tokenizer_id,
        vocab_size=vocab_size,
        model_id=f"{tokenizer_id}-model",
        model_family="synthetic",
    )


def _emit_target_contract(
    tmp_path: Path,
    tokenizer_id: str,
    vocab_size: int,
) -> VocabContract:
    store = emit_teacher_target_store(
        SyntheticTeacherBackend(
            tokenizer_id=tokenizer_id,
            tokenizer_hash=tokenizer_id,
            vocab_size=vocab_size,
        ),
        tmp_path / tokenizer_id,
        num_examples=2,
        sequence_length=3,
    )
    return vocab_contract_from_metadata(store.metadata)
