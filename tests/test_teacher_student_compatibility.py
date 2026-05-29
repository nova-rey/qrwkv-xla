from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from qrwkv_xla.contracts import (
    CompatibilityStatus,
    VocabContract,
    validate_direct_logit_eligibility,
    validate_store_for_student_config,
)
from qrwkv_xla.students import WKVRuntime, qrwkv_student_config_from_vocab_contract
from qrwkv_xla.teachers import SyntheticTeacherBackend, emit_teacher_target_store


def test_matching_contracts_full_logits_are_compatible() -> None:
    contract = _contract("synthetic-a", 8)

    result = validate_direct_logit_eligibility(
        teacher_contract=contract,
        student_contract=contract,
        target_type="full_logits",
    )

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.compatible is True
    assert "contracts match" in result.reason


def test_vocab_size_mismatch_is_incompatible() -> None:
    result = validate_direct_logit_eligibility(
        teacher_contract=_contract("synthetic-a", 8),
        student_contract=_contract("synthetic-a", 16),
        target_type="full_logits",
    )

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert result.compatible is False
    assert "vocab_size mismatch" in result.reason


def test_tokenizer_id_mismatch_is_incompatible() -> None:
    result = validate_direct_logit_eligibility(
        teacher_contract=_contract("synthetic-a", 8),
        student_contract=_contract("synthetic-b", 8),
        target_type="full_logits",
    )

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "tokenizer_id mismatch" in result.reason


def test_tokenizer_hash_mismatch_is_incompatible() -> None:
    result = validate_direct_logit_eligibility(
        teacher_contract=replace(_contract("synthetic-a", 8), tokenizer_hash="a"),
        student_contract=replace(_contract("synthetic-a", 8), tokenizer_hash="b"),
        target_type="full_logits",
    )

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "tokenizer_hash mismatch" in result.reason


def test_special_token_mismatch_is_incompatible() -> None:
    result = validate_direct_logit_eligibility(
        teacher_contract=replace(
            _contract("synthetic-a", 8),
            special_tokens={"eos": 1},
        ),
        student_contract=replace(
            _contract("synthetic-a", 8),
            special_tokens={"eos": 2},
        ),
        target_type="full_logits",
    )

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "special token mismatch" in result.reason


def test_hidden_states_target_type_is_unsupported() -> None:
    contract = _contract("synthetic-a", 8)

    result = validate_direct_logit_eligibility(
        teacher_contract=contract,
        student_contract=contract,
        target_type="hidden_states",
    )

    assert result.status is CompatibilityStatus.UNSUPPORTED
    assert "target_type 'hidden_states'" in result.reason


def test_unsupported_loss_mode_is_unsupported() -> None:
    contract = _contract("synthetic-a", 8)

    result = validate_direct_logit_eligibility(
        teacher_contract=contract,
        student_contract=contract,
        target_type="full_logits",
        loss_mode="hidden_projection",
    )

    assert result.status is CompatibilityStatus.UNSUPPORTED
    assert "loss_mode 'hidden_projection'" in result.reason


def test_store_artifact_a_and_selected_student_a_are_compatible(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, tokenizer_id="synthetic-a", vocab_size=8)
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-a", 8))

    result = validate_store_for_student_config(store, selected)

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert result.compatible is True


def test_store_artifact_a_and_selected_student_b_are_incompatible(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, tokenizer_id="synthetic-a", vocab_size=8)
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-a", 16))

    result = validate_store_for_student_config(store, selected)

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "vocab_size mismatch" in result.reason


def test_store_helper_reports_missing_student_contract_as_unsupported(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, tokenizer_id="synthetic-a", vocab_size=8)

    result = validate_store_for_student_config(store, object())

    assert result.status is CompatibilityStatus.UNSUPPORTED
    assert "does not expose a vocab_contract" in result.reason


def test_reference_remains_default() -> None:
    selected = qrwkv_student_config_from_vocab_contract(_contract("synthetic-a", 8))

    assert selected.runtime is WKVRuntime.REFERENCE
    assert selected.config.effective_wkv_runtime is WKVRuntime.REFERENCE


def test_pallas_remains_explicit_opt_in() -> None:
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
