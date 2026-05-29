from __future__ import annotations

import pytest

from qrwkv_xla.contracts import (
    VocabContract,
    validate_vocab_compatibility,
    vocab_contract_from_metadata,
)
from qrwkv_xla.teachers import SyntheticTeacherBackend


def test_vocab_contract_requires_non_empty_tokenizer_id() -> None:
    with pytest.raises(ValueError, match="tokenizer_id"):
        VocabContract(tokenizer_id="", vocab_size=8)


def test_vocab_contract_rejects_non_positive_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        VocabContract(tokenizer_id="synthetic-a", vocab_size=0)


def test_vocab_contract_rejects_out_of_range_special_token_ids() -> None:
    with pytest.raises(ValueError, match="outside vocab_size"):
        VocabContract(
            tokenizer_id="synthetic-a",
            vocab_size=8,
            special_tokens={"eos": 8},
        )


def test_vocab_contract_from_metadata_extracts_target_fields() -> None:
    metadata = SyntheticTeacherBackend(
        model_id="synthetic-model-a",
        model_family="synthetic-family",
        tokenizer_id="synthetic-a",
        tokenizer_hash="hash-a",
        vocab_size=8,
    ).build_metadata(num_examples=2, sequence_length=3)

    contract = vocab_contract_from_metadata(metadata)

    assert contract.tokenizer_id == "synthetic-a"
    assert contract.tokenizer_hash == "hash-a"
    assert contract.vocab_size == 8
    assert contract.model_id == "synthetic-model-a"
    assert contract.model_family == "synthetic-family"


def test_vocab_compatibility_passes_for_matching_contracts() -> None:
    contract = VocabContract(
        tokenizer_id="synthetic-a",
        vocab_size=8,
        tokenizer_hash="hash-a",
    )

    result = validate_vocab_compatibility(contract, contract)

    assert result.compatible is True
    assert result.reason == "vocab contracts match"


def test_vocab_compatibility_fails_for_vocab_size_mismatch() -> None:
    target = VocabContract(tokenizer_id="synthetic-a", vocab_size=8)
    student = VocabContract(tokenizer_id="synthetic-a", vocab_size=16)

    result = validate_vocab_compatibility(target, student)

    assert result.compatible is False
    assert "vocab_size mismatch" in result.reason


def test_vocab_compatibility_fails_for_tokenizer_id_mismatch() -> None:
    target = VocabContract(tokenizer_id="synthetic-a", vocab_size=8)
    student = VocabContract(tokenizer_id="synthetic-b", vocab_size=8)

    result = validate_vocab_compatibility(target, student)

    assert result.compatible is False
    assert "tokenizer_id mismatch" in result.reason


def test_vocab_compatibility_fails_for_tokenizer_hash_mismatch() -> None:
    target = VocabContract(
        tokenizer_id="synthetic-a",
        vocab_size=8,
        tokenizer_hash="hash-a",
    )
    student = VocabContract(
        tokenizer_id="synthetic-a",
        vocab_size=8,
        tokenizer_hash="hash-b",
    )

    result = validate_vocab_compatibility(target, student)

    assert result.compatible is False
    assert "tokenizer_hash mismatch" in result.reason
