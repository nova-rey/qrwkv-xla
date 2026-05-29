from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qrwkv_xla.contracts import (
    VocabContract,
    validate_direct_logit_eligibility,
)
from qrwkv_xla.students import (
    CURRENT_QRWKV_ARCHITECTURE_ID,
    CurrentQRWKVStudentBackend,
    WKVRuntime,
    available_student_architectures,
    create_student_backend,
    create_student_runtime,
)


def test_available_student_architectures_includes_current_qrwkv() -> None:
    assert CURRENT_QRWKV_ARCHITECTURE_ID == "current_qrwkv"
    assert CURRENT_QRWKV_ARCHITECTURE_ID in available_student_architectures()


def test_default_architecture_creates_current_qrwkv_backend() -> None:
    backend = create_student_backend(vocab_contract=_contract("synthetic-a", 8))

    assert isinstance(backend, CurrentQRWKVStudentBackend)


def test_explicit_current_qrwkv_architecture_creates_backend() -> None:
    backend = create_student_backend(
        architecture_id=CURRENT_QRWKV_ARCHITECTURE_ID,
        vocab_contract=_contract("synthetic-a", 8),
    )

    assert isinstance(backend, CurrentQRWKVStudentBackend)


def test_unknown_architecture_id_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown student architecture_id"):
        create_student_backend(
            architecture_id="missing_backend",
            vocab_contract=_contract("synthetic-a", 8),
        )


def test_vocab_contract_a_drives_backend_logits_vocab_size_8() -> None:
    backend = create_student_backend(vocab_contract=_contract("synthetic-a", 8))

    logits = _logits_for_backend(backend)

    assert backend.student.config.vocab_size == 8
    assert logits.shape[-1] == 8


def test_vocab_contract_b_drives_backend_logits_vocab_size_16() -> None:
    backend = create_student_backend(vocab_contract=_contract("synthetic-b", 16))

    logits = _logits_for_backend(backend)

    assert backend.student.config.vocab_size == 16
    assert logits.shape[-1] == 16


def test_default_runtime_remains_reference() -> None:
    backend = create_student_backend(vocab_contract=_contract("synthetic-a", 8))

    assert backend.runtime.wkv_runtime is WKVRuntime.REFERENCE
    assert backend.student.config.effective_wkv_runtime is WKVRuntime.REFERENCE


def test_explicit_pallas_runtime_remains_opt_in() -> None:
    backend = create_student_backend(
        vocab_contract=_contract("synthetic-a", 8),
        runtime=WKVRuntime.PALLAS,
    )

    assert backend.runtime.wkv_runtime is WKVRuntime.PALLAS
    assert backend.student.config.effective_wkv_runtime is WKVRuntime.PALLAS


def test_student_runtime_object_keeps_runtime_separate() -> None:
    runtime = create_student_runtime("reference")

    backend = create_student_backend(
        vocab_contract=_contract("synthetic-a", 8),
        runtime=runtime,
    )

    assert backend.runtime is runtime
    assert backend.runtime.wkv_runtime is WKVRuntime.REFERENCE


def test_registry_path_preserves_direct_logit_compatibility_expectation() -> None:
    contract = _contract("synthetic-a", 8)
    backend = create_student_backend(vocab_contract=contract)

    result = validate_direct_logit_eligibility(
        teacher_contract=contract,
        student_contract=contract,
        target_type="full_logits",
    )

    assert isinstance(backend, CurrentQRWKVStudentBackend)
    assert result.compatible is True


def test_registry_path_requires_no_hf_qwen_internet_or_accelerator() -> None:
    backend = create_student_backend(vocab_contract=_contract("synthetic-a", 8))

    assert isinstance(backend, CurrentQRWKVStudentBackend)
    assert backend.student.config.vocab_size == 8


def _contract(tokenizer_id: str, vocab_size: int) -> VocabContract:
    return VocabContract(
        tokenizer_id=tokenizer_id,
        tokenizer_hash=tokenizer_id,
        vocab_size=vocab_size,
        model_id=f"{tokenizer_id}-model",
        model_family="synthetic",
    )


def _logits_for_backend(backend: CurrentQRWKVStudentBackend) -> jax.Array:
    params = backend.init_params(jax.random.PRNGKey(0))
    output, _state = backend.forward_full(
        params,
        jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
    )
    return backend.logits(output)
