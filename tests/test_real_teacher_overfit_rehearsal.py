from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from qrwkv_xla.contracts import (
    CompatibilityStatus,
    validate_store_for_student_config,
    vocab_contract_from_metadata,
)
from qrwkv_xla.students import (
    CURRENT_QRWKV_ARCHITECTURE_ID,
    TINY_DEBUG_ARCHITECTURE_ID,
    CurrentQRWKVStudentBackend,
    TinyDebugStudentBackend,
    WKVRuntime,
    create_student_backend,
    qrwkv_student_config_from_vocab_contract,
)
from qrwkv_xla.targets import TeacherTargetStore
from qrwkv_xla.teachers import HFTeacherBackend, emit_teacher_target_store
from qrwkv_xla.training import run_tiny_real_teacher_overfit_rehearsal


def test_fake_hf_teacher_backend_style_artifact_is_used(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=11)

    result = run_tiny_real_teacher_overfit_rehearsal(store=store)

    assert store.metadata.created_by == "HFTeacherBackend"
    assert result.teacher_created_by == "HFTeacherBackend"
    assert result.teacher_model_id == "local/fake-model"


def test_artifact_contract_selects_backend_through_registry(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=11)
    contract = vocab_contract_from_metadata(store.metadata)

    backend = create_student_backend(vocab_contract=contract)

    assert isinstance(backend, CurrentQRWKVStudentBackend)
    assert backend.student.config.vocab_size == 11


def test_p99_compatibility_gate_passes_before_update(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=11)
    selected = qrwkv_student_config_from_vocab_contract(
        vocab_contract_from_metadata(store.metadata)
    )

    compatibility = validate_store_for_student_config(store, selected)
    result = run_tiny_real_teacher_overfit_rehearsal(store=store)

    assert compatibility.status is CompatibilityStatus.COMPATIBLE
    assert result.compatibility_status == "compatible"
    assert result.compatibility_reason.startswith("compatible:")


def test_initial_and_final_losses_are_finite(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_overfit_rehearsal(
        store=_emit_fake_hf_store(tmp_path, vocab_size=11)
    )

    assert result.status == "pass"
    assert result.initial_loss is not None
    assert result.final_loss is not None
    assert result.loss_finite is True
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)


def test_deterministic_tiny_rehearsal_moves_loss_down(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_overfit_rehearsal(
        store=_emit_fake_hf_store(tmp_path, vocab_size=11)
    )

    assert result.initial_loss is not None
    assert result.final_loss is not None
    assert result.final_loss < result.initial_loss
    assert result.loss_moved is True


def test_result_identifies_tiny_trainable_head_path(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_overfit_rehearsal(
        store=_emit_fake_hf_store(tmp_path, vocab_size=11)
    )

    assert result.path_used == "tiny_trainable_logit_head"
    assert result.training_kind == "tiny_controlled_rehearsal"
    assert result.steps == 3


def test_mismatched_student_contract_blocks_before_update(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=11)
    contract = vocab_contract_from_metadata(store.metadata)

    result = run_tiny_real_teacher_overfit_rehearsal(
        store=store,
        student_vocab_contract=replace(contract, vocab_size=13),
    )

    assert result.status == "incompatible"
    assert result.initial_loss is None
    assert result.final_loss is None
    assert result.loss_moved is False
    assert result.loss_finite is False
    assert result.path_used == "blocked_before_update"
    assert "vocab_size mismatch" in result.compatibility_reason


def test_baseline_requires_no_transformers_internet_or_downloads(
    tmp_path: Path,
) -> None:
    result = run_tiny_real_teacher_overfit_rehearsal(
        store=_emit_fake_hf_store(tmp_path, vocab_size=11)
    )
    report = result.to_report()

    assert report["hf_required_for_baseline_ci"] is False
    assert report["internet_required"] is False
    assert report["gpu_or_tpu_required"] is False


def test_no_qwen_specific_code_path_is_required(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_overfit_rehearsal(
        store=_emit_fake_hf_store(tmp_path, vocab_size=11)
    )

    assert "qwen_specific_support" in result.claims_not_made
    assert result.teacher_model_id == "local/fake-model"


def test_reference_remains_default(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_overfit_rehearsal(
        store=_emit_fake_hf_store(tmp_path, vocab_size=11)
    )

    assert result.student_architecture_id == CURRENT_QRWKV_ARCHITECTURE_ID
    assert result.student_runtime == WKVRuntime.REFERENCE.value


def test_pallas_remains_opt_in(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=11)
    selected = qrwkv_student_config_from_vocab_contract(
        vocab_contract_from_metadata(store.metadata),
        runtime=WKVRuntime.PALLAS,
    )

    compatibility = validate_store_for_student_config(store, selected)

    assert compatibility.compatible is True
    assert selected.runtime is WKVRuntime.PALLAS


def test_tiny_debug_can_be_selected_without_contract_or_runtime_changes(
    tmp_path: Path,
) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=11)
    contract = vocab_contract_from_metadata(store.metadata)
    backend = create_student_backend(
        vocab_contract=contract,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    result = run_tiny_real_teacher_overfit_rehearsal(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert isinstance(backend, TinyDebugStudentBackend)
    assert result.status == "pass"
    assert result.student_architecture_id == TINY_DEBUG_ARCHITECTURE_ID
    assert result.student_runtime == WKVRuntime.REFERENCE.value


def _emit_fake_hf_store(tmp_path: Path, *, vocab_size: int) -> TeacherTargetStore:
    return emit_teacher_target_store(
        _fake_backend(vocab_size=vocab_size),
        tmp_path / f"fake_hf_v{vocab_size}",
        num_examples=2,
        sequence_length=3,
    )


def _fake_backend(*, vocab_size: int) -> HFTeacherBackend:
    return HFTeacherBackend(
        "local/fake-model",
        tokenizer=_FakeTokenizer(vocab_size=vocab_size),
        model=_FakeCausalLM(vocab_size=vocab_size),
        prompts=("alpha", "beta"),
    )


class _FakeTokenizer:
    name_or_path = "local/fake-tokenizer"
    eos_token = "<eos>"
    pad_token = None
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = 0
    unk_token_id = 2

    def __init__(self, *, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        prompts: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        rows = []
        for row, _prompt in enumerate(prompts):
            rows.append([(row + col) % self.vocab_size for col in range(max_length)])
        input_ids = np.asarray(rows, dtype=np.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=np.int32),
        }


class _FakeCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids: Any, attention_mask: Any | None) -> object:
        del attention_mask
        input_array = np.asarray(input_ids, dtype=np.float32)
        vocab = np.arange(self.vocab_size, dtype=np.float32)[None, None, :]
        logits = input_array[:, :, None] * 0.25 + vocab * 0.125
        return SimpleNamespace(logits=logits.astype(np.float32))
