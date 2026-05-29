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
from qrwkv_xla.students import WKVRuntime, qrwkv_student_config_from_vocab_contract
from qrwkv_xla.targets import TeacherTargetStore, load_offline_target_batch
from qrwkv_xla.targets.real_teacher_consumption import (
    compatibility_passed,
    run_real_teacher_offline_consumption_smoke,
)
from qrwkv_xla.teachers import HFTeacherBackend, emit_teacher_target_store


def test_fake_hf_teacher_backend_emits_store_artifact(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)

    assert isinstance(store, TeacherTargetStore)
    assert store.metadata.target_type == "full_logits"
    assert store.metadata.model_family == "hf"
    assert store.metadata.vocab_size == 13
    store.validate()


def test_artifact_contract_selects_compatible_student_config(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)
    contract = vocab_contract_from_metadata(store.metadata)

    selected = qrwkv_student_config_from_vocab_contract(contract)

    assert selected.config.vocab_size == store.metadata.vocab_size
    assert selected.vocab_contract.tokenizer_id == store.metadata.tokenizer_id


def test_compatibility_gate_passes_before_loss_computation(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)
    selected = qrwkv_student_config_from_vocab_contract(
        vocab_contract_from_metadata(store.metadata)
    )

    compatibility = validate_store_for_student_config(store, selected)
    result = run_real_teacher_offline_consumption_smoke(
        store=store,
        selected_student=selected,
    )

    assert compatibility.status is CompatibilityStatus.COMPATIBLE
    assert result.compatibility_status == "compatible"
    assert compatibility_passed(result) is True


def test_offline_batch_loads_through_p95_path(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)

    batch = load_offline_target_batch(store)

    assert batch.input_ids.shape == (2, 3)
    assert batch.attention_mask.shape == (2, 3)
    assert batch.teacher_logits.shape == (2, 3, 13)


def test_current_backend_logits_match_artifact_vocab_and_loss_is_finite(
    tmp_path: Path,
) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)

    result = run_real_teacher_offline_consumption_smoke(store=store)

    assert result.status == "pass"
    assert result.loss is not None
    assert result.loss_finite is True
    assert np.isfinite(result.loss)
    assert result.teacher_logits_shape == (2, 3, 13)
    assert result.student_logits_shape == (2, 3, 13)
    assert result.training_performed is False


def test_mismatched_student_contract_blocks_before_loss(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)
    contract = vocab_contract_from_metadata(store.metadata)
    selected = qrwkv_student_config_from_vocab_contract(
        replace(contract, vocab_size=17)
    )

    result = run_real_teacher_offline_consumption_smoke(
        store=store,
        selected_student=selected,
    )

    assert result.status == "incompatible"
    assert result.loss is None
    assert result.loss_finite is False
    assert result.teacher_logits_shape is None
    assert result.student_logits_shape is None
    assert "vocab_size mismatch" in result.compatibility_reason


def test_no_training_or_optimizer_update_occurs(tmp_path: Path) -> None:
    result = run_real_teacher_offline_consumption_smoke(
        store=_emit_fake_hf_store(tmp_path, vocab_size=13)
    )

    assert result.training_performed is False
    assert "training_ready" in result.claims_not_made


def test_fake_hf_path_requires_no_transformers_or_downloads(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)

    assert store.metadata.source == {"kind": "hf", "local_files_only": "True"}
    assert store.metadata.model_id == "local/fake-model"


def test_reference_remains_default(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)
    result = run_real_teacher_offline_consumption_smoke(store=store)

    assert result.student_runtime == WKVRuntime.REFERENCE.value


def test_pallas_remains_opt_in(tmp_path: Path) -> None:
    store = _emit_fake_hf_store(tmp_path, vocab_size=13)
    selected = qrwkv_student_config_from_vocab_contract(
        vocab_contract_from_metadata(store.metadata),
        runtime=WKVRuntime.PALLAS,
    )

    compatibility = validate_store_for_student_config(store, selected)

    assert compatibility.compatible is True
    assert selected.runtime is WKVRuntime.PALLAS


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
