from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.artifacts.cascaded_soft_labels import encode_cascaded_soft_labels
from qrwkv_xla.contracts import vocab_contract_from_metadata
from qrwkv_xla.targets import TeacherTargetStore
from qrwkv_xla.teachers import (
    HFTeacherBackend,
    HFTeacherUnavailable,
    emit_teacher_target_store,
)


def test_importing_teachers_does_not_require_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    import qrwkv_xla.teachers as teachers

    assert teachers.HFTeacherBackend is HFTeacherBackend


def test_hf_teacher_backend_constructs_without_loading() -> None:
    backend = HFTeacherBackend("local/fake")

    assert backend.model_id == "local/fake"
    assert backend.local_files_only is True


def test_fake_hf_teacher_emits_full_logits_store(tmp_path: Path) -> None:
    backend = _fake_backend(vocab_size=11)

    store = emit_teacher_target_store(
        backend,
        tmp_path / "hf_targets",
        num_examples=2,
        sequence_length=4,
    )
    arrays = store.read_shard(0)

    assert isinstance(store, TeacherTargetStore)
    assert arrays["input_ids"].shape == (2, 4)
    assert arrays["attention_mask"].shape == (2, 4)
    assert arrays["logits"].shape == (2, 4, store.metadata.vocab_size)
    assert arrays["logits"].dtype == np.float32
    assert store.metadata.target_type == "full_logits"
    assert store.metadata.model_id == "local/fake-model"
    assert store.metadata.tokenizer_id == "local/fake-tokenizer"
    assert store.metadata.vocab_size == 11
    store.validate()


def test_hf_teacher_backend_emits_targets_from_encoded_without_retokenizing() -> None:
    backend = _fake_backend(vocab_size=11)
    tokenizer = backend.tokenizer
    assert isinstance(tokenizer, _FakeTokenizer)

    encoded = backend.encode_prompts(("alpha", "beta"), sequence_length=4)
    first_tokenizer_calls = tokenizer.call_count
    emitted = backend.emit_targets_from_encoded(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    assert tokenizer.call_count == first_tokenizer_calls
    assert first_tokenizer_calls == 1
    assert emitted["input_ids"].shape == (2, 4)
    assert emitted["attention_mask"].shape == (2, 4)
    assert emitted["logits"].shape == (2, 4, 11)


def test_hf_teacher_backend_compact_targets_match_cascaded_reference() -> None:
    backend = _fake_backend(vocab_size=11)
    tokenizer = backend.tokenizer
    assert isinstance(tokenizer, _FakeTokenizer)

    encoded = backend.encode_prompts(("alpha", "beta"), sequence_length=4)
    compact = backend.emit_compact_targets_from_encoded(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        top_k=5,
        bucket_edges=(1.0, 0.1, 0.01, 0.0),
    )
    full = backend.emit_targets_from_encoded(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    assert tokenizer.call_count == 1
    assert compact.input_ids.shape == (2, 4)
    assert compact.top_token_ids.shape == (2, 4, 5)
    assert compact.bucket_masses.shape == (2, 4, 3)
    assert compact.estimated_raw_logits_bytes == full["logits"].nbytes
    for batch_index in range(2):
        for position in range(4):
            reference = encode_cascaded_soft_labels(
                full["logits"][batch_index, position],
                top_k=5,
                bucket_edges=(1.0, 0.1, 0.01, 0.0),
                top_log_probs_dtype="float32",
            )
            assert compact.top_token_ids[batch_index, position].tolist() == (
                reference.top_token_ids.tolist()
            )
            np.testing.assert_allclose(
                compact.top_log_probs[batch_index, position],
                reference.top_log_probs.astype(np.float32),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                compact.bucket_masses[batch_index, position],
                reference.bucket_mass.astype(np.float32),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                compact.bucket_mean_log_probs[batch_index, position],
                reference.bucket_mean_logp.astype(np.float32),
                atol=1e-6,
            )
            assert compact.entropy[batch_index, position] == pytest.approx(
                float(reference.teacher_entropy), abs=1e-6
            )


def test_fake_hf_vocab_contract_round_trips_from_metadata(tmp_path: Path) -> None:
    backend = _fake_backend(vocab_size=13)
    expected = backend.vocab_contract()

    store = emit_teacher_target_store(
        backend,
        tmp_path / "hf_targets",
        num_examples=2,
        sequence_length=3,
    )
    from_metadata = vocab_contract_from_metadata(store.metadata)

    assert from_metadata.tokenizer_id == expected.tokenizer_id
    assert from_metadata.vocab_size == expected.vocab_size
    assert from_metadata.model_id == expected.model_id
    assert from_metadata.model_family == expected.model_family
    assert from_metadata.tokenizer_hash is None


def test_unavailable_transformers_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    backend = HFTeacherBackend("local/missing")

    with pytest.raises(HFTeacherUnavailable, match="transformers is not installed"):
        backend.load()


def test_unavailable_local_model_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("not cached")

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=_raise),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=_raise),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    backend = HFTeacherBackend("local/missing", local_files_only=True)

    with pytest.raises(HFTeacherUnavailable, match="local_files_only=True"):
        backend.load()


def test_hf_teacher_backend_uses_generic_model_id_not_qwen() -> None:
    backend = _fake_backend(vocab_size=7)

    assert backend.model_id == "local/fake-model"
    assert "qwen" not in backend.model_id.lower()
    assert backend.vocab_contract().model_family == "hf"


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
        self.call_count = 0

    def __call__(
        self,
        prompts: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        self.call_count += 1
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
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, *, input_ids: Any, attention_mask: Any | None) -> object:
        del attention_mask
        input_array = np.asarray(input_ids, dtype=np.float32)
        vocab = np.arange(self.vocab_size, dtype=np.float32)[None, None, :]
        logits = input_array[:, :, None] * 0.25 + vocab * 0.125
        return SimpleNamespace(logits=logits.astype(np.float32))
