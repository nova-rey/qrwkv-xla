from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.teachers import (
    DEFAULT_HF_SPECIMEN_MODEL_ID,
    HFTeacherBackend,
    run_hf_teacher_specimen_smoke,
)


def test_unavailable_transformers_reports_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        model_id="local/missing",
    )

    assert result.status == "unavailable"
    assert result.reason == "transformers_not_installed"
    assert result.target_store_validated is False
    assert result.vocab_contract_extracted is False
    assert result.error_type == "HFTeacherUnavailable"


def test_report_schema_includes_claims_not_made(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9),
        model_id="local/fake-model",
        prompts=("alpha", "beta"),
        sequence_length=4,
    )

    report = result.to_report()

    assert report["phase"] == "P104"
    assert report["scope"] == "tiny_hf_causal_lm_teacher_specimen_smoke"
    assert "qwen_specific_support" in report["claims_not_made"]
    assert "gpt2_specific_architecture" in report["claims_not_made"]
    assert "student_consumption_proven" in report["claims_not_made"]
    assert "training_ready" in report["claims_not_made"]


def test_local_files_only_defaults_true(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9),
    )

    assert result.local_files_only is True
    assert result.allow_downloads is False


def test_allow_downloads_is_opt_in_only(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9),
        allow_downloads=True,
    )

    assert result.allow_downloads is True
    assert result.local_files_only is False


def test_no_qwen_specific_behavior_is_required(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9),
        model_id="local/fake-model",
    )

    assert result.status == "pass"
    assert "qwen" not in result.model_id.lower()
    assert "qwen_specific_support" in result.claims_not_made


def test_fake_hf_specimen_emits_valid_target_store_report(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9),
        model_id="local/fake-model",
        prompts=("alpha", "beta"),
        sequence_length=4,
    )

    assert result.status == "pass"
    assert result.model_id == "local/fake-model"
    assert result.target_store_validated is True
    assert result.vocab_contract_extracted is True
    assert result.tokenizer_id == "local/fake-tokenizer"
    assert result.vocab_size == 9
    assert result.sequence_length == 4
    assert result.num_examples == 2
    assert result.target_type == "full_logits"
    assert result.logits_shape == (2, 4, 9)


def test_report_can_be_written_to_json(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9),
    )
    report_path = result.write_json(tmp_path / "report.json")

    assert report_path.is_file()
    assert '"phase": "P104"' in report_path.read_text(encoding="utf-8")


def test_default_specimen_model_id_is_configurable(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(
        target_store=tmp_path / "targets",
        backend=_fake_backend(vocab_size=9, model_id="local/custom-specimen"),
        model_id="local/custom-specimen",
    )

    assert DEFAULT_HF_SPECIMEN_MODEL_ID
    assert result.model_id == "local/custom-specimen"


def _fake_backend(
    *,
    vocab_size: int,
    model_id: str = "local/fake-model",
) -> HFTeacherBackend:
    return HFTeacherBackend(
        model_id,
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
