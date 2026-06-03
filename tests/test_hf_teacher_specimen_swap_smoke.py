from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.teachers import (
    HFTeacherBackend,
    HFTeacherSpecimenConfig,
    run_hf_teacher_specimen_swap_smoke,
)


def test_two_fake_hf_specimens_use_same_swap_path(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    assert report.status == "pass"
    assert report.specimen_count == 2
    assert report.passed == 2
    assert report.unavailable == 0
    assert report.failed == 0


def test_specimen_metadata_stays_distinct(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)
    first, second = report.specimens

    assert first.model_id == "fake-hf-specimen-a"
    assert first.tokenizer_id == "fake-tokenizer-a"
    assert first.vocab_size == 8
    assert first.logits_shape == (2, 4, 8)
    assert second.model_id == "fake-hf-specimen-b"
    assert second.tokenizer_id == "fake-tokenizer-b"
    assert second.vocab_size == 16
    assert second.logits_shape == (2, 4, 16)


def test_each_specimen_target_store_validates(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    assert all(specimen.status == "pass" for specimen in report.specimens)
    assert all(specimen.target_store_validated for specimen in report.specimens)
    assert all(specimen.vocab_contract_extracted for specimen in report.specimens)
    assert all(specimen.target_type == "full_logits" for specimen in report.specimens)


def test_aggregate_report_includes_both_specimens(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)
    payload = report.to_report()

    assert payload["phase"] == "P105"
    assert payload["scope"] == "second_teacher_specimen_swap_smoke"
    assert payload["model_ids"] == ("fake-hf-specimen-a", "fake-hf-specimen-b")
    assert [item["model_id"] for item in payload["specimens"]] == [
        "fake-hf-specimen-a",
        "fake-hf-specimen-b",
    ]


def test_no_specimen_is_treated_as_special_by_model_logic(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    assert {specimen.model_id for specimen in report.specimens} == set(report.model_ids)
    assert all("qwen" not in specimen.model_id.lower() for specimen in report.specimens)


def test_local_files_only_defaults_true() -> None:
    config = HFTeacherSpecimenConfig(model_id="fake-hf-specimen-a")

    assert config.local_files_only is True
    assert config.allow_downloads is False


def test_downloads_are_opt_in_only() -> None:
    config = HFTeacherSpecimenConfig(
        model_id="fake-hf-specimen-a",
        allow_downloads=True,
    )

    assert config.allow_downloads is True
    assert config.local_files_only is True


def test_unavailable_optional_specimen_reports_without_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    report = run_hf_teacher_specimen_swap_smoke(
        (HFTeacherSpecimenConfig(model_id="local/missing"),),
        target_store_root=tmp_path / "stores",
    )

    assert report.status == "unavailable"
    assert report.unavailable == 1
    assert report.failed == 0
    assert report.specimens[0].reason == "transformers_not_installed"


def test_report_includes_claims_not_made(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    assert "qwen_specific_support" in report.claims_not_made
    assert "gpt2_specific_architecture" in report.claims_not_made
    assert "student_consumption_proven" in report.claims_not_made
    assert "training_ready" in report.claims_not_made


def test_no_student_consumption_or_training_is_performed(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    assert "student_consumption_proven" in report.claims_not_made
    assert "training_ready" in report.claims_not_made
    assert all(specimen.status == "pass" for specimen in report.specimens)


def test_baseline_requires_no_live_hf_or_accelerator(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    assert report.status == "pass"
    assert all(specimen.local_files_only for specimen in report.specimens)
    assert all(specimen.allow_downloads is False for specimen in report.specimens)


def test_swap_report_can_be_written_to_json(tmp_path: Path) -> None:
    report = _run_fake_swap(tmp_path)

    report_path = report.write_json(tmp_path / "report.json")

    assert report_path.is_file()
    assert '"phase": "P105"' in report_path.read_text(encoding="utf-8")


def _run_fake_swap(tmp_path: Path):
    specimens = (
        HFTeacherSpecimenConfig(
            model_id="fake-hf-specimen-a",
            prompts=("alpha", "beta"),
            sequence_length=4,
        ),
        HFTeacherSpecimenConfig(
            model_id="fake-hf-specimen-b",
            prompts=("alpha", "beta"),
            sequence_length=4,
        ),
    )
    return run_hf_teacher_specimen_swap_smoke(
        specimens,
        target_store_root=tmp_path / "stores",
        backends={
            "fake-hf-specimen-a": _fake_backend(
                model_id="fake-hf-specimen-a",
                tokenizer_id="fake-tokenizer-a",
                vocab_size=8,
            ),
            "fake-hf-specimen-b": _fake_backend(
                model_id="fake-hf-specimen-b",
                tokenizer_id="fake-tokenizer-b",
                vocab_size=16,
            ),
        },
    )


def _fake_backend(
    *,
    model_id: str,
    tokenizer_id: str,
    vocab_size: int,
) -> HFTeacherBackend:
    return HFTeacherBackend(
        model_id,
        tokenizer=_FakeTokenizer(
            tokenizer_id=tokenizer_id,
            vocab_size=vocab_size,
        ),
        model=_FakeCausalLM(vocab_size=vocab_size),
        prompts=("alpha", "beta"),
    )


class _FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = 0
    unk_token_id = 2

    def __init__(self, *, tokenizer_id: str, vocab_size: int) -> None:
        self.name_or_path = tokenizer_id
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
