from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
)
from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    TinyRealTeacherFingerprintCaptureConfig,
    load_text_fixture,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.teachers import HFTeacherBackend

ROOT = Path(__file__).resolve().parents[1]
TEXTS = (
    ROOT
    / "tests"
    / "fixtures"
    / "fingerprint_capture_real_teacher"
    / "tiny_texts.jsonl"
)
SCRIPT = ROOT / "scripts" / "build_real_teacher_fingerprint_artifact.py"


def test_tiny_real_teacher_capture_smoke_with_fake_hf_backend(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=5)
    summary = _json(result.summary_path)

    assert result.status == "pass"
    assert result.artifact_validated is True
    assert result.teacher_real is True
    assert result.teacher_backend == "hf_causal_lm"
    assert result.local_files_only is True
    assert result.examples_processed > 0
    assert result.target_positions_processed > 0
    assert result.modes_discovered > 0
    assert result.exemplars_retained <= 5
    assert summary["phase"] == "P145"
    assert summary["capture_engine"] == "teacher_side_capture_skeleton_v0"
    assert summary["run_kind"] == "tiny_real_teacher_capture"


def test_artifact_validates_loads_and_summarizes(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=3)
    validation = validate_fingerprint_artifact(result.output_dir)
    targets = load_fingerprint_targets(result.output_dir, batch_size=2)
    exemplars = load_fingerprint_exemplars(result.output_dir, batch_size=2)
    artifact = summarize_fingerprint_artifact(result.output_dir)

    assert validation.ok is True
    assert targets.num_records == result.target_positions_processed
    assert exemplars.num_records == result.exemplars_retained
    assert artifact.num_corridor_records == result.target_positions_processed
    assert artifact.has_exemplars is True
    assert artifact.num_exemplar_records == result.exemplars_retained


def test_teacher_metadata_is_present_in_manifest_and_summary(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=2)
    manifest = _json(result.manifest_path)
    summary = _json(result.summary_path)

    assert manifest["teacher"]["backend"] == "hf_causal_lm"
    assert manifest["teacher"]["model_name_or_path"] == "local/fake-real-teacher"
    assert manifest["teacher"]["tokenizer_name_or_path"] == "local/fake-tokenizer"
    assert manifest["teacher"]["local_files_only"] is True
    assert manifest["teacher"]["teacher_real"] is True
    assert manifest["teacher"]["vocab_size"] == 16
    assert summary["teacher_real"] is True
    assert summary["teacher_backend"] == "hf_causal_lm"
    assert summary["teacher"]["vocab_size"] == 16
    assert summary["local_files_only"] is True


def test_summary_fields_and_consumer_sanity_are_recorded(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=4)
    summary = _json(result.summary_path)

    assert summary["examples_processed"] == 4
    assert summary["tokens_processed"] > 0
    assert summary["target_positions_processed"] == 16
    assert summary["positions_policy"] == "fixed_all_positions"
    assert summary["modes_discovered"] > 0
    assert summary["records_per_mode"]
    assert summary["corridor_bounds_method"] == "quantile"
    assert summary["exemplar_selection_policy"] == "stratified_interestingness_v0"
    assert summary["max_exemplars"] == 4
    assert summary["exemplars_retained"] <= 4
    assert summary["artifact_validated"] is True
    assert summary["targets_loadable"] is True
    assert summary["exemplars_loadable"] is True
    assert summary["consumer_sanity"]["kind"] in {
        "compressed_exemplar_optimizer_step",
        "p141_one_step",
        "p140_forward",
        "loader_only",
    }
    assert summary["consumer_sanity"]["status"] == "pass"


def test_exemplar_budget_is_honored(tmp_path: Path) -> None:
    result = _run_fake_capture(tmp_path, max_exemplars=3)
    manifest = _json(result.manifest_path)
    summary = _json(result.summary_path)

    assert result.exemplars_retained <= 3
    assert manifest["exemplar_reservoir"]["num_records"] <= 3
    assert summary["max_exemplars"] == 3
    assert summary["exemplars_retained"] <= 3


def test_large_vocab_records_loader_only_consumer_reason(tmp_path: Path) -> None:
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=2,
            max_target_positions=8,
            overwrite=True,
            max_exemplars=2,
            exemplar_target_type="dense_probs",
            consumer_vocab_limit=8,
        ),
        backend=_fake_backend(vocab_size=16),
    )
    summary = _json(result.summary_path)

    assert summary["consumer_sanity"]["kind"] == "loader_only"
    assert summary["consumer_sanity"]["status"] == "pass"
    assert "vocab too large" in summary["consumer_sanity"]["reason"]


def test_text_fixture_loads_expected_rows() -> None:
    texts = load_text_fixture(TEXTS)

    assert len(texts) == 4
    assert texts[0].startswith("The quick brown fox")


def test_config_defaults_to_local_files_only(tmp_path: Path) -> None:
    config = TinyRealTeacherFingerprintCaptureConfig(
        output_dir=tmp_path / "artifact",
        texts_path=TEXTS,
    )

    assert config.local_files_only is True
    assert config.allow_downloads is False
    assert config.exemplar_target_type == "cascaded_soft_labels_v1"
    assert config.exemplar_top_k == 256


def test_optional_cli_smoke_skips_when_local_teacher_unavailable(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cli_artifact"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--teacher-model",
            DEFAULT_TINY_REAL_TEACHER,
            "--texts",
            str(TEXTS),
            "--output-dir",
            str(output_dir),
            "--sequence-length",
            "8",
            "--max-examples",
            "2",
            "--max-target-positions",
            "8",
            "--max-exemplars",
            "2",
            "--local-files-only",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode == 2:
        pytest.skip("tiny HF teacher not available in local cache")
    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert validate_fingerprint_artifact(output_dir).ok is True


def test_p145_docs_do_not_claim_quality_improvement() -> None:
    doc = ROOT / "docs" / "P145_TINY_REAL_TEACHER_FINGERPRINT_CAPTURE.md"
    if not doc.exists():
        pytest.skip("P145 docs not written yet")
    text = doc.read_text(encoding="utf-8").lower()

    assert "quality improvement" in text
    assert "does not prove" in text
    assert "baseline" in text


def _run_fake_capture(
    tmp_path: Path,
    *,
    max_exemplars: int,
) -> object:
    return run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            overwrite=True,
            max_exemplars=max_exemplars,
            consumer_vocab_limit=64,
        ),
        backend=_fake_backend(vocab_size=16),
    )


def _fake_backend(*, vocab_size: int) -> HFTeacherBackend:
    return HFTeacherBackend(
        "local/fake-real-teacher",
        tokenizer=_FakeTokenizer(vocab_size=vocab_size),
        model=_FakeCausalLM(vocab_size=vocab_size),
        prompts=("placeholder",),
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
        masks = []
        for row, prompt in enumerate(prompts):
            prompt_len = min(max(1, len(prompt.split())), max_length)
            rows.append(
                [(row * 3 + col) % self.vocab_size for col in range(max_length)]
            )
            masks.append([1 if col < prompt_len else 0 for col in range(max_length)])
        return {
            "input_ids": np.asarray(rows, dtype=np.int32),
            "attention_mask": np.asarray(masks, dtype=np.int32),
        }


class _FakeCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids: Any, attention_mask: Any | None) -> object:
        del attention_mask
        ids = np.asarray(input_ids, dtype=np.float32)
        vocab = np.arange(self.vocab_size, dtype=np.float32)[None, None, :]
        logits = np.sin(ids[:, :, None] * 0.17 + vocab * 0.23)
        return SimpleNamespace(logits=logits.astype(np.float32))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
