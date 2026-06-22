from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qrwkv_xla.fingerprint import (
    FingerprintTrainedBaselineConfig,
    HeldOutFingerprintEvaluationConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    paired_bootstrap_interval,
    run_fingerprint_trained_baseline_comparison,
    run_held_out_fingerprint_evaluation,
    run_tiny_real_teacher_fingerprint_capture,
    select_held_out_winner,
    stable_hash,
    validate_fingerprint_provenance,
    write_fingerprint_provenance,
)
from qrwkv_xla.teachers import HFTeacherBackend

ROOT = Path(__file__).resolve().parents[1]
TRAIN_TEXTS = ROOT / "tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl"


def test_stable_hash_is_deterministic_and_order_sensitive() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert stable_hash(["a", "b"]) != stable_hash(["b", "a"])
    assert stable_hash([[1, 2]]) != stable_hash([[1, 3]])


def test_paired_bootstrap_is_deterministic() -> None:
    values = np.asarray([1.0, -0.5, 0.25, 2.0])
    first = paired_bootstrap_interval(values, samples=200, seed=7)
    second = paired_bootstrap_interval(values, samples=200, seed=7)

    assert first == second
    assert first[0] <= first[1]


def test_winner_logic_respects_direction_and_ties() -> None:
    assert select_held_out_winner(1.0, (0.2, 1.5), tolerance=1e-12) == "fingerprint"
    assert select_held_out_winner(-1.0, (-1.5, -0.2), tolerance=1e-12) == "baseline"
    assert select_held_out_winner(0.1, (-0.2, 0.4), tolerance=1e-12) == "inconclusive"
    assert select_held_out_winner(0.0, (0.0, 0.0), tolerance=1e-12) == "inconclusive"


def test_missing_provenance_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing JSON file"):
        validate_fingerprint_provenance(
            tmp_path / "missing",
            expected_role="held_out_evaluation",
        )


def test_held_out_evaluation_integration(tmp_path: Path) -> None:
    held_out_texts = tmp_path / "held_out.jsonl"
    held_out_texts.write_text(
        "".join(
            json.dumps({"text": text}) + "\n"
            for text in (
                "Quartz signals drift beneath a silent observatory.",
                "Copper circuits remember a winter theorem.",
                "Seven lanterns encode an unfamiliar sequence.",
                "Violet matrices rotate beyond the harbor wall.",
            )
        ),
        encoding="utf-8",
    )
    backend = _fake_backend(16)
    train_artifact = _capture(
        tmp_path / "train_artifact",
        TRAIN_TEXTS,
        "p145-real-teacher",
        backend,
    )
    held_out_artifact = _capture(
        tmp_path / "held_out_artifact",
        held_out_texts,
        "p152-held-out",
        backend,
    )
    write_fingerprint_provenance(
        train_artifact,
        source_file=TRAIN_TEXTS,
        artifact_role="training",
    )
    write_fingerprint_provenance(
        held_out_artifact,
        source_file=held_out_texts,
        artifact_role="held_out_evaluation",
    )
    p151 = run_fingerprint_trained_baseline_comparison(
        FingerprintTrainedBaselineConfig(
            fingerprint_artifact=train_artifact,
            source_texts=TRAIN_TEXTS,
            output_dir=tmp_path / "p151",
            steps=2,
            batch_size=2,
            optimizer="sgd",
            learning_rate=0.01,
            overwrite=True,
        )
    )
    result = run_held_out_fingerprint_evaluation(
        HeldOutFingerprintEvaluationConfig(
            baseline_checkpoint=p151.output_dir / "baseline/checkpoints/final",
            fingerprint_checkpoint=p151.output_dir / "fingerprint/checkpoints/final",
            held_out_fingerprint_artifact=held_out_artifact,
            train_fingerprint_artifact=train_artifact,
            p151_report=p151.report_path,
            output_dir=tmp_path / "p152",
            bootstrap_samples=100,
            bootstrap_seed=0,
            overwrite=True,
        )
    )
    report = _json(result.report_path)

    assert result.status == "pass"
    assert result.winner in {"baseline", "fingerprint", "inconclusive"}
    assert report["comparison_valid"] is True
    assert report["split_validation"]["id_overlap_count"] == 0
    assert report["split_validation"]["token_sequence_overlap_count"] == 0
    assert report["arms"]["baseline"]["parameters_unchanged"] is True
    assert report["arms"]["fingerprint"]["parameters_unchanged"] is True
    assert report["arms"]["baseline"]["records_evaluated"] > 0
    assert report["arms"]["baseline"]["primary_bootstrap_ci95"]
    assert report["paired_statistics"]["paired_delta_ci95"]
    assert report["general_quality_claim_made"] is False
    assert report["quality_per_byte_claim_made"] is False
    for name in (
        "held_out_evaluation_report.json",
        "held_out_evaluation_metrics.json",
        "held_out_evaluation_summary.md",
        "held_out_split_validation.json",
        "checkpoint_validation.json",
        "per_record_metrics.jsonl",
        "paired_deltas.jsonl",
        "provenance_manifest.json",
        "evaluation_receipt.json",
    ):
        assert (result.output_dir / name).is_file()


def test_overlapping_split_is_rejected(tmp_path: Path) -> None:
    backend = _fake_backend(16)
    train = _capture(
        tmp_path / "train",
        TRAIN_TEXTS,
        "p145-real-teacher",
        backend,
    )
    held_out = _capture(
        tmp_path / "held_out",
        TRAIN_TEXTS,
        "p145-real-teacher",
        backend,
    )
    write_fingerprint_provenance(
        train,
        source_file=TRAIN_TEXTS,
        artifact_role="training",
    )
    write_fingerprint_provenance(
        held_out,
        source_file=TRAIN_TEXTS,
        artifact_role="held_out_evaluation",
    )
    config = HeldOutFingerprintEvaluationConfig(
        baseline_checkpoint=tmp_path / "unused-baseline",
        fingerprint_checkpoint=tmp_path / "unused-fingerprint",
        held_out_fingerprint_artifact=held_out,
        train_fingerprint_artifact=train,
        output_dir=tmp_path / "output",
        overwrite=True,
    )

    with pytest.raises(ValueError) as error:
        run_held_out_fingerprint_evaluation(config)
    assert "example IDs overlap" in str(error.value)
    assert "tokenized inputs overlap" in str(error.value)


def _capture(
    output: Path,
    texts: Path,
    prefix: str,
    backend: HFTeacherBackend,
) -> Path:
    return run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=output,
            texts_path=texts,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            max_exemplars=16,
            example_id_prefix=prefix,
            overwrite=True,
        ),
        backend=backend,
    ).output_dir


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_backend(vocab_size: int) -> HFTeacherBackend:
    return HFTeacherBackend(
        "local/fake-real-teacher",
        tokenizer=_FakeTokenizer(vocab_size),
        model=_FakeCausalLM(vocab_size),
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

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, prompts: list[str], **kwargs) -> dict[str, np.ndarray]:
        length = int(kwargs["max_length"])
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            rows.append(
                [
                    3 + (digest[offset] % (self.vocab_size - 3))
                    for offset in range(length)
                ]
            )
        ids = np.asarray(rows, dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


class _FakeCausalLM:
    def __init__(self, vocab_size: int) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocab_size, model_type="fake_causal_lm"
        )

    def eval(self) -> _FakeCausalLM:
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        ids = np.asarray(input_ids)
        vocab = np.arange(self.config.vocab_size, dtype=np.float32)
        logits = -np.square(vocab[None, None, :] - ids[..., None]).astype(np.float32)
        return SimpleNamespace(logits=logits)
