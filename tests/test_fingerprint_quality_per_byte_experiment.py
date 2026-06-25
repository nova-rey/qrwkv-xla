from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    FingerprintQualityPerByteExperimentConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    run_fingerprint_quality_per_byte_experiment,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.teachers import HFTeacherBackend, HFTeacherUnavailable

ROOT = Path(__file__).resolve().parents[1]
TEXTS = (
    ROOT
    / "tests"
    / "fixtures"
    / "fingerprint_capture_real_teacher"
    / "tiny_texts.jsonl"
)
SCRIPT = ROOT / "scripts" / "run_fingerprint_quality_per_byte_experiment.py"


def test_quality_per_byte_report_computes_reference_delta(tmp_path: Path) -> None:
    result = run_fingerprint_quality_per_byte_experiment(
        _config(tmp_path, fingerprint_artifact=_build_fake_artifact(tmp_path))
    )
    report = _json(result.report_path)
    delta = report["quality_per_byte"]["reference_delta_vs_init_only"]

    assert result.status == "pass"
    assert report["phase"] == "P148"
    assert report["run_kind"] == "first_quality_per_byte_experiment"
    assert math.isfinite(delta["absolute_corridor_loss_delta"])
    assert math.isfinite(delta["relative_corridor_loss_delta"])
    assert math.isfinite(delta["corridor_loss_delta_per_mb"])
    assert math.isfinite(delta["inside_all_rate_delta"])
    assert math.isfinite(delta["inside_all_rate_delta_per_mb"])


def test_fairness_labels_init_only_correctly(tmp_path: Path) -> None:
    result = run_fingerprint_quality_per_byte_experiment(
        _config(tmp_path, fingerprint_artifact=_build_fake_artifact(tmp_path))
    )
    report = _json(result.report_path)
    fairness = report["fairness"]
    claims = report["claims"]

    assert fairness["trained_baseline_available"] is False
    assert fairness["comparison_fairness"] == "reference_only"
    assert fairness["baseline_init_only_is_competitive"] is False
    assert fairness["eval_split"] == "train_artifact_reuse"
    assert fairness["generalization_claim_made"] is False
    assert claims["winner_declared"] is False


def test_arms_have_corridor_eval_metrics(tmp_path: Path) -> None:
    result = run_fingerprint_quality_per_byte_experiment(
        _config(tmp_path, fingerprint_artifact=_build_fake_artifact(tmp_path))
    )
    report = _json(result.report_path)

    for arm_id in ("baseline_init_only", "fingerprint_corridor"):
        arm = _arm(report, arm_id)
        assert math.isfinite(arm["eval"]["corridor_loss_total"])
        assert math.isfinite(arm["eval"]["inside_all_rate"])
        assert math.isfinite(arm["eval"]["inside_entropy_rate"])
        assert math.isfinite(arm["eval"]["inside_top1_margin_rate"])
        assert math.isfinite(arm["eval"]["inside_top8_mass_rate"])
        assert math.isfinite(arm["eval"]["inside_top32_mass_rate"])
        assert math.isfinite(arm["eval"]["inside_tail_mass_rate"])
        assert arm["eval"]["metrics_finite"] is True
        assert arm["eval"]["records_evaluated"] > 0


def test_artifact_budget_and_claims_are_present(tmp_path: Path) -> None:
    result = run_fingerprint_quality_per_byte_experiment(
        _config(tmp_path, fingerprint_artifact=_build_fake_artifact(tmp_path))
    )
    report = _json(result.report_path)
    budget = report["artifact_budget"]
    claims = report["claims"]

    assert budget["fingerprint_artifact_size_bytes"] > 0
    assert budget["manifest_size_bytes"] > 0
    assert budget["targets_size_bytes"] > 0
    assert budget["modes_size_bytes"] > 0
    assert budget["target_records"] > 0
    assert budget["modes_discovered"] > 0
    assert claims["general_quality_claim_made"] is False
    assert claims["radlads_parity_claim_made"] is False
    assert claims["scale_claim_made"] is False
    assert claims["quality_per_byte_claim_scope"] == "tiny_smoke_only"


def test_build_then_run_with_fake_backend(tmp_path: Path) -> None:
    result = run_fingerprint_quality_per_byte_experiment(
        FingerprintQualityPerByteExperimentConfig(
            output_dir=tmp_path / "p148",
            build_real_teacher_artifact=True,
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            max_exemplars=4,
            steps=2,
            batch_size=2,
            learning_rate=0.01,
            seed=0,
            overwrite=True,
        ),
        backend=_fake_backend(vocab_size=16),
    )
    report = _json(result.report_path)

    assert result.status == "pass"
    assert report["artifact_budget"]["fingerprint_artifact_size_bytes"] > 0
    assert report["quality_per_byte"]["trained_baseline_available"] is False
    assert report["quality_per_byte"]["delta_vs_trained_baseline"] is None


def test_cli_smoke_with_existing_artifact(tmp_path: Path) -> None:
    artifact = _build_fake_artifact(tmp_path)
    output_dir = tmp_path / "cli_p148"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fingerprint-artifact",
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--seed",
            "0",
            "--eval-split",
            "train_artifact_reuse",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "comparison_fairness=reference_only" in completed.stdout
    assert "winner_declared=false" in completed.stdout
    assert (output_dir / "p148_quality_per_byte_report.json").is_file()
    assert (output_dir / "p148_quality_per_byte_summary.md").is_file()


def test_optional_local_cache_real_hf_build_then_run(tmp_path: Path) -> None:
    try:
        result = run_fingerprint_quality_per_byte_experiment(
            FingerprintQualityPerByteExperimentConfig(
                output_dir=tmp_path / "real_local_cache",
                build_real_teacher_artifact=True,
                texts_path=TEXTS,
                teacher_model=DEFAULT_TINY_REAL_TEACHER,
                sequence_length=8,
                max_examples=1,
                max_target_positions=8,
                max_exemplars=2,
                steps=1,
                batch_size=1,
                learning_rate=0.01,
                local_files_only=True,
                overwrite=True,
            )
        )
    except HFTeacherUnavailable:
        pytest.skip(
            "sshleifer/tiny-gpt2 is not available in local HF cache; skipping "
            "P148 real local-cache quality-per-byte smoke"
        )

    report = _json(result.report_path)
    assert result.status == "pass"
    assert report["fairness"]["comparison_fairness"] == "reference_only"


def _config(
    tmp_path: Path,
    *,
    fingerprint_artifact: Path,
) -> FingerprintQualityPerByteExperimentConfig:
    return FingerprintQualityPerByteExperimentConfig(
        output_dir=tmp_path / "p148",
        fingerprint_artifact=fingerprint_artifact,
        texts_path=TEXTS,
        sequence_length=4,
        max_examples=4,
        max_target_positions=16,
        max_exemplars=4,
        steps=2,
        batch_size=2,
        learning_rate=0.01,
        seed=0,
        eval_split="train_artifact_reuse",
        overwrite=True,
    )


def _build_fake_artifact(tmp_path: Path) -> Path:
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            max_exemplars=4,
            overwrite=True,
        ),
        backend=_fake_backend(vocab_size=16),
    )
    return result.output_dir


def _arm(report: dict[str, Any], arm_id: str) -> dict[str, Any]:
    return next(arm for arm in report["arms"] if arm["arm_id"] == arm_id)


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
        for prompt in prompts:
            prompt_len = min(max(1, len(prompt.split())), max_length)
            prompt_seed = sum(ord(char) for char in prompt) % self.vocab_size
            rows.append(
                [(prompt_seed + col) % self.vocab_size for col in range(max_length)]
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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
