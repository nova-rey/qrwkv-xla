from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    FingerprintBaselineComparisonConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    run_fingerprint_baseline_comparison,
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
SCRIPT = ROOT / "scripts" / "run_fingerprint_baseline_comparison.py"


def test_comparison_runs_with_existing_tiny_artifact(tmp_path: Path) -> None:
    artifact = _build_fake_artifact(tmp_path)
    result = run_fingerprint_baseline_comparison(
        _config(tmp_path, fingerprint_artifact=artifact)
    )
    report = _json(result.report_path)

    assert result.status == "pass"
    assert result.arms_run == ("baseline_init_only", "fingerprint_corridor")
    assert result.report_path.is_file()
    assert result.summary_path.is_file()
    assert report["status"] == "pass"
    assert {arm["arm_id"] for arm in report["arms"]} == {
        "baseline_init_only",
        "fingerprint_corridor",
    }


def test_fingerprint_arm_uses_main_runner(tmp_path: Path) -> None:
    artifact = _build_fake_artifact(tmp_path)
    result = run_fingerprint_baseline_comparison(
        _config(tmp_path, fingerprint_artifact=artifact)
    )
    arm = _arm(_json(result.report_path), "fingerprint_corridor")

    assert arm["distill_mode"] == "fingerprint_corridor"
    assert arm["training_path_kind"] == "main_runner_fingerprint_corridor"
    assert arm["main_runner_integrated"] is True
    assert arm["real_student_backend_integrated"] is True
    assert arm["teacher_required_during_training"] is False
    assert arm["optimizer_steps_completed"] == 2
    assert arm["params_changed"] is True
    assert arm["param_delta_norm"] > 0.0
    assert arm["fingerprint/corridor/loss_total"] is not None
    assert arm["fingerprint/corridor/inside_all_rate"] is not None


def test_baseline_arm_is_recorded_honestly(tmp_path: Path) -> None:
    artifact = _build_fake_artifact(tmp_path)
    result = run_fingerprint_baseline_comparison(
        _config(tmp_path, fingerprint_artifact=artifact)
    )
    arm = _arm(_json(result.report_path), "baseline_init_only")

    assert arm["status"] == "pass"
    assert arm["optimizer_steps_completed"] == 0
    assert arm["params_changed"] is False
    assert arm["param_delta_norm"] == 0.0
    assert arm["initial_loss"] is None
    assert arm["final_loss"] is None
    assert arm["checkpoint_written"] is True
    assert arm["checkpoint_loadable"] is True


def test_comparability_metadata_claims_and_artifact_budget(
    tmp_path: Path,
) -> None:
    artifact = _build_fake_artifact(tmp_path)
    result = run_fingerprint_baseline_comparison(
        _config(tmp_path, fingerprint_artifact=artifact)
    )
    report = _json(result.report_path)
    controls = report["comparison_controls"]
    budget = report["artifact"]
    claims = report["claims"]

    assert controls["same_student_backend"] is True
    assert controls["same_student_config"] is True
    assert controls["same_seed"] is True
    assert controls["same_training_steps_where_applicable"] is True
    assert controls["same_batch_size_where_applicable"] is True
    assert controls["same_eval_texts"] is True
    assert controls["limitations"]
    assert claims["quality_claim_made"] is False
    assert claims["baseline_winner_declared"] is False
    assert claims["quality_per_byte_claim_made"] is False
    assert budget["modes_discovered"] > 0
    assert budget["target_positions_processed"] > 0
    assert budget["exemplars_retained"] > 0
    assert budget["artifact_size_bytes"] > 0


def test_build_then_compare_with_fake_backend(tmp_path: Path) -> None:
    result = run_fingerprint_baseline_comparison(
        FingerprintBaselineComparisonConfig(
            output_dir=tmp_path / "compare",
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
    assert report["artifact"]["source"] == "built_from_tiny_real_teacher"
    assert report["artifact"]["teacher_real"] is True
    assert report["claims"]["baseline_winner_declared"] is False


def test_cli_smoke_with_existing_artifact(tmp_path: Path) -> None:
    artifact = _build_fake_artifact(tmp_path)
    output_dir = tmp_path / "cli_compare"
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
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "quality_claim_made=false" in completed.stdout
    assert (output_dir / "p147_comparison_report.json").is_file()


def test_optional_local_cache_real_hf_build_then_compare(tmp_path: Path) -> None:
    try:
        result = run_fingerprint_baseline_comparison(
            FingerprintBaselineComparisonConfig(
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
            "P147 real local-cache build-then-compare smoke"
        )

    report = _json(result.report_path)
    assert result.status == "pass"
    assert report["artifact"]["source"] == "built_from_tiny_real_teacher"
    assert _arm(report, "fingerprint_corridor")["params_changed"] is True


def _config(
    tmp_path: Path,
    *,
    fingerprint_artifact: Path,
) -> FingerprintBaselineComparisonConfig:
    return FingerprintBaselineComparisonConfig(
        output_dir=tmp_path / "compare",
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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
