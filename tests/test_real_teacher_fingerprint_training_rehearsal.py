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

from qrwkv_xla.artifacts import validate_fingerprint_artifact
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    RealTeacherFingerprintTrainingRehearsalConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    run_real_teacher_fingerprint_training_rehearsal,
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
SCRIPT = ROOT / "scripts" / "run_real_teacher_fingerprint_training_rehearsal.py"


def test_build_then_train_rehearsal_with_fake_backend(tmp_path: Path) -> None:
    result = run_real_teacher_fingerprint_training_rehearsal(
        _config(
            tmp_path,
            build_real_teacher_artifact=True,
            training_steps=2,
        ),
        backend=_fake_backend(vocab_size=16),
    )
    report = _json(result.report_path)

    assert result.status == "pass"
    assert result.artifact_source == "built_from_tiny_real_teacher"
    assert result.teacher_real is True
    assert result.teacher_required_during_training is False
    assert validate_fingerprint_artifact(result.artifact_dir).ok is True
    _assert_p146_report_passes(report, requested_steps=2)
    assert (
        report["capture"]["vocab_size"] == report["training"]["student"]["vocab_size"]
    )


def test_existing_artifact_reuse_mode_trains_without_teacher(
    tmp_path: Path,
) -> None:
    capture = run_tiny_real_teacher_fingerprint_capture(
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
    result = run_real_teacher_fingerprint_training_rehearsal(
        _config(
            tmp_path,
            fingerprint_artifact=capture.output_dir,
            training_steps=2,
        )
    )
    report = _json(result.report_path)

    assert result.status == "pass"
    assert result.artifact_source == "existing_artifact"
    assert report["artifact_source"] == "existing_artifact"
    assert report["training"]["teacher_required_during_training"] is False
    _assert_p146_report_passes(report, requested_steps=2)


def test_checkpoint_is_written_and_loadable(tmp_path: Path) -> None:
    result = run_real_teacher_fingerprint_training_rehearsal(
        _config(tmp_path, build_real_teacher_artifact=True, training_steps=1),
        backend=_fake_backend(vocab_size=16),
    )
    report = _json(result.report_path)

    assert result.checkpoint_dir is not None
    checkpoint = load_checkpoint(result.checkpoint_dir)
    assert checkpoint.manifest.step == 1
    assert checkpoint.manifest.student_architecture == "current_qrwkv"
    assert report["training"]["checkpoint_written"] is True
    assert report["training"]["checkpoint_loadable"] is True


def test_loss_diagnostics_are_present_and_finite(tmp_path: Path) -> None:
    result = run_real_teacher_fingerprint_training_rehearsal(
        _config(tmp_path, build_real_teacher_artifact=True, training_steps=2),
        backend=_fake_backend(vocab_size=16),
    )
    report = _json(result.report_path)
    training = report["training"]

    assert math.isfinite(training["initial_loss"])
    assert math.isfinite(training["final_loss"])
    assert math.isfinite(training["loss_delta"])
    assert isinstance(training["loss_non_increasing"], bool)
    assert training["loss_non_increasing_required"] is False


def test_cli_smoke_with_existing_artifact(tmp_path: Path) -> None:
    capture = run_tiny_real_teacher_fingerprint_capture(
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
    output_dir = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fingerprint-artifact",
            str(capture.output_dir),
            "--output-dir",
            str(output_dir),
            "--training-steps",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "artifact_source=existing_artifact" in completed.stdout
    assert (output_dir / "p146_rehearsal_report.json").is_file()


def test_optional_local_cache_real_hf_build_then_train(tmp_path: Path) -> None:
    try:
        result = run_real_teacher_fingerprint_training_rehearsal(
            RealTeacherFingerprintTrainingRehearsalConfig(
                output_dir=tmp_path / "real_local_cache",
                build_real_teacher_artifact=True,
                texts_path=TEXTS,
                teacher_model=DEFAULT_TINY_REAL_TEACHER,
                sequence_length=8,
                max_examples=1,
                max_target_positions=8,
                max_exemplars=2,
                training_steps=1,
                batch_size=1,
                learning_rate=0.01,
                local_files_only=True,
                overwrite=True,
            )
        )
    except HFTeacherUnavailable:
        pytest.skip(
            "sshleifer/tiny-gpt2 is not available in local HF cache; skipping "
            "P146 real local-cache integration smoke"
        )

    report = _json(result.report_path)
    assert result.status == "pass"
    assert report["artifact_source"] == "built_from_tiny_real_teacher"
    assert report["teacher_real"] is True
    assert report["training"]["params_changed"] is True


def _config(
    tmp_path: Path,
    *,
    build_real_teacher_artifact: bool = False,
    fingerprint_artifact: Path | None = None,
    training_steps: int = 2,
) -> RealTeacherFingerprintTrainingRehearsalConfig:
    return RealTeacherFingerprintTrainingRehearsalConfig(
        output_dir=tmp_path / "p146",
        fingerprint_artifact=fingerprint_artifact,
        build_real_teacher_artifact=build_real_teacher_artifact,
        texts_path=TEXTS,
        sequence_length=4,
        max_examples=4,
        max_target_positions=16,
        max_exemplars=4,
        training_steps=training_steps,
        batch_size=2,
        learning_rate=0.01,
        overwrite=True,
    )


def _assert_p146_report_passes(
    report: dict[str, Any],
    *,
    requested_steps: int,
) -> None:
    assert report["phase"] == "P146"
    assert report["run_kind"] == "real_teacher_artifact_student_training_rehearsal"
    assert report["status"] == "pass"
    assert report["teacher_real"] is True
    assert report["capture"]["artifact_validated"] is True
    assert report["capture"]["modes_discovered"] > 0
    assert report["capture"]["target_positions_processed"] > 0
    assert report["training"]["distill_mode"] == "fingerprint_corridor"
    assert (
        report["training"]["training_path_kind"] == "main_runner_fingerprint_corridor"
    )
    assert report["training"]["main_runner_integrated"] is True
    assert report["training"]["real_student_backend_integrated"] is True
    assert report["training"]["teacher_required_during_training"] is False
    assert report["training"]["optimizer_steps_completed"] == requested_steps
    assert report["training"]["batches_consumed"] == requested_steps
    assert report["training"]["params_changed"] is True
    assert report["training"]["param_delta_norm"] > 0.0
    assert report["training"]["metrics_finite"] is True
    assert report["training"]["checkpoint_written"] is True
    assert report["training"]["checkpoint_loadable"] is True


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
