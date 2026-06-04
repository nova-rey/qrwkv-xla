from __future__ import annotations

from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts import (
    validate_student_artifact,
    write_student_artifact_validation_report,
)
from qrwkv_xla.artifacts._json import write_json
from tests.test_teacher_textbook_artifact import create_fake_teacher_textbook


def test_fake_student_artifact_validates(tmp_path: Path) -> None:
    teacher_textbook = create_fake_teacher_textbook(tmp_path)
    student_artifact = create_fake_student_artifact(tmp_path)

    report = validate_student_artifact(
        student_artifact,
        teacher_textbook_path=teacher_textbook,
    )

    assert report.status == "pass"
    assert report.student_config_ok is True
    assert report.vocab_contract_ok is True
    assert report.checkpoint_ok is True
    assert report.vocab_matches_teacher_textbook is True
    assert report.pallas_not_default is True


def test_missing_student_config_fails(tmp_path: Path) -> None:
    student_artifact = create_fake_student_artifact(tmp_path)
    (student_artifact / "student_config.json").unlink()

    report = validate_student_artifact(student_artifact)

    assert report.status == "fail"
    assert any("student_config.json" in blocker for blocker in report.blockers)


def test_vocab_mismatch_with_teacher_textbook_fails(tmp_path: Path) -> None:
    teacher_textbook = create_fake_teacher_textbook(tmp_path)
    student_artifact = create_fake_student_artifact(tmp_path)
    write_json(
        student_artifact / "vocab_contract.json",
        {
            "tokenizer_id": "other-tokenizer",
            "tokenizer_hash": None,
            "vocab_size": 7,
        },
    )

    report = validate_student_artifact(
        student_artifact,
        teacher_textbook_path=teacher_textbook,
    )

    assert report.status == "fail"
    assert report.vocab_matches_teacher_textbook is False


def test_student_artifact_report_serializes(tmp_path: Path) -> None:
    student_artifact = create_fake_student_artifact(tmp_path)
    report = validate_student_artifact(student_artifact)
    report_path = student_artifact / "validation_report.json"

    write_student_artifact_validation_report(report, report_path)

    assert report_path.is_file()
    assert validate_student_artifact(student_artifact).validation_report_ok is True


def test_student_artifact_requires_no_tpu_gpu_hf_or_internet(tmp_path: Path) -> None:
    student_artifact = create_fake_student_artifact(tmp_path)

    report = validate_student_artifact(student_artifact)

    assert report.status == "pass"
    assert "no_hf_native_model_claim" in report.claims_not_made


def create_fake_student_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "student_artifact"
    root.mkdir()
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    np.savez(checkpoint / "params.npz", logits=np.zeros((1, 1, 5), dtype=np.float32))

    write_json(
        root / "vocab_contract.json",
        {
            "tokenizer_id": "sshleifer/tiny-gpt2",
            "tokenizer_hash": None,
            "vocab_size": 5,
            "model_id": "sshleifer/tiny-gpt2",
            "model_family": "hf-causal-lm",
        },
    )
    write_json(
        root / "student_config.json",
        {
            "artifact_type": "student_artifact",
            "artifact_version": 0,
            "architecture_id": "current_qrwkv",
            "student_family": "rwkv_family",
            "vocab_size": 5,
            "vocab_contract_path": "vocab_contract.json",
            "runtime": "reference",
            "reference_runtime_default": True,
            "pallas_opt_in": True,
            "target_type": "full_logits",
            "checkpoint_format": "checkpoint/params.npz",
            "forward_input_shape": "[batch, time]",
            "forward_output_shape": "[batch, time, vocab_size]",
            "created_from_teacher_textbook": "../teacher_textbook",
            "claims_not_made": [
                "no_hf_native_model_claim",
                "no_generation_claim",
                "no_model_quality_claim",
            ],
        },
    )
    write_json(
        root / "runtime_metadata.json",
        {
            "runtime": "reference",
            "jax_version": None,
            "jaxlib_version": None,
            "platform": "cpu",
            "devices": [],
            "pallas_enabled": False,
            "reference_runtime_default": True,
        },
    )
    write_json(root / "burn_report.json", {"status": "unit_pass"})
    write_json(root / "eval_report.json", {"status": "unit_pass"})
    write_json(root / "export_report.json", {"status": "unit_pass"})
    return root
