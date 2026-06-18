from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.training import (
    REAL_STUDENT_FINGERPRINT_FORWARD_METRIC_KEYS,
    FingerprintMixedSmokeConfig,
    FingerprintTrainingSmokeConfig,
    RealStudentFingerprintForwardConfig,
    run_mixed_fingerprint_training_smoke,
    run_real_student_fingerprint_forward_smoke,
    run_tiny_fingerprint_training_smoke,
    validate_fingerprint_smoke_report,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)
CORRIDOR_ONLY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "behavioral_fingerprint" / "v0_1_valid_tiny"
)


def test_real_student_fingerprint_forward_smoke_completes(tmp_path: Path) -> None:
    result = run_real_student_fingerprint_forward_smoke(
        RealStudentFingerprintForwardConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "real_student",
            batch_size=2,
            seed=0,
        )
    )
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert result.student_backend_name == "CurrentQRWKVStudentBackend"
    assert result.logits_shape == (2, 8, 16)
    assert result.optimizer_steps_completed == 0
    assert result.requested_steps == 0
    assert result.smoke_student_kind == "real_student_backend"
    assert result.smoke_student_uses_input_ids is True
    assert result.real_student_backend_integrated is True
    assert result.main_runner_integrated is False
    assert result.teacher_required is False
    assert result.exemplar_forward_enabled is False
    assert set(REAL_STUDENT_FINGERPRINT_FORWARD_METRIC_KEYS).issubset(result.metrics)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert validate_fingerprint_smoke_report(report) == []
    assert report["phase"] == "P140"
    assert report["training_path_kind"] == "real_student_fingerprint_forward_smoke"
    assert report["hf_required"] is False
    assert report["accelerator_required"] is False


def test_real_student_forward_report_and_summary_written(tmp_path: Path) -> None:
    result = run_real_student_fingerprint_forward_smoke(
        RealStudentFingerprintForwardConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "real_student",
        )
    )
    summary = Path(result.summary_path).read_text(encoding="utf-8")
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))

    assert Path(result.metrics_path).is_file()
    assert Path(result.report_path).is_file()
    assert "Real Student Fingerprint Forward Smoke Summary" in summary
    assert "Uses input IDs: true" in summary
    assert "Main runner integrated: false" in summary
    assert "Teacher required: false" in summary
    assert "No optimizer steps were run." in summary
    assert report["student"]["backend_name"] == "CurrentQRWKVStudentBackend"
    assert report["forward"]["logits_shape"] == [2, 8, 16]
    assert report["corridor"]["loss_finite"] is True


def test_real_student_forward_rejects_vocab_mismatch(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="fingerprint artifact vocab_size=16 but student vocab_size=17",
    ):
        run_real_student_fingerprint_forward_smoke(
            RealStudentFingerprintForwardConfig(
                artifact_dir=FIXTURE,
                output_dir=tmp_path / "mismatch",
                student_vocab_size=17,
            )
        )


def test_real_student_forward_rejects_position_out_of_range(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    targets_path = artifact / "targets" / "targets-00000.jsonl"
    rows = [
        json.loads(line)
        for line in targets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["position"] = 8
    targets_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="fingerprint target position outside student logits sequence",
    ):
        run_real_student_fingerprint_forward_smoke(
            RealStudentFingerprintForwardConfig(
                artifact_dir=artifact,
                output_dir=tmp_path / "bad_position",
            )
        )


def test_real_student_forward_cli(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_student_fingerprint_forward_smoke.py",
            "--artifact",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path / "cli"),
            "--batch-size",
            "2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "backend=CurrentQRWKVStudentBackend" in completed.stdout
    assert (tmp_path / "cli" / "metrics.json").is_file()
    assert (tmp_path / "cli" / "real_student_fingerprint_forward_report.json").is_file()
    assert (tmp_path / "cli" / "fingerprint_run_summary.md").is_file()


def test_tiny_fingerprint_smokes_still_work(tmp_path: Path) -> None:
    corridor = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=CORRIDOR_ONLY_FIXTURE,
            output_dir=tmp_path / "corridor",
            steps=1,
            batch_size=2,
        )
    )
    mixed = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=1,
            corridor_batch_size=2,
            exemplar_batch_size=2,
        )
    )

    assert corridor.status == "pass"
    assert mixed.status == "pass"
