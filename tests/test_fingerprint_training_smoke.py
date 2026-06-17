from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.training import (
    FINGERPRINT_SMOKE_METRIC_KEYS,
    FingerprintTrainingSmokeConfig,
    run_tiny_fingerprint_training_smoke,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "behavioral_fingerprint" / "v0_1_valid_tiny"
)


def test_tiny_fingerprint_training_smoke_completes(tmp_path: Path) -> None:
    result = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "smoke",
            steps=3,
            batch_size=2,
            seed=0,
        )
    )

    assert result.status == "pass"
    assert result.steps == 3
    assert result.loss_finite is True
    assert result.loss_non_negative is True
    assert result.loss_non_increasing is True
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    assert result.final_loss <= result.initial_loss + 1e-6
    assert Path(result.metrics_path).is_file()
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.report_path).is_file()


def test_tiny_fingerprint_training_smoke_metrics_keys_exist(tmp_path: Path) -> None:
    result = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "smoke",
            steps=1,
            batch_size=2,
        )
    )

    assert set(FINGERPRINT_SMOKE_METRIC_KEYS).issubset(result.metrics)
    assert "fingerprint/loss_total" in result.metrics
    assert "fingerprint/inside_all_rate" in result.metrics
    assert all(np.isfinite(value) for value in result.metrics.values())
    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert set(FINGERPRINT_SMOKE_METRIC_KEYS).issubset(payload)


def test_tiny_fingerprint_training_smoke_requires_no_teacher_or_accelerator(
    tmp_path: Path,
) -> None:
    report = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "smoke",
            steps=1,
            batch_size=2,
        )
    ).to_report()

    assert report["fingerprint_only"] is True
    assert report["teacher_required"] is False
    assert report["hf_download_required"] is False
    assert report["gpu_or_tpu_required"] is False
    assert "real_teacher_required" in report["claims_not_made"]
    assert "exemplar_reservoir" in report["claims_not_made"]


def test_tiny_fingerprint_training_smoke_is_deterministic(tmp_path: Path) -> None:
    config_a = FingerprintTrainingSmokeConfig(
        artifact_dir=FIXTURE,
        output_dir=tmp_path / "a",
        steps=3,
        batch_size=2,
        seed=11,
    )
    config_b = FingerprintTrainingSmokeConfig(
        artifact_dir=FIXTURE,
        output_dir=tmp_path / "b",
        steps=3,
        batch_size=2,
        seed=11,
    )

    result_a = run_tiny_fingerprint_training_smoke(config_a)
    result_b = run_tiny_fingerprint_training_smoke(config_b)

    np.testing.assert_allclose(result_a.initial_loss, result_b.initial_loss, rtol=1e-7)
    np.testing.assert_allclose(result_a.final_loss, result_b.final_loss, rtol=1e-7)
    assert result_a.metrics == result_b.metrics


def test_run_fingerprint_smoke_cli(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_fingerprint_smoke.py",
            "--artifact",
            str(FIXTURE),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path / "cli"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "status=pass" in completed.stdout
    assert (tmp_path / "cli" / "metrics.json").is_file()
    assert (tmp_path / "cli" / "checkpoint.json").is_file()
