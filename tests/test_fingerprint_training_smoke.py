from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.training import (
    FINGERPRINT_SMOKE_METRIC_KEYS,
    FingerprintTrainingSmokeConfig,
    classify_fingerprint_smoke_status,
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
    assert result.completed_steps == 3
    assert result.train_batches_consumed == 3
    assert result.loss_finite is True
    assert result.loss_non_negative is True
    assert isinstance(result.loss_non_increasing, bool)
    assert np.isfinite(result.loss_delta)
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
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
    assert report["smoke_student_kind"] == "tiny_position_logit_head"
    assert report["smoke_student_uses_input_ids"] is False
    assert report["main_runner_integrated"] is False
    assert report["teacher_required"] is False
    assert report["exemplar_reservoir_enabled"] is False
    assert report["artifact_kind"] == "behavioral_fingerprint"
    assert report["training_path_kind"] == "standalone_fingerprint_smoke"
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


def test_fingerprint_smoke_status_does_not_require_loss_non_increase() -> None:
    status = classify_fingerprint_smoke_status(
        completed_steps=3,
        requested_steps=3,
        train_batches_consumed=3,
        initial_loss=1.0,
        final_loss=1.1,
        metrics_finite=True,
    )

    assert status == "pass"


def test_fingerprint_smoke_status_requires_non_negative_initial_loss() -> None:
    status = classify_fingerprint_smoke_status(
        completed_steps=3,
        requested_steps=3,
        train_batches_consumed=3,
        initial_loss=-0.1,
        final_loss=0.1,
        metrics_finite=True,
    )

    assert status == "fail"


def test_tiny_fingerprint_training_smoke_rejects_zero_optimizer_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="zero optimizer batches"):
        run_tiny_fingerprint_training_smoke(
            FingerprintTrainingSmokeConfig(
                artifact_dir=FIXTURE,
                output_dir=tmp_path / "smoke",
                steps=1,
                batch_size=999,
                drop_remainder=True,
            )
        )


def test_fingerprint_fixture_jsonl_physical_lines_match_manifest() -> None:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["target_shards"]:
        shard_path = FIXTURE / shard["path"]
        lines = shard_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == shard["num_records"]
        for line in lines:
            assert isinstance(json.loads(line), dict)


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
