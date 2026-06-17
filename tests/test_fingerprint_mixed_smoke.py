from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.training import (
    FINGERPRINT_MIXED_SMOKE_METRIC_KEYS,
    FingerprintMixedSmokeConfig,
    FingerprintTrainingSmokeConfig,
    classify_fingerprint_mixed_smoke_status,
    run_mixed_fingerprint_training_smoke,
    run_tiny_fingerprint_training_smoke,
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


def test_mixed_fingerprint_smoke_completes(tmp_path: Path) -> None:
    result = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=3,
            corridor_batch_size=2,
            exemplar_batch_size=2,
            seed=0,
        )
    )

    assert result.status == "pass"
    assert result.requested_steps == 3
    assert result.optimizer_steps_completed == 3
    assert result.corridor_batches_consumed == 3
    assert result.exemplar_batches_consumed == 3
    assert result.mixed_loss_finite is True
    assert result.mixed_loss_non_negative is True
    assert np.isfinite(result.final_mixed_loss)
    assert Path(result.metrics_path).is_file()
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.report_path).is_file()


def test_mixed_fingerprint_smoke_metrics_keys_exist(tmp_path: Path) -> None:
    result = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=1,
            corridor_batch_size=2,
            exemplar_batch_size=2,
        )
    )

    assert set(FINGERPRINT_MIXED_SMOKE_METRIC_KEYS).issubset(result.metrics)
    assert "fingerprint/mixed_loss_total" in result.metrics
    assert "fingerprint/corridor_loss_total" in result.metrics
    assert "fingerprint/exemplar_loss_total" in result.metrics
    assert "fingerprint/exemplar_kl_loss" in result.metrics
    assert "fingerprint/corridor_inside_all_rate" in result.metrics
    assert all(np.isfinite(value) for value in result.metrics.values())
    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert set(FINGERPRINT_MIXED_SMOKE_METRIC_KEYS).issubset(payload)


def test_mixed_fingerprint_smoke_reports_no_teacher_or_accelerator(
    tmp_path: Path,
) -> None:
    report = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=1,
            corridor_batch_size=2,
            exemplar_batch_size=2,
        )
    ).to_report()

    assert report["phase"] == "P138"
    assert report["smoke_student_kind"] == "tiny_position_logit_head"
    assert report["smoke_student_uses_input_ids"] is False
    assert report["main_runner_integrated"] is False
    assert report["real_student_backend_integrated"] is False
    assert report["teacher_required"] is False
    assert report["exemplar_reservoir_enabled"] is True
    assert report["artifact_kind"] == "behavioral_fingerprint"
    assert report["training_path_kind"] == "standalone_mixed_fingerprint_smoke"
    assert report["hf_download_required"] is False
    assert report["gpu_or_tpu_required"] is False


def test_mixed_smoke_status_does_not_require_loss_non_increase() -> None:
    status = classify_fingerprint_mixed_smoke_status(
        optimizer_steps_completed=3,
        requested_steps=3,
        corridor_batches_consumed=3,
        exemplar_batches_consumed=3,
        initial_mixed_loss=1.0,
        final_mixed_loss=1.1,
        metrics_finite=True,
    )

    assert status == "pass"


def test_mixed_fingerprint_smoke_rejects_zero_corridor_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="zero corridor batches"):
        run_mixed_fingerprint_training_smoke(
            FingerprintMixedSmokeConfig(
                artifact_dir=FIXTURE,
                output_dir=tmp_path / "mixed",
                steps=1,
                corridor_batch_size=999,
                exemplar_batch_size=2,
                corridor_drop_remainder=True,
            )
        )


def test_mixed_fingerprint_smoke_rejects_zero_exemplar_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="zero exemplar batches"):
        run_mixed_fingerprint_training_smoke(
            FingerprintMixedSmokeConfig(
                artifact_dir=FIXTURE,
                output_dir=tmp_path / "mixed",
                steps=1,
                corridor_batch_size=2,
                exemplar_batch_size=999,
                exemplar_drop_remainder=True,
            )
        )


def test_mixed_fingerprint_smoke_rejects_zero_loss_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one mixed smoke loss weight"):
        run_mixed_fingerprint_training_smoke(
            FingerprintMixedSmokeConfig(
                artifact_dir=FIXTURE,
                output_dir=tmp_path / "mixed",
                steps=1,
                corridor_loss_weight=0.0,
                exemplar_loss_weight=0.0,
            )
        )


@pytest.mark.parametrize(
    ("corridor_weight", "exemplar_weight"),
    [(1.0, 0.0), (0.0, 1.0)],
)
def test_mixed_fingerprint_smoke_accepts_single_branch_weight(
    tmp_path: Path,
    corridor_weight: float,
    exemplar_weight: float,
) -> None:
    result = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=FIXTURE,
            output_dir=tmp_path / f"mixed_{corridor_weight}_{exemplar_weight}",
            steps=1,
            corridor_loss_weight=corridor_weight,
            exemplar_loss_weight=exemplar_weight,
        )
    )

    assert result.status == "pass"
    assert result.metrics["fingerprint/corridor_loss_weight"] == corridor_weight
    assert result.metrics["fingerprint/exemplar_loss_weight"] == exemplar_weight


def test_mixed_fingerprint_smoke_cli(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_fingerprint_smoke.py",
            "--mode",
            "mixed",
            "--artifact",
            str(FIXTURE),
            "--steps",
            "2",
            "--corridor-batch-size",
            "2",
            "--exemplar-batch-size",
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

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "corridor_batches=2" in completed.stdout
    assert "exemplar_batches=2" in completed.stdout
    assert (tmp_path / "cli" / "metrics.json").is_file()
    assert (tmp_path / "cli" / "checkpoint.json").is_file()
    assert (tmp_path / "cli" / "fingerprint_mixed_smoke_report.json").is_file()


def test_existing_corridor_only_smoke_still_works(tmp_path: Path) -> None:
    result = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=CORRIDOR_ONLY_FIXTURE,
            output_dir=tmp_path / "corridor",
            steps=1,
            batch_size=2,
        )
    )

    assert result.status == "pass"
    assert result.exemplar_reservoir_enabled is False
