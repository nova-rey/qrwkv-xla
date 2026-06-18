from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts import summarize_fingerprint_artifact
from qrwkv_xla.training import (
    FingerprintMixedSmokeConfig,
    FingerprintTrainingSmokeConfig,
    render_fingerprint_smoke_summary,
    run_mixed_fingerprint_training_smoke,
    run_tiny_fingerprint_training_smoke,
    validate_fingerprint_smoke_report,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "behavioral_fingerprint"
CORRIDOR_FIXTURE = FIXTURE_ROOT / "v0_1_valid_tiny"
MIXED_FIXTURE = FIXTURE_ROOT / "v0_1_with_exemplars_tiny"


def test_artifact_summary_helper_reports_corridor_fixture() -> None:
    summary = summarize_fingerprint_artifact(CORRIDOR_FIXTURE)

    assert summary.artifact_type == "behavioral_fingerprint"
    assert summary.artifact_version == "0.1"
    assert summary.vocab_size == 128
    assert summary.max_seq_len == 16
    assert summary.num_corridor_records == 8
    assert summary.has_exemplars is False
    assert summary.num_exemplar_records == 0


def test_artifact_summary_helper_reports_exemplar_fixture() -> None:
    summary = summarize_fingerprint_artifact(MIXED_FIXTURE)

    assert summary.artifact_type == "behavioral_fingerprint"
    assert summary.vocab_size == 16
    assert summary.max_seq_len == 8
    assert summary.num_corridor_records == 4
    assert summary.has_exemplars is True
    assert summary.exemplar_payload_type == "dense_probs"
    assert summary.num_exemplar_records == 4


def test_inspect_fingerprint_artifact_cli_reports_summary() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_fingerprint_artifact.py",
            str(MIXED_FIXTURE),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "artifact_type=behavioral_fingerprint" in completed.stdout
    assert "num_corridor_records=4" in completed.stdout
    assert "has_exemplars=true" in completed.stdout
    assert "num_exemplar_records=4" in completed.stdout


def test_corridor_smoke_writes_p139_report_and_summary(tmp_path: Path) -> None:
    result = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=CORRIDOR_FIXTURE,
            output_dir=tmp_path / "corridor",
            steps=2,
            batch_size=2,
        )
    )
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    summary = Path(result.summary_path).read_text(encoding="utf-8")

    assert validate_fingerprint_smoke_report(report) == []
    assert report["report_schema_phase"] == "P139"
    assert report["report_type"] == "corridor_only_smoke_report"
    assert report["artifact"]["artifact_version"] == "0.1"
    assert report["corridor_targets"]["num_records"] == 8
    assert report["loss"]["finite"] is True
    assert "Fingerprint Corridor Smoke Summary" in summary
    assert "Main runner integrated: false" in summary
    assert "Teacher required: false" in summary
    assert "Inside all rate" in summary
    assert "Limitations" in summary


def test_mixed_smoke_writes_p139_report_and_summary(tmp_path: Path) -> None:
    result = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=MIXED_FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=2,
            corridor_batch_size=2,
            exemplar_batch_size=2,
        )
    )
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    summary = Path(result.summary_path).read_text(encoding="utf-8")

    assert validate_fingerprint_smoke_report(report) == []
    assert report["report_schema_phase"] == "P139"
    assert report["report_type"] == "mixed_fingerprint_smoke_report"
    assert report["exemplar_reservoir"]["enabled"] is True
    assert report["exemplar_reservoir"]["payload_type"] == "dense_probs"
    assert report["loss_weights"] == {"corridor": 1.0, "exemplar": 1.0}
    assert report["mixed_loss"]["finite"] is True
    assert set(report["corridor_metrics"]).issuperset({"inside_all_rate", "loss_total"})
    assert set(report["exemplar_metrics"]).issuperset(
        {"kl_loss", "cross_entropy", "teacher_entropy"}
    )
    assert "Mixed Fingerprint Smoke Summary" in summary
    assert "Exemplar reservoir enabled: true" in summary
    assert "Corridor loss weight" in summary
    assert "Exemplar loss weight" in summary
    assert "KL" in summary
    assert "Limitations" in summary


def test_canonical_metric_aliases_exist_for_both_smoke_modes(tmp_path: Path) -> None:
    corridor = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=CORRIDOR_FIXTURE,
            output_dir=tmp_path / "corridor",
            steps=1,
            batch_size=2,
        )
    )
    mixed = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=MIXED_FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=1,
            corridor_batch_size=2,
            exemplar_batch_size=2,
        )
    )

    assert "fingerprint/corridor/loss_total" in corridor.metrics
    assert "fingerprint/corridor/inside_all_rate" in corridor.metrics
    assert "fingerprint/mixed/loss_total" in mixed.metrics
    assert "fingerprint/mixed/corridor_loss_weight" in mixed.metrics
    assert "fingerprint/exemplar/kl_loss" in mixed.metrics
    assert all(np.isfinite(value) for value in corridor.metrics.values())
    assert all(np.isfinite(value) for value in mixed.metrics.values())


def test_report_limitation_flags_remain_false(tmp_path: Path) -> None:
    report = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=MIXED_FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=1,
        )
    ).to_report()

    assert report["main_runner_integrated"] is False
    assert report["real_student_backend_integrated"] is False
    assert report["smoke_student_uses_input_ids"] is False
    assert report["teacher_required"] is False
    assert report["hf_required"] is False
    assert report["accelerator_required"] is False


def test_summary_renderer_accepts_report_dict(tmp_path: Path) -> None:
    report = run_mixed_fingerprint_training_smoke(
        FingerprintMixedSmokeConfig(
            artifact_dir=MIXED_FIXTURE,
            output_dir=tmp_path / "mixed",
            steps=1,
        )
    ).to_report()

    rendered = render_fingerprint_smoke_summary(report)

    assert "Mixed Fingerprint Smoke Summary" in rendered
    assert "Standalone smoke only." in rendered
