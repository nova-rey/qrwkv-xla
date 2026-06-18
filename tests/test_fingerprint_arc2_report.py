from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from qrwkv_xla.fingerprint import (
    FingerprintArc2ReportConfig,
    build_fingerprint_arc2_report,
    run_fingerprint_arc2_report,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fingerprint_arc2_report.py"
SNAPSHOT = ROOT / "docs" / "QRWKV_SNAPSHOT.yaml"


def test_arc2_report_emits_constrained_go(tmp_path: Path) -> None:
    result = run_fingerprint_arc2_report(
        FingerprintArc2ReportConfig(
            output_dir=tmp_path / "p149",
            snapshot_path=SNAPSHOT,
            overwrite=True,
        )
    )
    report = _json(result.report_path)

    assert result.status == "pass"
    assert report["phase"] == "P149"
    assert report["run_kind"] == "arc2_report_go_no_go"
    assert report["go_no_go"]["recommendation"] == "go_with_constraints"
    assert report["go_no_go"]["go"] is True
    assert report["go_no_go"]["constraints"]
    assert report["go_no_go"]["no_go_blockers"] == []
    assert result.summary_path.is_file()


def test_report_covers_p140_through_p148_evidence() -> None:
    report = build_fingerprint_arc2_report(_snapshot())
    evidence = {item["flag"]: item for item in report["evidence"]}

    for phase in range(140, 149):
        assert f"P{phase}" in report["arc"]["covered_phases"]
    assert evidence["p140_real_student_forward_smoke"]["present"] is True
    assert evidence["p141_main_runner_fingerprint_mode"]["present"] is True
    assert evidence["p148_quality_per_byte_experiment"]["present"] is True


def test_claims_block_prevents_overclaim() -> None:
    report = build_fingerprint_arc2_report(_snapshot())
    claims = report["claims"]

    assert claims["general_quality_claim_made"] is False
    assert claims["trained_baseline_win_claim_made"] is False
    assert claims["radlads_parity_claim_made"] is False
    assert claims["scale_readiness_claim_made"] is False
    assert claims["production_readiness_claim_made"] is False
    assert claims["pallas_default_claim_made"] is False


def test_missing_required_evidence_blocks_go() -> None:
    snapshot = _snapshot()
    snapshot["main_contains"]["p148_quality_per_byte_experiment"] = False
    report = build_fingerprint_arc2_report(snapshot)

    assert report["status"] == "fail"
    assert report["go_no_go"]["recommendation"] == "no_go"
    assert report["go_no_go"]["go"] is False
    assert (
        "p148_quality_per_byte_experiment missing"
        in report["go_no_go"]["no_go_blockers"]
    )


def test_report_records_open_gaps_and_next_steps() -> None:
    report = build_fingerprint_arc2_report(_snapshot())
    gaps = {gap["id"] for gap in report["open_gaps"]}

    assert "trained_baseline" in gaps
    assert "generalization_eval" in gaps
    assert "scale" in gaps
    assert report["recommended_next_steps"]


def test_cli_writes_report_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_p149"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--snapshot",
            str(SNAPSHOT),
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "recommendation=go_with_constraints" in completed.stdout
    assert (output_dir / "p149_arc2_report.json").is_file()
    assert (output_dir / "p149_arc2_summary.md").is_file()


def _snapshot() -> dict:
    payload = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
