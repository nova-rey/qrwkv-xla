from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qrwkv_xla.parity.radlads_clean_loader import (
    audit_clean_payload_loader,
    export_qrwkv_clean_payload_outputs,
    export_radlads_clean_payload_outputs,
    load_radlads_clean_payload,
    write_audit_reports,
)
from qrwkv_xla.parity.radlads_head_to_head import compare_radlads_qrwkv_head_to_head

ROOT = Path(__file__).resolve().parents[1]
RADLADS_REPO = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")
FIXTURE_MANIFEST = ROOT / "artifacts/p53_confirmation/head_to_head/manifest.json"
FIXTURE_PARAMETERS = (
    ROOT / "artifacts/p53_confirmation/head_to_head/radlads_parameters.npz"
)


@pytest.mark.skipif(
    not RADLADS_REPO.exists(), reason="RADLADS reference repo is missing"
)
def test_loader_audit_and_gate_adapters(tmp_path: Path) -> None:
    result = load_radlads_clean_payload(
        FIXTURE_PARAMETERS,
        radlads_source_path=RADLADS_REPO,
        seed=5353,
        run_smoke=False,
    )

    assert result.overall_status == "pass"
    assert len(result.shape_mismatches) == 4
    assert len(result.excluded) == 34
    assert len(result.unsupported) == 0
    assert len(result.missing_required) == 0
    assert all(row.get("adapter") == "truncate_rank" for row in result.shape_mismatches)

    report = audit_clean_payload_loader(RADLADS_REPO, FIXTURE_PARAMETERS, seed=5353)
    out_dir = tmp_path / "audit"
    write_audit_reports(report, out_dir)

    assert report["status"] == "pass"
    assert report["blockers_after"]["shape_mismatches"] == 4
    assert report["blockers_after"]["unsupported"] == 0
    assert report["counts"]["excluded_not_runtime_critical"] == 34
    assert (out_dir / "radlads_loader_audit.json").is_file()
    assert (out_dir / "P54_RADLADS_LOADER_AUDIT.md").is_file()
    payload = json.loads((out_dir / "radlads_loader_audit.json").read_text())
    assert payload["status"] == "pass"


@pytest.mark.skipif(
    not RADLADS_REPO.exists(), reason="RADLADS reference repo is missing"
)
def test_export_and_compare_outputs(tmp_path: Path) -> None:
    radlads_out = tmp_path / "radlads_outputs"
    qrwkv_out = tmp_path / "qrwkv_outputs"
    comparison_out = tmp_path / "comparison"

    radlads_manifest = export_radlads_clean_payload_outputs(
        FIXTURE_PARAMETERS,
        radlads_out,
        radlads_source_path=RADLADS_REPO,
        seed=5353,
        overwrite=True,
    ).output_manifest
    qrwkv_manifest = export_qrwkv_clean_payload_outputs(
        FIXTURE_PARAMETERS,
        qrwkv_out,
        seed=5353,
        overwrite=True,
    )

    assert radlads_manifest["overall_status"] == "pass"
    assert qrwkv_manifest["overall_status"] == "pass"
    assert (radlads_out / "manifest.json").is_file()
    assert (qrwkv_out / "manifest.json").is_file()

    report = compare_radlads_qrwkv_head_to_head(
        FIXTURE_MANIFEST,
        radlads_outputs=radlads_out,
        qrwkv_outputs=qrwkv_out,
        out_dir=comparison_out,
        report_prefix="P54",
    )

    assert report["cases_ran_both_sides"] > 0
    assert report["surface_comparisons_count"] > 0
    assert (comparison_out / "head_to_head_comparison_report.json").is_file()
    assert (comparison_out / "P54_RESULTS.md").is_file()
    assert (comparison_out / "P54_SURFACE_COMPARISON.md").is_file()


@pytest.mark.skipif(
    not RADLADS_REPO.exists(), reason="RADLADS reference repo is missing"
)
def test_clean_loader_cli_help() -> None:
    for script in (
        "audit_radlads_clean_payload_loader.py",
        "export_radlads_clean_payload_outputs.py",
        "compare_radlads_qrwkv_head_to_head.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P54" in result.stdout or "RADLADS" in result.stdout


@pytest.mark.skipif(
    not RADLADS_REPO.exists(), reason="RADLADS reference repo is missing"
)
def test_missing_repo_reports_source_unavailable(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-radlads"
    result = load_radlads_clean_payload(
        FIXTURE_PARAMETERS,
        radlads_source_path=missing_repo,
        seed=5353,
        run_smoke=False,
    )

    assert result.overall_status == "blocked"
    assert result.reason
