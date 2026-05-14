from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qrwkv_xla.parity.radlads_wkv_state_provenance import load_provenance_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_real_radlads_qrwkv_wkv_state_provenance.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_real_radlads_qrwkv_wkv_state_provenance.py"


def _run_p60(tmp_path: Path, *extra: str) -> Path:
    out = tmp_path / "p60"
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--out",
            str(out),
            "--overwrite",
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P60 real WKV state provenance" in result.stdout
    return out


def _compare_p60(out: Path, *extra: str) -> Path:
    comparison = out / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--radlads-trace",
            str(out / "real_wkv_state_provenance_radlads.jsonl"),
            "--qrwkv-trace",
            str(out / "real_wkv_state_provenance_qrwkv.jsonl"),
            "--out",
            str(comparison),
            "--overwrite",
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P60 real WKV state provenance comparison" in result.stdout
    return comparison


def test_p60_runner_labels_cached_real_artifacts_and_valid_schema(
    tmp_path: Path,
) -> None:
    out = _run_p60(tmp_path)
    metadata = json.loads((out / "real_provenance_metadata.json").read_text())
    assert metadata["real_artifact_trace"] is True
    assert metadata["synthetic_trace"] is False
    assert metadata["derived_from_cached_outputs"] is True
    assert metadata["regenerated_live_outputs"] is False

    for filename in [
        "real_wkv_state_provenance_radlads.jsonl",
        "real_wkv_state_provenance_qrwkv.jsonl",
    ]:
        rows = load_provenance_jsonl(out / filename)
        assert rows
        assert all(row["real_artifact_trace"] is True for row in rows)
        assert all(row["synthetic_trace"] is False for row in rows)
        assert all(row["derived_from_cached_outputs"] is True for row in rows)

    for filename in [
        "P60_RESULTS.md",
        "TRACE_PROVENANCE.md",
        "TINY_NO_MASK_REAL_STATE.md",
        "TINY_STEPWISE_REAL_STATE.md",
        "REAL_MASK_PADDING_STATE.md",
        "HIDDEN_STATE_DEPENDENCY.md",
        "p60_real_state_provenance_report.json",
    ]:
        assert (out / filename).is_file()
    assert not (out / "P60_FIX_NOTE.md").exists()


def test_p60_strict_mode_fails_instead_of_synthetic_fallback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--out",
            str(tmp_path / "strict"),
            "--strict-real-artifacts",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "will not synthesize replacements" in result.stderr


def test_p60_missing_path_failure(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--out",
            str(tmp_path / "missing"),
            "--radlads-outputs",
            str(tmp_path / "missing-radlads"),
            "--case",
            "tiny_no_mask",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing real artifact paths" in result.stderr


def test_p60_compare_deterministic_first_divergence(tmp_path: Path) -> None:
    out = _run_p60(tmp_path)
    comparison = _compare_p60(out)
    report = json.loads(
        (comparison / "p60_real_wkv_state_provenance_report.json").read_text()
    )
    first = report["first_divergence"]
    assert first == report["first_mismatch"]
    assert first["case"] == "tiny_attention_mask"
    assert first["comparison"] == "initial_state_handoff"
    assert first["state_name"] == "wkv_matrix_state"
    assert (comparison / "P60_REAL_WKV_STATE_PROVENANCE.md").is_file()


def test_p60_case_and_mode_filtering(tmp_path: Path) -> None:
    out = _run_p60(
        tmp_path,
        "--case",
        "tiny_stepwise_state",
        "--mode",
        "stepwise",
    )
    radlads = load_provenance_jsonl(out / "real_wkv_state_provenance_radlads.jsonl")
    qrwkv = load_provenance_jsonl(out / "real_wkv_state_provenance_qrwkv.jsonl")
    assert {row["case"] for row in radlads + qrwkv} == {"tiny_stepwise_state"}
    assert {row["comparison"] for row in radlads + qrwkv} == {"full_vs_stepwise"}

    comparison = _compare_p60(out, "--case", "tiny_stepwise_state")
    report = json.loads(
        (comparison / "p60_real_wkv_state_provenance_report.json").read_text()
    )
    assert report["case_counts"] == {"tiny_stepwise_state": report["row_count"]}


def test_p60_hidden_state_dependency_schema(tmp_path: Path) -> None:
    out = _run_p60(
        tmp_path,
        "--case",
        "tiny_stepwise_state",
        "--mode",
        "stepwise",
    )
    payload = json.loads((out / "hidden_state_dependency.json").read_text())
    assert payload["schema"] == "radlads_qrwkv_p60_hidden_state_dependency.v1"
    assert payload["hidden_state_rows"] == 2
    required = set(payload["required_fields"])
    assert {"real_artifact_trace", "derived_from_cached_outputs"} <= required
    assert all(row["state_name"] == "hidden_states" for row in payload["rows"])


def test_p60_script_help() -> None:
    for script in [RUN_SCRIPT, COMPARE_SCRIPT]:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P60" in result.stdout
        assert "provenance" in result.stdout
