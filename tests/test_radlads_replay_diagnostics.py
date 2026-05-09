from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import (
    build_diagnostic_report,
    find_first_nonfinite,
    summarize_array,
    summarize_parameter_payload,
    write_diagnostic_reports,
    write_parameter_sanity_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def test_summarize_array_detects_finite_arrays() -> None:
    summary = summarize_array("finite", np.array([[1.0, -2.0]], dtype=np.float32))
    assert summary.nonfinite_count == 0
    assert summary.abs_max == 2.0
    assert summary.first_nonfinite_index is None


def test_summarize_array_detects_nan_and_inf() -> None:
    summary = summarize_array(
        "bad",
        np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32),
    )
    assert summary.nonfinite_count == 3
    assert summary.nan_count == 1
    assert summary.posinf_count == 1
    assert summary.neginf_count == 1
    assert summary.first_nonfinite_index == [1]


def test_find_first_nonfinite_uses_ordered_summaries() -> None:
    rows = [
        summarize_array("good", np.array([1.0], dtype=np.float32)),
        summarize_array("bad", np.array([np.nan], dtype=np.float32)),
        summarize_array("later", np.array([np.inf], dtype=np.float32)),
    ]
    first = find_first_nonfinite(rows)
    assert first is not None
    assert first["name"] == "bad"


def test_parameter_sanity_flags_nonfinite_and_huge_parameter() -> None:
    report = summarize_parameter_payload(
        {
            "ok.weight": np.array([1.0], dtype=np.float32),
            "huge.weight": np.array([1e9], dtype=np.float32),
            "bad.weight": np.array([np.nan], dtype=np.float32),
        },
        mapping_entries=[
            {"radlads": "ok.weight", "qrwkv": "ok.weight", "qrwkv_shape": [1]},
            {
                "radlads": "huge.weight",
                "qrwkv": "huge.weight",
                "qrwkv_shape": [1],
            },
            {"radlads": "bad.weight", "qrwkv": "bad.weight", "qrwkv_shape": [1]},
            {
                "status": "defaulted",
                "qrwkv": "layers.self_attn.time_bias",
                "reason": "x",
            },
        ],
        active_defaulted_surfaces={"layers.self_attn.time_bias"},
    )
    assert report["nonfinite_parameter_count"] == 1
    suspicious = {row["name"]: row for row in report["suspicious_parameters"]}
    assert "huge.weight" in suspicious
    assert "bad.weight" in suspicious
    defaults = {row["qrwkv"]: row for row in report["defaulted_parameters"]}
    assert (
        defaults["layers.self_attn.time_bias"]["qrwkv_only_default_used_in_active_path"]
        is True
    )


def test_diagnostic_and_parameter_report_writers(tmp_path: Path) -> None:
    parameter_report = summarize_parameter_payload(
        {"ok.weight": np.array([1.0], dtype=np.float32)},
        mapping_entries=[
            {"radlads": "ok.weight", "qrwkv": "ok.weight", "qrwkv_shape": [1]}
        ],
    )
    write_parameter_sanity_reports(parameter_report, tmp_path)
    report = build_diagnostic_report(
        case_reports=[
            {
                "case": "tiny_no_mask",
                "first_nonfinite": None,
                "final_outputs_finite": True,
                "instrumented_stages": ["input_embeddings"],
                "suspected_root_cause": "active_path_profile_mismatch",
            }
        ],
        parameter_sanity=parameter_report,
    )
    write_diagnostic_reports(report, tensor_summaries=[], out_dir=tmp_path)
    assert (tmp_path / "parameter_sanity_report.json").is_file()
    assert (tmp_path / "P51_PARAMETER_SANITY.md").is_file()
    assert (tmp_path / "replay_diagnostics.json").is_file()
    assert (tmp_path / "tensor_summaries.jsonl").is_file()
    assert (tmp_path / "P51_DIAGNOSTIC_REPORT.md").is_file()
    payload = json.loads((tmp_path / "replay_diagnostics.json").read_text())
    assert payload["cases"][0]["case"] == "tiny_no_mask"


def test_diagnose_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "diagnose_radlads_replay_nonfinite.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P51" in result.stdout
    assert "--all-cases" in result.stdout
