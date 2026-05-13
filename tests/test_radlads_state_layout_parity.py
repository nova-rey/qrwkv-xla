from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P54_ROOT = ROOT / "artifacts/p54_confirmation"
RADLADS_OUTPUTS = P54_ROOT / "radlads_outputs"
QRWKV_OUTPUTS = P54_ROOT / "qrwkv_outputs"


@pytest.mark.parametrize(
    ("script", "needle"),
    [
        ("audit_radlads_qrwkv_surface_layouts.py", "P55"),
        ("analyze_radlads_qrwkv_layout_candidates.py", "P55"),
        ("compare_radlads_qrwkv_head_to_head.py", "radlads_qrwkv"),
    ],
)
def test_cli_help(script: str, needle: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert needle.lower() in result.stdout.lower()


def test_surface_layout_audit_and_candidate_analysis(tmp_path: Path) -> None:
    layout_out = tmp_path / "layout"
    candidates_out = tmp_path / "candidates"

    audit = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_radlads_qrwkv_surface_layouts.py"),
            "--radlads-outputs",
            str(RADLADS_OUTPUTS),
            "--qrwkv-outputs",
            str(QRWKV_OUTPUTS),
            "--out",
            str(layout_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote surface layout audit" in audit.stdout

    payload = json.loads((layout_out / "surface_layout_audit.json").read_text())
    rows = payload["rows"]
    hidden = next(
        row
        for row in rows
        if row["case"] == "tiny_no_mask" and row["surface"] == "hidden_states"
    )
    wkv = next(
        row
        for row in rows
        if row["case"] == "tiny_no_mask" and row["surface"] == "wkv_matrix_state"
    )
    assert hidden["radlads_shape"] == [2, 4, 8]
    assert hidden["qrwkv_shape"] == [2, 2, 4, 8]
    assert hidden["shape_relation"] == "layer_major_all_hidden"
    assert wkv["shape_relation"] == "same_shape"

    cand = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_radlads_qrwkv_layout_candidates.py"),
            "--radlads-outputs",
            str(RADLADS_OUTPUTS),
            "--qrwkv-outputs",
            str(QRWKV_OUTPUTS),
            "--out",
            str(candidates_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote layout candidate analysis" in cand.stdout
    candidate_payload = json.loads(
        (candidates_out / "layout_candidate_report.json").read_text()
    )
    candidate_rows = candidate_payload["rows"]
    hidden_final = next(
        row
        for row in candidate_rows
        if row["case"] == "tiny_no_mask"
        and row["surface"] == "hidden_states"
        and row["candidate_name"] == "select_final_layer_if_all_layers_present"
    )
    wkv_transpose = next(
        row
        for row in candidate_rows
        if row["case"] == "tiny_no_mask"
        and row["surface"] == "wkv_matrix_state"
        and row["candidate_name"] == "transpose_last_two_matrix_dims"
    )
    stepwise_na = next(
        row
        for row in candidate_rows
        if row["case"] == "tiny_no_mask"
        and row["surface"] == "stepwise_hidden_states"
        and row["candidate_name"] == "full_final_vs_stepwise_final"
    )
    assert hidden_final["applicable"] is True
    assert hidden_final["shape_match"] is True
    assert hidden_final["finite_both"] is True
    assert hidden_final["status"] == "fail"
    assert hidden_final["max_abs_error"] and hidden_final["max_abs_error"] > 0
    assert wkv_transpose["applicable"] is True
    assert wkv_transpose["shape_match"] is True
    assert wkv_transpose["finite_both"] is True
    assert stepwise_na["status"] == "not_applicable"

    assert (layout_out / "P55_SURFACE_LAYOUT_AUDIT.md").is_file()
    assert (layout_out / "surface_layout_audit.json").is_file()
    assert (candidates_out / "P55_LAYOUT_CANDIDATES.md").is_file()
    assert (candidates_out / "layout_candidate_report.json").is_file()


def test_head_to_head_normalizes_hidden_states_and_classifies_stepwise(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_radlads_qrwkv_head_to_head.py"),
            "--manifest",
            str(P54_ROOT / "fixtures" / "manifest.json"),
            "--radlads-outputs",
            str(RADLADS_OUTPUTS),
            "--qrwkv-outputs",
            str(QRWKV_OUTPUTS),
            "--out",
            str(out_dir),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote P53 comparison reports" in result.stdout

    report = json.loads((out_dir / "head_to_head_comparison_report.json").read_text())
    tiny = next(case for case in report["cases"] if case["name"] == "tiny_no_mask")
    hidden = next(row for row in tiny["comparisons"] if row["name"] == "hidden_states")
    stepwise = next(
        row for row in tiny["comparisons"] if row["name"] == "stepwise_hidden_states"
    )
    assert hidden["status"] == "fail"
    assert hidden["shape_match"] is True
    assert stepwise["status"] == "not_applicable"
    assert report["cases_ran_both_sides"] > 0
    assert report["surface_comparisons_count"] > 0
    assert (
        report["surface_conventions"]["hidden_states"]["comparison"]
        == "final_hidden_selected_from_layer_major"
    )
