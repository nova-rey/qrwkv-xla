from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_wkv_composite_hook import (
    WKV_COMPOSITE_BALANCE_HOOK_COMPARISON_SCHEMA,
    build_composite_hook_trace,
    compare_composite_hook_traces,
    load_composite_hook_jsonl,
    write_composite_hook_reports,
    write_composite_hook_trace,
)

ROOT = Path(__file__).resolve().parents[1]
LOCATE_SCRIPT = ROOT / "scripts" / "locate_radlads_qrwkv_wkv_composite_hook.py"
EXTRACT_SCRIPT = ROOT / "scripts" / "extract_radlads_qrwkv_wkv_composite_hook.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_radlads_qrwkv_wkv_composite_hook.py"


def _row(
    side: str, stage: str, value: object, token_index: int = 0
) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "case": "tiny_no_mask",
        "side": side,
        "layer": 0,
        "head": 0,
        "token_index": token_index,
        "stage": stage,
        "name": f"{side}.{stage}",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": True,
        "min": float(np.min(array)) if array.size else 0.0,
        "max": float(np.max(array)) if array.size else 0.0,
        "mean": float(np.mean(array)) if array.size else 0.0,
        "std": float(np.std(array)) if array.size else 0.0,
        "abs_max": float(np.max(np.abs(array))) if array.size else 0.0,
        "array": array.tolist(),
    }


def _source_rows(
    side: str, *, include_state_after: bool = True
) -> list[dict[str, object]]:
    state_before = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    decay_value = np.array([[0.5, 0.25]], dtype=np.float32)
    decayed_state = state_before * decay_value[:, None, :]
    update_outer = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    balance = np.array([[[0.01, 0.02], [0.03, 0.04]]], dtype=np.float32)
    state_after = decayed_state + update_outer + balance
    rows = [
        _row(side, "state_before", state_before),
        _row(side, "decay_value", decay_value),
        _row(side, "update_outer_product", update_outer),
    ]
    if include_state_after:
        rows.append(_row(side, "state_after", state_after))
    return rows


def test_source_locator_report_schema_validates(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(LOCATE_SCRIPT),
            "--qrwkv-root",
            str(ROOT),
            "--out",
            str(tmp_path / "locator"),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (tmp_path / "locator" / "composite_hook_source_locator.json").read_text()
    )
    assert payload["schema"] == "radlads_qrwkv_wkv_composite_hook_locator.v1"
    assert payload["sources"][0]["comparison_label"] == "composite_balance_update_term"
    assert "wrote P64" in result.stdout


def test_composite_hook_record_schema_validates(tmp_path: Path) -> None:
    rad = build_composite_hook_trace(
        _source_rows("radlads"), side="radlads", allow_exact_reconstruction=True
    )
    qrw = build_composite_hook_trace(
        _source_rows("qrwkv"), side="qrwkv", allow_exact_reconstruction=True
    )
    report = compare_composite_hook_traces(rad, qrw)
    write_composite_hook_reports(
        radlads_entries=rad,
        qrwkv_entries=qrw,
        comparison_report=report,
        out_dir=tmp_path / "reports",
    )
    payload = json.loads(
        (tmp_path / "reports" / "composite_balance_hook_report.json").read_text()
    )
    assert payload["schema"] == WKV_COMPOSITE_BALANCE_HOOK_COMPARISON_SCHEMA
    assert (tmp_path / "reports" / "P64_COMPOSITE_BALANCE_HOOK.md").is_file()


def test_capture_kind_enum_validates() -> None:
    rows = build_composite_hook_trace(
        _source_rows("radlads"), side="radlads", allow_exact_reconstruction=True
    )
    kinds = {row["capture_kind"] for row in rows}
    assert kinds <= {
        "live_captured",
        "exact_reconstruction",
        "partial_reconstruction",
        "unavailable",
    }


def test_unavailable_hook_is_reported_clearly() -> None:
    rows = build_composite_hook_trace(
        _source_rows("radlads", include_state_after=False), side="radlads"
    )
    missing = next(
        row
        for row in rows
        if row["comparison_label"] == "composite_balance_update_term"
    )
    assert missing["capture_kind"] == "unavailable"
    assert missing["status"] == "unavailable"


def test_partial_reconstruction_is_not_labeled_exact() -> None:
    rows = build_composite_hook_trace(
        [
            _row(
                "radlads",
                "state_before",
                np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            ),
        ],
        side="radlads",
        allow_exact_reconstruction=True,
        allow_partial_reconstruction=True,
    )
    decayed = next(row for row in rows if row["comparison_label"] == "decayed_state")
    assert decayed["capture_kind"] in {"partial_reconstruction", "unavailable"}
    assert decayed["capture_kind"] != "exact_reconstruction"


def test_exact_reconstruction_requires_all_ingredients() -> None:
    rows = build_composite_hook_trace(
        _source_rows("radlads", include_state_after=True),
        side="radlads",
        allow_exact_reconstruction=True,
    )
    composite = next(
        row
        for row in rows
        if row["comparison_label"] == "composite_balance_update_term"
    )
    assert composite["capture_kind"] == "exact_reconstruction"
    reconstructed = next(
        row
        for row in rows
        if row["comparison_label"] == "composite_balance_update_term_reconstructed"
    )
    assert reconstructed["capture_kind"] == "exact_reconstruction"
    formula = next(
        row
        for row in rows
        if row["comparison_label"] == "state_after_from_full_source_formula"
    )
    assert formula["capture_kind"] == "exact_reconstruction"


def test_comparison_report_schema_validates(tmp_path: Path) -> None:
    rad = build_composite_hook_trace(
        _source_rows("radlads"), side="radlads", allow_exact_reconstruction=True
    )
    qrw = build_composite_hook_trace(
        _source_rows("qrwkv"), side="qrwkv", allow_exact_reconstruction=True
    )
    report = compare_composite_hook_traces(rad, qrw)
    assert report["schema"] == WKV_COMPOSITE_BALANCE_HOOK_COMPARISON_SCHEMA
    assert report["composite_balance_update_term_match"] is True
    assert report["state_after_from_full_source_formula_match"] is True
    write_composite_hook_reports(
        radlads_entries=rad,
        qrwkv_entries=qrw,
        comparison_report=report,
        out_dir=tmp_path / "reports",
    )
    assert (tmp_path / "reports" / "composite_balance_hook_report.json").is_file()


def test_decision_gate_emits_exactly_one_recommendation(tmp_path: Path) -> None:
    rad = build_composite_hook_trace(
        _source_rows("radlads"), side="radlads", allow_exact_reconstruction=True
    )
    qrw = build_composite_hook_trace(
        _source_rows("qrwkv"), side="qrwkv", allow_exact_reconstruction=True
    )
    report = compare_composite_hook_traces(rad, qrw)
    write_composite_hook_reports(
        radlads_entries=rad,
        qrwkv_entries=qrw,
        comparison_report=report,
        out_dir=tmp_path / "reports",
    )
    text = (tmp_path / "reports" / "P64_DECISION_GATE.md").read_text()
    assert text.count("recommendation:") == 1


def test_script_help_works() -> None:
    for script in (LOCATE_SCRIPT, EXTRACT_SCRIPT, COMPARE_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P64" in result.stdout
        assert "WKV" in result.stdout


def test_jsonl_roundtrip_works(tmp_path: Path) -> None:
    rows = build_composite_hook_trace(
        _source_rows("radlads"), side="radlads", allow_exact_reconstruction=True
    )
    path = tmp_path / "trace.jsonl"
    write_composite_hook_trace(rows, path)
    loaded = load_composite_hook_jsonl(path)
    assert loaded[0]["case"] == "tiny_no_mask"
    assert loaded[0]["comparison_label"] == rows[0]["comparison_label"]
