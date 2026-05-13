from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_wkv_trace import (
    WKVTraceCollector,
    compare_trace_arrays,
    compare_trace_entries,
    load_trace_jsonl,
    write_trace_comparison_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def test_trace_collector_splits_headwise(tmp_path: Path) -> None:
    collector = WKVTraceCollector(
        case="tiny_no_mask", side="qrwkv", include_arrays=True
    )
    collector.record(
        "test",
        np.arange(16, dtype=np.float32).reshape(2, 2, 4),
        stage="wkv_state_after",
        layer=1,
        token_index=3,
    )
    assert len(collector.entries) == 2
    assert {row["head"] for row in collector.entries} == {0, 1}
    assert all(row["stage"] == "wkv_state_after" for row in collector.entries)

    out = tmp_path / "trace.jsonl"
    collector.write_jsonl(out)
    loaded = load_trace_jsonl(out)
    assert loaded[0]["case"] == "tiny_no_mask"


def test_compare_trace_entries_detects_first_divergence(tmp_path: Path) -> None:
    left = [
        {
            "case": "tiny_no_mask",
            "layer": 0,
            "head": 0,
            "token_index": 0,
            "stage": "wkv_state_after",
            "array": [[1.0, 2.0]],
        }
    ]
    right = [
        {
            "case": "tiny_no_mask",
            "layer": 0,
            "head": 0,
            "token_index": 0,
            "stage": "wkv_state_after",
            "array": [[1.0, 3.0]],
        }
    ]
    report = compare_trace_entries(left, right)
    assert report["first_divergent_stage"] == "wkv_state_after"
    assert report["first_divergent_layer"] == 0
    assert report["first_divergent_head"] == 0
    assert report["first_divergent_token"] == 0
    assert report["first_divergent_max_abs_error"] == 1.0

    out = tmp_path / "comparison"
    write_trace_comparison_reports(report, out)
    assert (out / "wkv_trace_comparison_report.json").is_file()
    assert (out / "P56_WKV_TRACE_COMPARISON.md").is_file()


def test_compare_trace_arrays_handles_shape_mismatch() -> None:
    stats = compare_trace_arrays(np.zeros((2, 2)), np.zeros((2, 3)))
    assert stats["status"] == "shape_mismatch"
    assert stats["shape_match"] is False


def test_cli_help_scripts() -> None:
    for script in [
        "trace_radlads_qrwkv_wkv_update.py",
        "compare_radlads_qrwkv_wkv_trace.py",
        "analyze_wkv_update_order_candidates.py",
    ]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P56" in result.stdout or "WKV" in result.stdout
