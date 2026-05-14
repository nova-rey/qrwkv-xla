from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_wkv_live_update_hooks import (
    WKV_LIVE_UPDATE_HOOK_COMPARISON_SCHEMA,
    build_live_update_hook_trace,
    compare_live_update_hook_traces,
    load_live_update_hook_jsonl,
    write_live_update_hook_reports,
    write_live_update_hook_trace,
)

ROOT = Path(__file__).resolve().parents[1]
TRACE_SCRIPT = ROOT / "scripts" / "trace_radlads_qrwkv_wkv_live_update_hooks.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_radlads_qrwkv_wkv_live_update_hooks.py"


def _source_row(
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


def _minimal_source(side: str, *, residual: float = 0.0) -> list[dict[str, object]]:
    before = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    decay = np.array([[0.5, 0.25]], dtype=np.float32)
    decayed = before * decay[:, None, :]
    outer = np.array([[[0.01, 0.02], [0.03, 0.04]]], dtype=np.float32)
    after = decayed + outer + residual
    return [
        _source_row(side, "wkv_state_before", before),
        _source_row(side, "decay_after_transform", decay),
        _source_row(side, "wkv_decay_applied", decayed),
        _source_row(side, "k_a", np.array([[1.0, 2.0]], dtype=np.float32)),
        _source_row(side, "v", np.array([[3.0, 4.0]], dtype=np.float32)),
        _source_row(side, "wkv_update_outer_or_term", outer),
        _source_row(side, "wkv_state_after", after),
        _source_row(side, "wkv_state_before", after, token_index=1),
        _source_row(side, "wkv_state_after", after, token_index=1),
    ]


def test_live_hook_record_schema_validates() -> None:
    rows = build_live_update_hook_trace(_minimal_source("radlads"), side="radlads")
    first = rows[0]
    assert first["capture_kind"] in {"live_captured", "reconstructed", "unavailable"}
    assert first["source_file"].endswith("rwkv7_radlads_reference.py")
    assert first["status"] in {"pass", "unavailable", "fail"}


def test_missing_hook_is_reported_not_omitted() -> None:
    rows = build_live_update_hook_trace(
        _minimal_source("radlads"),
        side="radlads",
        allow_reconstructed=False,
    )
    missing = next(row for row in rows if row["stage"] == "balance_state_matmul")
    assert missing["capture_kind"] == "unavailable"
    assert missing["status"] == "unavailable"


def test_reconstructed_hook_is_labeled() -> None:
    rows = build_live_update_hook_trace(
        _minimal_source("radlads"),
        side="radlads",
        allow_reconstructed=True,
    )
    reconstructed = next(row for row in rows if row["stage"] == "balance_state_matmul")
    assert reconstructed["capture_kind"] == "reconstructed"


def test_hook_availability_matrix_schema_validates(tmp_path: Path) -> None:
    rad = build_live_update_hook_trace(_minimal_source("radlads"), side="radlads")
    qrw = build_live_update_hook_trace(_minimal_source("qrwkv"), side="qrwkv")
    report = compare_live_update_hook_traces(rad, qrw)
    write_live_update_hook_reports(
        radlads_entries=rad,
        qrwkv_entries=qrw,
        comparison_report=report,
        out_dir=tmp_path / "reports",
    )
    payload = json.loads(
        (tmp_path / "reports" / "wkv_live_update_hooks_report.json").read_text()
    )
    assert payload["schema"] == WKV_LIVE_UPDATE_HOOK_COMPARISON_SCHEMA
    assert (tmp_path / "reports" / "HOOK_AVAILABILITY_MATRIX.md").is_file()


def test_first_live_divergence_selector_is_deterministic() -> None:
    rad = build_live_update_hook_trace(_minimal_source("radlads"), side="radlads")
    qrw = build_live_update_hook_trace(
        _minimal_source("qrwkv", residual=0.01), side="qrwkv"
    )
    first = compare_live_update_hook_traces(rad, qrw)
    second = compare_live_update_hook_traces(rad, qrw)
    assert first["first_divergent_stage"] == second["first_divergent_stage"]
    assert first["first_divergent_case"] == "tiny_no_mask"


def test_jsonl_roundtrip_works(tmp_path: Path) -> None:
    rows = build_live_update_hook_trace(_minimal_source("radlads"), side="radlads")
    path = tmp_path / "trace.jsonl"
    write_live_update_hook_trace(rows, path)
    loaded = load_live_update_hook_jsonl(path)
    assert loaded[0]["case"] == "tiny_no_mask"
    assert loaded[0]["stage"] == rows[0]["stage"]


def test_compare_script_help_works() -> None:
    for script in (TRACE_SCRIPT, COMPARE_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P63" in result.stdout
        assert "WKV" in result.stdout
