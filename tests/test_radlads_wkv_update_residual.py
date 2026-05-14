from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_wkv_update_residual import (
    WKV_UPDATE_RESIDUAL_COMPARISON_SCHEMA,
    build_update_residual_trace,
    compare_update_residual_traces,
    load_update_residual_jsonl,
    reconstruct_update_residual,
    write_update_residual_reports,
    write_update_residual_trace,
)

ROOT = Path(__file__).resolve().parents[1]


def _source_row(
    *,
    side: str,
    stage: str,
    value: object,
    token_index: int = 0,
    head: int = 0,
) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "case": "tiny_no_mask",
        "side": side,
        "layer": 0,
        "head": head,
        "token_index": token_index,
        "stage": stage,
        "name": f"{side}.{stage}",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": True,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "abs_max": float(np.max(np.abs(array))),
        "array": array.tolist(),
    }


def _minimal_source(side: str, *, residual: float = 0.0) -> list[dict[str, object]]:
    before = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    decay = np.array([[0.5, 0.25]], dtype=np.float32)
    decayed = before * decay[:, None, :]
    outer = np.array([[[0.01, 0.02], [0.03, 0.04]]], dtype=np.float32)
    after = decayed + outer + residual
    return [
        _source_row(side=side, stage="wkv_state_before", value=before),
        _source_row(side=side, stage="decay_after_transform", value=decay),
        _source_row(side=side, stage="wkv_decay_applied", value=decayed),
        _source_row(side=side, stage="k_a", value=np.array([[1.0, 2.0]])),
        _source_row(side=side, stage="v", value=np.array([[3.0, 4.0]])),
        _source_row(side=side, stage="wkv_update_outer_or_term", value=outer),
        _source_row(side=side, stage="wkv_state_after", value=after),
        _source_row(
            side=side,
            stage="wkv_state_before",
            value=after,
            token_index=1,
        ),
        _source_row(
            side=side,
            stage="wkv_state_after",
            value=after,
            token_index=1,
        ),
    ]


def test_build_update_residual_trace_marks_required_stages_and_unavailable() -> None:
    rows = build_update_residual_trace(_minimal_source("radlads"), side="radlads")
    stages = {row["stage"] for row in rows}
    assert "state_before" in stages
    assert "decay_value" in stages
    assert "state_after_for_next_token" in stages
    update_term = next(row for row in rows if row["stage"] == "update_term")
    assert update_term["available"] is False
    assert "Composite update_term" in update_term["unavailable_reason"]


def test_compare_update_residual_traces_reports_first_residual() -> None:
    radlads = build_update_residual_trace(_minimal_source("radlads"), side="radlads")
    qrwkv = build_update_residual_trace(
        _minimal_source("qrwkv", residual=0.01),
        side="qrwkv",
    )
    report = compare_update_residual_traces(radlads, qrwkv)
    assert report["schema"] == WKV_UPDATE_RESIDUAL_COMPARISON_SCHEMA
    assert report["kernel_ready"] == "no"
    assert report["first_divergent_stage"] in {
        "update_term",
        "state_after",
        "state_after_for_next_token",
    }
    assert report["audit"]["outer_product_convention"]["radlads_available"] is True
    assert report["audit"]["decay_application"]["qrwkv_available"] is True


def test_reconstruct_update_residual_is_diagnostic_when_composite_missing() -> None:
    rows = build_update_residual_trace(_minimal_source("radlads"), side="radlads")
    report = reconstruct_update_residual(rows)
    first = report["first_residual"]
    assert report["status"] == "pass"
    assert first is None

    with_balance = build_update_residual_trace(
        _minimal_source("radlads", residual=0.5),
        side="radlads",
    )
    residual_report = reconstruct_update_residual(with_balance)
    assert residual_report["status"] == "fail"
    assert residual_report["first_residual"]["max_abs_error"] > 0.0


def test_writer_roundtrip_and_reports(tmp_path: Path) -> None:
    radlads = build_update_residual_trace(_minimal_source("radlads"), side="radlads")
    qrwkv = build_update_residual_trace(_minimal_source("qrwkv"), side="qrwkv")
    rad_path = tmp_path / "rad.jsonl"
    qrw_path = tmp_path / "qrw.jsonl"
    write_update_residual_trace(radlads, rad_path)
    write_update_residual_trace(qrwkv, qrw_path)
    assert load_update_residual_jsonl(rad_path)

    report = compare_update_residual_traces(
        load_update_residual_jsonl(rad_path),
        load_update_residual_jsonl(qrw_path),
    )
    write_update_residual_reports(
        radlads_entries=radlads,
        qrwkv_entries=qrwkv,
        comparison_report=report,
        out_dir=tmp_path / "reports",
    )
    assert (
        tmp_path / "reports" / "wkv_update_residual_comparison_report.json"
    ).is_file()
    assert (tmp_path / "reports" / "P62_WKV_UPDATE_RESIDUAL.md").is_file()
    assert (tmp_path / "reports" / "wkv_update_residual_values.npz").is_file()


def test_compare_script_end_to_end_with_tmp_path_traces(tmp_path: Path) -> None:
    rad_source = tmp_path / "source_radlads.jsonl"
    qrw_source = tmp_path / "source_qrwkv.jsonl"
    rad_source.write_text(
        "\n".join(json.dumps(row) for row in _minimal_source("radlads")) + "\n",
        encoding="utf-8",
    )
    qrw_source.write_text(
        "\n".join(json.dumps(row) for row in _minimal_source("qrwkv")) + "\n",
        encoding="utf-8",
    )
    trace_out = tmp_path / "trace"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "trace_radlads_qrwkv_wkv_update_residual.py"),
            "--radlads-trace",
            str(rad_source),
            "--qrwkv-trace",
            str(qrw_source),
            "--out",
            str(trace_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compare_out = tmp_path / "compare"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_radlads_qrwkv_wkv_update_residual.py"),
            "--radlads-trace",
            str(trace_out / "wkv_update_residual_radlads.jsonl"),
            "--qrwkv-trace",
            str(trace_out / "wkv_update_residual_qrwkv.jsonl"),
            "--out",
            str(compare_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(
        (compare_out / "wkv_update_residual_comparison_report.json").read_text()
    )
    assert report["schema"] == WKV_UPDATE_RESIDUAL_COMPARISON_SCHEMA
    assert report["kernel_ready"] == "no"


def test_p62_script_help() -> None:
    for script in (
        "trace_radlads_qrwkv_wkv_update_residual.py",
        "compare_radlads_qrwkv_wkv_update_residual.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P62" in result.stdout
        assert "WKV" in result.stdout
