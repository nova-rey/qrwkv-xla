from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.parity.radlads_wkv_state_provenance import (
    WKV_STATE_PROVENANCE_SCHEMA,
    compare_provenance_records,
    compare_state_arrays,
    load_provenance_jsonl,
    make_provenance_record,
    trace_qrwkv_state_provenance,
    validate_provenance_record,
    write_provenance_jsonl,
    write_provenance_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_student():
    jax = pytest.importorskip("jax")
    from qrwkv_xla.students.rwkv7_qwen_reference import (
        RWKV7QwenReferenceConfig,
        RWKV7QwenReferenceStudent,
    )

    config = RWKV7QwenReferenceConfig(
        vocab_size=24,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        use_rope=False,
        emit_logits=True,
        attention_qkv_bias=True,
    )
    student = RWKV7QwenReferenceStudent(config)
    return student, student.init_params(jax.random.PRNGKey(59))


def test_jsonl_writer_reader_roundtrip_and_schema_validation(tmp_path: Path) -> None:
    record = make_provenance_record(
        case="tiny",
        side="qrwkv",
        comparison="initial_state",
        state_name="wkv_matrix_state",
        left_label="left",
        right_label="right",
        left=np.zeros((1, 2), dtype=np.float32),
        right=np.zeros((1, 2), dtype=np.float32),
    )
    assert record["schema"] == WKV_STATE_PROVENANCE_SCHEMA
    validate_provenance_record(record)

    path = tmp_path / "trace.jsonl"
    write_provenance_jsonl([record], path)
    assert load_provenance_jsonl(path) == [record]

    bad = dict(record)
    bad.pop("comparison")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_provenance_record(bad)

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps({**record, "schema": "wrong"}) + "\n")
    with pytest.raises(ValueError, match="unsupported schema"):
        load_provenance_jsonl(bad_path)


def test_compare_state_arrays_detects_mismatch() -> None:
    stats = compare_state_arrays(
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 3.0]], dtype=np.float32),
    )
    assert stats["status"] == "fail"
    assert stats["max_abs_error"] == 1.0

    shape = compare_state_arrays(np.zeros((1, 2)), np.zeros((2, 1)))
    assert shape["status"] == "shape_mismatch"
    assert shape["shape_match"] is False


def test_trace_provenance_covers_required_comparisons(tmp_path: Path) -> None:
    student, params = _synthetic_student()
    input_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    attention_mask = np.array([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=np.int32)
    records = trace_qrwkv_state_provenance(
        student,
        params,
        input_ids,
        attention_mask=attention_mask,
        case="synthetic",
        max_inline_values=4096,
    )

    comparisons = {record["comparison"] for record in records}
    assert {
        "initial_state",
        "initial_state_handoff",
        "token_carry",
        "full_vs_stepwise",
        "mask_behavior",
    } <= comparisons
    assert any(
        record["comparison"] == "initial_state" and record["status"] == "pass"
        for record in records
    )
    assert all(
        record["status"] == "pass"
        for record in records
        if record["comparison"]
        in {
            "initial_state",
            "initial_state_handoff",
            "token_carry",
            "full_vs_stepwise",
        }
    )
    assert any(record["comparison"] == "mask_behavior" for record in records)

    out = tmp_path / "reports"
    report = write_provenance_reports(records, out)
    assert report["schema"].endswith("provenance_report.v1")
    assert (out / "wkv_state_provenance_report.json").is_file()
    assert (out / "P59_WKV_STATE_PROVENANCE.md").is_file()


def test_compare_provenance_records_uses_inline_arrays() -> None:
    base = make_provenance_record(
        case="tiny",
        side="qrwkv",
        comparison="full_vs_stepwise",
        state_name="wkv_matrix_state",
        left_label="full",
        right_label="step",
        left=np.array([1.0], dtype=np.float32),
        right=np.array([1.0], dtype=np.float32),
    )
    changed = {
        **base,
        "right_array": [2.0],
    }
    report = compare_provenance_records([base], [changed])
    assert report["status"] == "fail"
    assert report["first_mismatch"]["max_abs_error"] == 1.0


def test_trace_script_generates_tmp_path_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    out = tmp_path / "p59"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "trace_radlads_qrwkv_wkv_state_provenance.py"),
            "--out",
            str(out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P59 WKV state provenance" in result.stdout
    assert (out / "wkv_state_provenance.jsonl").is_file()
    assert (out / "wkv_state_provenance_report.json").is_file()
    assert (out / "P59_WKV_STATE_PROVENANCE.md").is_file()


def test_script_help() -> None:
    for script in [
        "trace_radlads_qrwkv_wkv_state_provenance.py",
        "compare_radlads_qrwkv_wkv_state_provenance.py",
    ]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P59" in result.stdout
        assert "provenance" in result.stdout
