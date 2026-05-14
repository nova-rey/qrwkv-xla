from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.parity.radlads_log_w_parity import (
    LOG_W_CANDIDATE_SCHEMA,
    LOG_W_PARITY_SCHEMA,
    LogWRecord,
    capture_qrwkv_log_w_from_current_run,
    compare_log_w_records,
    evaluate_log_w_candidate_variants,
    load_radlads_log_w_from_jsonl,
    log_w_replay_profile_for_case,
    write_log_w_reports,
)
from qrwkv_xla.parity.radlads_numerical_fixtures import (
    generate_radlads_tiny_numerical_fixtures,
    load_numerical_manifest,
)
from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
)
from qrwkv_xla.parity.radlads_replay import student_for_replay_profile

ROOT = Path(__file__).resolve().parents[1]


def _row(
    value: object,
    *,
    side: str = "radlads",
    stage: str = "log_w",
    head: int | None = 0,
) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "case": "tiny",
        "side": side,
        "layer": 0,
        "head": head,
        "token_index": 0,
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


def _record(
    value: object,
    *,
    side: str = "radlads",
    head: int | None = 0,
) -> LogWRecord:
    array = np.asarray(value, dtype=np.float32)
    return LogWRecord(
        case="tiny",
        side=side,
        layer=0,
        head=head,
        token_index=0,
        name=f"{side}.log_w",
        shape=list(array.shape),
        dtype=str(array.dtype),
        finite=True,
        array=array.tolist(),
    )


def test_load_radlads_log_w_schema_validation(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(_row([[-0.25, -0.5]])) + "\n", encoding="utf-8")
    rows = load_radlads_log_w_from_jsonl(trace)
    assert len(rows) == 1
    assert rows[0].shape == [1, 2]

    bad = tmp_path / "bad.jsonl"
    payload = _row([[-0.25]])
    payload.pop("array")
    bad.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        load_radlads_log_w_from_jsonl(bad)


def test_compare_log_w_exact_match_and_mismatch_candidates() -> None:
    w = np.array([[0.0, 1.0]], dtype=np.float32)
    target = -np.exp(np.float32(-0.5)) * (1.0 / (1.0 + np.exp(-w)))
    rad = _record(target)
    qrw = _record(target, side="qrwkv")
    report = compare_log_w_records([rad], [qrw])
    assert report["schema"] == LOG_W_PARITY_SCHEMA
    assert report["status"] == "pass"

    mismatch = _record(target + 0.01, side="qrwkv")
    mismatch_report = compare_log_w_records([rad], [mismatch])
    assert mismatch_report["status"] == "fail"
    assert mismatch_report["first_mismatch"]["max_abs_error"] > 0

    source = _record(w, side="qrwkv")
    candidates = evaluate_log_w_candidate_variants(
        radlads_rows=[rad],
        qrwkv_w_rows=[source],
    )
    assert candidates["schema"] == LOG_W_CANDIDATE_SCHEMA
    assert candidates["best_candidate_status"] == "pass"
    assert candidates["best_candidate"].startswith("as_is__negative__")

    wrong_target = _record(target + 0.25)
    wrong_candidates = evaluate_log_w_candidate_variants(
        radlads_rows=[wrong_target],
        qrwkv_w_rows=[source],
    )
    assert wrong_candidates["best_candidate_max_abs_error"] > 0
    assert any(row["status"] == "fail" for row in wrong_candidates["rows"])


def test_capture_qrwkv_log_w_finite_path() -> None:
    jax = pytest.importorskip("jax")
    from qrwkv_xla.students.rwkv7_qwen_reference import (
        RWKV7QwenReferenceConfig,
        RWKV7QwenReferenceStudent,
    )

    config = RWKV7QwenReferenceConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        use_rope=False,
        emit_logits=True,
        attention_qkv_bias=True,
    )
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(7))
    capture = capture_qrwkv_log_w_from_current_run(
        student,
        params,
        np.array([[1, 2, 3]], dtype=np.int32),
        case="finite",
    )
    assert capture["log_w"]
    assert all(row["finite"] for row in capture["log_w"])
    assert capture["w_source"]


def test_tiny_no_mask_profile_keeps_low_rank_decay_and_matches_source(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jax")

    fixture_root = tmp_path / "p54_confirmation" / "fixtures"
    generate_radlads_tiny_numerical_fixtures(
        fixture_root,
        overwrite=True,
        init_policy="deterministic_finite",
    )
    manifest = load_numerical_manifest(fixture_root / "manifest.json")
    case = next(case for case in manifest["cases"] if case["name"] == "tiny_no_mask")
    profile = log_w_replay_profile_for_case(case)
    assert profile.low_rank_decay is True

    import_result = import_radlads_parameters_for_replay(
        fixture_root / "radlads_parameters.npz"
    )
    student = student_for_replay_profile(import_result.qrwkv_config, profile)
    fixture = np.load(fixture_root / "tiny_no_mask.npz")
    capture = capture_qrwkv_log_w_from_current_run(
        student,
        import_result.params,
        fixture["input_ids"],
        attention_mask=None,
        case="tiny_no_mask",
    )
    assert any(
        row["name"] == "layers.0.self_attn.w2_projection" for row in capture["w_source"]
    )
    assert any(
        row["name"] == "layers.0.self_attn.w_head_split" for row in capture["w_source"]
    )
    assert all(
        row["name"] != "layers.0.self_attn.w_projection" for row in capture["w_source"]
    )

    radlads_trace = tmp_path / "p54_confirmation" / "radlads_trace.jsonl"
    radlads_trace.write_text(
        "\n".join(
            json.dumps({**row, "side": "radlads", "stage": "log_w"})
            for row in capture["log_w"]
        )
        + "\n",
        encoding="utf-8",
    )
    radlads_rows = load_radlads_log_w_from_jsonl(radlads_trace)
    qrwkv_rows = [LogWRecord(**row) for row in capture["log_w"]]
    report = compare_log_w_records(radlads_rows, qrwkv_rows)
    assert report["status"] == "pass"
    assert all(row["status"] == "pass" for row in report["rows"])


def test_script_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_radlads_qrwkv_log_w.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P57" in result.stdout
    assert "--radlads-trace" in result.stdout


def test_report_writers(tmp_path: Path) -> None:
    rad = _record([[-0.25]])
    qrw = _record([[-0.25]], side="qrwkv")
    source = _record([[0.0]], side="qrwkv")
    parity = compare_log_w_records([rad], [qrw])
    candidates = evaluate_log_w_candidate_variants(
        radlads_rows=[rad],
        qrwkv_w_rows=[source],
    )
    write_log_w_reports(
        parity_report=parity,
        candidate_report=candidates,
        radlads_rows=[rad],
        qrwkv_rows=[qrw],
        qrwkv_w_rows=[source],
        out_dir=tmp_path,
    )
    assert (tmp_path / "log_w_parity_report.json").is_file()
    assert (tmp_path / "P57_LOG_W_PARITY.md").is_file()
    assert (tmp_path / "log_w_values.npz").is_file()
    assert (tmp_path / "P57_LOG_W_CANDIDATES.md").is_file()
    assert (tmp_path / "log_w_candidate_report.json").is_file()
