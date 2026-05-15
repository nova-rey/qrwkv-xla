from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_same_run_update_ingredients import (
    DEPENDENCY_ORDER,
    SAME_RUN_UPDATE_INGREDIENT_COMPARISON_SCHEMA,
    SAME_RUN_UPDATE_INGREDIENT_SCHEMA,
    build_same_run_update_ingredient_trace,
    compare_same_run_update_ingredients,
    load_same_run_update_ingredient_jsonl,
    run_same_run_update_ingredient_trace,
    write_same_run_update_ingredient_trace,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_same_run_update_ingredient_trace.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_same_run_update_ingredients.py"


def _source_row(
    side: str,
    stage: str,
    value: object,
    *,
    token_index: int = 0,
    run_id: str = "run-a",
    capture_kind: str = "live_captured",
) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "case": "tiny_no_mask",
        "side": side,
        "run_id": run_id,
        "lineage_key": run_id,
        "layer": 0,
        "head": 0,
        "token_index": token_index,
        "stage": stage,
        "comparison_label": stage,
        "capture_kind": capture_kind,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": True,
        "array": array.tolist(),
    }


def _minimal_source(
    side: str,
    *,
    run_id: str = "run-a",
    k_delta: float = 0.0,
) -> list[dict[str, object]]:
    matrix = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    decay = np.array([[0.5, 0.25]], dtype=np.float32)
    return [
        _source_row(side, "pre_attention_norm", [[1.0, 2.0]], run_id=run_id),
        _source_row(side, "k_head_split", [[1.0 + k_delta, 2.0]], run_id=run_id),
        _source_row(side, "v_head_split", [[3.0, 4.0]], run_id=run_id),
        _source_row(side, "v_first", [[2.5, 3.5]], run_id=run_id),
        _source_row(side, "mixed_value", [[3.5, 4.5]], run_id=run_id),
        _source_row(side, "iclr_update_rate", [[0.1, 0.2]], run_id=run_id),
        _source_row(side, "k_k", [[0.01, 0.02]], run_id=run_id),
        _source_row(side, "k_a", [[0.03, 0.04]], run_id=run_id),
        _source_row(side, "low_rank_decay", [[-1.0, -2.0]], run_id=run_id),
        _source_row(side, "decay_applied_weights", decay, run_id=run_id),
        _source_row(side, "wkv_state_before", matrix, run_id=run_id),
        _source_row(
            side,
            "wkv_decay_applied",
            matrix * decay[:, None, :],
            run_id=run_id,
        ),
        _source_row(side, "wkv_update_outer_or_term", matrix + 0.01, run_id=run_id),
        _source_row(
            side,
            "composite_balance_update_term",
            matrix + 0.02,
            run_id=run_id,
            capture_kind="exact_reconstruction",
        ),
        _source_row(
            side,
            "state_after_from_full_source_formula",
            matrix + 0.03,
            run_id=run_id,
            capture_kind="exact_reconstruction",
        ),
        _source_row(side, "wkv_state_after", matrix + 0.04, run_id=run_id),
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_trace_reports_unavailable_without_omitting() -> None:
    rows = _minimal_source("radlads")
    rows = [row for row in rows if row["stage"] != "v_first"]

    trace = build_same_run_update_ingredient_trace(rows, side="radlads")

    assert [row["stage"] for row in trace] == list(DEPENDENCY_ORDER)
    missing = next(row for row in trace if row["stage"] == "v_first")
    assert missing["status"] == "unavailable"
    assert missing["capture_kind"] == "unavailable"


def test_reconstructed_values_are_not_labeled_live() -> None:
    trace = build_same_run_update_ingredient_trace(
        _minimal_source("radlads"), side="radlads"
    )

    balance = next(row for row in trace if row["stage"] == "balance_state_term")
    composite = next(row for row in trace if row["stage"] == "composite_update_term")
    assert balance["capture_kind"] == "exact_reconstruction"
    assert composite["capture_kind"] == "exact_reconstruction"


def test_first_differing_ingredient_follows_dependency_order() -> None:
    rad = build_same_run_update_ingredient_trace(
        _minimal_source("radlads"), side="radlads"
    )
    off = build_same_run_update_ingredient_trace(
        _minimal_source("qrwkv_off", k_delta=0.5), side="qrwkv_off"
    )
    exp = build_same_run_update_ingredient_trace(
        _minimal_source("qrwkv_experimental", k_delta=0.5),
        side="qrwkv_experimental",
    )

    report = compare_same_run_update_ingredients(
        radlads_entries=rad,
        qrwkv_off_entries=off,
        qrwkv_experimental_entries=exp,
    )

    assert report["schema"] == SAME_RUN_UPDATE_INGREDIENT_COMPARISON_SCHEMA
    assert report["first_divergent_stage"] == "k_head_split"
    assert report["first_divergent_dependency_index"] == DEPENDENCY_ORDER.index(
        "k_head_split"
    )


def test_strict_same_run_rejects_mixed_lineage() -> None:
    rad = build_same_run_update_ingredient_trace(
        _minimal_source("radlads", run_id="run-a"), side="radlads"
    )
    off = build_same_run_update_ingredient_trace(
        _minimal_source("qrwkv_off", run_id="run-b"), side="qrwkv_off"
    )
    exp = build_same_run_update_ingredient_trace(
        _minimal_source("qrwkv_experimental", run_id="run-a"),
        side="qrwkv_experimental",
    )

    report = compare_same_run_update_ingredients(
        radlads_entries=rad,
        qrwkv_off_entries=off,
        qrwkv_experimental_entries=exp,
        strict_same_run=True,
    )

    assert report["same_run_valid"] is False
    assert report["mixed_lineage_rejected"] is True
    assert report["same_run_validity"]["reason"] == "mixed lineage"


def test_jsonl_roundtrip_and_runner_reports(tmp_path: Path) -> None:
    rad_source = _minimal_source("radlads")
    off_source = _minimal_source("qrwkv_off")
    exp_source = _minimal_source("qrwkv_experimental")
    rad_path = tmp_path / "rad.jsonl"
    off_path = tmp_path / "off.jsonl"
    exp_path = tmp_path / "exp.jsonl"
    metadata_path = tmp_path / "metadata.json"
    _write_jsonl(rad_path, rad_source)
    _write_jsonl(off_path, off_source)
    _write_jsonl(exp_path, exp_source)
    metadata_path.write_text(
        json.dumps({"run_id": "run-a", "lineage_key": "run-a"}) + "\n",
        encoding="utf-8",
    )

    report = run_same_run_update_ingredient_trace(
        out_dir=tmp_path / "out",
        radlads_trace=rad_path,
        qrwkv_off_trace=off_path,
        qrwkv_experimental_trace=exp_path,
        metadata_path=metadata_path,
        overwrite=True,
    )

    assert report["same_run_valid"] is True
    out = tmp_path / "out"
    assert (out / "same_run_update_ingredients_report.json").is_file()
    assert (out / "P67_SAME_RUN_UPDATE_INGREDIENTS.md").is_file()
    assert (out / "FIRST_DIFFERING_INGREDIENT.md").is_file()
    payload = json.loads((out / "same_run_update_ingredients_report.json").read_text())
    assert payload["schema"] == SAME_RUN_UPDATE_INGREDIENT_COMPARISON_SCHEMA
    trace = load_same_run_update_ingredient_jsonl(
        out / "same_run_update_ingredients_radlads.jsonl"
    )
    assert trace[0]["schema"] == SAME_RUN_UPDATE_INGREDIENT_SCHEMA


def test_compare_script_reads_three_jsonl_traces(tmp_path: Path) -> None:
    rad = build_same_run_update_ingredient_trace(
        _minimal_source("radlads"), side="radlads"
    )
    off = build_same_run_update_ingredient_trace(
        _minimal_source("qrwkv_off"), side="qrwkv_off"
    )
    exp = build_same_run_update_ingredient_trace(
        _minimal_source("qrwkv_experimental"), side="qrwkv_experimental"
    )
    rad_path = tmp_path / "rad_ingredients.jsonl"
    off_path = tmp_path / "off_ingredients.jsonl"
    exp_path = tmp_path / "exp_ingredients.jsonl"
    write_same_run_update_ingredient_trace(rad, rad_path)
    write_same_run_update_ingredient_trace(off, off_path)
    write_same_run_update_ingredient_trace(exp, exp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--radlads-trace",
            str(rad_path),
            "--qrwkv-off-trace",
            str(off_path),
            "--qrwkv-experimental-trace",
            str(exp_path),
            "--out-dir",
            str(tmp_path / "compare"),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "same_run_valid=True" in result.stdout
    assert (tmp_path / "compare" / "P67_RESULTS.md").is_file()


def test_script_help_works() -> None:
    for script in (RUN_SCRIPT, COMPARE_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P67" in result.stdout
        assert "same-run" in result.stdout
