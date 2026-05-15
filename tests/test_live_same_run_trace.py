from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_live_same_run_trace import (
    DEPENDENCY_ORDER,
    LIVE_SAME_RUN_TRACE_SCHEMA,
    build_live_same_run_trace,
    compare_live_same_run_traces,
    deterministic_fixture_id,
    deterministic_parameter_id,
    load_live_same_run_trace_jsonl,
    new_same_run_group_id,
    run_live_same_run_trace,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_live_same_run_update_trace.py"


def _source_row(
    side: str,
    stage: str,
    value: object,
    *,
    group: str = "group-a",
    fixture: str = "fixture-a",
    parameter: str = "parameter-a",
    token: int = 0,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "case": "tiny_no_mask",
        "side": side,
        "same_run_group_id": group,
        "fixture_id": fixture,
        "parameter_id": parameter,
        "mode": None,
        "layer": 0,
        "head": 0,
        "token": token,
        "stage": stage,
        "source_stage_name": stage,
        "capture_kind": "live_captured",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "array": array.tolist(),
        "live_config": config or {"seed": 7, "dtype": "float32"},
    }


def _minimal_source(
    side: str,
    *,
    group: str = "group-a",
    fixture: str = "fixture-a",
    parameter: str = "parameter-a",
    k_delta: float = 0.0,
    decay_delta: float = 0.0,
    config: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    matrix = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    decay = np.array([[0.5 + decay_delta, 0.25]], dtype=np.float32)
    values = {
        "pre_attention_norm": [[1.0, 2.0]],
        "k_head_split": [[1.0 + k_delta, 2.0]],
        "v_head_split": [[3.0, 4.0]],
        "v_first": [[2.5, 3.5]],
        "mixed_value": [[3.5, 4.5]],
        "iclr_update_rate": [[0.1, 0.2]],
        "k_k": [[0.01, 0.02]],
        "k_a": [[0.03, 0.04]],
        "low_rank_decay": [[-1.0 + decay_delta, -2.0]],
        "decay_applied_weights": decay,
        "wkv_state_before": matrix,
        "wkv_decay_applied": matrix * decay[:, None, :],
        "wkv_update_outer_or_term": matrix + 0.01,
        "balance_state_term": matrix + 0.02,
        "composite_update_term": matrix + 0.03,
        "wkv_state_after": matrix + 0.04,
        "state_after_exported": matrix + 0.04,
    }
    return [
        _source_row(
            side,
            stage,
            value,
            group=group,
            fixture=fixture,
            parameter=parameter,
            config=config,
        )
        for stage, value in values.items()
    ]


def _traces(
    *,
    group: str = "group-a",
    fixture: str = "fixture-a",
    parameter: str = "parameter-a",
    off_group: str | None = None,
    off_fixture: str | None = None,
    off_parameter: str | None = None,
    k_delta: float = 0.0,
    decay_delta: float = 0.0,
    off_config: dict[str, object] | None = None,
) -> dict[str, list[dict[str, object]]]:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    return {
        "radlads": build_live_same_run_trace(
            _minimal_source(
                "radlads",
                group=group,
                fixture=fixture,
                parameter=parameter,
            ),
            side="radlads",
            same_run_group_id=group,
            fixture_id=fixture,
            parameter_id=parameter,
            contexts=contexts,
        ),
        "qrwkv_off": build_live_same_run_trace(
            _minimal_source(
                "qrwkv_off",
                group=off_group or group,
                fixture=off_fixture or fixture,
                parameter=off_parameter or parameter,
                k_delta=k_delta,
                decay_delta=decay_delta,
                config=off_config,
            ),
            side="qrwkv_off",
            same_run_group_id=group,
            fixture_id=fixture,
            parameter_id=parameter,
            contexts=contexts,
        ),
        "qrwkv_experimental": build_live_same_run_trace(
            _minimal_source(
                "qrwkv_experimental",
                group=group,
                fixture=fixture,
                parameter=parameter,
                k_delta=k_delta,
                decay_delta=decay_delta,
            ),
            side="qrwkv_experimental",
            same_run_group_id=group,
            fixture_id=fixture,
            parameter_id=parameter,
            contexts=contexts,
        ),
    }


def _metadata(
    *,
    group: str = "group-a",
    fixture: str = "fixture-a",
    parameter: str = "parameter-a",
) -> dict[str, str]:
    return {
        "same_run_group_id": group,
        "fixture_id": fixture,
        "parameter_id": parameter,
    }


def test_same_run_group_id_reused_and_schema_valid() -> None:
    traces = _traces()
    ids = {
        row["same_run_group_id"]
        for rows in traces.values()
        for row in rows
    }
    assert ids == {"group-a"}
    first = traces["radlads"][0]
    assert first["schema"] == LIVE_SAME_RUN_TRACE_SCHEMA
    assert first["phase"] == "P68"
    for field in (
        "same_run_group_id",
        "fixture_id",
        "parameter_id",
        "side",
        "case",
        "mode",
        "layer",
        "token",
        "head",
        "stage",
        "source_stage_name",
        "capture_kind",
        "shape",
        "dtype",
        "array",
        "summary",
    ):
        assert field in first


def test_deterministic_ids_are_content_stable(tmp_path: Path) -> None:
    fixture = tmp_path / "manifest.json"
    params = tmp_path / "params.json"
    fixture.write_text(json.dumps({"cases": ["tiny_no_mask"]}), encoding="utf-8")
    params.write_text(json.dumps({"a": 1}), encoding="utf-8")

    assert deterministic_fixture_id(fixture) == deterministic_fixture_id(fixture)
    assert deterministic_parameter_id(parameters=params) == deterministic_parameter_id(
        parameters=params
    )
    assert deterministic_parameter_id(
        fixture_parameter_key="tiny:key"
    ) == deterministic_parameter_id(fixture_parameter_key="tiny:key")


def test_same_run_group_id_is_deterministic_for_inputs() -> None:
    first = new_same_run_group_id(
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        cases=["tiny_no_mask"],
        mode="both",
        layer=0,
        head=0,
        max_tokens=1,
        strict_live=True,
    )
    second = new_same_run_group_id(
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        cases=["tiny_no_mask"],
        mode="both",
        layer=0,
        head=0,
        max_tokens=1,
        strict_live=True,
    )
    different = new_same_run_group_id(
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        cases=["other_case"],
        mode="both",
        layer=0,
        head=0,
        max_tokens=1,
        strict_live=True,
    )

    assert first == second
    assert first != different


def test_strict_live_invalidates_mixed_ids_and_config_deltas() -> None:
    mixed = compare_live_same_run_traces(
        traces=_traces(off_group="group-b"),
        metadata=_metadata(),
        strict_live=True,
    )
    assert mixed["same_run_valid"] is False
    assert mixed["same_run_validity"]["identity"]["status"] == "fail"

    config_delta = compare_live_same_run_traces(
        traces=_traces(off_config={"seed": 8, "dtype": "float32"}),
        metadata=_metadata(),
        strict_live=True,
    )
    assert config_delta["same_run_valid"] is False
    assert config_delta["same_run_validity"]["live_config"]["status"] == "fail"


def test_unavailable_critical_stage_invalidates() -> None:
    traces = _traces()
    traces["radlads"] = [
        row for row in traces["radlads"] if row["stage"] != "wkv_state_after"
    ]
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    traces["radlads"] = build_live_same_run_trace(
        [
            row
            for row in _minimal_source("radlads")
            if row["stage"] != "wkv_state_after"
        ],
        side="radlads",
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        contexts=contexts,
    )

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )
    assert report["same_run_valid"] is False
    assert report["same_run_validity"]["critical_availability"]["status"] == "fail"
    assert report["recommended_next_phase"].startswith("P69 targeted live RADLADS")


def test_decay_log_w_precondition_invalidates() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(decay_delta=0.5),
        metadata=_metadata(),
        strict_live=True,
    )
    assert report["same_run_valid"] is False
    assert report["decay_precondition_pass"] is False
    assert report["same_run_validity"]["decay_log_w_precondition"]["status"] == "fail"


def test_first_difference_uses_dependency_order() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(k_delta=0.5),
        metadata=_metadata(),
        strict_live=True,
    )
    assert report["first_divergent_stage"] == "k_head_split"
    assert report["first_divergent_dependency_index"] == DEPENDENCY_ORDER.index(
        "k_head_split"
    )


def test_runner_writes_invalid_unavailable_live_report(tmp_path: Path) -> None:
    fixture = tmp_path / "manifest.json"
    params = tmp_path / "params.json"
    fixture.write_text(json.dumps({"cases": ["tiny_no_mask"]}), encoding="utf-8")
    params.write_text(json.dumps({"a": 1}), encoding="utf-8")

    report = run_live_same_run_trace(
        fixture_manifest=fixture,
        parameters=params,
        out_dir=tmp_path / "out",
        strict_live=True,
        overwrite=True,
    )

    assert report["same_run_valid"] is False
    out = tmp_path / "out"
    assert (out / "P68_RESULTS.md").is_file()
    assert (out / "live_same_run_trace_metadata.json").is_file()
    assert (out / "live_trace_radlads.jsonl").is_file()
    assert (out / "live_trace_qrwkv_off.jsonl").is_file()
    assert (out / "live_trace_qrwkv_experimental.jsonl").is_file()
    assert (out / "live_trace_combined.jsonl").is_file()
    assert (out / "live_same_run_update_ingredients_report.json").is_file()
    assert (out / "LIVE_SAME_RUN_VALIDITY.md").is_file()
    assert (out / "STAGE_AVAILABILITY_MATRIX.md").is_file()
    assert (out / "FIRST_DIFFERING_INGREDIENT.md").is_file()
    assert (out / "P68_DECISION.md").is_file()
    rows = load_live_same_run_trace_jsonl(out / "live_trace_radlads.jsonl")
    assert rows
    assert all(row["capture_kind"] == "unavailable" for row in rows)
    assert report["same_run_group_id"] == new_same_run_group_id(
        fixture_id=report["fixture_id"],
        parameter_id=report["parameter_id"],
        cases=None,
        mode="both",
        layer=None,
        head=None,
        max_tokens=None,
        strict_live=True,
    )


def test_runner_reuses_same_run_group_id_for_same_inputs(tmp_path: Path) -> None:
    fixture = tmp_path / "manifest.json"
    params = tmp_path / "params.json"
    fixture.write_text(json.dumps({"cases": ["tiny_no_mask"]}), encoding="utf-8")
    params.write_text(json.dumps({"a": 1}), encoding="utf-8")

    first = run_live_same_run_trace(
        fixture_manifest=fixture,
        parameters=params,
        out_dir=tmp_path / "out-a",
        strict_live=True,
        overwrite=True,
    )
    second = run_live_same_run_trace(
        fixture_manifest=fixture,
        parameters=params,
        out_dir=tmp_path / "out-b",
        strict_live=True,
        overwrite=True,
    )

    assert first["same_run_group_id"] == second["same_run_group_id"]


def test_script_help_works() -> None:
    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P68" in result.stdout
    assert "--strict-live" in result.stdout
