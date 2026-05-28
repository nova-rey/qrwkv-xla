from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

from qrwkv_xla.parity.radlads_live_same_run_trace import (
    DEPENDENCY_ORDER,
    LIVE_SAME_RUN_TRACE_SCHEMA,
    MINIMUM_STAGES,
    P71_STRETCH_STAGES,
    FixtureExpectationMetadata,
    LiveTraceCollector,
    _capture_radlads_case,
    _p79_matrix_markdown,
    _trace_key,
    build_live_same_run_trace,
    build_p75_residual_impact_gate,
    build_p79_broader_fixture_residual_matrix,
    classify_balance_state_lane,
    compare_live_same_run_traces,
    deterministic_fixture_id,
    deterministic_parameter_id,
    load_live_same_run_trace_jsonl,
    new_same_run_group_id,
    run_live_same_run_trace,
)
from qrwkv_xla.students import (
    PallasRuntimeUnavailableError,
    RWKV7QwenReferenceConfig,
    RWKV7QwenReferenceStudent,
    WKVRuntime,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_live_same_run_update_trace.py"
BALANCE_TERMS_CONFIG = {
    "seed": 7,
    "dtype": "float32",
    "radlads_balance_state_terms": True,
    "radlads_balance_state": False,
}
DIRECT_BALANCE_CONFIG = {
    "seed": 7,
    "dtype": "float32",
    "radlads_balance_state_terms": True,
    "radlads_balance_state": True,
}


def _write_live_case_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    payload = tmp_path / "tiny_no_mask.npz"
    np.savez(payload, input_ids=np.array([[1, 2]], dtype=np.int32))
    case = {
        "name": "tiny_no_mask",
        "payload": payload.name,
        "all_radlads_math": True,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    return manifest, case


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
        "live_config": config or BALANCE_TERMS_CONFIG,
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
        "kk": [[0.05, 0.06]],
        "k_for_update": [[1.0 + k_delta, 2.0]],
        "v_for_update": [[3.0, 4.0]],
        "low_rank_decay": [[-1.0 + decay_delta, -2.0]],
        "decay_applied_weights": decay,
        "wkv_state_before": matrix,
        "wkv_decay_applied": matrix * decay[:, None, :],
        "wkv_update_outer_or_term": matrix + 0.01,
        "ab": matrix + 0.015,
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
            [
                *_minimal_source(
                    "radlads",
                    group=group,
                    fixture=fixture,
                    parameter=parameter,
                ),
                *[
                    row
                    for row in _minimal_source(
                        "radlads",
                        group=group,
                        fixture=fixture,
                        parameter=parameter,
                        config=DIRECT_BALANCE_CONFIG,
                    )
                    if row["stage"] not in {"k_k", "k_a"}
                ],
            ],
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
                config={**BALANCE_TERMS_CONFIG, **(off_config or {})},
            ),
            side="qrwkv_off",
            same_run_group_id=group,
            fixture_id=fixture,
            parameter_id=parameter,
            contexts=contexts,
        ),
        "qrwkv_experimental": build_live_same_run_trace(
            [
                row
                for row in _minimal_source(
                    "qrwkv_experimental",
                    group=group,
                    fixture=fixture,
                    parameter=parameter,
                    k_delta=k_delta,
                    decay_delta=decay_delta,
                    config=DIRECT_BALANCE_CONFIG,
                )
                if row["stage"] not in {"k_k", "k_a"}
            ],
            side="qrwkv_experimental",
            same_run_group_id=group,
            fixture_id=fixture,
            parameter_id=parameter,
            contexts=contexts,
        ),
    }


def _traces_for_case(case: str) -> dict[str, list[dict[str, object]]]:
    traces = _traces()
    for rows in traces.values():
        for row in rows:
            row["case"] = case
    return traces


def _evidence_rows_for_case(
    rows: list[dict[str, object]], case: str
) -> list[dict[str, object]]:
    copied = [dict(row) for row in rows]
    for row in copied:
        row["case"] = case
    return copied


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


def _p77_evidence_rows(
    *,
    group: str = "group-a",
    fixture: str = "fixture-a",
    parameter: str = "parameter-a",
) -> list[dict[str, object]]:
    rows = []
    surfaces = (
        ("radlads", "balance_state_terms"),
        ("qrwkv_off", "balance_state_terms"),
        ("radlads", "direct_balance_state"),
        ("qrwkv_experimental", "direct_balance_state"),
    )
    for side, lane in surfaces:
        for stage in ("state_after_live", "state_after_exported"):
            rows.append(
                {
                    "same_run_group_id": group,
                    "fixture_id": fixture,
                    "parameter_id": parameter,
                    "case": "tiny_no_mask",
                    "side": side,
                    "balance_state_lane": lane,
                    "comparison": "full_vs_stepwise",
                    "mode": "full_vs_stepwise",
                    "full_mode": "full",
                    "stepwise_mode": "stepwise",
                    "layer": 0,
                    "head": 0,
                    "token": 1,
                    "final_token": 1,
                    "stage": stage,
                    "state_slot": "wkv_matrix_state",
                    "status": "pass",
                    "reason": "allclose",
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "max_relative_error": 0.0,
                    "shape_match": True,
                    "dtype_match": True,
                    "finite_both": True,
                    "allclose": True,
                }
            )
    return rows


def _p78_output_evidence_rows(
    *,
    group: str = "group-a",
    fixture: str = "fixture-a",
    parameter: str = "parameter-a",
    hidden_delta: float = 0.0,
    include_logits: bool = False,
) -> list[dict[str, object]]:
    rows = _p77_evidence_rows(group=group, fixture=fixture, parameter=parameter)
    surfaces = (
        ("radlads", "balance_state_terms"),
        ("qrwkv_off", "balance_state_terms"),
        ("radlads", "direct_balance_state"),
        ("qrwkv_experimental", "direct_balance_state"),
    )
    for side, lane in surfaces:
        delta = hidden_delta if side in {"qrwkv_off", "qrwkv_experimental"} else 0.0
        full_hidden = np.array([[[1.0 + delta, 2.0]]], dtype=np.float32)
        rows.append(
            {
                "same_run_group_id": group,
                "fixture_id": fixture,
                "parameter_id": parameter,
                "case": "tiny_no_mask",
                "side": side,
                "balance_state_lane": lane,
                "comparison": "full_vs_stepwise_output",
                "mode": "full_vs_stepwise",
                "full_mode": "full",
                "stepwise_mode": "stepwise",
                "layer": None,
                "head": None,
                "token": 1,
                "final_token": 1,
                "stage": "post_block_hidden_output",
                "status": "pass" if delta == 0.0 else "fail",
                "reason": "allclose" if delta == 0.0 else "fail",
                "max_abs_error": abs(delta),
                "mean_abs_error": abs(delta) / 2.0,
                "max_relative_error": abs(delta),
                "shape_match": True,
                "dtype_match": True,
                "finite_both": True,
                "allclose": delta == 0.0,
                "full_array": full_hidden.tolist(),
                "stepwise_array": full_hidden.tolist(),
            }
        )
        rows.append(
            {
                "same_run_group_id": group,
                "fixture_id": fixture,
                "parameter_id": parameter,
                "case": "tiny_no_mask",
                "side": side,
                "balance_state_lane": lane,
                "comparison": "full_vs_stepwise_output",
                "mode": "full_vs_stepwise",
                "layer": None,
                "head": None,
                "token": 1,
                "final_token": 1,
                "stage": "final_normalized_hidden",
                "status": "unavailable",
                "reason": "missing_stepwise_final_norm_capture",
                "full_array": None,
                "stepwise_array": None,
            }
        )
        if include_logits:
            logits = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
            selected = logits[:, -1, :]
            for stage, value in (
                ("final_lm_head_logits", logits),
                ("selected_token_logits", selected),
            ):
                rows.append(
                    {
                        "same_run_group_id": group,
                        "fixture_id": fixture,
                        "parameter_id": parameter,
                        "case": "tiny_no_mask",
                        "side": side,
                        "balance_state_lane": lane,
                        "comparison": "full_vs_stepwise_output",
                        "mode": "full_vs_stepwise",
                        "layer": None,
                        "head": None,
                        "token": 1,
                        "final_token": 1,
                        "stage": stage,
                        "status": "pass",
                        "reason": "allclose",
                        "max_abs_error": 0.0,
                        "mean_abs_error": 0.0,
                        "max_relative_error": 0.0,
                        "shape_match": True,
                        "dtype_match": True,
                        "finite_both": True,
                        "allclose": True,
                        "full_array": value.tolist(),
                        "stepwise_array": value.tolist(),
                    }
                )
        else:
            rows.append(
                {
                    "same_run_group_id": group,
                    "fixture_id": fixture,
                    "parameter_id": parameter,
                    "case": "tiny_no_mask",
                    "side": side,
                    "balance_state_lane": lane,
                    "comparison": "full_vs_stepwise_output",
                    "mode": "full_vs_stepwise",
                    "layer": None,
                    "head": None,
                    "token": 1,
                    "final_token": 1,
                    "stage": "final_lm_head_logits",
                    "status": "unavailable",
                    "reason": "missing_lm_head_logits_path",
                    "full_array": None,
                    "stepwise_array": None,
                }
            )
    return rows


def _p79_matrix(
    *,
    traces: dict[str, list[dict[str, object]]] | None = None,
    manifest_cases: list[str] | None = None,
    expected_cases: tuple[str, ...] | None = ("tiny_no_mask",),
    p77_rows: list[dict[str, object]] | None = None,
    p78_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    metadata = _metadata()
    metadata["p77_full_vs_stepwise_evidence"] = (
        p77_rows if p77_rows is not None else _p77_evidence_rows()
    )
    metadata["p78_logits_output_evidence"] = (
        p78_rows
        if p78_rows is not None
        else _p78_output_evidence_rows(include_logits=True)
    )
    return build_p79_broader_fixture_residual_matrix(
        fixture_manifest_data={
            "cases": [{"name": case} for case in (manifest_cases or ["tiny_no_mask"])]
        },
        traces=traces or _traces(),
        metadata=metadata,
        strict_live=True,
        atol=1e-5,
        rtol=1e-5,
        expected_cases=expected_cases,
    )


def test_p73_balance_state_lane_classifier() -> None:
    assert classify_balance_state_lane(BALANCE_TERMS_CONFIG) == "balance_state_terms"
    assert classify_balance_state_lane(DIRECT_BALANCE_CONFIG) == "direct_balance_state"
    assert classify_balance_state_lane({"seed": 7}) == "native_or_unknown"
    assert (
        classify_balance_state_lane({"use_radlads_balance_state_terms": True})
        == "balance_state_terms"
    )


def test_same_run_group_id_reused_and_schema_valid() -> None:
    traces = _traces()
    ids = {row["same_run_group_id"] for rows in traces.values() for row in rows}
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
        "balance_state_lane",
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


def test_live_collector_copies_array_and_normalizes_stage() -> None:
    value = np.array([[[1.0, 2.0]]], dtype=np.float32)
    collector = LiveTraceCollector(
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        case="tiny_no_mask",
        side="qrwkv_off",
    )
    collector.record(
        "k",
        value,
        layer=0,
        token=0,
        stage="k_head_split",
    )
    value[0, 0, 0] = 99.0

    assert collector.entries[0]["stage"] == "raw_k"
    assert collector.entries[0]["source_stage_name"] == "k_head_split"
    assert collector.entries[0]["array"] == [[1.0, 2.0]]


def test_radlads_live_capture_path_appends_minimum_rows(tmp_path: Path) -> None:
    manifest, case = _write_live_case_fixture(tmp_path)
    config = RWKV7QwenReferenceConfig(
        vocab_size=8,
        hidden_size=4,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        intermediate_size=8,
        use_rope=False,
        emit_logits=True,
        emit_mixer_outputs=True,
        radlads_compatible_math=True,
        radlads_replay_mode=True,
        radlads_attention_group_norm=True,
        radlads_balance_state=True,
        attention_qkv_bias=True,
        radlads_low_rank_gate=True,
        lora_rank_decay=2,
        lora_rank_iclr=2,
        lora_rank_value_residual_mix=2,
        lora_rank_gate=2,
    )
    params = RWKV7QwenReferenceStudent(config).init_params(jax.random.PRNGKey(7))
    collector = LiveTraceCollector(
        same_run_group_id="group-rad",
        fixture_id="fixture-rad",
        parameter_id="parameter-rad",
        case="tiny_no_mask",
        side="radlads",
        live_config={"radlads_compatible_math": True},
    )

    _capture_radlads_case(
        fixture_manifest=manifest,
        case=case,
        params=params,
        config=config,
        collector=collector,
        max_tokens=1,
    )

    stages = {row["stage"] for row in collector.entries}
    assert set(MINIMUM_STAGES).issubset(stages)
    assert {row["side"] for row in collector.entries} == {"radlads"}
    assert {row["same_run_group_id"] for row in collector.entries} == {"group-rad"}
    assert {row["fixture_id"] for row in collector.entries} == {"fixture-rad"}
    assert {row["parameter_id"] for row in collector.entries} == {"parameter-rad"}
    assert {
        "pre_attention_norm",
        "k_head_split",
        "v_head_split",
        "low_rank_decay",
        "decay_applied_weights",
        "wkv_state_before",
        "wkv_update_outer_or_term",
        "wkv_state_after",
    }.issubset({row["source_stage_name"] for row in collector.entries})
    assert {"mixed_value", "kk", "k_for_update", "v_for_update", "ab"} & stages
    exported = [
        row
        for row in collector.entries
        if row["stage"] == "state_after_exported"
        and row["capture_kind"] == "exported_state"
    ]
    assert exported
    assert all(
        "export_reference_state_object" in row["export_path"] for row in exported
    )
    assert all(row["import_roundtrip_status"] == "pass" for row in exported)


def test_p76_exported_state_rows_are_lane_aware_and_feed_gate() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    p76 = report["p76_state_export_import_residual"]

    assert p76["schema"] == "qrwkv_xla.p76_state_export_import_residual.v1"
    assert (
        report["p75_residual_impact_gate"]["output_gates"]["exported_state"]["status"]
        == "pass"
    )
    assert {row["lane"] for row in p76["inter_side_exported_state"]} == {
        "balance_state_terms",
        "direct_balance_state",
    }
    assert report["recommended_next_phase"] == (
        "P77 targeted full-vs-stepwise residual fix"
    )
    assert p76["recommended_next_phase"] == (
        "P77 targeted state export/import convention fix"
    )


def test_p76_missing_exported_state_rows_preserves_p75_missing_evidence() -> None:
    traces = {
        side: [row for row in rows if row["stage"] != "state_after_exported"]
        for side, rows in _traces().items()
    }
    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    exported_gate = report["p75_residual_impact_gate"]["output_gates"]["exported_state"]
    p76 = report["p76_state_export_import_residual"]

    assert exported_gate["status"] == "unavailable"
    assert exported_gate["reason"] == "missing_evidence:exported_state"
    assert p76["status"] == "unavailable"
    assert p76["lane_pair_status"]["balance_state_terms"]["reason"] == (
        "missing_exported_state_rows"
    )
    assert p76["recommended_next_phase"] == (
        "P77 targeted state export/import convention fix"
    )


def test_minimum_stages_are_counted_separately_from_stretch() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    assert set(report["minimum_stage_availability"]) == set(MINIMUM_STAGES)
    assert "v_first" not in report["minimum_stage_availability"]
    assert report["live_rows_captured_radlads"] > len(MINIMUM_STAGES)


def test_p73_lane_map_classifies_all_three_sides() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )

    lane_map = report["balance_state_lane_map"]
    assert lane_map["radlads"]["lane"] == "balance_state_terms"
    assert lane_map["qrwkv_off"]["lane"] == "balance_state_terms"
    assert lane_map["qrwkv_experimental"]["lane"] == "direct_balance_state"
    assert report["lane_mixed_comparison_valid"] is False


def test_p73_direct_lane_not_applicable_does_not_block_terms_math() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    traces = _traces()
    traces["qrwkv_experimental"] = build_live_same_run_trace(
        [
            row
            for row in _minimal_source(
                "qrwkv_experimental",
                config=DIRECT_BALANCE_CONFIG,
            )
            if row["stage"] not in {"k_k", "k_a"}
        ],
        side="qrwkv_experimental",
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

    assert report["first_overall_non_applicable_stage"] == "k_k"
    assert report["first_comparable_differing_stage"] is None
    assert report["balance_state_terms_lane_valid"] is True
    assert report["math_conclusion_valid"] is True
    assert report["direct_balance_state_lane_valid"] is True
    assert report["recommended_next_phase"] == (
        "P77 targeted full-vs-stepwise residual fix"
    )


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
    assert (
        report["recommended_next_phase"]
        == "P72 targeted live missing-stage hook completion"
    )
    assert report["unavailable_minimum_stages"]
    assert report["unavailable_minimum_stages"][0]["reason"].startswith(
        "missing_live_hook:radlads:"
    )


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
    assert report["first_divergent_stage"] == "raw_k"
    assert report["first_divergent_dependency_index"] == DEPENDENCY_ORDER.index("raw_k")


def test_p71_stretch_stage_normalization_preserves_source_stage_name() -> None:
    collector = LiveTraceCollector(
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        case="tiny_no_mask",
        side="qrwkv_off",
    )

    collector.record(
        "k",
        np.array([[[1.0, 2.0]]], dtype=np.float32),
        layer=0,
        token=0,
        stage="k_after_balance",
    )

    assert collector.entries[0]["stage"] == "k_for_update"
    assert collector.entries[0]["source_stage_name"] == "k_after_balance"
    assert collector.entries[0]["head"] == 0


def test_p72_k_k_and_k_a_aliases_preserve_allowed_source_names() -> None:
    collector = LiveTraceCollector(
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        case="tiny_no_mask",
        side="qrwkv_off",
    )

    collector.record(
        "k_k",
        np.array([[[1.0, 2.0]]], dtype=np.float32),
        layer=0,
        token=0,
        stage="key_norm_factor",
    )
    collector.record(
        "k_a",
        np.array([[[0.5, 0.75]]], dtype=np.float32),
        layer=0,
        token=0,
        stage="key_balance_adjustment",
    )

    assert collector.entries[0]["stage"] == "k_k"
    assert collector.entries[0]["source_stage_name"] == "key_norm_factor"
    assert collector.entries[1]["stage"] == "k_a"
    assert collector.entries[1]["source_stage_name"] == "key_balance_adjustment"


def test_p71_stretch_rows_are_classified_separately() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )

    assert set(report["stretch_stage_availability"]) == set(P71_STRETCH_STAGES)
    assert "v_first" in report["stretch_stage_availability"]
    assert "mixed_value" in report["stretch_stage_availability"]
    assert "v_first" not in report["minimum_stage_availability"]
    assert report["minimum_stage_valid"] is True


def test_k_and_v_for_update_are_not_reconstructed() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    source = [
        row
        for row in _minimal_source("radlads")
        if row["stage"] not in {"k_for_update", "v_for_update"}
    ]

    rows = build_live_same_run_trace(
        source,
        side="radlads",
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        contexts=contexts,
    )

    assert (
        next(row for row in rows if row["stage"] == "k_for_update")["capture_kind"]
        == "unavailable"
    )
    assert (
        next(row for row in rows if row["stage"] == "v_for_update")["capture_kind"]
        == "unavailable"
    )


def test_balance_terms_use_exact_reconstruction_from_live_inputs() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    source = [
        row
        for row in _minimal_source("radlads")
        if row["stage"] not in {"balance_state_term", "composite_update_term"}
    ]

    rows = build_live_same_run_trace(
        source,
        side="radlads",
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        contexts=contexts,
    )

    balance = next(row for row in rows if row["stage"] == "balance_state_term")
    composite = next(row for row in rows if row["stage"] == "composite_update_term")
    assert balance["capture_kind"] == "exact_reconstruction"
    assert composite["capture_kind"] == "exact_reconstruction"
    assert {item["stage"] for item in composite["reconstruction_sources"]} == {
        "wkv_state_before",
        "ab",
        "wkv_update_outer_or_term",
    }


def test_balance_term_reconstruction_requires_live_ab() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    source = [
        row
        for row in _minimal_source("radlads")
        if row["stage"] not in {"ab", "balance_state_term"}
    ]

    rows = build_live_same_run_trace(
        source,
        side="radlads",
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        contexts=contexts,
    )

    assert (
        next(row for row in rows if row["stage"] == "balance_state_term")[
            "capture_kind"
        ]
        == "unavailable"
    )


def test_unavailable_stretch_does_not_invalidate_minimum_but_blocks_math() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    traces = {}
    for side in ("radlads", "qrwkv_off", "qrwkv_experimental"):
        traces[side] = build_live_same_run_trace(
            [row for row in _minimal_source(side) if row["stage"] != "kk"],
            side=side,
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

    assert report["same_run_valid"] is True
    assert report["minimum_stage_valid"] is True
    assert report["stretch_stages_available"] is False
    assert report["math_conclusion_valid"] is False
    assert report["first_divergent_stage"] == "kk"
    assert report["recommended_next_phase"] == (
        "P72 targeted live missing-stage hook completion"
    )


def test_p72_experimental_missing_k_k_k_a_uses_inactive_path_reason() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    rows = build_live_same_run_trace(
        [
            row
            for row in _minimal_source("qrwkv_experimental")
            if row["stage"] not in {"k_k", "k_a"}
        ],
        side="qrwkv_experimental",
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        contexts=contexts,
    )

    unavailable = {row["stage"]: row for row in rows if row["stage"] in {"k_k", "k_a"}}
    assert unavailable["k_k"]["capture_kind"] == "unavailable"
    assert unavailable["k_k"]["reason"] == (
        "not_active_in_fixture_path:qrwkv_experimental:k_k"
    )


def test_p73_experimental_direct_lane_k_k_k_a_are_not_applicable() -> None:
    contexts = [("tiny_no_mask", None, 0, 0, 0)]
    rows = build_live_same_run_trace(
        [
            row
            for row in _minimal_source(
                "qrwkv_experimental",
                config=DIRECT_BALANCE_CONFIG,
            )
            if row["stage"] not in {"k_k", "k_a"}
        ],
        side="qrwkv_experimental",
        same_run_group_id="group-a",
        fixture_id="fixture-a",
        parameter_id="parameter-a",
        contexts=contexts,
    )

    unavailable = {row["stage"]: row for row in rows if row["stage"] in {"k_k", "k_a"}}
    assert unavailable["k_k"]["capture_kind"] == "not_applicable"
    assert unavailable["k_k"]["reason"] == (
        "not_active_in_lane:direct_balance_state:k_k"
    )
    assert unavailable["k_a"]["reason"] == (
        "not_active_in_lane:direct_balance_state:k_a"
    )
    assert unavailable["k_k"]["balance_state_lane"] == "direct_balance_state"


def test_p74_radlads_terms_and_direct_rows_do_not_collide() -> None:
    traces = _traces()
    radlads_rows = traces["radlads"]
    lanes = {row["balance_state_lane"] for row in radlads_rows}

    assert lanes == {"balance_state_terms", "direct_balance_state"}
    assert len({_trace_key(row) for row in radlads_rows}) == len(radlads_rows)
    same_context = [
        row
        for row in radlads_rows
        if row["case"] == "tiny_no_mask"
        and row["token"] == 0
        and row["head"] == 0
        and row["stage"] == "kk"
    ]
    assert {row["balance_state_lane"] for row in same_context} == {
        "balance_state_terms",
        "direct_balance_state",
    }


def test_p74_radlads_direct_k_k_k_a_are_not_applicable() -> None:
    rows = [
        row
        for row in _traces()["radlads"]
        if row["balance_state_lane"] == "direct_balance_state"
        and row["stage"] in {"k_k", "k_a"}
    ]

    assert {row["stage"] for row in rows} == {"k_k", "k_a"}
    assert {row["capture_kind"] for row in rows} == {"not_applicable"}
    assert all(row["reason"].startswith("not_active_in_lane:") for row in rows)


def test_p74_direct_lane_mismatch_recommends_direct_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_experimental"]:
        if row["balance_state_lane"] == "direct_balance_state" and row["stage"] == "kk":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["direct_balance_state_first_differing_stage"] == "kk"
    assert report["recommended_next_phase"] == (
        "P75 targeted direct-lane kk construction fix"
    )


def test_p74_both_lanes_passing_recommends_residual_gate() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["balance_state_terms_lane_valid"] is True
    assert report["direct_balance_state_lane_valid"] is True
    assert report["recommended_next_phase"] == (
        "P77 targeted full-vs-stepwise residual fix"
    )


def test_p75_gate_preserves_p74_lanes_and_reports_both_residual_lanes() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    gate = report["p75_residual_impact_gate"]

    assert gate["schema"] == "qrwkv_xla.p75_residual_impact_gate.v1"
    assert gate["lane_comparisons"]["balance_state_terms"]["left"] == "RADLADS terms"
    assert gate["lane_comparisons"]["balance_state_terms"]["right"] == "QRWKV off terms"
    assert gate["lane_comparisons"]["direct_balance_state"]["left"] == (
        "RADLADS direct"
    )
    assert gate["lane_comparisons"]["direct_balance_state"]["right"] == (
        "QRWKV experimental direct"
    )
    assert gate["lane_comparisons_valid"] is True
    assert set(gate["residuals"]) == {
        "balance_state_terms",
        "direct_balance_state",
    }
    assert (
        gate["residuals"]["balance_state_terms"]["measurements"]["state_after_live"][
            "status"
        ]
        == "pass"
    )
    assert (
        gate["residuals"]["direct_balance_state"]["measurements"]["state_after_live"][
            "status"
        ]
        == "pass"
    )


def test_p75_missing_required_output_evidence_blocks_kernel_ready() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    gate = report["p75_residual_impact_gate"]

    assert gate["kernel_ready"] == "no"
    assert "missing_evidence:full_vs_stepwise" in gate["blocking_gates"]
    assert (
        "output_gate_unavailable:logits_output:missing_stepwise_output_capture"
        in gate["blocking_gates"]
    )
    assert gate["recommended_next_phase"] == (
        "P77 targeted full-vs-stepwise residual fix"
    )


def test_p77_full_vs_stepwise_evidence_feeds_p75_gate() -> None:
    metadata = {
        **_metadata(),
        "p77_full_vs_stepwise_evidence": _p77_evidence_rows(),
    }
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=metadata,
        strict_live=True,
    )
    p77 = report["p77_full_vs_stepwise_residual"]
    gate = report["p75_residual_impact_gate"]

    assert p77["schema"] == "qrwkv_xla.p77_full_vs_stepwise_residual.v1"
    assert p77["status"] == "pass"
    assert p77["full_path"] == "RWKV7QwenReferenceStudent.apply_with_state"
    assert p77["stepwise_path"] == "RWKV7QwenReferenceStudent.step"
    assert p77["final_state"]["status"] == "pass"
    assert p77["exported_state"]["status"] == "pass"
    assert p77["outputs"]["status"] == "unavailable"
    assert p77["blocking_gates"] == []
    assert p77["lanes"]["balance_state_terms"]["left"] == "RADLADS terms"
    assert p77["lanes"]["balance_state_terms"]["right"] == "QRWKV off terms"
    assert p77["lanes"]["direct_balance_state"]["left"] == "RADLADS direct"
    assert p77["lanes"]["direct_balance_state"]["right"] == (
        "QRWKV experimental direct"
    )
    assert gate["output_gates"]["state_after"]["status"] == "pass"
    assert gate["output_gates"]["exported_state"]["status"] == "pass"
    assert gate["output_gates"]["full_vs_stepwise"]["status"] == "pass"
    assert gate["output_gates"]["logits_output"]["status"] == "unavailable"
    assert gate["blocking_gates"] == [
        "output_gate_unavailable:logits_output:missing_stepwise_output_capture"
    ]
    assert gate["recommended_next_phase"] == (
        "P79 targeted logits/output hook completion"
    )


def test_p78_hidden_output_evidence_feeds_p75_gate_without_fake_logits() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata={
            **_metadata(),
            "p77_full_vs_stepwise_evidence": _p78_output_evidence_rows(),
        },
        strict_live=True,
    )
    p78 = report["p78_logits_output_residual"]
    gate = report["p75_residual_impact_gate"]

    assert p78["schema"] == "qrwkv_xla.p78_logits_output_residual.v1"
    assert p78["status"] == "unavailable"
    assert p78["lane_aware_keys"] is True
    assert p78["hidden_path"]["status"] == "pass"
    assert p78["logits_path"] == {
        "status": "unavailable",
        "reason": "missing_lm_head_logits_path",
        "row_count": 4,
    }
    assert p78["lanes"]["balance_state_terms"]["left"] == "RADLADS terms"
    assert p78["lanes"]["balance_state_terms"]["right"] == "QRWKV off terms"
    assert p78["lanes"]["direct_balance_state"]["left"] == "RADLADS direct"
    assert p78["lanes"]["direct_balance_state"]["right"] == (
        "QRWKV experimental direct"
    )
    assert gate["output_gates"]["full_vs_stepwise"]["status"] == "pass"
    assert gate["output_gates"]["logits_output"]["status"] == "unavailable"
    assert gate["output_gates"]["logits_output"]["reason"] == (
        "missing_lm_head_logits_path"
    )
    assert "missing_evidence:logits_output" not in gate["blocking_gates"]
    assert (
        "output_gate_unavailable:logits_output:missing_lm_head_logits_path"
        in gate["blocking_gates"]
    )
    assert gate["kernel_ready"] == "no"
    assert p78["recommended_next_phase"] == (
        "P79 targeted logits/output hook completion"
    )


def test_p78_true_logits_availability_is_reported_separately() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata={
            **_metadata(),
            "p77_full_vs_stepwise_evidence": _p78_output_evidence_rows(
                include_logits=True
            ),
        },
        strict_live=True,
    )
    p78 = report["p78_logits_output_residual"]

    assert p78["status"] == "pass"
    assert p78["hidden_path"]["status"] == "pass"
    assert p78["logits_path"]["status"] == "pass"
    assert p78["logits_path"]["reason"] == "true_lm_head_logits_allclose"
    assert p78["full_vs_stepwise_output"]["status"] == "pass"


def test_p78_output_mismatch_blocks_p75_gate_and_recommends_p78() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata={
            **_metadata(),
            "p77_full_vs_stepwise_evidence": _p78_output_evidence_rows(
                hidden_delta=1.0
            ),
        },
        strict_live=True,
    )
    p78 = report["p78_logits_output_residual"]
    gate = report["p75_residual_impact_gate"]

    assert p78["status"] == "fail"
    assert p78["inter_side_parity"]["balance_state_terms"]["status"] == "fail"
    assert gate["output_gates"]["logits_output"]["status"] == "fail"
    assert "output_gate_failed:logits_output" in gate["blocking_gates"]
    assert gate["recommended_next_phase"] == "P79 targeted logits/output residual fix"


def test_p78_missing_output_evidence_remains_unavailable() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata={**_metadata(), "p77_full_vs_stepwise_evidence": _p77_evidence_rows()},
        strict_live=True,
    )
    p78 = report["p78_logits_output_residual"]
    gate = report["p75_residual_impact_gate"]

    assert p78["status"] == "unavailable"
    assert p78["blocking_gates"] == ["missing_stepwise_output_capture"]
    assert gate["output_gates"]["logits_output"]["status"] == "unavailable"
    assert (
        "output_gate_unavailable:logits_output:missing_stepwise_output_capture"
        in gate["blocking_gates"]
    )


def test_p77_full_vs_stepwise_mismatch_blocks_gate() -> None:
    rows = _p77_evidence_rows()
    rows[0] = {
        **rows[0],
        "status": "fail",
        "reason": "fail",
        "allclose": False,
        "max_abs_error": 1.0,
    }
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata={**_metadata(), "p77_full_vs_stepwise_evidence": rows},
        strict_live=True,
    )
    p77 = report["p77_full_vs_stepwise_residual"]
    gate = report["p75_residual_impact_gate"]

    assert p77["status"] == "fail"
    assert p77["surface_status"]["radlads_terms"]["status"] == "fail"
    assert gate["output_gates"]["full_vs_stepwise"]["status"] == "fail"
    assert "output_gate_failed:full_vs_stepwise" in gate["blocking_gates"]


def test_p77_missing_stepwise_path_is_unavailable_not_pass() -> None:
    rows = _p77_evidence_rows()
    rows = [row for row in rows if row["side"] != "qrwkv_experimental"]
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata={**_metadata(), "p77_full_vs_stepwise_evidence": rows},
        strict_live=True,
    )
    p77 = report["p77_full_vs_stepwise_residual"]
    gate = report["p75_residual_impact_gate"]

    assert p77["status"] == "unavailable"
    assert p77["surface_status"]["qrwkv_experimental_direct"] == {
        "status": "unavailable",
        "reason": "missing_full_vs_stepwise_rows",
    }
    assert gate["output_gates"]["full_vs_stepwise"]["status"] == "unavailable"


def test_p75_kernel_ready_yes_requires_all_output_gates_passing() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    report = {
        **report,
        "p75_output_gates": {
            "state_after": {"status": "pass", "reason": "synthetic_unit"},
            "exported_state": {"status": "pass", "reason": "synthetic_unit"},
            "full_vs_stepwise": {"status": "pass", "reason": "synthetic_unit"},
            "logits_output": {"status": "pass", "reason": "synthetic_unit"},
        },
    }
    gate = build_p75_residual_impact_gate(report)

    assert gate["kernel_ready"] == "yes"
    assert gate["blocking_gates"] == []
    assert gate["recommended_next_phase"] == (
        "P79 broader fixture residual-impact validation"
    )


def test_p75_shape_mismatch_is_blocking_state_after_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["balance_state_lane"] == "balance_state_terms" and row["stage"] == (
            "state_after_live"
        ):
            row["array"] = [[1.0, 2.0]]
            row["shape"] = [1, 2]
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )
    gate = report["p75_residual_impact_gate"]

    assert (
        "residual_blocking:balance_state_terms:state_after_live"
        in gate["blocking_gates"]
    )
    assert gate["recommended_next_phase"] == "P77 targeted lane-aware state layout fix"


def test_p75_non_finite_residual_is_blocking() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["balance_state_lane"] == "balance_state_terms" and row["stage"] == "vk":
            row["array"] = np.full(np.asarray(row["array"]).shape, np.nan).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )
    gate = report["p75_residual_impact_gate"]

    assert "residual_blocking:balance_state_terms:vk" in gate["blocking_gates"]
    assert gate["kernel_ready"] == "no"


def test_p75_logits_failure_recommends_output_fix() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    gate = build_p75_residual_impact_gate(
        {
            **report,
            "p75_output_gates": {
                "state_after": {"status": "pass"},
                "exported_state": {"status": "pass"},
                "full_vs_stepwise": {"status": "pass"},
                "logits_output": {"status": "fail", "reason": "unit_mismatch"},
            },
        }
    )

    assert gate["recommended_next_phase"] == ("P79 targeted logits/output residual fix")


def test_first_live_mismatch_in_k_for_update_recommends_balance_prep_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["stage"] == "k_for_update":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["first_divergent_stage"] == "k_for_update"
    assert report["recommended_next_phase"] == (
        "P75 targeted terms-lane k_for_update/v_for_update balance-prep fix"
    )


def test_first_live_mismatch_in_k_k_recommends_p73_k_k_k_a_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["stage"] == "k_k":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["first_divergent_stage"] == "k_k"
    assert report["recommendation"] == "P73 targeted k_k/k_a construction fix"


def test_first_live_mismatch_in_k_a_recommends_p73_k_k_k_a_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["stage"] == "k_a":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["first_divergent_stage"] == "k_a"
    assert report["recommendation"] == "P73 targeted k_k/k_a construction fix"


def test_first_live_mismatch_in_kk_recommends_construction_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["stage"] == "kk":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["first_divergent_stage"] == "kk"
    assert (
        report["recommended_next_phase"]
        == "P75 targeted terms-lane kk construction fix"
    )


def test_first_live_mismatch_in_ab_recommends_ab_fix() -> None:
    traces = _traces()
    for row in traces["qrwkv_off"]:
        if row["stage"] == "ab":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
            break

    report = compare_live_same_run_traces(
        traces=traces,
        metadata=_metadata(),
        strict_live=True,
    )

    assert report["first_divergent_stage"] == "ab"
    assert (
        report["recommended_next_phase"]
        == "P75 targeted terms-lane ab construction/orientation fix"
    )


def test_p79_matrix_includes_requested_found_and_missing_cases() -> None:
    matrix = _p79_matrix(
        manifest_cases=["tiny_no_mask"],
        expected_cases=("tiny_no_mask", "missing_case"),
    )

    by_case = matrix["cases"]
    assert by_case["tiny_no_mask"]["kernel_ready_for_case"] == "yes"
    assert by_case["missing_case"]["kernel_ready_for_case"] == "unavailable"
    assert by_case["missing_case"]["blocking_gates"] == ["fixture_case_not_found"]
    assert by_case["missing_case"]["recommended_action"] == (
        "P80 targeted fixture lineage/harness repair"
    )
    assert matrix["summary"]["cases_unavailable"] == 1
    assert (
        matrix["recommended_next_phase"]
        == "P80 targeted fixture lineage/harness repair"
    )


def test_structured_expected_fixture_cases_support_aliases() -> None:
    matrix = _p79_matrix(
        manifest_cases=["tiny_no_mask", "tiny_prefix_or_left_padding"],
        expected_cases=None,
    )

    assert matrix["active_expected_cases"] == [
        "tiny_no_mask",
        "tiny_attention_mask",
        "tiny_stepwise_state",
        "tiny_prefix_or_left_padding",
        "tiny_all_radlads_math_enabled",
    ]
    assert matrix["accepted_aliases"] == {
        "tiny_prefix_padding_or_left_padding": "tiny_prefix_or_left_padding"
    }
    assert matrix["deprecated_cases"] == []
    assert matrix["optional_cases"] == []
    assert matrix["cases"]["tiny_prefix_padding_or_left_padding"]["resolution"] in {
        "alias",
        "missing",
    }


def test_tiny_prefix_padding_alias_resolves_to_current_case() -> None:
    matrix = _p79_matrix(
        traces=_traces_for_case("tiny_prefix_or_left_padding"),
        manifest_cases=["tiny_no_mask", "tiny_prefix_or_left_padding"],
        expected_cases=("tiny_prefix_padding_or_left_padding",),
        p77_rows=_evidence_rows_for_case(
            _p77_evidence_rows(), "tiny_prefix_or_left_padding"
        ),
        p78_rows=_evidence_rows_for_case(
            _p78_output_evidence_rows(include_logits=True),
            "tiny_prefix_or_left_padding",
        ),
    )
    row = matrix["cases"]["tiny_prefix_padding_or_left_padding"]

    assert row["canonical_case"] == "tiny_prefix_or_left_padding"
    assert row["resolved_case"] == "tiny_prefix_or_left_padding"
    assert row["resolution"] == "alias"
    assert row["resolved_by_alias"] is True
    assert row["kernel_ready_for_case"] == "yes"


def test_alias_resolution_reuses_canonical_evidence_without_duplication() -> None:
    matrix = _p79_matrix(
        traces=_traces_for_case("tiny_prefix_or_left_padding"),
        manifest_cases=["tiny_no_mask", "tiny_prefix_or_left_padding"],
        expected_cases=("tiny_prefix_padding_or_left_padding",),
        p77_rows=_evidence_rows_for_case(
            _p77_evidence_rows(), "tiny_prefix_or_left_padding"
        ),
        p78_rows=_evidence_rows_for_case(
            _p78_output_evidence_rows(include_logits=True),
            "tiny_prefix_or_left_padding",
        ),
    )
    row = matrix["cases"]["tiny_prefix_padding_or_left_padding"]

    assert row["evidence_source_case"] == "tiny_prefix_or_left_padding"
    assert row["duplicates_fixture_evidence"] is False
    assert row["fixture_id"] == "fixture-a"
    assert row["same_run_group_id"] == "group-a"


def test_missing_active_fixture_without_alias_remains_unavailable() -> None:
    matrix = _p79_matrix(
        manifest_cases=["tiny_no_mask"],
        expected_cases=("tiny_no_mask", "not_an_alias"),
    )
    row = matrix["cases"]["not_an_alias"]

    assert row["resolution"] == "missing"
    assert row["blocking_gates"] == ["fixture_case_not_found"]
    assert row["kernel_ready_for_case"] == "unavailable"
    assert matrix["remaining_missing_cases"] == ["not_an_alias"]


def test_deprecated_fixture_does_not_block_active_matrix() -> None:
    matrix = build_p79_broader_fixture_residual_matrix(
        fixture_manifest_data={"cases": [{"name": "tiny_no_mask"}]},
        traces=_traces(),
        metadata={
            **_metadata(),
            "p77_full_vs_stepwise_evidence": _p77_evidence_rows(),
            "p78_logits_output_evidence": _p78_output_evidence_rows(
                include_logits=True
            ),
        },
        strict_live=True,
        atol=1e-5,
        rtol=1e-5,
        fixture_expectations=FixtureExpectationMetadata(
            active_expected_cases=("tiny_no_mask",),
            accepted_aliases={},
            deprecated_cases=("old_case",),
        ),
    )

    assert matrix["cases"]["old_case"]["resolution"] == "deprecated"
    assert matrix["cases"]["old_case"]["blocking_gates"] == []
    assert matrix["all_expected_cases_pass"] is True


def test_optional_absent_fixture_does_not_block_active_matrix() -> None:
    matrix = build_p79_broader_fixture_residual_matrix(
        fixture_manifest_data={"cases": [{"name": "tiny_no_mask"}]},
        traces=_traces(),
        metadata={
            **_metadata(),
            "p77_full_vs_stepwise_evidence": _p77_evidence_rows(),
            "p78_logits_output_evidence": _p78_output_evidence_rows(
                include_logits=True
            ),
        },
        strict_live=True,
        atol=1e-5,
        rtol=1e-5,
        fixture_expectations=FixtureExpectationMetadata(
            active_expected_cases=("tiny_no_mask",),
            accepted_aliases={},
            optional_cases=("optional_case",),
        ),
    )

    assert matrix["cases"]["optional_case"]["resolution"] == "optional_absent"
    assert matrix["cases"]["optional_case"]["blocking_gates"] == []
    assert matrix["all_expected_cases_pass"] is True


def test_matrix_reports_direct_and_alias_resolution_separately() -> None:
    traces = _traces()
    prefix_traces = _traces_for_case("tiny_prefix_or_left_padding")
    for side in traces:
        traces[side].extend(prefix_traces[side])
    matrix = _p79_matrix(
        traces=traces,
        manifest_cases=["tiny_no_mask", "tiny_prefix_or_left_padding"],
        expected_cases=("tiny_no_mask", "tiny_prefix_padding_or_left_padding"),
        p77_rows=[
            *_p77_evidence_rows(),
            *_evidence_rows_for_case(
                _p77_evidence_rows(), "tiny_prefix_or_left_padding"
            ),
        ],
        p78_rows=[
            *_p78_output_evidence_rows(include_logits=True),
            *_evidence_rows_for_case(
                _p78_output_evidence_rows(include_logits=True),
                "tiny_prefix_or_left_padding",
            ),
        ],
    )

    assert matrix["cases"]["tiny_no_mask"]["resolution"] == "direct"
    assert (
        matrix["cases"]["tiny_prefix_padding_or_left_padding"]["resolution"] == "alias"
    )


def test_p79_same_run_invalid_marks_lineage_repair() -> None:
    matrix = _p79_matrix(traces=_traces(off_group="group-b"))
    row = matrix["case_rows"][0]

    assert row["same_run_valid"] is False
    assert row["recommended_action"] == "P80 targeted fixture lineage/harness repair"


def test_p79_gate_failures_recommend_exact_p80_fixes() -> None:
    state_traces = _traces()
    for row in state_traces["qrwkv_off"]:
        if row["stage"] == "state_after_live":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
    assert _p79_matrix(traces=state_traces)["case_rows"][0]["recommended_action"] == (
        "P80 targeted broader-fixture state_after residual fix"
    )

    exported_traces = _traces()
    for row in exported_traces["qrwkv_off"]:
        if row["stage"] == "state_after_exported":
            row["array"] = (np.asarray(row["array"]) + 1.0).tolist()
    assert _p79_matrix(traces=exported_traces)["case_rows"][0][
        "recommended_action"
    ] == ("P80 targeted broader-fixture exported_state residual fix")

    p77_rows = _p77_evidence_rows()
    p77_rows[0]["status"] = "fail"
    p77_rows[0]["max_abs_error"] = 1.0
    assert _p79_matrix(p77_rows=p77_rows)["case_rows"][0]["recommended_action"] == (
        "P80 targeted broader-fixture full-vs-stepwise residual fix"
    )

    p78_rows = _p78_output_evidence_rows(hidden_delta=1.0, include_logits=True)
    assert _p79_matrix(p78_rows=p78_rows)["case_rows"][0]["recommended_action"] == (
        "P80 targeted broader-fixture logits/output residual fix"
    )


def test_p79_all_expected_passing_recommends_pallas_or_scaffold() -> None:
    matrix = _p79_matrix()

    assert matrix["all_expected_cases_pass"] is True
    assert matrix["recommended_action"] in {
        "P81 Pallas prototype behind known-caveat flag",
        "P81 kernel/reference parity scaffold",
    }


def test_p79_matrix_markdown_table_renders() -> None:
    markdown = _p79_matrix_markdown(_p79_matrix())

    assert (
        "| requested_case | canonical_case | resolved_case | resolution |" in markdown
    )
    assert "tiny_no_mask" in markdown


def test_all_active_expected_cases_passing_recommends_p81_pallas() -> None:
    traces = _traces()
    prefix_traces = _traces_for_case("tiny_prefix_or_left_padding")
    for side in traces:
        traces[side].extend(prefix_traces[side])
    matrix = _p79_matrix(
        traces=traces,
        manifest_cases=["tiny_no_mask", "tiny_prefix_or_left_padding"],
        expected_cases=("tiny_no_mask", "tiny_prefix_padding_or_left_padding"),
        p77_rows=[
            *_p77_evidence_rows(),
            *_evidence_rows_for_case(
                _p77_evidence_rows(), "tiny_prefix_or_left_padding"
            ),
        ],
        p78_rows=[
            *_p78_output_evidence_rows(include_logits=True),
            *_evidence_rows_for_case(
                _p78_output_evidence_rows(include_logits=True),
                "tiny_prefix_or_left_padding",
            ),
        ],
    )

    assert matrix["all_expected_cases_pass"] is True
    assert (
        matrix["recommended_next_phase"]
        == "P81 Pallas prototype behind known-caveat flag"
    )


def test_wkv_runtime_reference_is_default_and_preserves_outputs() -> None:
    default_config = RWKV7QwenReferenceConfig(
        vocab_size=16,
        hidden_size=4,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        emit_logits=True,
    )
    explicit_config = RWKV7QwenReferenceConfig(
        vocab_size=16,
        hidden_size=4,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        emit_logits=True,
        wkv_runtime="reference",
    )
    assert default_config.wkv_runtime is WKVRuntime.REFERENCE
    assert explicit_config.wkv_runtime is WKVRuntime.REFERENCE

    params = RWKV7QwenReferenceStudent(default_config).init_params(
        jax.random.PRNGKey(0)
    )
    tokens = np.array([[1, 2]], dtype=np.int32)
    default_output, default_state = RWKV7QwenReferenceStudent(
        default_config
    ).apply_with_state(params, tokens)
    explicit_output, explicit_state = RWKV7QwenReferenceStudent(
        explicit_config
    ).apply_with_state(params, tokens)

    np.testing.assert_allclose(
        default_output.hidden_states, explicit_output.hidden_states
    )
    np.testing.assert_allclose(default_output.logits, explicit_output.logits)
    np.testing.assert_allclose(
        default_state.wkv_matrix_state, explicit_state.wkv_matrix_state
    )


def test_wkv_runtime_invalid_value_fails_clearly() -> None:
    try:
        RWKV7QwenReferenceConfig(wkv_runtime="bogus")
    except ValueError as exc:
        assert "wkv_runtime must be one of reference, pallas" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("invalid wkv_runtime should fail")


def test_wkv_runtime_pallas_is_explicit_unavailable_opt_in() -> None:
    config = RWKV7QwenReferenceConfig(
        vocab_size=16,
        hidden_size=4,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        wkv_runtime="pallas",
    )
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(0))

    try:
        student.apply_with_state(params, np.array([[1]], dtype=np.int32))
    except PallasRuntimeUnavailableError as exc:
        assert "wkv_runtime='pallas' was requested" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Pallas runtime should fail closed while unavailable")


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
        broader_fixture_report=True,
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
    assert (out / "P73_BALANCE_STATE_LANE_MAP.md").is_file()
    assert (out / "balance_state_lane_map.json").is_file()
    assert (out / "P74_DIRECT_BALANCE_LANE_REPORT.md").is_file()
    assert (out / "direct_balance_lane_comparison.json").is_file()
    assert (out / "P75_RESIDUAL_IMPACT_GATE.md").is_file()
    assert (out / "residual_impact_gate.json").is_file()
    assert (out / "P75_KERNEL_READINESS_DECISION.md").is_file()
    assert (out / "P77_FULL_VS_STEPWISE_REPORT.md").is_file()
    assert (out / "full_vs_stepwise_residual.json").is_file()
    assert (out / "P78_LOGITS_OUTPUT_REPORT.md").is_file()
    assert (out / "logits_output_residual.json").is_file()
    assert (out / "P79_BROADER_FIXTURE_VALIDATION_REPORT.md").is_file()
    assert (out / "broader_fixture_residual_matrix.json").is_file()
    assert (out / "broader_fixture_residual_matrix.md").is_file()
    assert (out / "P80_FIXTURE_LINEAGE_REPAIR_REPORT.md").is_file()
    assert (out / "fixture_lineage_resolution.json").is_file()
    assert (out / "P80_FIX_NOTE.md").is_file()
    assert (out / "P81_PALLAS_PROTOTYPE_REPORT.md").is_file()
    assert (out / "P82_PALLAS_RUNTIME_SCAFFOLD_COMPLETION_REPORT.md").is_file()
    assert (out / "P83_PALLAS_REFERENCE_PARITY_REPORT.md").is_file()
    assert (out / "pallas_runtime_probe.json").is_file()
    assert (out / "pallas_reference_parity_probe.json").is_file()
    assert (out / "P81_FIX_NOTE.md").is_file()
    assert "| requested_case | canonical_case | resolved_case | resolution |" in (
        out / "broader_fixture_residual_matrix.md"
    ).read_text(encoding="utf-8")
    resolution = json.loads(
        (out / "fixture_lineage_resolution.json").read_text(encoding="utf-8")
    )
    assert resolution["schema"] == "qrwkv_xla.p80_fixture_lineage_resolution.v1"
    probe = json.loads((out / "pallas_runtime_probe.json").read_text(encoding="utf-8"))
    parity_probe = json.loads(
        (out / "pallas_reference_parity_probe.json").read_text(encoding="utf-8")
    )
    assert parity_probe == probe
    assert probe["schema"] == "qrwkv_xla.p83_pallas_wkv_parity_probe.v1"
    assert probe["phase"] == "P83"
    assert probe["default_runtime"] == "reference"
    assert probe["allowed_runtimes"] == ["reference", "pallas"]
    assert probe["wkv_runtime_requested"] == "reference"
    assert probe["wkv_runtime_effective"] == "reference"
    assert probe["prototype_status"] == "not_requested"
    assert probe["parity_status"] == "not_requested"
    assert probe["parity_scope"] == "tiny_one_step_wkv_update"
    assert probe["fallback_used"] is False
    assert probe["kernel_parity_claimed"] is False
    rows = load_live_same_run_trace_jsonl(out / "live_trace_radlads.jsonl")
    assert rows
    assert all(row["capture_kind"] == "unavailable" for row in rows)
    assert report["live_rows_captured_radlads"] == 0
    assert "live_rows_captured_qrwkv_off" in report
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


def test_runner_accepts_pallas_opt_in_and_skips_reference_capture(
    tmp_path: Path,
) -> None:
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
        wkv_runtime="pallas",
    )

    probe = report["p83_pallas_wkv_parity_probe"]
    out = tmp_path / "out"
    assert report["p82_pallas_runtime_probe"] == probe
    assert probe["wkv_runtime_requested"] == "pallas"
    assert probe["wkv_runtime_effective"] in {"pallas", "unavailable"}
    assert probe["fallback_used"] is False
    assert probe["prototype_status"] in {"pass", "unavailable", "failed"}
    assert probe["parity_status"] in {"pass", "unavailable", "failed"}
    assert probe["parity_scope"] == "tiny_one_step_wkv_update"
    assert probe["kernel_parity_claimed"] is (probe["parity_status"] == "pass")
    assert report["pallas_requested_reference_trace_contamination"] is False
    assert report["reference_trace_capture_skipped"] is True
    assert not (out / "live_trace_qrwkv_off.jsonl").exists()
    assert (out / "P82_PALLAS_RUNTIME_SCAFFOLD_COMPLETION_REPORT.md").is_file()
    assert (out / "P83_PALLAS_REFERENCE_PARITY_REPORT.md").is_file()
    persisted = json.loads((out / "pallas_runtime_probe.json").read_text())
    parity_persisted = json.loads(
        (out / "pallas_reference_parity_probe.json").read_text()
    )
    assert persisted == parity_persisted
    assert persisted["schema"] == "qrwkv_xla.p83_pallas_wkv_parity_probe.v1"
    if probe["prototype_status"] == "pass":
        assert probe["pallas_available"] is True
        assert probe["finite"] is True
        assert probe["shape_match"] is True
        assert probe["max_abs_error"] <= probe["atol"]
        assert probe["max_rel_error"] <= probe["rtol"]
        assert probe["probe_shapes"]["state"] == [1, 1, 2, 2]
        assert (
            probe["recommended_next_phase"]
            == "P84 broader Pallas WKV shape/dtype parity"
        )
        assert not (out / "P82_BLOCKER_REPORT.md").exists()
    else:
        assert (out / "P82_BLOCKER_REPORT.md").is_file()


def test_script_help_works() -> None:
    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P68" in result.stdout
    assert "--strict-live" in result.stdout
    assert "--broader-fixture-report" in result.stdout
    assert "--wkv-runtime" in result.stdout


def test_script_invalid_wkv_runtime_fails_clearly(tmp_path: Path) -> None:
    fixture = tmp_path / "manifest.json"
    params = tmp_path / "params.json"
    fixture.write_text(json.dumps({"cases": ["tiny_no_mask"]}), encoding="utf-8")
    params.write_text(json.dumps({"a": 1}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--fixture-manifest",
            str(fixture),
            "--parameters",
            str(params),
            "--strict-live",
            "--wkv-runtime",
            "bogus",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice: 'bogus'" in result.stderr
