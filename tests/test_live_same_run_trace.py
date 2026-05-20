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
    LiveTraceCollector,
    _capture_radlads_case,
    build_live_same_run_trace,
    compare_live_same_run_traces,
    deterministic_fixture_id,
    deterministic_parameter_id,
    load_live_same_run_trace_jsonl,
    new_same_run_group_id,
    run_live_same_run_trace,
)
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_live_same_run_update_trace.py"


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


def test_minimum_stages_are_counted_separately_from_stretch() -> None:
    report = compare_live_same_run_traces(
        traces=_traces(),
        metadata=_metadata(),
        strict_live=True,
    )
    assert set(report["minimum_stage_availability"]) == set(MINIMUM_STAGES)
    assert "v_first" not in report["minimum_stage_availability"]
    assert report["live_rows_captured_radlads"] > len(MINIMUM_STAGES)


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
    assert (
        report["recommended_next_phase"]
        == "P72 targeted k_for_update/v_for_update balance-prep fix"
    )


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
        report["recommended_next_phase"] == "P72 targeted kk/k_k/k_a construction fix"
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
        == "P72 targeted ab construction/orientation fix"
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


def test_script_help_works() -> None:
    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P68" in result.stdout
    assert "--strict-live" in result.stdout
