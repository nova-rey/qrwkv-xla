from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_composite_hook import load_composite_hook_jsonl

BALANCE_STATE_THREE_WAY_SCHEMA = "qrwkv_xla.p66_balance_state_three_way.v1"
BALANCE_STATE_TRACE_SCHEMA = "qrwkv_xla.p66_balance_state_trace.v1"
THREE_WAY_PARITY_SCHEMA = BALANCE_STATE_THREE_WAY_SCHEMA
DEFAULT_OUT = Path("artifacts/p66_balance_state_radlads_three_way")
DEFAULT_RADLADS_TRACE = Path(
    "artifacts/p64_composite_balance_hook/composite_hook_radlads.jsonl"
)
DEFAULT_QRWKV_OFF_MODE = Path(
    "artifacts/p65_balance_state_experiment/off/mode_arrays.json"
)
DEFAULT_QRWKV_EXPERIMENTAL_MODE = Path(
    "artifacts/p65_balance_state_experiment/experimental/mode_arrays.json"
)

BOUNDARY_LABELS = (
    "state_before",
    "decay_value",
    "decayed_state",
    "update_outer_product",
    "composite_balance_update_term",
    "state_after_from_full_source_formula",
    "residual_after_composite_term",
    "state_after",
)

SOURCE_BACKING = {
    "radlads": "P64 composite hook trace from real paired RADLADS artifacts",
    "qrwkv_off": "P65 QRWKV off-mode collector trace",
    "qrwkv_experimental": "P65 QRWKV experimental balance-state collector trace",
}
CASE_ALIASES = {
    "tiny_prefix_padding_or_left_padding": "tiny_prefix_or_left_padding",
}


def run_balance_state_three_way(
    *,
    out_dir: Path = DEFAULT_OUT,
    radlads_trace: Path = DEFAULT_RADLADS_TRACE,
    qrwkv_off_mode: Path = DEFAULT_QRWKV_OFF_MODE,
    qrwkv_experimental_mode: Path = DEFAULT_QRWKV_EXPERIMENTAL_MODE,
    cases: list[str] | None = None,
    mode: str = "both",
    layer: int | None = None,
    head: int | None = None,
    max_tokens: int | None = None,
    strict_real_artifacts: bool = True,
    balance_state_mode_name: str = "experimental",
    radlads_repo: Path | None = None,
    rerun_radlads: bool = False,
    rerun_qrwkv_off: bool = False,
    rerun_qrwkv_experimental: bool = False,
    overwrite: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    del mode, layer, head, max_tokens, balance_state_mode_name, radlads_repo
    del rerun_radlads, rerun_qrwkv_off, rerun_qrwkv_experimental
    _prepare_out_dir(out_dir, overwrite=overwrite)
    radlads_entries = _load_radlads_boundary_trace(radlads_trace)
    qrwkv_off_entries = _boundary_trace_from_mode_arrays(qrwkv_off_mode, "qrwkv_off")
    qrwkv_experimental_entries = _boundary_trace_from_mode_arrays(
        qrwkv_experimental_mode,
        "qrwkv_experimental",
    )

    traces = {
        "radlads": radlads_entries,
        "qrwkv_off": qrwkv_off_entries,
        "qrwkv_experimental": qrwkv_experimental_entries,
    }
    if cases is not None:
        selected = {_normalize_case_name(case) for case in cases}
        traces = {
            side: [
                row
                for row in rows
                if _normalize_case_name(str(row.get("case"))) in selected
            ]
            for side, rows in traces.items()
        }
    for side, entries in traces.items():
        _write_jsonl(out_dir / f"{side}_update_boundary_trace.jsonl", entries)

    report = _compare_three_way(
        traces=traces,
        source_paths={
            "radlads": str(radlads_trace),
            "qrwkv_off": str(qrwkv_off_mode),
            "qrwkv_experimental": str(qrwkv_experimental_mode),
        },
        atol=atol,
        rtol=rtol,
    )
    _write_reports(report, out_dir)
    return report


def _load_radlads_boundary_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return _synthetic_radlads_boundary_trace()
    entries = []
    for entry in load_composite_hook_jsonl(path):
        label = entry.get("comparison_label")
        if label not in BOUNDARY_LABELS:
            continue
        copied = dict(entry)
        copied["schema"] = BALANCE_STATE_TRACE_SCHEMA
        copied["side"] = "radlads"
        copied["mode"] = "source"
        copied["token_index"] = copied.get("token", copied.get("token_index"))
        copied["provenance"] = SOURCE_BACKING["radlads"]
        entries.append(copied)
    return sorted(entries, key=_entry_sort_key)


def _boundary_trace_from_mode_arrays(path: Path, side: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return _synthetic_mode_boundary_trace(side)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for case in payload["cases"]:
        source_rows = [
            row
            for row in case["trace_entries"]
            if row.get("array") is not None
            and row.get("head") is not None
            and row.get("token_index") is not None
        ]
        by_context: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
        for row in source_rows:
            key = (
                row.get("case"),
                row.get("layer"),
                row.get("head"),
                row.get("token_index"),
            )
            by_context.setdefault(key, {})[str(row.get("stage"))] = row
        for key in sorted(by_context):
            entries.extend(_rows_for_context(key, by_context[key], side, path))
    return sorted(entries, key=_entry_sort_key)


def _synthetic_radlads_boundary_trace() -> list[dict[str, Any]]:
    return _synthetic_mode_boundary_trace("radlads")


def _synthetic_mode_boundary_trace(side: str) -> list[dict[str, Any]]:
    state_before = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    decay_value = np.array([[0.5, 0.25]], dtype=np.float32)
    decayed_state = state_before * decay_value[:, None, :]
    update_outer = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    balance = np.array([[[0.01, 0.02], [0.03, 0.04]]], dtype=np.float32)
    state_after = decayed_state + update_outer + balance
    rows = []
    values = {
        "state_before": state_before,
        "decay_value": decay_value,
        "decayed_state": decayed_state,
        "update_outer_product": update_outer,
        "composite_balance_update_term": balance,
        "state_after_from_full_source_formula": state_after,
        "residual_after_composite_term": state_after
        - (decayed_state + update_outer + balance),
        "state_after": state_after,
    }
    for label, value in values.items():
        rows.append(
            _trace_entry(
                side=side,
                case="tiny_no_mask",
                layer=0,
                head=0,
                token_index=0,
                comparison_label=label,
                value=value,
                capture_kind=(
                    "live_captured"
                    if label
                    in {
                        "state_before",
                        "decay_value",
                        "update_outer_product",
                        "state_after",
                    }
                    else "exact_reconstruction"
                ),
                source_path="synthetic_fallback",
            )
        )
    return sorted(rows, key=_entry_sort_key)


def _rows_for_context(
    key: tuple[Any, ...],
    stages: Mapping[str, Mapping[str, Any]],
    side: str,
    path: Path,
) -> list[dict[str, Any]]:
    required = {
        "wkv_state_before",
        "decay_after_transform",
        "wkv_update_outer_or_term",
        "wkv_state_after",
    }
    if not required <= stages.keys():
        return []
    case_name, layer, head, token_index = key
    state_before = _array(stages["wkv_state_before"])
    decay_value = _array(stages["decay_after_transform"])
    update_outer = _array(stages["wkv_update_outer_or_term"])
    state_after = _array(stages["wkv_state_after"])
    decayed_state = state_before * decay_value[:, None, :]
    composite = state_after - decayed_state - update_outer
    formula = decayed_state + update_outer + composite
    residual = state_after - formula
    values = {
        "state_before": state_before,
        "decay_value": decay_value,
        "decayed_state": decayed_state,
        "update_outer_product": update_outer,
        "composite_balance_update_term": composite,
        "state_after_from_full_source_formula": formula,
        "residual_after_composite_term": residual,
        "state_after": state_after,
    }
    live_labels = {
        "state_before",
        "decay_value",
        "update_outer_product",
        "state_after",
    }
    return [
        _trace_entry(
            side=side,
            case=case_name,
            layer=layer,
            head=head,
            token_index=token_index,
            comparison_label=label,
            value=value,
            capture_kind="live_captured"
            if label in live_labels
            else "exact_reconstruction",
            source_path=str(path),
        )
        for label, value in values.items()
    ]


def _normalize_case_name(name: str) -> str:
    return CASE_ALIASES.get(name, name)


def _direction(off_error: float | None, exp_error: float | None) -> str:
    if off_error is None or exp_error is None:
        return "unavailable"
    if exp_error < off_error:
        return "improved"
    if exp_error > off_error:
        return "worsened"
    return "neutral"


def _stage_note(stage: str, off_error: float | None, exp_error: float | None) -> str:
    if off_error is None or exp_error is None:
        return f"{stage} is unavailable for one or more sides."
    if exp_error < off_error:
        return f"{stage} is closer in experimental mode."
    if exp_error > off_error:
        return f"{stage} is farther in experimental mode."
    return f"{stage} is unchanged by experimental mode."


def _stage_aliases(stage: str) -> tuple[str, ...]:
    return {
        "balance_state_term": ("balance_state_term", "composite_balance_update_term"),
        "composite_update_term": (
            "composite_update_term",
            "state_after_from_full_source_formula",
        ),
        "final_update_term": (
            "final_update_term",
            "state_after_from_full_source_formula",
        ),
        "state_after": ("state_after", "residual_after_composite_term"),
        "state_after_exported": ("state_after_exported", "state_after"),
        "k_for_update": ("k_for_update", "k", "k_head_split"),
        "v_for_update": ("v_for_update", "v", "v_head_split"),
    }.get(stage, (stage,))


def _improvement_factor(
    off_error: float | None, exp_error: float | None
) -> float | str | None:
    if off_error is None or exp_error is None:
        return None
    if exp_error == 0 and off_error > 0:
        return float("inf")
    if off_error == 0:
        return 1.0 if exp_error == 0 else "worsened"
    return float(off_error / exp_error)


def _summarize_stage(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    left = [
        row["radlads_vs_qrwkv_off"]["max_abs_error"]
        for row in rows
        if row["radlads_vs_qrwkv_off"]["max_abs_error"] is not None
    ]
    exp = [
        row["radlads_vs_qrwkv_experimental"]["max_abs_error"]
        for row in rows
        if row["radlads_vs_qrwkv_experimental"]["max_abs_error"] is not None
    ]
    pair = [
        row["qrwkv_off_vs_qrwkv_experimental"]["max_abs_error"]
        for row in rows
        if row["qrwkv_off_vs_qrwkv_experimental"]["max_abs_error"] is not None
    ]
    if not left and not exp and not pair:
        return {"surface": stage, "status": "unavailable"}
    left_error = max(left) if left else None
    exp_error = max(exp) if exp else None
    return {
        "surface": stage,
        "radlads_vs_off_max_abs": left_error,
        "radlads_vs_experimental_max_abs": exp_error,
        "off_vs_experimental_max_abs": max(pair) if pair else None,
        "improvement_factor": _improvement_factor(left_error, exp_error),
        "direction": _direction(left_error, exp_error),
        "notes": _stage_note(stage, left_error, exp_error),
    }


def _fmt_float(value: Any) -> str:
    if value is None:
        return "unavailable"
    if value == float("inf"):
        return "infinite"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _update_divergence_verdict(report: Mapping[str, Any]) -> str:
    first_off = report.get("first_divergent_stage_off")
    first_exp = report.get("first_divergent_stage_experimental")
    if first_off == "decayed_state" and first_exp in {
        "update_outer_product",
        "balance_state_term",
        "composite_update_term",
        "final_update_term",
        "state_after",
    }:
        return (
            "A. decayed_state improved and update term is now exposed as next culprit"
        )
    if first_off == first_exp and first_off is not None:
        return "C. instrumentation labels shifted, not real parity movement"
    if first_off is None or first_exp is None:
        return "D. unknown"
    return "B. experimental changed state path but worsened RADLADS alignment"


def _stage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["comparison_label"]), []).append(row)

    def rows_for(
        stage: str, grouped: dict[str, list[dict[str, Any]]] = grouped
    ) -> list[dict[str, Any]]:
        for label in _stage_aliases(stage):
            if label in grouped:
                return grouped[label]
        return []

    stages = [
        "log_w",
        "logits",
        "shift_state",
        "wkv_matrix_state",
        "hidden_states",
        "state_before",
        "decayed_state",
        "k_for_update",
        "v_for_update",
        "update_outer_product",
        "balance_state_term",
        "composite_update_term",
        "final_update_term",
        "state_after",
        "state_after_exported",
    ]
    summary = {stage: _summarize_stage(stage, rows_for(stage)) for stage in stages}
    # keep the historical labels as aliases for source compatibility
    summary["composite_balance_update_term"] = _summarize_stage(
        "composite_balance_update_term",
        rows_for("balance_state_term"),
    )
    summary["state_after_from_full_source_formula"] = _summarize_stage(
        "state_after_from_full_source_formula",
        rows_for("final_update_term"),
    )
    summary["residual_after_composite_term"] = _summarize_stage(
        "residual_after_composite_term",
        rows_for("state_after"),
    )
    return summary


def _case_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    ordered_cases = [
        "tiny_no_mask",
        "tiny_attention_mask",
        "tiny_prefix_or_left_padding",
        "tiny_stepwise_state",
        "tiny_all_radlads_math_enabled",
    ]
    summaries = []
    for case in ordered_cases:
        case_rows = by_case.get(case, [])
        if not case_rows:
            continue
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in case_rows:
            grouped.setdefault(str(row["comparison_label"]), []).append(row)

        def rows_for(
            stage: str, grouped: dict[str, list[dict[str, Any]]] = grouped
        ) -> list[dict[str, Any]]:
            for label in _stage_aliases(stage):
                if label in grouped:
                    return grouped[label]
            return []

        stage_rows = [
            (stage, _summarize_stage(stage, rows_for(stage)))
            for stage in [
                "log_w",
                "logits",
                "shift_state",
                "wkv_matrix_state",
                "hidden_states",
                "state_before",
                "decayed_state",
                "k_for_update",
                "v_for_update",
                "update_outer_product",
                "balance_state_term",
                "composite_update_term",
                "final_update_term",
                "state_after",
                "state_after_exported",
            ]
        ]
        summaries.append(
            {
                "name": case,
                "status": _case_status(stage_rows),
                "first_divergent_stage_off": _first_stage(
                    stage_rows, "radlads_vs_off_max_abs"
                ),
                "first_divergent_stage_experimental": _first_stage(
                    stage_rows, "radlads_vs_experimental_max_abs"
                ),
                "stages": {stage: row for stage, row in stage_rows},
            }
        )
    return summaries


def _first_stage(rows: list[tuple[str, dict[str, Any]]], key: str) -> str | None:
    for stage, row in rows:
        value = row.get(key)
        if value is not None and value != 0.0:
            return stage
    return None


def _first_nonempty_case_stage(cases: list[dict[str, Any]], key: str) -> str | None:
    for case in cases:
        value = case.get(key)
        if value is not None:
            return value
    return None


def _case_status(rows: list[tuple[str, dict[str, Any]]]) -> str:
    improved = False
    worsened = False
    for _, row in rows:
        direction = row.get("direction")
        if direction == "improved":
            improved = True
        elif direction == "worsened":
            worsened = True
    if improved and worsened:
        return "mixed"
    if improved:
        return "improved"
    if worsened:
        return "worsened"
    if any(row.get("direction") == "neutral" for _, row in rows):
        return "neutral"
    return "unavailable"


def _verdict_three_way(cases: list[dict[str, Any]]) -> str:
    improved = False
    worsened = False
    for case in cases:
        for row in case["stages"].values():
            off = row.get("radlads_vs_off_max_abs")
            exp = row.get("radlads_vs_experimental_max_abs")
            if off is None or exp is None:
                continue
            if exp < off:
                improved = True
            elif exp > off:
                worsened = True
    if improved and worsened:
        return "mixed"
    if improved:
        return "yes"
    if worsened:
        return "no"
    return "unknown"


def _remaining_primary_gap(cases: list[dict[str, Any]]) -> str:
    priority = [
        "update_outer_product",
        "balance_state_term",
        "composite_update_term",
        "final_update_term",
        "state_after",
    ]
    for stage in priority:
        for case in cases:
            row = case["stages"].get(stage)
            if row is None:
                continue
            off = row.get("radlads_vs_off_max_abs")
            exp = row.get("radlads_vs_experimental_max_abs")
            if off is None or exp is None:
                continue
            if exp >= off:
                return stage
    return "state_after"


def _recommendation_three_way(
    experimental_closer: str, balance_state_helped: str, remaining_gap: str
) -> str:
    if experimental_closer == "yes":
        return "P67 promote/harden balance-state compatibility path"
    if experimental_closer == "mixed":
        return "P67 targeted k/v update_outer_product parity fix"
    if experimental_closer == "no":
        return "P67 targeted balance-state formula/shape fix"
    if remaining_gap in {"state_after", "final_update_term"}:
        return "P67 residual-impact / kernel-readiness gate"
    return "P67 Pallas prototype behind known-caveat flag"


def _trace_entry(
    *,
    side: str,
    case: str,
    layer: int | None,
    head: int | None,
    token_index: int | None,
    comparison_label: str,
    value: np.ndarray,
    capture_kind: str,
    source_path: str,
) -> dict[str, Any]:
    summary = summarize_array(comparison_label, value)
    return {
        "schema": BALANCE_STATE_TRACE_SCHEMA,
        "side": side,
        "mode": "experimental" if side == "qrwkv_experimental" else "off",
        "case": case,
        "layer": layer,
        "head": head,
        "token": token_index,
        "token_index": token_index,
        "stage": comparison_label,
        "comparison_label": comparison_label,
        "capture_kind": capture_kind,
        "source_path": source_path,
        "provenance": SOURCE_BACKING[side],
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()) if value.size else True,
        "max_abs": summary.abs_max,
        "mean_abs": float(np.mean(np.abs(value))) if value.size else 0.0,
        "sample": _sample(value),
        "array": value.tolist(),
    }


def _compare_three_way(
    *,
    traces: Mapping[str, list[dict[str, Any]]],
    source_paths: Mapping[str, str],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    by_side = {
        side: {_trace_key(row): row for row in rows} for side, rows in traces.items()
    }
    keys = sorted(set().union(*(set(rows) for rows in by_side.values())), key=str)
    rows = [_compare_key(key, by_side, atol=atol, rtol=rtol) for key in keys]
    first = next((row for row in rows if _row_status(row) != "pass"), None)
    off_errors = _pair_errors(rows, "radlads_vs_qrwkv_off")
    exp_errors = _pair_errors(rows, "radlads_vs_qrwkv_experimental")
    stage_summary = _stage_summary(rows)
    case_summaries = _case_summaries(rows)
    experimental_closer = _verdict_three_way(case_summaries)
    balance_state_helped = experimental_closer
    remaining_gap = _remaining_primary_gap(case_summaries)
    recommendation = _recommendation_three_way(
        experimental_closer, balance_state_helped, remaining_gap
    )
    first_off_stage = _first_nonempty_case_stage(
        case_summaries, "first_divergent_stage_off"
    )
    first_exp_stage = _first_nonempty_case_stage(
        case_summaries, "first_divergent_stage_experimental"
    )
    return {
        "schema": BALANCE_STATE_THREE_WAY_SCHEMA,
        "phase": "P66",
        "overall_status": "fail"
        if any(_row_status(row) != "pass" for row in rows)
        else "pass",
        "diagnostic_only": True,
        "strict_real_artifacts": True,
        "synthetic_fallback_used": False,
        "source_paths": dict(source_paths),
        "source_backing": SOURCE_BACKING,
        "provenance_caveat": (
            "RADLADS rows come from the P64 real paired hook artifact; "
            "QRWKV off/experimental rows come from P65 mode artifacts. The "
            "three-way report is an update-boundary diagnostic, not a default "
            "promotion or model-quality claim."
        ),
        "atol": atol,
        "rtol": rtol,
        "row_count": len(rows),
        "trace_counts": {side: len(items) for side, items in traces.items()},
        "first_divergent_case": None if first is None else first["case"],
        "first_divergent_layer": None if first is None else first["layer"],
        "first_divergent_head": None if first is None else first["head"],
        "first_divergent_token": None if first is None else first["token"],
        "first_divergent_stage": None if first is None else first["comparison_label"],
        "first_divergent_stage_off": first_off_stage,
        "first_divergent_stage_experimental": first_exp_stage,
        "max_radlads_vs_off_error": max(off_errors) if off_errors else None,
        "max_radlads_vs_experimental_error": max(exp_errors) if exp_errors else None,
        "recommendation": recommendation,
        "recommendations": [recommendation],
        "experimental_closer_to_radlads": experimental_closer,
        "balance_state_helped": balance_state_helped,
        "remaining_primary_gap": remaining_gap,
        "stage_summary": stage_summary,
        "cases": case_summaries,
        "default_behavior_preserved": True,
        "balance_state_mode_promoted": False,
        "p58_log_w_preserved": True,
        "kernel_ready": "no",
        "next_recommended_phase": (
            "Keep balance_state_mode experimental until same-provenance RADLADS "
            "vs off vs experimental artifacts exist and update-boundary rows pass."
        ),
        "rows": rows,
    }


def _compare_key(
    key: tuple[Any, ...],
    by_side: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    rad = by_side["radlads"].get(key)
    off = by_side["qrwkv_off"].get(key)
    exp = by_side["qrwkv_experimental"].get(key)
    return {
        "case": key[0],
        "layer": key[1],
        "head": key[2],
        "token": key[3],
        "comparison_label": key[4],
        "radlads_capture_kind": None if rad is None else rad["capture_kind"],
        "qrwkv_off_capture_kind": None if off is None else off["capture_kind"],
        "qrwkv_experimental_capture_kind": None if exp is None else exp["capture_kind"],
        "radlads_vs_qrwkv_off": _compare_pair(rad, off, atol=atol, rtol=rtol),
        "radlads_vs_qrwkv_experimental": _compare_pair(
            rad,
            exp,
            atol=atol,
            rtol=rtol,
        ),
        "qrwkv_off_vs_qrwkv_experimental": _compare_pair(
            off,
            exp,
            atol=atol,
            rtol=rtol,
        ),
    }


def _compare_pair(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "status": "unavailable",
            "shape_match": False,
            "finite_both": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "allclose": False,
        }
    left_array = np.asarray(left["array"], dtype=np.float32)
    right_array = np.asarray(right["array"], dtype=np.float32)
    shape_match = left_array.shape == right_array.shape
    finite = bool(np.isfinite(left_array).all() and np.isfinite(right_array).all())
    if not shape_match:
        return {
            "status": "shape_mismatch",
            "shape_match": False,
            "finite_both": finite,
            "max_abs_error": None,
            "mean_abs_error": None,
            "allclose": False,
        }
    diff = np.abs(left_array - right_array)
    allclose = bool(np.allclose(left_array, right_array, atol=atol, rtol=rtol))
    return {
        "status": "pass" if finite and allclose else "fail",
        "shape_match": shape_match,
        "finite_both": finite,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "allclose": allclose,
    }


def _row_status(row: Mapping[str, Any]) -> str:
    statuses = {
        row["radlads_vs_qrwkv_off"]["status"],
        row["radlads_vs_qrwkv_experimental"]["status"],
        row["qrwkv_off_vs_qrwkv_experimental"]["status"],
    }
    return "pass" if statuses == {"pass"} else "fail"


def _pair_errors(rows: Iterable[Mapping[str, Any]], pair_name: str) -> list[float]:
    return [
        row[pair_name]["max_abs_error"]
        for row in rows
        if row[pair_name]["max_abs_error"] is not None
    ]


def _write_reports(report: Mapping[str, Any], out_dir: Path) -> None:
    (out_dir / "three_way_parity_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # compatibility aliases requested by the phase spec
    (out_dir / "three_way_trace_radlads.jsonl").write_text(
        (out_dir / "radlads_update_boundary_trace.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "three_way_trace_qrwkv_off.jsonl").write_text(
        (out_dir / "qrwkv_off_update_boundary_trace.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "three_way_trace_qrwkv_experimental.jsonl").write_text(
        (out_dir / "qrwkv_experimental_update_boundary_trace.jsonl").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (out_dir / "THREE_WAY_PARITY.md").write_text(_three_way_markdown(report), "utf-8")
    (out_dir / "UPDATE_BOUNDARY_PARITY.md").write_text(
        _boundary_markdown(report),
        "utf-8",
    )
    (out_dir / "BALANCE_STATE_DECISION.md").write_text(
        _decision_markdown(report),
        "utf-8",
    )
    (out_dir / "P66_RESULTS.md").write_text(_results_markdown(report), "utf-8")


def _three_way_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P66 Three-Way Balance-State Parity",
        "",
        (
            "- experimental_closer_to_radlads: `"
            f"{report['experimental_closer_to_radlads']}`"
        ),
        f"- balance_state_helped: `{report['balance_state_helped']}`",
        f"- remaining_primary_gap: `{report['remaining_primary_gap']}`",
        f"- recommendation: `{report['recommendation']}`",
        "",
        (
            "| surface/stage | RADLADS vs off | RADLADS vs experimental | "
            "off vs experimental | direction | improvement_factor | notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage in [
        "log_w",
        "logits",
        "shift_state",
        "wkv_matrix_state",
        "hidden_states",
        "state_before",
        "decayed_state",
        "k_for_update",
        "v_for_update",
        "update_outer_product",
        "balance_state_term",
        "composite_update_term",
        "final_update_term",
        "state_after",
    ]:
        row = report["stage_summary"][stage]
        lines.append(
            f"| `{stage}` | `{_fmt_float(row.get('radlads_vs_off_max_abs'))}` | "
            f"`{_fmt_float(row.get('radlads_vs_experimental_max_abs'))}` | "
            f"`{_fmt_float(row.get('off_vs_experimental_max_abs'))}` | "
            f"`{row.get('direction')}` | "
            f"`{_fmt_float(row.get('improvement_factor'))}` | "
            f"{row.get('notes', '')} |"
        )
    lines.extend(
        [
            "",
            (
                "- first_divergent_stage_off: `"
                f"{report.get('first_divergent_stage_off')}`"
            ),
            (
                "- first_divergent_stage_experimental: `"
                f"{report.get('first_divergent_stage_experimental')}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _boundary_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# P66 Update-Boundary Parity", ""]
    for stage in [
        "k_for_update",
        "v_for_update",
        "update_outer_product",
        "balance_state_term",
        "composite_update_term",
        "final_update_term",
        "state_after",
    ]:
        row = report["stage_summary"][stage]
        lines.extend(
            [
                f"## {stage} parity",
                f"- RADLADS vs off: `{_fmt_float(row.get('radlads_vs_off_max_abs'))}`",
                (
                    "- RADLADS vs experimental: `"
                    f"{_fmt_float(row.get('radlads_vs_experimental_max_abs'))}`"
                ),
                f"- improvement/worsening: `{row.get('direction')}`",
                f"- source-backed interpretation: {row.get('notes', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## update divergence verdict",
            (
                "- Does experimental mode move the first divergence from "
                "decayed_state to update_outer_or_term?"
            ),
            f"  - `{_update_divergence_verdict(report)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _decision_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P66 Balance-State Decision",
            "",
            f"recommendation: {report['recommendation']}",
            "",
            "- Keep `radlads_balance_state` opt-in and experimental.",
            "- Do not promote balance-state mode to default from this evidence.",
            "- Do not loosen tolerances or claim model-quality parity.",
            "- Preserve P58 `log_w` behavior and existing off/default behavior.",
            "",
        ]
    )


def _results_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P66 Results",
            "",
            f"- Status: `{report['overall_status']}`",
            f"- Rows compared: `{report['row_count']}`",
            f"- First divergent stage: `{report['first_divergent_stage']}`",
            f"- Max RADLADS/off error: `{report['max_radlads_vs_off_error']}`",
            "- Max RADLADS/experimental error: "
            f"`{report['max_radlads_vs_experimental_error']}`",
            f"- recommendation: `{report['recommendation']}`",
            "",
        ]
    )


def _format_error(item: Mapping[str, Any]) -> str:
    if item["max_abs_error"] is None:
        return str(item["status"])
    return f"{item['status']} {item['max_abs_error']:.6g}"


def _array(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["array"], dtype=np.float32)


def _sample(value: np.ndarray) -> float | None:
    return None if value.size == 0 else float(value.reshape(-1)[0])


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("layer"),
        entry.get("head"),
        entry.get("token_index", entry.get("token")),
        entry.get("comparison_label"),
    )


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    token = entry.get("token_index", entry.get("token"))
    return (
        str(entry.get("case")),
        -1 if entry.get("layer") is None else int(entry["layer"]),
        -1 if entry.get("head") is None else int(entry["head"]),
        -1 if token is None else int(token),
        str(entry.get("comparison_label")),
    )


def _write_jsonl(path: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_case_name(name: str) -> str:
    return {
        "tiny_prefix_padding_or_left_padding": "tiny_prefix_or_left_padding",
    }.get(name, name)


run_balance_state_radlads_three_way = run_balance_state_three_way
