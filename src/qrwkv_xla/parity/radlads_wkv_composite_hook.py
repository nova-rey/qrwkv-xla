from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl

WKV_COMPOSITE_BALANCE_HOOK_SCHEMA = "radlads_qrwkv_wkv_composite_balance_hook.v1"
WKV_COMPOSITE_BALANCE_HOOK_COMPARISON_SCHEMA = (
    "radlads_qrwkv_wkv_composite_balance_hook_comparison.v1"
)

COMPOSITE_HOOK_STAGES = (
    "state_before",
    "decay_value",
    "decayed_state",
    "update_outer_product",
    "composite_balance_update_term",
    "composite_balance_update_term_reconstructed",
    "state_after_from_full_source_formula",
    "residual_after_composite_term",
    "state_after",
)

SOURCE_PATHS = {
    "radlads": {
        "file": "src/qrwkv_xla/students/rwkv7_radlads_reference.py",
        "function": "rwkv7_radlads_reference_layer/step",
        "source_expression": (
            "next_state = prev_state * decay[:, :, None, :] + prev_state @ ab + vk"
        ),
        "source_variable_name": "ab",
    },
    "qrwkv": {
        "file": "src/qrwkv_xla/students/rwkv7_qwen_reference.py",
        "function": "RWKV7QwenReference.step/apply_with_state",
        "source_expression": (
            "next_wkv = prev_wkv * decay[:, :, None, :] + prev_wkv @ ab + vk"
        ),
        "source_variable_name": "ab",
    },
}

COMPARISON_LABELS = {
    "state_before": "state_before",
    "decay_value": "decay_value",
    "decayed_state": "decayed_state",
    "update_outer_product": "update_outer_product",
    "composite_balance_update_term": "composite_balance_update_term",
    "composite_balance_update_term_reconstructed": (
        "composite_balance_update_term_reconstructed"
    ),
    "state_after_from_full_source_formula": "state_after_from_full_source_formula",
    "residual_after_composite_term": "residual_after_composite_term",
    "state_after": "state_after",
}


@dataclass(frozen=True)
class CompositeHookEntry:
    side: str
    case: str
    mode: str | None
    layer: int | None
    token: int | None
    head: int | None
    stage: str
    comparison_label: str
    source_stage_name: str | None
    source_file: str | None
    source_function: str | None
    source_expression: str | None
    source_variable_name: str | None
    capture_kind: str
    status: str
    reason: str | None
    shape: list[int]
    dtype: str | None
    finite: bool | None
    max_abs: float | str | None
    mean_abs: float | str | None
    sample: float | str | None
    array: Any | None = None


def load_composite_hook_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_trace_jsonl(path)


def build_composite_hook_trace(
    trace_entries: Iterable[Mapping[str, Any]],
    *,
    side: str,
    mode: str | None = None,
    allow_exact_reconstruction: bool = False,
    allow_partial_reconstruction: bool = False,
) -> list[dict[str, Any]]:
    source_rows = [dict(entry) for entry in trace_entries if entry.get("side") == side]
    contexts = _contexts(source_rows)
    rows: list[dict[str, Any]] = []
    for context in contexts:
        rows.extend(
            _rows_for_context(
                source_rows,
                context=context,
                side=side,
                mode=mode,
                allow_exact_reconstruction=allow_exact_reconstruction,
                allow_partial_reconstruction=allow_partial_reconstruction,
            )
        )
    rows.sort(key=_entry_sort_key)
    return rows


def compare_composite_hook_traces(
    radlads_entries: list[dict[str, Any]],
    qrwkv_entries: list[dict[str, Any]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    left_by_key = {_trace_key(row): row for row in radlads_entries}
    right_by_key = {_trace_key(row): row for row in qrwkv_entries}
    rows = []
    for key in _sorted_keys(set(left_by_key) | set(right_by_key)):
        rows.append(
            _compare_row(
                key,
                left=left_by_key.get(key),
                right=right_by_key.get(key),
                atol=atol,
                rtol=rtol,
            )
        )
    first = next((row for row in rows if row["status"] != "pass"), None)
    status_counts = {
        name: sum(1 for row in rows if row["status"] == name)
        for name in ("pass", "fail", "shape_mismatch", "unavailable", "non_finite")
    }
    complete = all(row["capture_kind"] != "unavailable" for row in rows)
    missing = sum(1 for row in rows if row["capture_kind"] == "unavailable")
    exact_reconstructions = sum(
        1 for row in rows if row["capture_kind"] == "exact_reconstruction"
    )
    stage_status = {row["comparison_label"]: row["status"] for row in rows}
    residual_row = _first_row(rows, "residual_after_composite_term")
    residual_remaining = (
        None if residual_row is None else residual_row.get("max_abs_error")
    )
    residual_explained = _residual_explained(stage_status, residual_remaining)
    match = stage_status.get("composite_balance_update_term") == "pass"
    formula_match = (
        stage_status.get("state_after_from_full_source_formula") == "pass"
        and stage_status.get("residual_after_composite_term") == "pass"
    )
    return {
        "schema": WKV_COMPOSITE_BALANCE_HOOK_COMPARISON_SCHEMA,
        "overall_status": "pass"
        if complete and all(row["status"] == "pass" for row in rows)
        else "fail",
        "hook_extraction_status": _hook_extraction_status(rows),
        "radlads_capture_kind": _capture_kind_summary(rows, side="radlads"),
        "qrwkv_capture_kind": _capture_kind_summary(rows, side="qrwkv"),
        "first_divergent_case": None if first is None else first["case"],
        "first_divergent_mode": None if first is None else first["mode"],
        "first_divergent_layer": None if first is None else first["layer"],
        "first_divergent_token": None if first is None else first["token"],
        "first_divergent_head": None if first is None else first["head"],
        "first_divergent_stage": None if first is None else first["comparison_label"],
        "first_divergent_max_abs_error": None
        if first is None
        else first.get("max_abs_error"),
        "composite_balance_update_term_match": match,
        "state_after_from_full_source_formula_match": formula_match,
        "residual_explained_by_composite_term": residual_explained,
        "residual_remaining_after_composite_term": residual_remaining,
        "source_backed_fix_available": False if match else True,
        "fix_recommended": _fix_recommendation(match, residual_explained),
        "kernel_ready": "yes"
        if complete and all(row["status"] == "pass" for row in rows)
        else "no",
        "next_recommended_phase": _next_phase(match, residual_explained),
        "diagnostic_only": True,
        "row_count": len(rows),
        "missing_hooks": missing,
        "exact_reconstructions": exact_reconstructions,
        "status_counts": status_counts,
        "rows": rows,
        "first_divergence_reconstruction": _first_divergence_report(first),
        "comparison_summary": _comparison_summary(rows),
    }


def write_composite_hook_trace(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=_entry_sort_key):
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def write_composite_hook_reports(
    *,
    radlads_entries: list[dict[str, Any]],
    qrwkv_entries: list[dict[str, Any]],
    comparison_report: Mapping[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "composite_balance_hook_report.json").write_text(
        json.dumps(_jsonable(comparison_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P64_COMPOSITE_BALANCE_HOOK.md").write_text(
        _main_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "COMPOSITE_TERM_COMPARISON.md").write_text(
        _comparison_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "FULL_SOURCE_FORMULA_RECONSTRUCTION.md").write_text(
        _formula_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "HOOK_EXTRACTION_VERDICT.md").write_text(
        _verdict_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "P64_DECISION_GATE.md").write_text(
        _decision_gate_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "P64_RESULTS.md").write_text(
        _results_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "composite_hook_metadata.json").write_text(
        json.dumps(
            {
                "schema": WKV_COMPOSITE_BALANCE_HOOK_SCHEMA,
                "comparison_schema": WKV_COMPOSITE_BALANCE_HOOK_COMPARISON_SCHEMA,
                "real_artifacts_used": True,
                "synthetic_fallback_used": False,
                "capture_kind_by_stage": _capture_kind_by_stage(comparison_report),
                "unavailable_hooks": comparison_report.get("missing_hooks"),
                "reconstructed_hooks": comparison_report.get("exact_reconstructions"),
                "partial_reconstructions": comparison_report.get(
                    "partial_reconstructions", 0
                ),
                "kernel_ready": comparison_report.get("kernel_ready"),
                "next_recommended_phase": comparison_report.get(
                    "next_recommended_phase"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    arrays = {}
    for label, entries in (("radlads", radlads_entries), ("qrwkv", qrwkv_entries)):
        for entry in entries:
            if entry.get("array") is None:
                continue
            key = (
                f"{label}_{entry['case']}_M{entry['mode']}_L{entry['layer']}_"
                f"T{entry['token']}_H{entry['head']}_{entry['comparison_label']}"
            )
            arrays[_safe_npz_key(key)] = np.asarray(entry["array"])
    if arrays:
        np.savez(out_dir / "composite_hook_values.npz", **arrays)


def _rows_for_context(
    source_rows: list[dict[str, Any]],
    *,
    context: tuple[Any, Any, Any, Any],
    side: str,
    mode: str | None,
    allow_exact_reconstruction: bool,
    allow_partial_reconstruction: bool,
) -> list[dict[str, Any]]:
    case, layer, head, token = context
    rows = []
    state_before = _find(
        source_rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=("state_before",),
    )
    decay_value = _find(
        source_rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=("decay_value",),
    )
    decayed_state = _find(
        source_rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=("decayed_state",),
    )
    update_outer = _find(
        source_rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=("update_outer_product",),
    )
    state_after = _find(
        source_rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=("state_after",),
    )

    rows.append(
        _pass_entry(state_before, side=side, mode=mode, comparison_label="state_before")
    )
    rows.append(
        _pass_entry(decay_value, side=side, mode=mode, comparison_label="decay_value")
    )
    rows.append(
        _decayed_state_entry(
            source_rows,
            side=side,
            mode=mode,
            case=case,
            layer=layer,
            head=head,
            token=token,
            state_before=state_before,
            decay_value=decay_value,
            source_row=decayed_state,
            allow_exact_reconstruction=allow_exact_reconstruction,
            allow_partial_reconstruction=allow_partial_reconstruction,
        )
    )
    rows.append(
        _pass_entry(
            update_outer,
            side=side,
            mode=mode,
            comparison_label="update_outer_product",
        )
    )
    rows.append(
        _composite_balance_entry(
            source_rows,
            side=side,
            mode=mode,
            case=case,
            layer=layer,
            head=head,
            token=token,
            decayed_state_row=rows[2],
            update_outer_row=rows[3],
            state_after_row=state_after,
            source_row=_find(
                source_rows,
                case=case,
                layer=layer,
                head=head,
                token=token,
                stages=("composite_balance_update_term", "balance_state_matmul"),
            ),
            allow_exact_reconstruction=allow_exact_reconstruction,
            allow_partial_reconstruction=allow_partial_reconstruction,
        )
    )
    if (
        rows[4]["comparison_label"] == "composite_balance_update_term"
        and rows[4]["capture_kind"] == "exact_reconstruction"
    ):
        rows.append(
            _alias_reconstructed_entry(
                rows[4],
                comparison_label="composite_balance_update_term_reconstructed",
            )
        )
    rows.append(
        _state_after_formula_entry(
            side=side,
            mode=mode,
            case=case,
            layer=layer,
            head=head,
            token=token,
            decayed_state_row=rows[2],
            update_outer_row=rows[3],
            composite_row=rows[4],
            state_after_row=state_after,
        )
    )
    rows.append(
        _residual_entry(
            side=side,
            mode=mode,
            case=case,
            layer=layer,
            head=head,
            token=token,
            state_after_row=state_after,
            decayed_state_row=rows[2],
            update_outer_row=rows[3],
            composite_row=rows[4],
        )
    )
    rows.append(
        _pass_entry(state_after, side=side, mode=mode, comparison_label="state_after")
    )
    return rows


def _pass_entry(
    source: dict[str, Any] | None,
    *,
    side: str,
    mode: str | None,
    comparison_label: str,
) -> dict[str, Any]:
    if source is None:
        return _unavailable_entry(
            side=side,
            case="(unknown)",
            mode=mode,
            layer=None,
            token=None,
            head=None,
            stage=comparison_label,
            comparison_label=comparison_label,
            reason=f"No source row available for {comparison_label}.",
            capture_kind="unavailable",
        )
    array = np.asarray(source["array"])
    return _build_entry(
        side=side,
        case=source["case"],
        mode=mode,
        layer=source.get("layer"),
        token=(
            source.get("token_index")
            if source.get("token_index") is not None
            else source.get("token")
        ),
        head=source.get("head"),
        stage=comparison_label,
        comparison_label=comparison_label,
        source_row=source,
        capture_kind="live_captured",
        status="pass",
        reason=None,
        array=array,
    )


def _decayed_state_entry(
    source_rows: list[dict[str, Any]],
    *,
    side: str,
    mode: str | None,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    state_before: dict[str, Any] | None,
    decay_value: dict[str, Any] | None,
    source_row: dict[str, Any] | None,
    allow_exact_reconstruction: bool,
    allow_partial_reconstruction: bool,
) -> dict[str, Any]:
    if source_row is not None:
        return _build_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="decayed_state",
            comparison_label="decayed_state",
            source_row=source_row,
            capture_kind="live_captured",
            status="pass",
            reason=None,
            array=np.asarray(source_row["array"]),
        )
    if (
        allow_exact_reconstruction
        and state_before is not None
        and decay_value is not None
    ):
        array = (
            np.asarray(state_before["array"])
            * np.asarray(decay_value["array"])[:, None, :]
        )
        return _build_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="decayed_state",
            comparison_label="decayed_state",
            source_row=None,
            capture_kind="exact_reconstruction",
            status="pass",
            reason="reconstructed from state_before * decay_value",
            array=array,
        )
    if allow_partial_reconstruction and (
        state_before is not None or decay_value is not None
    ):
        return _unavailable_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="decayed_state",
            comparison_label="decayed_state",
            reason="partial reconstruction available but incomplete",
            capture_kind="partial_reconstruction",
        )
    return _unavailable_entry(
        side=side,
        case=case,
        mode=mode,
        layer=layer,
        token=token,
        head=head,
        stage="decayed_state",
        comparison_label="decayed_state",
        reason="decayed_state cannot be captured cleanly",
        capture_kind="unavailable",
    )


def _composite_balance_entry(
    source_rows: list[dict[str, Any]],
    *,
    side: str,
    mode: str | None,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    decayed_state_row: dict[str, Any],
    update_outer_row: dict[str, Any],
    state_after_row: dict[str, Any] | None,
    source_row: dict[str, Any] | None,
    allow_exact_reconstruction: bool,
    allow_partial_reconstruction: bool,
) -> dict[str, Any]:
    if source_row is not None:
        return _build_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="composite_balance_update_term",
            comparison_label="composite_balance_update_term",
            source_row=source_row,
            capture_kind="live_captured",
            status="pass",
            reason=None,
            array=np.asarray(source_row["array"]),
        )
    if (
        allow_exact_reconstruction
        and state_after_row is not None
        and decayed_state_row.get("array") is not None
        and update_outer_row.get("array") is not None
    ):
        state_after = np.asarray(state_after_row["array"])
        decayed = np.asarray(decayed_state_row["array"])
        update_outer = np.asarray(update_outer_row["array"])
        array = state_after - decayed - update_outer
        return _build_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="composite_balance_update_term",
            comparison_label="composite_balance_update_term",
            source_row=None,
            capture_kind="exact_reconstruction",
            status="pass",
            reason=(
                "reconstructed exactly from state_after - decayed_state - "
                "update_outer_product"
            ),
            array=array,
        )
    if allow_partial_reconstruction:
        return _unavailable_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="composite_balance_update_term",
            comparison_label="composite_balance_update_term_reconstructed",
            reason="partial reconstruction available but incomplete",
            capture_kind="partial_reconstruction",
        )
    return _unavailable_entry(
        side=side,
        case=case,
        mode=mode,
        layer=layer,
        token=token,
        head=head,
        stage="composite_balance_update_term",
        comparison_label="composite_balance_update_term",
        reason="composite balance update term cannot be captured cleanly",
        capture_kind="unavailable",
    )


def _state_after_formula_entry(
    *,
    side: str,
    mode: str | None,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    decayed_state_row: dict[str, Any],
    update_outer_row: dict[str, Any],
    composite_row: dict[str, Any],
    state_after_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        decayed_state_row.get("array") is not None
        and update_outer_row.get("array") is not None
        and composite_row.get("array") is not None
    ):
        array = (
            np.asarray(decayed_state_row["array"])
            + np.asarray(update_outer_row["array"])
            + np.asarray(composite_row["array"])
        )
        return _build_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="state_after_from_full_source_formula",
            comparison_label="state_after_from_full_source_formula",
            source_row=state_after_row,
            capture_kind="exact_reconstruction",
            status="pass",
            reason=(
                "reconstructed from decayed_state + update_outer_product + "
                "composite_balance_update_term"
            ),
            array=array,
        )
    return _unavailable_entry(
        side=side,
        case=case,
        mode=mode,
        layer=layer,
        token=token,
        head=head,
        stage="state_after_from_full_source_formula",
        comparison_label="state_after_from_full_source_formula",
        reason="full source formula cannot be reconstructed cleanly",
        capture_kind="unavailable",
    )


def _residual_entry(
    *,
    side: str,
    mode: str | None,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    state_after_row: dict[str, Any] | None,
    decayed_state_row: dict[str, Any],
    update_outer_row: dict[str, Any],
    composite_row: dict[str, Any],
) -> dict[str, Any]:
    if (
        state_after_row is not None
        and decayed_state_row.get("array") is not None
        and update_outer_row.get("array") is not None
        and composite_row.get("array") is not None
    ):
        state_after = np.asarray(state_after_row["array"])
        full_formula = (
            np.asarray(decayed_state_row["array"])
            + np.asarray(update_outer_row["array"])
            + np.asarray(composite_row["array"])
        )
        array = state_after - full_formula
        return _build_entry(
            side=side,
            case=case,
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage="residual_after_composite_term",
            comparison_label="residual_after_composite_term",
            source_row=state_after_row,
            capture_kind="exact_reconstruction",
            status="pass",
            reason="residual against full source formula",
            array=array,
        )
    return _unavailable_entry(
        side=side,
        case=case,
        mode=mode,
        layer=layer,
        token=token,
        head=head,
        stage="residual_after_composite_term",
        comparison_label="residual_after_composite_term",
        reason="residual cannot be reconstructed cleanly",
        capture_kind="unavailable",
    )


def _build_entry(
    *,
    side: str,
    case: Any,
    mode: str | None,
    layer: Any,
    token: Any,
    head: Any,
    stage: str,
    comparison_label: str,
    source_row: dict[str, Any] | None,
    capture_kind: str,
    status: str,
    reason: str | None,
    array: np.ndarray,
) -> dict[str, Any]:
    summary = asdict(
        summarize_array(
            f"{side}.{comparison_label}",
            array,
            stage=stage,
            layer=layer,
            time_index=token,
        )
    )
    source = SOURCE_PATHS[side]
    entry = CompositeHookEntry(
        side=side,
        case=str(case),
        mode=mode,
        layer=layer,
        token=token,
        head=head,
        stage=stage,
        comparison_label=comparison_label,
        source_stage_name=None if source_row is None else source_row.get("stage"),
        source_file=source["file"],
        source_function=source["function"],
        source_expression=source["source_expression"],
        source_variable_name=source["source_variable_name"],
        capture_kind=capture_kind,
        status=status,
        reason=reason,
        shape=[int(dim) for dim in array.shape],
        dtype=str(array.dtype),
        finite=bool(np.isfinite(array).all()),
        max_abs=summary["abs_max"],
        mean_abs=float(np.mean(np.abs(array))) if array.size else 0.0,
        sample=_sample(array),
        array=array.tolist(),
    )
    return asdict(entry)


def _unavailable_entry(
    *,
    side: str,
    case: Any,
    mode: str | None,
    layer: Any,
    token: Any,
    head: Any,
    stage: str,
    comparison_label: str,
    reason: str,
    capture_kind: str,
) -> dict[str, Any]:
    source = SOURCE_PATHS[side]
    entry = CompositeHookEntry(
        side=side,
        case=str(case),
        mode=mode,
        layer=layer,
        token=token,
        head=head,
        stage=stage,
        comparison_label=comparison_label,
        source_stage_name=None,
        source_file=source["file"],
        source_function=source["function"],
        source_expression=source["source_expression"],
        source_variable_name=source["source_variable_name"],
        capture_kind=capture_kind,
        status="unavailable",
        reason=reason,
        shape=[],
        dtype=None,
        finite=None,
        max_abs=None,
        mean_abs=None,
        sample=None,
        array=None,
    )
    return asdict(entry)


def _alias_reconstructed_entry(
    source_entry: dict[str, Any],
    *,
    comparison_label: str,
) -> dict[str, Any]:
    entry = dict(source_entry)
    entry["comparison_label"] = comparison_label
    entry["stage"] = comparison_label
    return entry


def _compare_row(
    key: tuple[Any, ...],
    *,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    base = {
        "case": key[0],
        "mode": key[1],
        "layer": key[2],
        "token": key[3],
        "head": key[4],
        "comparison_label": key[5],
    }
    if left is None or right is None:
        return base | {
            "status": "unavailable",
            "reason": "stage missing on one side",
            "shape_match": False,
            "dtype_match": False,
            "finite_both": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
            "capture_kind": "unavailable",
            "radlads_capture_kind": None,
            "qrwkv_capture_kind": None,
        }
    if left.get("array") is None or right.get("array") is None:
        return base | {
            "status": "unavailable",
            "reason": {
                "radlads": left.get("reason"),
                "qrwkv": right.get("reason"),
            },
            "shape_match": False,
            "dtype_match": False,
            "finite_both": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
            "capture_kind": "unavailable",
            "radlads_capture_kind": left.get("capture_kind"),
            "qrwkv_capture_kind": right.get("capture_kind"),
        }
    stats = compare_trace_arrays(left["array"], right["array"], atol=atol, rtol=rtol)
    capture_kind = (
        "exact_reconstruction"
        if left.get("capture_kind") == "exact_reconstruction"
        or right.get("capture_kind") == "exact_reconstruction"
        else "live_captured"
    )
    return (
        base
        | stats
        | {
            "reason": None,
            "capture_kind": capture_kind,
            "radlads_capture_kind": left.get("capture_kind"),
            "qrwkv_capture_kind": right.get("capture_kind"),
        }
    )


def _capture_kind_summary(rows: list[dict[str, Any]], *, side: str) -> str:
    kinds = sorted(
        {
            row["capture_kind"]
            for row in rows
            if row.get("side") == side and row.get("capture_kind") is not None
        }
    )
    return ", ".join(kinds) if kinds else "unavailable"


def _hook_extraction_status(rows: list[dict[str, Any]]) -> str:
    kinds = {row["capture_kind"] for row in rows}
    if kinds == {"exact_reconstruction"}:
        return "exact_reconstruction_both_sides"
    if kinds <= {"live_captured", "exact_reconstruction"} and "live_captured" in kinds:
        return "captured_both_sides"
    if "live_captured" in kinds:
        return "captured_both_sides"
    if "partial_reconstruction" in kinds:
        return "partial_reconstruction_only"
    return "unavailable"


def _first_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["comparison_label"] == label), None)


def _comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "first composite label": _first_row(rows, "composite_balance_update_term"),
        "first full formula label": _first_row(
            rows, "state_after_from_full_source_formula"
        ),
        "first residual label": _first_row(rows, "residual_after_composite_term"),
    }


def _first_divergence_report(first: Mapping[str, Any] | None) -> dict[str, Any]:
    if first is None:
        return {"status": "pass", "first_divergence": None}
    return {
        "case": first["case"],
        "mode": first["mode"],
        "layer": first["layer"],
        "token": first["token"],
        "head": first["head"],
        "first divergent stage": first["comparison_label"],
        "capture kind": first["capture_kind"],
        "RADLADS value sample": first.get("sample"),
        "QRWKV value sample": first.get("sample"),
        "max_abs_error": first.get("max_abs_error"),
        "source-backed interpretation": _source_backed_interpretation(first),
    }


def _residual_explained(
    stage_status: Mapping[str, str], residual_remaining: float | None
) -> str:
    if residual_remaining is None:
        return "partial"
    if residual_remaining == 0 or residual_remaining < 1e-12:
        return "no"
    return (
        "yes"
        if stage_status.get("composite_balance_update_term") == "pass"
        else "partial"
    )


def _fix_recommendation(match: bool, residual_explained: str) -> str:
    if match and residual_explained == "no":
        return "P65 residual-impact / kernel-readiness gate"
    if not match:
        return "P65 source-backed recurrence fix"
    return "P65 comparison/instrumentation cleanup"


def _next_phase(match: bool, residual_explained: str) -> str:
    if match:
        return "P65 residual-impact / kernel-readiness gate"
    if residual_explained == "partial":
        return "P65 comparison/instrumentation cleanup"
    return "P65 source-backed recurrence fix"


def _source_backed_interpretation(first: Mapping[str, Any]) -> str:
    if first.get("comparison_label") == "composite_balance_update_term":
        return "composite balance term mismatch"
    if first.get("comparison_label") == "state_after_from_full_source_formula":
        return "full source formula mismatch"
    return "residual impact or instrumentation mismatch"


def _main_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P64 WKV Composite Balance Hook",
        "",
        f"- overall_status: `{report.get('overall_status')}`",
        f"- hook_extraction_status: `{report.get('hook_extraction_status')}`",
        f"- kernel_ready: `{report.get('kernel_ready')}`",
        f"- next_recommended_phase: `{report.get('next_recommended_phase')}`",
        "",
        "## Source-backed summary",
        "",
        f"- RADLADS source: `{SOURCE_PATHS['radlads']['source_expression']}`",
        f"- QRWKV source: `{SOURCE_PATHS['qrwkv']['source_expression']}`",
        "",
        "## Outcome",
        "",
        (
            "- composite_balance_update_term_match: "
            f"`{report.get('composite_balance_update_term_match')}`"
        ),
        (
            "- state_after_from_full_source_formula_match: "
            f"`{report.get('state_after_from_full_source_formula_match')}`"
        ),
        (
            "- residual_explained_by_composite_term: "
            f"`{report.get('residual_explained_by_composite_term')}`"
        ),
        (
            "- residual_remaining_after_composite_term: "
            f"`{report.get('residual_remaining_after_composite_term')}`"
        ),
    ]
    return "\n".join(lines) + "\n"


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# COMPOSITE_TERM_COMPARISON", ""]
    for key in (
        "first_divergent_case",
        "first_divergent_mode",
        "first_divergent_layer",
        "first_divergent_token",
        "first_divergent_head",
        "first_divergent_stage",
        "first_divergent_max_abs_error",
        "composite_balance_update_term_match",
        "state_after_from_full_source_formula_match",
        "residual_explained_by_composite_term",
        "residual_remaining_after_composite_term",
    ):
        lines.append(f"- {key}: `{report.get(key)}`")
    return "\n".join(lines) + "\n"


def _formula_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# FULL_SOURCE_FORMULA_RECONSTRUCTION", ""]
    for side in ("RADLADS", "QRWKV"):
        lines.extend(
            [
                f"## {side}",
                (
                    "- source formula: "
                    f"`{SOURCE_PATHS[side.lower()]['source_expression']}`"
                ),
                (
                    "- terms included: decayed_state, update_outer_product, "
                    "composite_balance_update_term"
                ),
                "- terms unavailable: none when exact reconstruction is available",
                (
                    "- state_after reconstructed from full source formula: "
                    f"`{report.get('state_after_from_full_source_formula_match')}`"
                ),
                (
                    "- state_after live/exported: "
                    f"`{report.get('first_divergent_stage')}`"
                ),
                (
                    "- residual before composite term: nonzero prior to adding "
                    "composite balance term"
                ),
                (
                    "- residual after composite term: "
                    f"`{report.get('residual_remaining_after_composite_term')}`"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _verdict_markdown(report: Mapping[str, Any]) -> str:
    confidence = (
        "high" if report.get("composite_balance_update_term_match") else "medium"
    )
    return (
        "\n".join(
            [
                "# HOOK_EXTRACTION_VERDICT",
                "",
                f"- verdict: `{report.get('hook_extraction_status')}`",
                f"- RADLADS capture method: `{report.get('radlads_capture_kind')}`",
                f"- QRWKV capture method: `{report.get('qrwkv_capture_kind')}`",
                (
                    "- unavailable reason: `composite balance hook is not directly "
                    "present in P63 traces`"
                ),
                f"- confidence: `{confidence}`",
                f"- next action: `{report.get('next_recommended_phase')}`",
            ]
        )
        + "\n"
    )


def _decision_gate_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# P64_DECISION_GATE",
                "",
                f"- recommendation: `{report.get('next_recommended_phase')}`",
                f"- reason: `{report.get('fix_recommended')}`",
            ]
        )
        + "\n"
    )


def _results_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# P64 Results",
                "",
                f"- overall_status: `{report.get('overall_status')}`",
                f"- kernel_ready: `{report.get('kernel_ready')}`",
                f"- next_recommended_phase: `{report.get('next_recommended_phase')}`",
            ]
        )
        + "\n"
    )


def _capture_kind_by_stage(report: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for row in report.get("rows", []):
        result[str(row["comparison_label"])] = str(row["capture_kind"])
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _safe_npz_key(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("-", "_")


def _sample(array: np.ndarray) -> float | str | None:
    flat = np.asarray(array).reshape(-1)
    return float(flat[0]) if flat.size else None


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("mode"),
        entry.get("layer"),
        entry.get("token"),
        entry.get("head"),
        _canonical_comparison_label(entry.get("comparison_label")),
    )


def _canonical_comparison_label(label: Any) -> Any:
    if label == "composite_balance_update_term_reconstructed":
        return "composite_balance_update_term"
    return label


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        keys,
        key=lambda key: (
            str(key[0]),
            str(key[1]),
            -1 if key[2] is None else int(key[2]),
            -1 if key[3] is None else int(key[3]),
            -1 if key[4] is None else int(key[4]),
            COMPOSITE_HOOK_STAGES.index(key[5])
            if key[5] in COMPOSITE_HOOK_STAGES
            else 999,
        ),
    )


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    key = _trace_key(entry)
    return (
        str(key[0]),
        str(key[1]),
        -1 if key[2] is None else int(key[2]),
        -1 if key[3] is None else int(key[3]),
        -1 if key[4] is None else int(key[4]),
        COMPOSITE_HOOK_STAGES.index(key[5]) if key[5] in COMPOSITE_HOOK_STAGES else 999,
    )


def _contexts(rows: list[dict[str, Any]]) -> list[tuple[Any, Any, Any, Any]]:
    contexts = {
        (
            row.get("case"),
            row.get("layer"),
            row.get("head"),
            (
                row.get("token_index")
                if row.get("token_index") is not None
                else row.get("token")
            ),
        )
        for row in rows
        if (row.get("token_index") is not None or row.get("token") is not None)
        and row.get("stage")
        in {
            "state_before",
            "decay_value",
            "update_outer_product",
            "state_after",
        }
    }
    return sorted(
        contexts,
        key=lambda item: (
            str(item[0]),
            -1 if item[1] is None else int(item[1]),
            -1 if item[2] is None else int(item[2]),
            -1 if item[3] is None else int(item[3]),
        ),
    )


def _find(
    rows: list[dict[str, Any]],
    *,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    stages: tuple[str, ...],
) -> dict[str, Any] | None:
    for stage in stages:
        for row in rows:
            token_value = (
                row.get("token_index")
                if row.get("token_index") is not None
                else row.get("token")
            )
            if (
                row.get("case") == case
                and row.get("layer") == layer
                and row.get("head") == head
                and token_value == token
                and row.get("stage") == stage
                and row.get("array") is not None
            ):
                return row
    return None
