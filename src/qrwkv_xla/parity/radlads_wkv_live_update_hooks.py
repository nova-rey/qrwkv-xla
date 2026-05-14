from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl

WKV_LIVE_UPDATE_HOOK_SCHEMA = "radlads_qrwkv_wkv_live_update_hooks.v1"
WKV_LIVE_UPDATE_HOOK_COMPARISON_SCHEMA = (
    "radlads_qrwkv_wkv_live_update_hooks_comparison.v1"
)

LIVE_UPDATE_STAGES = (
    "state_before",
    "decay_value",
    "decayed_state",
    "k_for_update",
    "v_for_update",
    "update_outer_product",
    "balance_state_matmul",
    "composite_balance_update_term",
    "composite_update_term",
    "update_term",
    "state_after",
    "state_after_for_next_token",
    "state_after_exported",
)

SOURCE_STAGE_MAP = {
    "state_before": ("wkv_state_before",),
    "decay_value": ("decay_after_transform",),
    "decayed_state": ("wkv_decay_applied",),
    "k_for_update": ("k_a", "k", "k_projection", "k_head_split"),
    "v_for_update": ("v", "v_projection", "value_after_v_first_mix", "v_head_split"),
    "update_outer_product": ("wkv_update_outer_or_term",),
    "balance_state_matmul": (
        "wkv_balance_state_matmul",
        "balance_state_matmul",
        "composite_balance_update_term",
        "wkv_composite_balance_update_term",
    ),
    "composite_balance_update_term": (
        "composite_balance_update_term",
        "wkv_composite_balance_update_term",
        "wkv_balance_state_matmul",
        "balance_state_matmul",
    ),
    "composite_update_term": (
        "wkv_composite_update_term",
        "composite_update_term",
    ),
    "update_term": ("wkv_update_outer_or_term",),
    "state_after": ("wkv_state_after",),
    "state_after_for_next_token": ("wkv_state_before",),
    "state_after_exported": ("returned_wkv_matrix_state",),
}

SOURCE_PATHS = {
    "radlads": {
        "file": "src/qrwkv_xla/students/rwkv7_radlads_reference.py",
        "function": "RWKV7RadladsReference.step/apply_with_state",
    },
    "qrwkv": {
        "file": "src/qrwkv_xla/students/rwkv7_qwen_reference.py",
        "function": "RWKV7QwenReference.step/apply_with_state",
    },
}


@dataclass(frozen=True)
class WKVLiveUpdateHookEntry:
    side: str
    case: str
    mode: str | None
    layer: int | None
    token: int | None
    head: int | None
    stage: str
    source_stage_name: str | None
    source_file: str | None
    source_function: str | None
    shape: list[int]
    dtype: str | None
    finite: bool | None
    max_abs: float | str | None
    sample: float | str | None
    status: str
    reason: str | None
    capture_kind: str
    array: Any | None = None


def load_live_update_hook_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_trace_jsonl(path)


def build_live_update_hook_trace(
    trace_entries: Iterable[Mapping[str, Any]],
    *,
    side: str,
    mode: str | None = None,
    allow_reconstructed: bool = False,
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
                allow_reconstructed=allow_reconstructed,
            )
        )
    rows.sort(key=_entry_sort_key)
    return rows


def compare_live_update_hook_traces(
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
        left = left_by_key.get(key)
        right = right_by_key.get(key)
        rows.append(_compare_row(key, left=left, right=right, atol=atol, rtol=rtol))
    first = next((row for row in rows if row["status"] != "pass"), None)
    status_counts = {
        name: sum(1 for row in rows if row["status"] == name)
        for name in ("pass", "fail", "shape_mismatch", "unavailable", "non_finite")
    }
    complete = all(row["capture_kind"] != "unavailable" for row in rows)
    missing = sum(1 for row in rows if row["capture_kind"] == "unavailable")
    reconstructed = sum(1 for row in rows if row["capture_kind"] == "reconstructed")
    stage_status = {row["stage"]: row["status"] for row in rows}
    return {
        "schema": WKV_LIVE_UPDATE_HOOK_COMPARISON_SCHEMA,
        "status": "pass"
        if complete and all(row["status"] == "pass" for row in rows)
        else "fail",
        "kernel_ready": "yes"
        if complete and all(row["status"] == "pass" for row in rows)
        else "no",
        "diagnostic_only": True,
        "overall_status": "pass"
        if complete and all(row["status"] == "pass" for row in rows)
        else "fail",
        "live_hooks_complete": complete,
        "missing_hooks": missing,
        "reconstructed_hooks": reconstructed,
        "atol": atol,
        "rtol": rtol,
        "row_count": len(rows),
        "status_counts": status_counts,
        "first_divergent_case": None if first is None else first["case"],
        "first_divergent_mode": None if first is None else first["mode"],
        "first_divergent_layer": None if first is None else first["layer"],
        "first_divergent_token": None if first is None else first["token"],
        "first_divergent_head": None if first is None else first["head"],
        "first_divergent_stage": None if first is None else first["stage"],
        "first_divergent_capture_kind": None
        if first is None
        else first["capture_kind"],
        "first_divergent_max_abs_error": None
        if first is None
        else first.get("max_abs_error"),
        "decayed_state_match": stage_status.get("decayed_state") == "pass",
        "update_outer_product_match": stage_status.get("update_outer_product")
        == "pass",
        "balance_state_matmul_match": stage_status.get("balance_state_matmul")
        == "pass",
        "composite_update_term_match": stage_status.get("composite_update_term")
        == "pass",
        "update_term_match": stage_status.get("update_term") == "pass",
        "state_after_match": stage_status.get("state_after") == "pass",
        "suspected_root_cause": _suspected_root_cause(first),
        "fix_recommended": "no source-backed fix proven",
        "rows": rows,
        "hook_availability": _hook_availability(rows),
        "first_divergence_reconstruction": _reconstruct_first_divergence(rows, first),
        "composite_update_term_audit": _composite_update_audit(rows),
        "decayed_state_audit": _decayed_state_audit(rows),
        "state_after_assembly_audit": _state_after_assembly_audit(rows),
        "next_phase_recommendation": (
            "P64 should be a source-backed recurrence fix or comparison cleanup "
            "phase once the live balance-state/composite term is visible."
        ),
    }


def write_live_update_hook_trace(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=_entry_sort_key):
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def write_live_update_hook_reports(
    *,
    radlads_entries: list[dict[str, Any]],
    qrwkv_entries: list[dict[str, Any]],
    comparison_report: Mapping[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wkv_live_update_hooks_report.json").write_text(
        json.dumps(_jsonable(comparison_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P63_WKV_LIVE_UPDATE_HOOKS.md").write_text(
        _markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "HOOK_AVAILABILITY_MATRIX.md").write_text(
        _hook_availability_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "FIRST_LIVE_DIVERGENCE.md").write_text(
        _first_divergence_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "COMPOSITE_UPDATE_TERM_AUDIT.md").write_text(
        _composite_update_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "DECAYED_STATE_AUDIT.md").write_text(
        _decayed_state_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "STATE_AFTER_ASSEMBLY.md").write_text(
        _state_after_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "P63_RESULTS.md").write_text(
        _results_markdown(comparison_report),
        encoding="utf-8",
    )
    arrays = {}
    for label, entries in (("radlads", radlads_entries), ("qrwkv", qrwkv_entries)):
        for entry in entries:
            if entry.get("array") is None or entry.get("capture_kind") == "unavailable":
                continue
            key = (
                f"{label}_{entry['case']}_M{entry['mode']}_L{entry['layer']}_"
                f"T{entry['token']}_H{entry['head']}_{entry['stage']}"
            )
            arrays[_safe_npz_key(key)] = np.asarray(entry["array"])
    if arrays:
        np.savez(out_dir / "live_update_hooks_values.npz", **arrays)
    (out_dir / "live_update_hooks_metadata.json").write_text(
        json.dumps(
            {
                "schema": WKV_LIVE_UPDATE_HOOK_SCHEMA,
                "comparison_schema": WKV_LIVE_UPDATE_HOOK_COMPARISON_SCHEMA,
                "kernel_ready": comparison_report.get("kernel_ready"),
                "diagnostic_only": comparison_report.get("diagnostic_only", True),
                "first_divergent_stage": comparison_report.get("first_divergent_stage"),
                "next_phase_recommendation": comparison_report.get(
                    "next_phase_recommendation"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _rows_for_context(
    source_rows: list[dict[str, Any]],
    *,
    context: tuple[Any, Any, Any, Any],
    side: str,
    mode: str | None,
    allow_reconstructed: bool,
) -> list[dict[str, Any]]:
    case, layer, head, token = context
    rows = []
    for stage in LIVE_UPDATE_STAGES:
        source = _source_for_stage(
            source_rows,
            case=case,
            layer=layer,
            head=head,
            token=token,
            stage=stage,
        )
        if source is None:
            if _is_reconstructable_composite_stage(stage) and allow_reconstructed:
                rows.append(
                    _reconstructed_entry(
                        source_rows,
                        case=case,
                        side=side,
                        mode=mode,
                        layer=layer,
                        token=token,
                        head=head,
                        stage=stage,
                        source_stage_name=None,
                        reason="composite update term not captured in live trace rows",
                    )
                )
            else:
                rows.append(
                    _unavailable_entry(
                        case=case,
                        side=side,
                        mode=mode,
                        layer=layer,
                        token=token,
                        head=head,
                        stage=stage,
                        reason=_unavailable_reason(stage),
                    )
                )
        else:
            rows.append(_available_entry(source, side=side, mode=mode, stage=stage))
    return rows


def _source_for_stage(
    rows: list[dict[str, Any]],
    *,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    stage: str,
) -> dict[str, Any] | None:
    if stage == "state_after_for_next_token":
        return _find(
            rows,
            case=case,
            layer=layer,
            head=head,
            token=None if token is None else int(token) + 1,
            stages=SOURCE_STAGE_MAP["state_before"],
        )
    return _find(
        rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=SOURCE_STAGE_MAP.get(stage, ()),
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
            if (
                row.get("case") == case
                and row.get("layer") == layer
                and row.get("head") == head
                and row.get("token_index") == token
                and row.get("stage") == stage
                and row.get("array") is not None
            ):
                return row
    return None


def _contexts(rows: list[dict[str, Any]]) -> list[tuple[Any, Any, Any, Any]]:
    contexts = {
        (row.get("case"), row.get("layer"), row.get("head"), row.get("token_index"))
        for row in rows
        if row.get("stage") == "wkv_state_after" and row.get("token_index") is not None
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


def _available_entry(
    source: Mapping[str, Any], *, side: str, mode: str | None, stage: str
) -> dict[str, Any]:
    array = np.asarray(source["array"])
    summary = asdict(
        summarize_array(
            str(source.get("name", stage)),
            array,
            stage=stage,
            layer=source.get("layer"),
            time_index=source.get("token_index"),
        )
    )
    return asdict(
        WKVLiveUpdateHookEntry(
            side=side,
            case=str(source["case"]),
            mode=mode,
            layer=source.get("layer"),
            token=source.get("token_index"),
            head=source.get("head"),
            stage=stage,
            source_stage_name=source.get("stage"),
            source_file=SOURCE_PATHS[side]["file"],
            source_function=SOURCE_PATHS[side]["function"],
            shape=[int(dim) for dim in array.shape],
            dtype=str(array.dtype),
            finite=bool(np.isfinite(array).all()),
            max_abs=summary["abs_max"],
            sample=_sample(array),
            status="pass",
            reason=None,
            capture_kind="live_captured",
            array=array.tolist(),
        )
    )


def _reconstructed_entry(
    source_rows: list[dict[str, Any]],
    *,
    case: Any,
    side: str,
    mode: str | None,
    layer: Any,
    token: Any,
    head: Any,
    stage: str,
    source_stage_name: str | None,
    reason: str,
) -> dict[str, Any]:
    array = _reconstruct_composite_array(
        source_rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stage=stage,
    )
    if array is not None:
        array = np.asarray(array)
        summary = asdict(
            summarize_array(
                f"{side}.{stage}.reconstructed",
                array,
                stage=stage,
                layer=layer,
                time_index=token,
            )
        )
        shape = [int(dim) for dim in array.shape]
        dtype = str(array.dtype)
        finite = bool(np.isfinite(array).all())
        max_abs = summary["abs_max"]
        sample = _sample(array)
        status = "pass"
        payload = array.tolist()
    else:
        shape = []
        dtype = None
        finite = None
        max_abs = None
        sample = None
        status = "unavailable"
        payload = None
    return asdict(
        WKVLiveUpdateHookEntry(
            side=side,
            case=str(case),
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage=stage,
            source_stage_name=source_stage_name,
            source_file=SOURCE_PATHS[side]["file"],
            source_function=SOURCE_PATHS[side]["function"],
            shape=shape,
            dtype=dtype,
            finite=finite,
            max_abs=max_abs,
            sample=sample,
            status=status,
            reason=reason,
            capture_kind="reconstructed",
            array=payload,
        )
    )


def _is_reconstructable_composite_stage(stage: str) -> bool:
    return stage in {
        "balance_state_matmul",
        "composite_balance_update_term",
        "composite_update_term",
    }


def _reconstruct_composite_array(
    rows: list[dict[str, Any]],
    *,
    case: Any,
    layer: Any,
    head: Any,
    token: Any,
    stage: str,
) -> np.ndarray | None:
    balance = _find(
        rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=SOURCE_STAGE_MAP["balance_state_matmul"],
    )
    composite_balance = _find(
        rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=SOURCE_STAGE_MAP["composite_balance_update_term"],
    )
    outer = _find(
        rows,
        case=case,
        layer=layer,
        head=head,
        token=token,
        stages=SOURCE_STAGE_MAP["update_outer_product"],
    )
    if stage in {"balance_state_matmul", "composite_balance_update_term"}:
        source = balance if balance is not None else composite_balance
        return None if source is None else np.asarray(source["array"])
    if stage == "composite_update_term":
        balance_source = balance if balance is not None else composite_balance
        if balance_source is not None and outer is not None:
            return np.asarray(balance_source["array"]) + np.asarray(outer["array"])
    return None


def _unavailable_entry(
    *,
    case: Any,
    side: str,
    mode: str | None,
    layer: Any,
    token: Any,
    head: Any,
    stage: str,
    reason: str,
) -> dict[str, Any]:
    return asdict(
        WKVLiveUpdateHookEntry(
            side=side,
            case=str(case),
            mode=mode,
            layer=layer,
            token=token,
            head=head,
            stage=stage,
            source_stage_name=None,
            source_file=SOURCE_PATHS[side]["file"],
            source_function=SOURCE_PATHS[side]["function"],
            shape=[],
            dtype=None,
            finite=None,
            max_abs=None,
            sample=None,
            status="unavailable",
            reason=reason,
            capture_kind="unavailable",
            array=None,
        )
    )


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
        "stage": key[5],
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
    if (
        left.get("capture_kind") == "unavailable"
        or right.get("capture_kind") == "unavailable"
        or left.get("array") is None
        or right.get("array") is None
    ):
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
        "reconstructed"
        if left.get("capture_kind") == "reconstructed"
        or right.get("capture_kind") == "reconstructed"
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


def _hook_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = []
    for stage in LIVE_UPDATE_STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage]
        matrix.append(
            {
                "stage": stage,
                "radlads_live_captured": any(
                    row.get("radlads_capture_kind") == "live_captured"
                    for row in stage_rows
                ),
                "qrwkv_live_captured": any(
                    row.get("qrwkv_capture_kind") == "live_captured"
                    for row in stage_rows
                ),
                "radlads_reconstructed": any(
                    row.get("radlads_capture_kind") == "reconstructed"
                    for row in stage_rows
                ),
                "qrwkv_reconstructed": any(
                    row.get("qrwkv_capture_kind") == "reconstructed"
                    for row in stage_rows
                ),
                "source_name": ", ".join(SOURCE_STAGE_MAP.get(stage, ())) or "(none)",
                "notes": _hook_note(stage_rows),
            }
        )
    return {"matrix": matrix}


def _reconstruct_first_divergence(
    rows: list[dict[str, Any]], first: Mapping[str, Any] | None
) -> dict[str, Any]:
    if first is None:
        return {"status": "pass", "first_divergence": None}
    matching = [
        row
        for row in rows
        if row["case"] == first["case"]
        and row["mode"] == first["mode"]
        and row["layer"] == first["layer"]
        and row["token"] == first["token"]
        and row["head"] == first["head"]
    ]
    prior = None
    for row in matching:
        if row["stage"] == first["stage"]:
            break
        if row["status"] == "pass":
            prior = row["stage"]
    return {
        "case": first["case"],
        "mode": first["mode"],
        "layer": first["layer"],
        "token": first["token"],
        "head": first["head"],
        "first divergent live stage": first["stage"],
        "capture kind": first["capture_kind"],
        "RADLADS value sample": _sample_row(first, side="radlads"),
        "QRWKV value sample": _sample_row(first, side="qrwkv"),
        "max_abs_error": first.get("max_abs_error"),
        "prior matching stage": prior,
        "next divergent stage": _next_divergent_stage(matching, first["stage"]),
        "source-backed interpretation": _suspected_root_cause(first),
    }


def _composite_update_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage = _rows_by_stage(rows)
    return {
        "RADLADS source formula": (
            "decayed_state + update_outer_product (+ balance_state_matmul if exposed)"
        ),
        "QRWKV source formula": (
            "decayed_state + update_outer_product (+ balance_state_matmul if exposed)"
        ),
        "source names": {
            "update_outer_product": SOURCE_STAGE_MAP["update_outer_product"],
            "balance_state_matmul": SOURCE_STAGE_MAP["balance_state_matmul"],
            "composite_balance_update_term": SOURCE_STAGE_MAP[
                "composite_balance_update_term"
            ],
            "composite_update_term": SOURCE_STAGE_MAP["composite_update_term"],
        },
        "whether update_outer_product is sufficient": by_stage.get(
            "update_outer_product"
        )
        == "pass",
        "whether balance-state/composite term is required": True,
        "whether each term matches": {
            stage: by_stage.get(stage) == "pass"
            for stage in (
                "update_outer_product",
                "balance_state_matmul",
                "composite_balance_update_term",
                "composite_update_term",
                "update_term",
            )
        },
        "which term explains residual": _first_divergent_stage(by_stage),
        "candidate fix, if any": "source-backed live hook completion only",
    }


def _decayed_state_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage = _rows_by_stage(rows)
    return {
        "decay_value matches": "yes" if by_stage.get("decay_value") == "pass" else "no",
        "state_before matches": "yes"
        if by_stage.get("state_before") == "pass"
        else "no",
        "decayed_state live captured matches": "yes"
        if any(
            row["stage"] == "decayed_state"
            and row["capture_kind"] == "live_captured"
            and row["status"] == "pass"
            for row in rows
        )
        else "no",
        "decayed_state reconstructed matches": "yes"
        if any(
            row["stage"] == "decayed_state"
            and row["capture_kind"] == "reconstructed"
            and row["status"] == "pass"
            for row in rows
        )
        else "no",
        "source-backed cause if mismatch": _first_divergent_stage(by_stage),
    }


def _state_after_assembly_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage = _rows_by_stage(rows)
    included = [
        stage
        for stage in (
            "state_before",
            "decayed_state",
            "update_outer_product",
            "balance_state_matmul",
            "composite_balance_update_term",
            "composite_update_term",
            "update_term",
        )
        if by_stage.get(stage) == "pass"
    ]
    omitted = [
        stage
        for stage in (
            "state_before",
            "decayed_state",
            "update_outer_product",
            "balance_state_matmul",
            "composite_balance_update_term",
            "composite_update_term",
            "update_term",
        )
        if by_stage.get(stage) != "pass"
    ]
    return {
        "RADLADS assembly": " + ".join(included),
        "QRWKV assembly": " + ".join(included),
        "terms included": included,
        "terms omitted": omitted,
        "assembly match": "yes" if not omitted else "no",
        "first missing/different term": omitted[0] if omitted else None,
    }


def _hook_note(rows: list[dict[str, Any]]) -> str:
    if any(row["capture_kind"] == "unavailable" for row in rows):
        return "one or more live hooks are unavailable in the captured trace"
    if any(row["capture_kind"] == "reconstructed" for row in rows):
        return "contains reconstructed composite terms"
    return "all rows live-captured"


def _first_divergent_stage(stage_status: Mapping[str, str]) -> str | None:
    for stage in LIVE_UPDATE_STAGES:
        if stage_status.get(stage) != "pass":
            return stage
    return None


def _next_divergent_stage(rows: list[dict[str, Any]], stage: str) -> str | None:
    seen = False
    for row in rows:
        if row["stage"] == stage:
            seen = True
            continue
        if seen and row["status"] != "pass":
            return row["stage"]
    return None


def _rows_by_stage(rows: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for stage in LIVE_UPDATE_STAGES:
        result[stage] = next(
            (row["status"] for row in rows if row["stage"] == stage), "unavailable"
        )
    return result


def _sample_row(row: Mapping[str, Any], *, side: str) -> Any:
    return row.get("sample")


def _sample(array: np.ndarray) -> float | str | None:
    flat = np.asarray(array).reshape(-1)
    return float(flat[0]) if flat.size else None


def _suspected_root_cause(first: Mapping[str, Any] | None) -> str:
    if first is None:
        return "no divergence detected"
    if first.get("stage") in {
        "balance_state_matmul",
        "composite_balance_update_term",
        "composite_update_term",
    }:
        return "missing composite live hook"
    if first.get("stage") in {"decayed_state", "state_after", "update_term"}:
        return "live hook completion or assembly mismatch"
    return "unresolved residual"


def _unavailable_reason(stage: str) -> str:
    if stage in {
        "balance_state_matmul",
        "composite_balance_update_term",
        "composite_update_term",
    }:
        return "live composite update term is not captured in the source trace rows"
    if stage == "state_after_for_next_token":
        return "no following token state_before row is available for this context"
    if stage == "state_after_exported":
        return "exported final state is not captured per layer/head/token in this trace"
    return f"No captured source row maps to required stage {stage!r}."


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("mode"),
        entry.get("layer"),
        entry.get("token"),
        entry.get("head"),
        entry.get("stage"),
    )


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        keys,
        key=lambda key: (
            str(key[0]),
            str(key[1]),
            -1 if key[2] is None else int(key[2]),
            -1 if key[3] is None else int(key[3]),
            -1 if key[4] is None else int(key[4]),
            LIVE_UPDATE_STAGES.index(key[5]) if key[5] in LIVE_UPDATE_STAGES else 999,
        ),
    )


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return _sorted_keys([_trace_key(entry)])[0]


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


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P63 WKV Live Update Hooks",
        "",
        f"- status: `{report.get('status')}`",
        f"- kernel_ready: `{report.get('kernel_ready')}`",
        f"- first divergent stage: `{report.get('first_divergent_stage')}`",
        (
            "- first divergent capture kind: "
            f"`{report.get('first_divergent_capture_kind')}`"
        ),
        (
            "- first divergent max abs error: "
            f"`{report.get('first_divergent_max_abs_error')}`"
        ),
        "",
        "## Live hook availability",
        "",
    ]
    for row in report.get("hook_availability", {}).get("matrix", []):
        lines.append(
            f"- {row['stage']}: radlads_live={row['radlads_live_captured']} "
            f"qrwkv_live={row['qrwkv_live_captured']} "
            f"radlads_recon={row['radlads_reconstructed']} "
            f"qrwkv_recon={row['qrwkv_reconstructed']}"
        )
    lines.extend(
        [
            "",
            "## First divergence",
            "",
            str(report.get("first_divergence_reconstruction")),
            "",
            "## Composite update audit",
            "",
            str(report.get("composite_update_term_audit")),
            "",
            "## Decayed state audit",
            "",
            str(report.get("decayed_state_audit")),
            "",
            "## State-after assembly audit",
            "",
            str(report.get("state_after_assembly_audit")),
        ]
    )
    return "\n".join(lines) + "\n"


def _hook_availability_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# HOOK_AVAILABILITY_MATRIX", ""]
    lines.append(
        "| stage | RADLADS live captured yes/no | QRWKV live captured yes/no | "
        "RADLADS reconstructed yes/no | QRWKV reconstructed yes/no | source "
        "name | notes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report.get("hook_availability", {}).get("matrix", []):
        lines.append(
            f"| {row['stage']} | {row['radlads_live_captured']} | "
            f"{row['qrwkv_live_captured']} | {row['radlads_reconstructed']} | "
            f"{row['qrwkv_reconstructed']} | {row['source_name']} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def _first_divergence_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# FIRST_LIVE_DIVERGENCE",
                "",
                *(
                    f"- {key}: `{value}`"
                    for key, value in report.get(
                        "first_divergence_reconstruction", {}
                    ).items()
                ),
            ]
        )
        + "\n"
    )


def _composite_update_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# COMPOSITE_UPDATE_TERM_AUDIT",
                "",
                *(
                    f"- {key}: `{value}`"
                    for key, value in report.get(
                        "composite_update_term_audit", {}
                    ).items()
                ),
            ]
        )
        + "\n"
    )


def _decayed_state_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# DECAYED_STATE_AUDIT",
                "",
                *(
                    f"- {key}: `{value}`"
                    for key, value in report.get("decayed_state_audit", {}).items()
                ),
            ]
        )
        + "\n"
    )


def _state_after_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# STATE_AFTER_ASSEMBLY",
                "",
                *(
                    f"- {key}: `{value}`"
                    for key, value in report.get(
                        "state_after_assembly_audit", {}
                    ).items()
                ),
            ]
        )
        + "\n"
    )


def _results_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# P63 Results",
                "",
                f"- status: `{report.get('status')}`",
                f"- kernel_ready: `{report.get('kernel_ready')}`",
                f"- first divergent stage: `{report.get('first_divergent_stage')}`",
                f"- next recommendation: `{report.get('next_phase_recommendation')}`",
            ]
        )
        + "\n"
    )
