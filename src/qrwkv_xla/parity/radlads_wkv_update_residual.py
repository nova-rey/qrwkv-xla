from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl

WKV_UPDATE_RESIDUAL_SCHEMA = "radlads_qrwkv_wkv_update_residual.v1"
WKV_UPDATE_RESIDUAL_COMPARISON_SCHEMA = (
    "radlads_qrwkv_wkv_update_residual_comparison.v1"
)

REQUIRED_UPDATE_STAGES = (
    "state_before",
    "decay_value",
    "decayed_state",
    "k_for_update",
    "v_for_update",
    "update_outer_product",
    "update_term",
    "state_after",
    "state_after_for_next_token",
    "state_after_exported",
)

SOURCE_STAGE_CANDIDATES = {
    "state_before": ("wkv_state_before",),
    "decay_value": ("decay_after_transform",),
    "decayed_state": ("wkv_decay_applied",),
    "k_for_update": ("k_a", "k", "k_projection"),
    "v_for_update": ("v", "v_projection", "value_after_v_first_mix"),
    "update_outer_product": ("wkv_update_outer_or_term",),
    "update_term": ("wkv_update_term",),
    "state_after": ("wkv_state_after",),
    "state_after_exported": ("returned_wkv_matrix_state",),
}


@dataclass(frozen=True)
class WKVUpdateResidualEntry:
    case: str
    side: str
    layer: int | None
    head: int | None
    token_index: int | None
    stage: str
    source_stage: str | None
    available: bool
    unavailable_reason: str | None
    shape: list[int]
    dtype: str | None
    finite: bool | None
    min: float | str | None
    max: float | str | None
    mean: float | str | None
    std: float | str | None
    abs_max: float | str | None
    array: Any | None = None


def load_update_residual_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_trace_jsonl(path)


def build_update_residual_trace(
    trace_entries: Iterable[Mapping[str, Any]],
    *,
    side: str,
) -> list[dict[str, Any]]:
    source_rows = [dict(entry) for entry in trace_entries if entry.get("side") == side]
    contexts = _contexts(source_rows)
    rows: list[dict[str, Any]] = []
    for context in contexts:
        rows.extend(_rows_for_context(source_rows, context=context, side=side))
    rows.sort(key=_entry_sort_key)
    return rows


def compare_update_residual_traces(
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
    reconstruction = {
        "radlads": reconstruct_update_residual(radlads_entries),
        "qrwkv": reconstruct_update_residual(qrwkv_entries),
    }
    status_counts = {
        name: sum(1 for row in rows if row["status"] == name)
        for name in ("pass", "fail", "shape_mismatch", "unavailable", "non_finite")
    }
    all_required_available = all(
        row["status"] != "unavailable"
        for row in rows
        if row.get("stage") in REQUIRED_UPDATE_STAGES
    )
    all_compared_pass = bool(rows) and all(row["status"] == "pass" for row in rows)
    stage_status = {row["stage"]: row["status"] for row in rows}
    return {
        "schema": WKV_UPDATE_RESIDUAL_COMPARISON_SCHEMA,
        "status": "pass" if all_required_available and all_compared_pass else "fail",
        "kernel_ready": "yes" if all_required_available and all_compared_pass else "no",
        "kernel_ready_reason": "all_required_update_residual_rows_pass"
        if all_required_available and all_compared_pass
        else "missing_or_failing_required_update_residual_rows",
        "all_required_available": all_required_available,
        "all_compared_pass": all_compared_pass,
        "diagnostic_only": True,
        "claim": (
            "P62 traces WKV update-term/state-after residual surfaces from real "
            "paired trace artifacts where available. Missing stages are reported "
            "as unavailable; no synthetic fallback or recurrence fix is applied."
        ),
        "atol": atol,
        "rtol": rtol,
        "row_count": len(rows),
        "status_counts": status_counts,
        "first_divergent_case": None if first is None else first["case"],
        "first_divergent_stage": None if first is None else first["stage"],
        "first_divergent_layer": None if first is None else first["layer"],
        "first_divergent_head": None if first is None else first["head"],
        "first_divergent_token": None if first is None else first["token_index"],
        "first_divergent_max_abs_error": None
        if first is None
        else first.get("max_abs_error"),
        "state_before_match": stage_status.get("state_before") == "pass",
        "decay_value_match": stage_status.get("decay_value") == "pass",
        "decayed_state_match": stage_status.get("decayed_state") == "pass",
        "k_for_update_match": stage_status.get("k_for_update") == "pass",
        "v_for_update_match": stage_status.get("v_for_update") == "pass",
        "update_outer_product_match": (
            stage_status.get("update_outer_product") == "pass"
        ),
        "update_term_match": stage_status.get("update_term") == "pass",
        "state_after_match": stage_status.get("state_after") == "pass",
        "state_after_for_next_token_match": (
            stage_status.get("state_after_for_next_token") == "pass"
        ),
        "state_after_exported_match": (
            stage_status.get("state_after_exported") == "pass"
        ),
        "suspected_root_cause": _suspected_root_cause(first),
        "fix_recommended": "no source-backed fix proven",
        "rows": rows,
        "audit": _audit_report(radlads_entries, qrwkv_entries, reconstruction),
        "reconstruction": reconstruction,
        "next_phase_recommendation": (
            "P63 should capture source-backed RADLADS and QRWKV `update_term` "
            "and `decayed_state` rows in the live recurrence hooks, including "
            "the balance-state matmul term, before any numeric correction."
        ),
    }


def reconstruct_update_residual(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    by_key = {_trace_key(entry): entry for entry in entries}
    contexts = {
        (entry["case"], entry["layer"], entry["head"], entry["token_index"])
        for entry in entries
        if entry.get("available") is True
        and entry.get("stage") == "decayed_state"
        and entry.get("token_index") is not None
    }
    for case, layer, head, token_index in sorted(
        contexts,
        key=lambda item: (
            str(item[0]),
            -1 if item[1] is None else int(item[1]),
            -1 if item[2] is None else int(item[2]),
            -1 if item[3] is None else int(item[3]),
        ),
    ):
        needed = {
            stage: by_key.get((case, layer, head, token_index, stage))
            for stage in ("decayed_state", "update_outer_product", "state_after")
        }
        if any(row is None or row.get("array") is None for row in needed.values()):
            rows.append(
                {
                    "case": case,
                    "layer": layer,
                    "head": head,
                    "token_index": token_index,
                    "status": "unavailable",
                    "reason": (
                        "decayed_state, update_outer_product, or state_after "
                        "unavailable"
                    ),
                }
            )
            continue
        reconstructed = np.asarray(needed["decayed_state"]["array"]) + np.asarray(
            needed["update_outer_product"]["array"]
        )
        stats = compare_trace_arrays(needed["state_after"]["array"], reconstructed)
        rows.append(
            {
                "case": case,
                "layer": layer,
                "head": head,
                "token_index": token_index,
                "status": stats["status"],
                "max_abs_error": stats["max_abs_error"],
                "mean_abs_error": stats["mean_abs_error"],
                "note": (
                    "Reconstruction omits state_before @ balance_state_term when "
                    "the composite update_term is unavailable."
                ),
            }
        )
    first = next((row for row in rows if row["status"] != "pass"), None)
    return {
        "status": "pass" if rows and first is None else "fail",
        "row_count": len(rows),
        "first_residual": first,
        "rows": rows,
    }


def write_update_residual_trace(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=_entry_sort_key):
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def write_update_residual_reports(
    *,
    radlads_entries: list[dict[str, Any]],
    qrwkv_entries: list[dict[str, Any]],
    comparison_report: Mapping[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wkv_update_residual_comparison_report.json").write_text(
        json.dumps(_jsonable(comparison_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "p62_wkv_update_residual_report.json").write_text(
        json.dumps(_jsonable(comparison_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P62_WKV_UPDATE_RESIDUAL.md").write_text(
        _markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "FIRST_RESIDUAL_POINT.md").write_text(
        _first_residual_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "OUTER_PRODUCT_CONVENTION.md").write_text(
        _outer_product_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "DECAY_APPLICATION.md").write_text(
        _decay_application_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "DTYPE_ACCUMULATION.md").write_text(
        _dtype_accumulation_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "MASK_UPDATE_INTERACTION.md").write_text(
        _mask_update_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "P62_RESULTS.md").write_text(
        _results_markdown(comparison_report),
        encoding="utf-8",
    )
    arrays = {}
    for label, entries in (("radlads", radlads_entries), ("qrwkv", qrwkv_entries)):
        for entry in entries:
            if entry.get("array") is None or entry.get("available") is not True:
                continue
            key = (
                f"{label}_{entry['case']}_L{entry['layer']}_H{entry['head']}_"
                f"T{entry['token_index']}_{entry['stage']}"
            )
            arrays[_safe_npz_key(key)] = np.asarray(entry["array"])
    if arrays:
        np.savez(out_dir / "wkv_update_residual_values.npz", **arrays)
    (out_dir / "wkv_update_residual_metadata.json").write_text(
        json.dumps(
            {
                "schema": WKV_UPDATE_RESIDUAL_SCHEMA,
                "comparison_schema": WKV_UPDATE_RESIDUAL_COMPARISON_SCHEMA,
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
) -> list[dict[str, Any]]:
    case, layer, head, token_index = context
    rows = []
    for stage in REQUIRED_UPDATE_STAGES:
        source = _source_for_stage(
            source_rows,
            case=case,
            layer=layer,
            head=head,
            token_index=token_index,
            stage=stage,
        )
        if source is None:
            rows.append(
                _unavailable_entry(
                    case=case,
                    side=side,
                    layer=layer,
                    head=head,
                    token_index=token_index,
                    stage=stage,
                    reason=_unavailable_reason(stage),
                )
            )
        else:
            rows.append(_available_entry(source, side=side, stage=stage))
    return rows


def _source_for_stage(
    rows: list[dict[str, Any]],
    *,
    case: Any,
    layer: Any,
    head: Any,
    token_index: Any,
    stage: str,
) -> dict[str, Any] | None:
    if stage == "state_after_for_next_token":
        return _find(
            rows,
            case=case,
            layer=layer,
            head=head,
            token_index=None if token_index is None else int(token_index) + 1,
            stages=SOURCE_STAGE_CANDIDATES["state_before"],
        )
    return _find(
        rows,
        case=case,
        layer=layer,
        head=head,
        token_index=token_index,
        stages=SOURCE_STAGE_CANDIDATES.get(stage, ()),
    )


def _find(
    rows: list[dict[str, Any]],
    *,
    case: Any,
    layer: Any,
    head: Any,
    token_index: Any,
    stages: tuple[str, ...],
) -> dict[str, Any] | None:
    for stage in stages:
        for row in rows:
            if (
                row.get("case") == case
                and row.get("layer") == layer
                and row.get("head") == head
                and row.get("token_index") == token_index
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
    source: Mapping[str, Any], *, side: str, stage: str
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
        WKVUpdateResidualEntry(
            case=str(source["case"]),
            side=side,
            layer=source.get("layer"),
            head=source.get("head"),
            token_index=source.get("token_index"),
            stage=stage,
            source_stage=source.get("stage"),
            available=True,
            unavailable_reason=None,
            shape=[int(dim) for dim in array.shape],
            dtype=str(array.dtype),
            finite=bool(np.isfinite(array).all()),
            min=summary["min"],
            max=summary["max"],
            mean=summary["mean"],
            std=summary["std"],
            abs_max=summary["abs_max"],
            array=array.tolist(),
        )
    )


def _unavailable_entry(
    *,
    case: Any,
    side: str,
    layer: Any,
    head: Any,
    token_index: Any,
    stage: str,
    reason: str,
) -> dict[str, Any]:
    return asdict(
        WKVUpdateResidualEntry(
            case=str(case),
            side=side,
            layer=layer,
            head=head,
            token_index=token_index,
            stage=stage,
            source_stage=None,
            available=False,
            unavailable_reason=reason,
            shape=[],
            dtype=None,
            finite=None,
            min=None,
            max=None,
            mean=None,
            std=None,
            abs_max=None,
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
        "layer": key[1],
        "head": key[2],
        "token_index": key[3],
        "stage": key[4],
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
        }
    if not left.get("available") or not right.get("available"):
        return base | {
            "status": "unavailable",
            "reason": {
                "radlads": left.get("unavailable_reason"),
                "qrwkv": right.get("unavailable_reason"),
            },
            "shape_match": False,
            "dtype_match": False,
            "finite_both": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
        }
    stats = compare_trace_arrays(left["array"], right["array"], atol=atol, rtol=rtol)
    return base | stats | {"reason": None}


def _audit_report(
    radlads_entries: list[dict[str, Any]],
    qrwkv_entries: list[dict[str, Any]],
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "first_residual_reconstruction": {
            "radlads": reconstruction["radlads"]["first_residual"],
            "qrwkv": reconstruction["qrwkv"]["first_residual"],
        },
        "outer_product_convention": _stage_availability(
            radlads_entries, qrwkv_entries, "update_outer_product"
        )
        | {
            "expected_expression": "einsum('bhi,bhj->bhij', v_for_update, k_for_update)"
        },
        "decay_application": _stage_availability(
            radlads_entries, qrwkv_entries, "decayed_state"
        )
        | {"expected_expression": "state_before * decay_value[:, :, None, :]"},
        "dtype_accumulation": {
            "radlads": _dtype_set(radlads_entries),
            "qrwkv": _dtype_set(qrwkv_entries),
            "note": (
                "P62 reports captured dtypes only; it does not change "
                "accumulation order."
            ),
        },
        "mask_update_interaction": {
            "status": "diagnostic_only",
            "note": (
                "The paired traces do not expose a separate mask gate on the "
                "update term; masked-case residual remains visible through "
                "state_after/exported-state surfaces."
            ),
        },
    }


def _stage_availability(
    radlads_entries: list[dict[str, Any]],
    qrwkv_entries: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "radlads_available": any(
            row.get("stage") == stage and row.get("available") is True
            for row in radlads_entries
        ),
        "qrwkv_available": any(
            row.get("stage") == stage and row.get("available") is True
            for row in qrwkv_entries
        ),
    }


def _dtype_set(entries: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(row["dtype"])
            for row in entries
            if row.get("available") is True and row.get("dtype") is not None
        }
    )


def _unavailable_reason(stage: str) -> str:
    if stage == "update_term":
        return "Composite update_term is not present in source trace rows."
    if stage == "state_after_for_next_token":
        return "No following token state_before row is available for this context."
    if stage == "state_after_exported":
        return (
            "Exported final state is not captured per layer/head/token in this trace."
        )
    return f"No captured source row maps to required stage {stage!r}."


def _suspected_root_cause(first: Mapping[str, Any] | None) -> str:
    if first is None:
        return "no divergence detected"
    stage = first.get("stage")
    if stage in {"k_for_update", "v_for_update", "update_outer_product"}:
        return "update-term surface mismatch"
    if stage == "decayed_state":
        return "decay application mismatch"
    if stage in {"state_after", "state_after_for_next_token", "state_after_exported"}:
        return "state handoff / export mismatch"
    return "unresolved residual"


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("layer"),
        entry.get("head"),
        entry.get("token_index"),
        entry.get("stage"),
    )


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        keys,
        key=lambda key: (
            str(key[0]),
            -1 if key[1] is None else int(key[1]),
            -1 if key[2] is None else int(key[2]),
            -1 if key[3] is None else int(key[3]),
            REQUIRED_UPDATE_STAGES.index(key[4])
            if key[4] in REQUIRED_UPDATE_STAGES
            else 999,
        ),
    )


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return _sorted_keys([_trace_key(entry)])[0]


def _safe_npz_key(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace("None", "none")
        .replace("-", "_")
    )


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


def _first_residual_markdown(report: Mapping[str, Any]) -> str:
    first_row = {
        "case": report.get("first_divergent_case"),
        "layer": report.get("first_divergent_layer"),
        "head": report.get("first_divergent_head"),
        "token_index": report.get("first_divergent_token"),
        "stage": report.get("first_divergent_stage"),
    }
    matching_rows = [
        row
        for row in report.get("rows", [])
        if row.get("case") == first_row.get("case")
        and row.get("layer") == first_row.get("layer")
        and row.get("head") == first_row.get("head")
        and row.get("token_index") == first_row.get("token_index")
    ]
    row_by_stage = {row.get("stage"): row for row in matching_rows}

    def _err(stage: str) -> Any:
        row = row_by_stage.get(stage)
        if row is None:
            return None
        return row.get("max_abs_error")

    lines = [
        "# FIRST_RESIDUAL_POINT",
        "",
        f"- case: `{first_row.get('case')}`",
        "- mode: `both`",
        f"- layer: `{first_row.get('layer')}`",
        f"- token: `{first_row.get('token_index')}`",
        f"- head: `{first_row.get('head')}`",
        f"- stage: `{first_row.get('stage')}`",
        f"- state_before_error: `{_err('state_before')}`",
        f"- decay_value_error: `{_err('decay_value')}`",
        f"- decayed_state_error: `{_err('decayed_state')}`",
        f"- k_for_update_error: `{_err('k_for_update')}`",
        f"- v_for_update_error: `{_err('v_for_update')}`",
        f"- update_outer_product_error: `{_err('update_outer_product')}`",
        f"- update_term_error: `{_err('update_term')}`",
        f"- state_after_error: `{_err('state_after')}`",
        f"- first_new_divergence: `{report.get('first_divergent_stage')}`",
    ]
    return "\n".join(lines) + "\n"


def _outer_product_markdown(report: Mapping[str, Any]) -> str:
    verdict = report.get("audit", {}).get("outer_product_convention", {})
    source_backed = bool(
        verdict.get("radlads_available") and verdict.get("qrwkv_available")
    )
    lines = [
        "# OUTER_PRODUCT_CONVENTION",
        "",
        f"- source-backed convention: `{source_backed}`",
        "- candidate_best: `outer(v_for_update, k_for_update)`",
        "- candidate_best_source_supported: `no`",
        "- numeric_improvement: `none`",
        "- accepted_or_rejected: `rejected`",
    ]
    return "\n".join(lines) + "\n"


def _decay_application_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DECAY_APPLICATION",
        "",
        "- RADLADS convention: `prev_state * decay_value[:, None, :]`",
        "- QRWKV convention: `prev_state * decay_value[:, None, :]`",
        "- match: `unknown`",
        "- suspected_issue: `no source-backed decay orientation mismatch proven`",
    ]
    return "\n".join(lines) + "\n"


def _dtype_accumulation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DTYPE_ACCUMULATION",
        "",
        "- dtype mismatch explains residual: `unknown`",
        "- candidate cast fix: `none`",
        "- source-backed: `no`",
    ]
    return "\n".join(lines) + "\n"


def _mask_update_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MASK_UPDATE_INTERACTION",
        "",
        "- RADLADS convention: `masked cases remain diagnostic-only`",
        "- QRWKV convention: `masked cases remain diagnostic-only`",
        "- match: `unknown`",
    ]
    return "\n".join(lines) + "\n"


def _results_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P62 Results",
        "",
        f"- status: `{report.get('status')}`",
        f"- kernel_ready: `{report.get('kernel_ready')}`",
        f"- first divergent stage: `{report.get('first_divergent_stage')}`",
        f"- next recommendation: `{report.get('next_phase_recommendation')}`",
    ]
    return "\n".join(lines) + "\n"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P62 WKV Update Residual",
        "",
        f"- status: `{report.get('status')}`",
        f"- kernel_ready: `{report.get('kernel_ready')}`",
        f"- first divergent stage: `{report.get('first_divergent_stage')}`",
        f"- first divergent layer: `{report.get('first_divergent_layer')}`",
        f"- first divergent head: `{report.get('first_divergent_head')}`",
        f"- first divergent token: `{report.get('first_divergent_token')}`",
        "- first divergent max abs error: "
        f"`{report.get('first_divergent_max_abs_error')}`",
        "",
        "## Audit",
        "",
        "- first residual reconstruction: "
        f"`{report.get('audit', {}).get('first_residual_reconstruction')}`",
        "- outer-product convention: "
        f"`{report.get('audit', {}).get('outer_product_convention')}`",
        f"- decay application: `{report.get('audit', {}).get('decay_application')}`",
        f"- dtype accumulation: `{report.get('audit', {}).get('dtype_accumulation')}`",
        "- mask/update interaction: "
        f"`{report.get('audit', {}).get('mask_update_interaction')}`",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("rows", [])[:200]:
        lines.append(
            f"- {row['case']} / L{row['layer']} / H{row['head']} / "
            f"T{row['token_index']} / {row['stage']}: {row['status']} "
            f"(max_abs={row.get('max_abs_error')}, reason={row.get('reason')})"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            str(report.get("next_phase_recommendation")),
        ]
    )
    return "\n".join(lines)
