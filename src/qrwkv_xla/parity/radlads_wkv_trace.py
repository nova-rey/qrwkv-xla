from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array

WKV_TRACE_SCHEMA = "radlads_qrwkv_wkv_trace.v1"
WKV_TRACE_COMPARISON_SCHEMA = "radlads_qrwkv_wkv_trace_comparison.v1"

SEMANTIC_STAGE_ALIASES = {
    "input_embeddings": "input_to_attention",
    "mixed_input": "input_to_attention",
    "pre_attention_norm": "pre_attention_norm",
    "rope_output_q": "q",
    "rope_output_k": "k",
    "q_projection": "q",
    "k_projection": "k",
    "v_projection": "v",
    "v_first": "value_before_v_first_mix",
    "mixed_value": "value_after_v_first_mix",
    "gate_output": "g_or_gate_after_activation",
    "g1_projection": "g_or_gate_raw",
    "g2_projection": "g_or_gate_after_activation",
    "iclr_update_rate": "a_or_iclr_after_transform",
    "low_rank_decay": "log_w",
    "decay_applied_weights": "decay_after_transform",
    "initial_matrix_state": "wkv_state_before",
    "update_term": "wkv_update_outer_or_term",
    "output_before_o_proj": "wkv_output_before_o_proj",
    "o_proj_output": "o_proj_output",
    "normalized_output": "attention_output",
    "final_hidden": "layer_output",
    "logits": "logits",
    "returned_wkv_matrix_state": "wkv_state_after",
    "returned_shift_state": "shift_state_after",
}

HEADWISE_STAGE_HINTS = {
    "q",
    "k",
    "v",
    "receptance_or_r",
    "decay_raw",
    "log_w",
    "decay_after_transform",
    "a_or_iclr_raw",
    "a_or_iclr_after_transform",
    "g_or_gate_raw",
    "g_or_gate_after_activation",
    "value_before_v_first_mix",
    "value_after_v_first_mix",
    "wkv_state_before",
    "wkv_update_outer_or_term",
    "wkv_decay_applied",
    "wkv_state_after",
    "wkv_output_before_o_proj",
    "o_proj_output",
    "attention_output",
}


@dataclass(frozen=True)
class WKVTraceEntry:
    case: str
    side: str
    layer: int | None
    head: int | None
    token_index: int | None
    stage: str
    name: str
    shape: list[int]
    dtype: str
    finite: bool
    min: float | str | None
    max: float | str | None
    mean: float | str | None
    std: float | str | None
    abs_max: float | str | None
    array: Any | None = None


class WKVTraceCollector:
    def __init__(
        self,
        *,
        case: str,
        side: str,
        include_arrays: bool = True,
        max_inline_values: int = 256,
    ) -> None:
        self.case = case
        self.side = side
        self.include_arrays = include_arrays
        self.max_inline_values = max_inline_values
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        name: str,
        value: Any,
        *,
        stage: str,
        layer: int | None = None,
        head: int | None = None,
        token_index: int | None = None,
        time_index: int | None = None,
    ) -> None:
        index = token_index if token_index is not None else time_index
        semantic_stage = SEMANTIC_STAGE_ALIASES.get(stage, stage)
        array = np.asarray(value)
        if semantic_stage in HEADWISE_STAGE_HINTS and head is None and array.ndim >= 3:
            head_axis = 1 if array.ndim in {3, 4, 5} else None
            if head_axis is not None and array.shape[head_axis] > 0:
                for head_index in range(int(array.shape[head_axis])):
                    self._append(
                        name=name,
                        value=np.take(array, head_index, axis=head_axis),
                        stage=semantic_stage,
                        layer=layer,
                        head=head_index,
                        token_index=index,
                    )
                return
        self._append(
            name=name,
            value=array,
            stage=semantic_stage,
            layer=layer,
            head=head,
            token_index=index,
        )

    def _append(
        self,
        *,
        name: str,
        value: np.ndarray,
        stage: str,
        layer: int | None,
        head: int | None,
        token_index: int | None,
    ) -> None:
        summary = asdict(
            summarize_array(
                name, value, stage=stage, layer=layer, time_index=token_index
            )
        )
        entry = WKVTraceEntry(
            case=self.case,
            side=self.side,
            layer=layer,
            head=head,
            token_index=token_index,
            stage=stage,
            name=name,
            shape=[int(dim) for dim in value.shape],
            dtype=str(value.dtype),
            finite=bool(np.isfinite(value).all()) if value.size else True,
            min=summary["min"],
            max=summary["max"],
            mean=summary["mean"],
            std=summary["std"],
            abs_max=summary["abs_max"],
            array=_maybe_inline_array(value, self.max_inline_values)
            if self.include_arrays
            else None,
        )
        self.entries.append(asdict(entry))

    def extend(self, entries: Iterable[dict[str, Any]]) -> None:
        self.entries.extend(entries)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


@dataclass(frozen=True)
class TraceComparisonRow:
    case: str
    layer: int | None
    head: int | None
    token_index: int | None
    stage: str
    shape_match: bool
    dtype_match: bool
    finite_both: bool
    max_abs_error: float | None
    mean_abs_error: float | None
    max_relative_error: float | None
    allclose: bool
    status: str
    reason: str | None = None


def _maybe_inline_array(value: np.ndarray, max_inline_values: int) -> Any | None:
    if value.size > max_inline_values:
        return None
    return value.tolist()


def load_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("layer"),
        entry.get("head"),
        entry.get("token_index"),
        entry.get("stage"),
    )


def compare_trace_arrays(
    left: Any, right: Any, *, atol: float = 1e-5, rtol: float = 1e-5
) -> dict[str, Any]:
    left_value = np.asarray(left)
    right_value = np.asarray(right)
    shape_match = tuple(left_value.shape) == tuple(right_value.shape)
    dtype_match = str(left_value.dtype) == str(right_value.dtype)
    finite_both = bool(np.isfinite(left_value).all() and np.isfinite(right_value).all())
    if not shape_match:
        return {
            "shape_match": False,
            "dtype_match": dtype_match,
            "finite_both": finite_both,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
            "status": "shape_mismatch",
        }
    diff = np.abs(left_value.astype(np.float64) - right_value.astype(np.float64))
    denom = np.maximum(np.abs(right_value.astype(np.float64)), 1e-12)
    allclose = bool(np.allclose(left_value, right_value, atol=atol, rtol=rtol))
    status = "pass" if allclose else ("non_finite" if not finite_both else "fail")
    return {
        "shape_match": True,
        "dtype_match": dtype_match,
        "finite_both": finite_both,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_relative_error": float(np.max(diff / denom)) if diff.size else 0.0,
        "allclose": allclose,
        "status": status,
    }


def compare_trace_entries(
    left_entries: list[dict[str, Any]],
    right_entries: list[dict[str, Any]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    left_by_key = {_trace_key(entry): entry for entry in left_entries}
    right_by_key = {_trace_key(entry): entry for entry in right_entries}
    keys = sorted(
        set(left_by_key) & set(right_by_key),
        key=lambda key: (
            "" if key[0] is None else str(key[0]),
            -1 if key[1] is None else int(key[1]),
            -1 if key[2] is None else int(key[2]),
            -1 if key[3] is None else int(key[3]),
            "" if key[4] is None else str(key[4]),
        ),
    )
    rows: list[dict[str, Any]] = []
    first_divergent = None
    for key in keys:
        left = left_by_key[key]
        right = right_by_key[key]
        stats = compare_trace_arrays(
            left.get("array"), right.get("array"), atol=atol, rtol=rtol
        )
        row = {
            "case": key[0],
            "layer": key[1],
            "head": key[2],
            "token_index": key[3],
            "stage": key[4],
            **stats,
        }
        rows.append(row)
        if first_divergent is None and row["status"] != "pass":
            first_divergent = row
    return {
        "schema": WKV_TRACE_COMPARISON_SCHEMA,
        "rows": rows,
        "row_count": len(rows),
        "first_divergent_stage": None
        if first_divergent is None
        else first_divergent["stage"],
        "first_divergent_layer": None
        if first_divergent is None
        else first_divergent["layer"],
        "first_divergent_head": None
        if first_divergent is None
        else first_divergent["head"],
        "first_divergent_token": None
        if first_divergent is None
        else first_divergent["token_index"],
        "first_divergent_max_abs_error": None
        if first_divergent is None
        else first_divergent["max_abs_error"],
    }


def write_trace_comparison_reports(
    report: dict[str, Any], out_dir: Path, *, report_prefix: str = "P56"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wkv_trace_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{report_prefix}_WKV_TRACE_COMPARISON.md").write_text(
        _trace_comparison_markdown(report, report_prefix=report_prefix),
        encoding="utf-8",
    )


def _trace_comparison_markdown(report: Mapping[str, Any], *, report_prefix: str) -> str:
    lines = [
        f"# {report_prefix} WKV Trace Comparison",
        "",
        f"- First divergent stage: `{report.get('first_divergent_stage')}`",
        f"- First divergent layer: `{report.get('first_divergent_layer')}`",
        f"- First divergent head: `{report.get('first_divergent_head')}`",
        f"- First divergent token: `{report.get('first_divergent_token')}`",
        "- First divergent max abs error: "
        f"`{report.get('first_divergent_max_abs_error')}`",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("rows", [])[:200]:
        lines.append(
            f"- {row['case']} / L{row['layer']} / H{row['head']} / "
            f"T{row['token_index']} / {row['stage']}: {row['status']} "
            f"(max_abs={row['max_abs_error']})"
        )
    return "\n".join(lines)


def trace_row_by_stage(
    entries: list[dict[str, Any]],
    *,
    stage: str,
    layer: int | None = None,
    head: int | None = None,
    token_index: int | None = None,
) -> dict[str, Any] | None:
    for entry in entries:
        if (
            entry.get("stage") == stage
            and entry.get("layer") == layer
            and entry.get("head") == head
            and entry.get("token_index") == token_index
        ):
            return entry
    return None
