from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WKV_STATE_PROVENANCE_SCHEMA = "radlads_qrwkv_wkv_state_provenance.v1"
WKV_STATE_PROVENANCE_REPORT_SCHEMA = "radlads_qrwkv_wkv_state_provenance_report.v1"

PROVENANCE_COMPARISONS = {
    "initial_state",
    "initial_state_handoff",
    "token_carry",
    "full_vs_stepwise",
    "mask_behavior",
}


@dataclass(frozen=True)
class WKVStateProvenanceRecord:
    schema: str
    case: str
    side: str
    comparison: str
    state_name: str
    left_label: str
    right_label: str
    layer: int | None
    token_index: int | None
    left_shape: list[int] | None
    right_shape: list[int] | None
    left_dtype: str | None
    right_dtype: str | None
    shape_match: bool
    dtype_match: bool
    finite_both: bool
    max_abs_error: float | None
    mean_abs_error: float | None
    max_relative_error: float | None
    allclose: bool
    status: str
    note: str | None = None
    left_array: Any | None = None
    right_array: Any | None = None


def compare_state_arrays(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    shape_match = tuple(left_array.shape) == tuple(right_array.shape)
    dtype_match = str(left_array.dtype) == str(right_array.dtype)
    finite_both = bool(np.isfinite(left_array).all() and np.isfinite(right_array).all())
    if not shape_match:
        return {
            "left_shape": [int(dim) for dim in left_array.shape],
            "right_shape": [int(dim) for dim in right_array.shape],
            "left_dtype": str(left_array.dtype),
            "right_dtype": str(right_array.dtype),
            "shape_match": False,
            "dtype_match": dtype_match,
            "finite_both": finite_both,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
            "status": "shape_mismatch",
        }
    diff = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
    denom = np.maximum(np.abs(right_array.astype(np.float64)), 1e-12)
    allclose = bool(np.allclose(left_array, right_array, atol=atol, rtol=rtol))
    return {
        "left_shape": [int(dim) for dim in left_array.shape],
        "right_shape": [int(dim) for dim in right_array.shape],
        "left_dtype": str(left_array.dtype),
        "right_dtype": str(right_array.dtype),
        "shape_match": True,
        "dtype_match": dtype_match,
        "finite_both": finite_both,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_relative_error": float(np.max(diff / denom)) if diff.size else 0.0,
        "allclose": allclose,
        "status": "pass" if allclose else ("non_finite" if not finite_both else "fail"),
    }


def make_provenance_record(
    *,
    case: str,
    side: str,
    comparison: str,
    state_name: str,
    left_label: str,
    right_label: str,
    left: Any,
    right: Any,
    layer: int | None = None,
    token_index: int | None = None,
    note: str | None = None,
    include_arrays: bool = True,
    max_inline_values: int = 256,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    if comparison not in PROVENANCE_COMPARISONS:
        raise ValueError(f"unknown provenance comparison: {comparison}")
    stats = compare_state_arrays(left, right, atol=atol, rtol=rtol)
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    record = WKVStateProvenanceRecord(
        schema=WKV_STATE_PROVENANCE_SCHEMA,
        case=case,
        side=side,
        comparison=comparison,
        state_name=state_name,
        left_label=left_label,
        right_label=right_label,
        layer=layer,
        token_index=token_index,
        note=note,
        left_array=_maybe_inline(left_array, include_arrays, max_inline_values),
        right_array=_maybe_inline(right_array, include_arrays, max_inline_values),
        **stats,
    )
    return asdict(record)


def trace_qrwkv_state_provenance(
    student: Any,
    params: Mapping[str, Any],
    input_ids: Any,
    *,
    attention_mask: Any | None = None,
    case: str = "synthetic",
    include_arrays: bool = True,
    max_inline_values: int = 256,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> list[dict[str, Any]]:
    token_ids = np.asarray(input_ids, dtype=np.int32)
    if token_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B,S], got {token_ids.shape}")
    mask = (
        None if attention_mask is None else np.asarray(attention_mask, dtype=np.int32)
    )
    if mask is not None and mask.shape != token_ids.shape:
        raise ValueError(
            f"attention_mask must have shape {token_ids.shape}, got {mask.shape}"
        )

    initial_state = student.init_state(int(token_ids.shape[0]))
    full_output, full_state = student.apply_with_state(
        dict(params),
        token_ids,
        attention_mask=mask,
        initial_state=initial_state,
    )
    implicit_output, implicit_state = student.apply_with_state(
        dict(params),
        token_ids,
        attention_mask=mask,
    )

    records: list[dict[str, Any]] = []
    records.extend(
        _state_records(
            case=case,
            side="qrwkv",
            comparison="initial_state",
            left_label="explicit_initial",
            right_label="fresh_initial",
            left_state=initial_state,
            right_state=student.init_state(int(token_ids.shape[0])),
            include_arrays=include_arrays,
            max_inline_values=max_inline_values,
            atol=atol,
            rtol=rtol,
        )
    )
    records.extend(
        _state_records(
            case=case,
            side="qrwkv",
            comparison="initial_state_handoff",
            left_label="explicit_initial_full_final",
            right_label="implicit_initial_full_final",
            left_state=full_state,
            right_state=implicit_state,
            include_arrays=include_arrays,
            max_inline_values=max_inline_values,
            atol=atol,
            rtol=rtol,
        )
    )
    records.extend(
        _output_records(
            case=case,
            side="qrwkv",
            comparison="initial_state_handoff",
            left_label="explicit_initial_full_output",
            right_label="implicit_initial_full_output",
            left_output=full_output,
            right_output=implicit_output,
            include_arrays=include_arrays,
            max_inline_values=max_inline_values,
            atol=atol,
            rtol=rtol,
        )
    )

    carry = initial_state
    step_states: list[Any] = []
    step_outputs = []
    for token_index in range(token_ids.shape[1]):
        before = carry
        step_mask = None if mask is None else mask[:, token_index : token_index + 1]
        step_output, carry = student.step(
            dict(params),
            token_ids[:, token_index : token_index + 1],
            before,
            attention_mask=step_mask,
        )
        step_outputs.append(step_output)
        expected_before = initial_state if token_index == 0 else step_states[-1]
        records.extend(
            _state_records(
                case=case,
                side="qrwkv",
                comparison="token_carry",
                left_label="step_input_state",
                right_label="previous_step_output_state",
                left_state=before,
                right_state=expected_before,
                token_index=token_index,
                include_arrays=include_arrays,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
        if mask is not None and not bool(np.all(step_mask)):
            records.extend(
                _state_records(
                    case=case,
                    side="qrwkv",
                    comparison="mask_behavior",
                    left_label="masked_step_input_state",
                    right_label="masked_step_output_state",
                    left_state=before,
                    right_state=carry,
                    token_index=token_index,
                    note=(
                        "Diagnostic delta across a masked token; this reports "
                        "current state handoff behavior and does not assert "
                        "that every cache surface must remain unchanged."
                    ),
                    include_arrays=include_arrays,
                    max_inline_values=max_inline_values,
                    atol=atol,
                    rtol=rtol,
                )
            )
        step_states.append(carry)

    stepwise_output = _concat_step_outputs(step_outputs)
    records.extend(
        _state_records(
            case=case,
            side="qrwkv",
            comparison="full_vs_stepwise",
            left_label="full_sequence_final_state",
            right_label="stepwise_final_state",
            left_state=full_state,
            right_state=carry,
            include_arrays=include_arrays,
            max_inline_values=max_inline_values,
            atol=atol,
            rtol=rtol,
        )
    )
    records.extend(
        _output_records(
            case=case,
            side="qrwkv",
            comparison="full_vs_stepwise",
            left_label="full_sequence_output",
            right_label="stepwise_output",
            left_output=full_output,
            right_output=stepwise_output,
            include_arrays=include_arrays,
            max_inline_values=max_inline_values,
            atol=atol,
            rtol=rtol,
        )
    )
    return records


def write_provenance_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_provenance_record(record)
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def load_provenance_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            validate_provenance_record(record)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        records.append(record)
    return records


def validate_provenance_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "case",
        "side",
        "comparison",
        "state_name",
        "left_label",
        "right_label",
        "shape_match",
        "dtype_match",
        "finite_both",
        "allclose",
        "status",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"provenance record missing required fields: {missing}")
    if record["schema"] != WKV_STATE_PROVENANCE_SCHEMA:
        raise ValueError(f"unsupported schema: {record['schema']}")
    if record["comparison"] not in PROVENANCE_COMPARISONS:
        raise ValueError(f"unknown comparison: {record['comparison']}")
    if record["status"] not in {"pass", "fail", "shape_mismatch", "non_finite"}:
        raise ValueError(f"unknown status: {record['status']}")


def compare_provenance_records(
    left_records: Iterable[Mapping[str, Any]],
    right_records: Iterable[Mapping[str, Any]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    left_by_key = {_record_key(record): record for record in left_records}
    right_by_key = {_record_key(record): record for record in right_records}
    rows = []
    for key in _sorted_keys(set(left_by_key) | set(right_by_key)):
        left = left_by_key.get(key)
        right = right_by_key.get(key)
        if left is None or right is None:
            rows.append(
                {
                    "case": key[0],
                    "comparison": key[1],
                    "state_name": key[2],
                    "layer": key[3],
                    "token_index": key[4],
                    "status": "missing_left" if left is None else "missing_right",
                    "max_abs_error": None,
                }
            )
            continue
        if (
            left.get("left_array") is None
            or right.get("left_array") is None
            or left.get("right_array") is None
            or right.get("right_array") is None
        ):
            rows.append(
                {
                    "case": key[0],
                    "comparison": key[1],
                    "state_name": key[2],
                    "layer": key[3],
                    "token_index": key[4],
                    "status": "missing_inline_array",
                    "max_abs_error": None,
                }
            )
            continue
        left_stats = compare_state_arrays(
            left["left_array"], right["left_array"], atol=atol, rtol=rtol
        )
        right_stats = compare_state_arrays(
            left["right_array"], right["right_array"], atol=atol, rtol=rtol
        )
        stats = dict(left_stats)
        stats["status"] = (
            "pass"
            if left_stats["status"] == "pass" and right_stats["status"] == "pass"
            else left_stats["status"]
            if left_stats["status"] != "pass"
            else right_stats["status"]
        )
        stats["allclose"] = bool(left_stats["allclose"] and right_stats["allclose"])
        stats["max_abs_error"] = max(
            float(left_stats["max_abs_error"] or 0.0),
            float(right_stats["max_abs_error"] or 0.0),
        )
        stats["mean_abs_error"] = max(
            float(left_stats["mean_abs_error"] or 0.0),
            float(right_stats["mean_abs_error"] or 0.0),
        )
        stats["max_relative_error"] = max(
            float(left_stats["max_relative_error"] or 0.0),
            float(right_stats["max_relative_error"] or 0.0),
        )
        rows.append(
            {
                "case": key[0],
                "comparison": key[1],
                "state_name": key[2],
                "layer": key[3],
                "token_index": key[4],
                **stats,
            }
        )
    first = next((row for row in rows if row["status"] != "pass"), None)
    return {
        "schema": WKV_STATE_PROVENANCE_REPORT_SCHEMA,
        "diagnostic_only": True,
        "row_count": len(rows),
        "status": "pass" if rows and first is None else "fail",
        "first_mismatch": first,
        "rows": rows,
        "atol": atol,
        "rtol": rtol,
    }


def summarize_provenance_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    status_counts: dict[str, int] = {}
    comparison_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        comparison = str(row["comparison"])
        comparison_counts[comparison] = comparison_counts.get(comparison, 0) + 1
    first = next((row for row in rows if row["status"] != "pass"), None)
    return {
        "schema": WKV_STATE_PROVENANCE_REPORT_SCHEMA,
        "diagnostic_only": True,
        "record_count": len(rows),
        "status": "pass" if rows and first is None else "fail",
        "status_counts": status_counts,
        "comparison_counts": comparison_counts,
        "first_mismatch": first,
        "records": rows,
    }


def write_provenance_reports(
    records: Iterable[Mapping[str, Any]],
    out_dir: Path,
    *,
    report_name: str = "wkv_state_provenance_report",
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    report = summarize_provenance_records(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report_name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P59_WKV_STATE_PROVENANCE.md").write_text(
        _provenance_markdown(report),
        encoding="utf-8",
    )
    return report


def _state_records(
    *,
    case: str,
    side: str,
    comparison: str,
    left_label: str,
    right_label: str,
    left_state: Any,
    right_state: Any,
    token_index: int | None = None,
    note: str | None = None,
    include_arrays: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    records = []
    for state_name in ("wkv_matrix_state", "shift_state", "next_position"):
        left = getattr(left_state, state_name)
        right = getattr(right_state, state_name)
        records.append(
            make_provenance_record(
                case=case,
                side=side,
                comparison=comparison,
                state_name=state_name,
                left_label=left_label,
                right_label=right_label,
                left=left,
                right=right,
                token_index=token_index,
                note=note,
                include_arrays=include_arrays,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
    return records


def _output_records(
    *,
    case: str,
    side: str,
    comparison: str,
    left_label: str,
    right_label: str,
    left_output: Any,
    right_output: Any,
    include_arrays: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    records = [
        make_provenance_record(
            case=case,
            side=side,
            comparison=comparison,
            state_name="hidden_states",
            left_label=left_label,
            right_label=right_label,
            left=left_output.hidden_states,
            right=right_output.hidden_states,
            include_arrays=include_arrays,
            max_inline_values=max_inline_values,
            atol=atol,
            rtol=rtol,
        )
    ]
    if (
        getattr(left_output, "logits", None) is not None
        or getattr(right_output, "logits", None) is not None
    ):
        records.append(
            make_provenance_record(
                case=case,
                side=side,
                comparison=comparison,
                state_name="logits",
                left_label=left_label,
                right_label=right_label,
                left=left_output.logits,
                right=right_output.logits,
                include_arrays=include_arrays,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
    return records


def _concat_step_outputs(step_outputs: list[Any]) -> Any:
    from qrwkv_xla.students.base import StudentOutput

    hidden = np.concatenate(
        [np.asarray(output.hidden_states) for output in step_outputs], axis=2
    )
    logits_values = [getattr(output, "logits", None) for output in step_outputs]
    logits = None
    if all(value is not None for value in logits_values):
        logits = np.concatenate([np.asarray(value) for value in logits_values], axis=1)
    return StudentOutput(hidden_states=hidden, logits=logits, mixer_outputs=None)


def _record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("case"),
        record.get("comparison"),
        record.get("state_name"),
        record.get("layer"),
        record.get("token_index"),
    )


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        keys,
        key=lambda key: (
            "" if key[0] is None else str(key[0]),
            "" if key[1] is None else str(key[1]),
            "" if key[2] is None else str(key[2]),
            -1 if key[3] is None else int(key[3]),
            -1 if key[4] is None else int(key[4]),
        ),
    )


def _maybe_inline(
    value: np.ndarray,
    include_arrays: bool,
    max_inline_values: int,
) -> Any | None:
    if not include_arrays or value.size > max_inline_values:
        return None
    return value.tolist()


def _provenance_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P59 WKV State Provenance",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Records: `{report.get('record_count')}`",
        f"- Diagnostic only: `{report.get('diagnostic_only')}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(dict(report.get("status_counts", {})).items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Comparison Counts", ""])
    for comparison, count in sorted(dict(report.get("comparison_counts", {})).items()):
        lines.append(f"- `{comparison}`: `{count}`")
    first = report.get("first_mismatch")
    if first is not None:
        lines.extend(
            [
                "",
                "## First Non-Passing Record",
                "",
                f"- Comparison: `{first.get('comparison')}`",
                f"- State: `{first.get('state_name')}`",
                f"- Token: `{first.get('token_index')}`",
                f"- Status: `{first.get('status')}`",
                f"- Max abs error: `{first.get('max_abs_error')}`",
            ]
        )
    return "\n".join(lines) + "\n"
