from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_trace import WKVTraceCollector, load_trace_jsonl

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - optional CI dependency
    jax = None

LOG_W_TRACE_SCHEMA = "radlads_log_w_trace.v1"
LOG_W_PARITY_SCHEMA = "radlads_qrwkv_log_w_parity.v1"
LOG_W_CANDIDATE_SCHEMA = "radlads_qrwkv_log_w_candidate_report.v1"

LOG_W_STAGE_NAMES = {"log_w", "low_rank_decay"}
W_SOURCE_STAGE_NAMES = {"w_head_split", "w2_projection", "w_projection", "decay_raw"}


def log_w_replay_profile_for_case(case: Mapping[str, Any]):
    from qrwkv_xla.parity.radlads_replay import replay_profile_for_case

    profile = replay_profile_for_case(case)
    if str(case.get("name")) == "tiny_no_mask":
        return replace(
            profile,
            low_rank_decay=True,
            reason=(
                "P58 log_w caliper keeps the RADLADS low-rank decay path active "
                "for the tiny_no_mask source trace."
            ),
        )
    return profile


@dataclass(frozen=True)
class LogWRecord:
    case: str
    side: str
    layer: int | None
    head: int | None
    token_index: int | None
    name: str
    shape: list[int]
    dtype: str
    finite: bool
    array: Any


def capture_qrwkv_log_w_from_current_run(
    student: Any,
    params: Mapping[str, Any],
    input_ids: Any,
    *,
    attention_mask: Any | None = None,
    case: str = "tiny_no_mask",
    max_inline_values: int = 4096,
) -> dict[str, Any]:
    collector = WKVTraceCollector(
        case=case,
        side="qrwkv",
        include_arrays=True,
        max_inline_values=max_inline_values,
    )
    output, state = student.apply_with_state(
        dict(params),
        input_ids,
        attention_mask=attention_mask,
        diagnostics=collector,
    )
    hidden = getattr(output, "hidden_states", None)
    state_array = None if state is None else getattr(state, "wkv_matrix_state", None)
    return {
        "schema": LOG_W_TRACE_SCHEMA,
        "case": case,
        "side": "qrwkv",
        "log_w": [
            asdict(row)
            for row in log_w_records_from_trace_entries(
                collector.entries,
                side="qrwkv",
            )
        ],
        "w_source": [
            asdict(row)
            for row in w_source_records_from_trace_entries(
                collector.entries,
                side="qrwkv",
            )
        ],
        "diagnostic_entry_count": len(collector.entries),
        "output_hidden_shape": _shape_or_none(hidden),
        "state_shape": _shape_or_none(state_array),
    }


def load_radlads_log_w_from_jsonl(path: Path) -> list[LogWRecord]:
    rows = log_w_records_from_trace_entries(load_trace_jsonl(path), side="radlads")
    if not rows:
        raise ValueError(f"{path} does not contain RADLADS log_w trace rows")
    return rows


def log_w_records_from_trace_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    side: str | None = None,
) -> list[LogWRecord]:
    rows = []
    for entry in entries:
        if side is not None and entry.get("side") != side:
            continue
        if entry.get("stage") not in LOG_W_STAGE_NAMES:
            continue
        rows.append(_record_from_entry(entry))
    return rows


def w_source_records_from_trace_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    side: str | None = None,
) -> list[LogWRecord]:
    rows = []
    for entry in entries:
        if side is not None and entry.get("side") != side:
            continue
        if entry.get("stage") not in W_SOURCE_STAGE_NAMES:
            continue
        if entry.get("array") is None:
            continue
        array = np.asarray(entry["array"])
        if entry.get("head") is None and array.ndim == 3:
            for head_index in range(array.shape[1]):
                head_entry = dict(entry)
                head_entry["head"] = head_index
                head_entry["array"] = array[:, head_index, :].tolist()
                rows.append(_record_from_entry(head_entry))
            continue
        rows.append(_record_from_entry(entry))
    return rows


def compare_log_w_records(
    radlads_rows: list[LogWRecord],
    qrwkv_rows: list[LogWRecord],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    left_by_key = {_record_key(row): row for row in radlads_rows}
    right_by_key = {_record_key(row): row for row in qrwkv_rows}
    rows = []
    for key in _sorted_keys(set(left_by_key) | set(right_by_key)):
        left = left_by_key.get(key)
        right = right_by_key.get(key)
        if left is None or right is None:
            missing = "source" if left is None else "qrwkv"
            rows.append(_missing_row(key, missing=missing))
            continue
        rows.append(
            {
                "case": key[0],
                "layer": key[1],
                "head": key[2],
                "token_index": key[3],
                **_compare_arrays(left.array, right.array, atol=atol, rtol=rtol),
            }
        )
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    fail_count = sum(1 for row in rows if row["status"] != "pass")
    first = next((row for row in rows if row["status"] != "pass"), None)
    return {
        "schema": LOG_W_PARITY_SCHEMA,
        "claim": (
            "P57 is a source-audit/caliper report for log_w only. It does not "
            "patch model math or claim broader RADLADS parity."
        ),
        "status": "pass" if fail_count == 0 and rows else "fail",
        "diagnostic_only": True,
        "counts": {"rows": len(rows), "pass": pass_count, "fail": fail_count},
        "all_finite": all(row.get("finite_both") is True for row in rows)
        if rows
        else False,
        "first_mismatch": first,
        "rows": rows,
        "atol": atol,
        "rtol": rtol,
    }


def evaluate_log_w_candidate_variants(
    *,
    radlads_rows: list[LogWRecord],
    qrwkv_w_rows: list[LogWRecord],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    targets = {_record_key(row): row for row in radlads_rows}
    sources = {_record_key(row): row for row in qrwkv_w_rows}
    rows = []
    for spec in _candidate_specs():
        variant_rows = []
        for key in _sorted_keys(set(targets) & set(sources)):
            target = targets[key]
            source = sources[key]
            candidate = _candidate_log_w(np.asarray(source.array), spec)
            candidate = _orient_candidate(candidate, np.asarray(target.array), spec)
            variant_rows.append(
                {
                    "case": key[0],
                    "layer": key[1],
                    "head": key[2],
                    "token_index": key[3],
                    **_compare_arrays(target.array, candidate, atol=atol, rtol=rtol),
                }
            )
        pass_count = sum(1 for row in variant_rows if row["status"] == "pass")
        max_abs = [
            row["max_abs_error"]
            for row in variant_rows
            if row.get("max_abs_error") is not None
        ]
        rows.append(
            {
                "candidate_name": _candidate_name(spec),
                **spec,
                "row_count": len(variant_rows),
                "pass_count": pass_count,
                "status": "pass"
                if variant_rows and pass_count == len(variant_rows)
                else "fail",
                "max_abs_error": None if not max_abs else float(max(max_abs)),
                "mean_abs_error": _mean_existing(
                    row.get("mean_abs_error") for row in variant_rows
                ),
                "rows": variant_rows,
            }
        )
    rows.sort(
        key=lambda row: (
            row["status"] != "pass",
            float("inf")
            if row["max_abs_error"] is None
            else float(row["max_abs_error"]),
            row["candidate_name"],
        )
    )
    best = rows[0] if rows else None
    return {
        "schema": LOG_W_CANDIDATE_SCHEMA,
        "claim": (
            "Candidate formulas are calipers against captured source traces. "
            "They are not model patches."
        ),
        "candidate_count": len(rows),
        "best_candidate": None if best is None else best["candidate_name"],
        "best_candidate_status": None if best is None else best["status"],
        "best_candidate_max_abs_error": None if best is None else best["max_abs_error"],
        "rows": rows,
        "atol": atol,
        "rtol": rtol,
    }


def write_log_w_reports(
    *,
    parity_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    radlads_rows: list[LogWRecord],
    qrwkv_rows: list[LogWRecord],
    qrwkv_w_rows: list[LogWRecord],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log_w_parity_report.json").write_text(
        json.dumps(parity_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "log_w_candidate_report.json").write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P57_LOG_W_PARITY.md").write_text(
        _parity_markdown(parity_report),
        encoding="utf-8",
    )
    (out_dir / "P57_LOG_W_CANDIDATES.md").write_text(
        _candidate_markdown(candidate_report),
        encoding="utf-8",
    )
    _write_npz(
        out_dir / "log_w_values.npz",
        radlads_rows=radlads_rows,
        qrwkv_rows=qrwkv_rows,
        qrwkv_w_rows=qrwkv_w_rows,
    )


def _record_from_entry(entry: Mapping[str, Any]) -> LogWRecord:
    for field in ("case", "side", "stage", "name", "array"):
        if field not in entry:
            raise ValueError(f"trace entry missing required field: {field}")
    if entry.get("array") is None:
        raise ValueError(f"trace entry has no inline array: {entry.get('name')}")
    array = np.asarray(entry["array"])
    summary = asdict(
        summarize_array(
            str(entry["name"]),
            array,
            stage=str(entry.get("stage")),
            layer=entry.get("layer"),
            time_index=entry.get("token_index"),
        )
    )
    return LogWRecord(
        case=str(entry["case"]),
        side=str(entry["side"]),
        layer=_optional_int(entry.get("layer")),
        head=_optional_int(entry.get("head")),
        token_index=_optional_int(entry.get("token_index")),
        name=str(entry["name"]),
        shape=[int(dim) for dim in array.shape],
        dtype=str(array.dtype),
        finite=bool(summary["nonfinite_count"] == 0),
        array=array.tolist(),
    )


def _compare_arrays(
    left: Any, right: Any, *, atol: float, rtol: float
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
            "reason": "array shapes differ",
            "left_shape": [int(dim) for dim in left_value.shape],
            "right_shape": [int(dim) for dim in right_value.shape],
        }
    diff = np.abs(left_value.astype(np.float64) - right_value.astype(np.float64))
    denom = np.maximum(np.abs(left_value.astype(np.float64)), 1e-12)
    allclose = bool(np.allclose(left_value, right_value, atol=atol, rtol=rtol))
    return {
        "shape_match": True,
        "dtype_match": dtype_match,
        "finite_both": finite_both,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_relative_error": float(np.max(diff / denom)) if diff.size else 0.0,
        "allclose": allclose,
        "status": "pass" if allclose else ("non_finite" if not finite_both else "fail"),
        "reason": None if allclose else "values differ",
    }


def _candidate_specs() -> list[dict[str, str]]:
    specs = []
    for orientation in ("as_is", "transpose_last_two"):
        for sign in ("negative", "positive"):
            for activation in ("sigmoid", "stable_sigmoid", "none"):
                for base_term in ("exp_neg_half", "one"):
                    for dtype in ("float32", "float64"):
                        for axis in ("as_is", "flatten_or_split_heads"):
                            specs.append(
                                {
                                    "orientation": orientation,
                                    "sign": sign,
                                    "activation": activation,
                                    "base_term": base_term,
                                    "dtype": dtype,
                                    "axis": axis,
                                }
                            )
    return specs


def _candidate_log_w(value: np.ndarray, spec: Mapping[str, str]) -> np.ndarray:
    dtype = np.float32 if spec["dtype"] == "float32" else np.float64
    w = np.asarray(value, dtype=dtype)
    if spec["activation"] == "sigmoid":
        activated = 1.0 / (1.0 + np.exp(-w))
    elif spec["activation"] == "stable_sigmoid":
        activated = np.exp(-np.logaddexp(0.0, -w))
    elif spec["activation"] == "none":
        activated = w
    else:  # pragma: no cover - guarded by local spec list
        raise ValueError(f"unknown activation: {spec['activation']}")
    base = np.exp(dtype(-0.5)) if spec["base_term"] == "exp_neg_half" else dtype(1.0)
    sign = dtype(-1.0) if spec["sign"] == "negative" else dtype(1.0)
    return np.asarray(sign * base * activated, dtype=dtype)


def _orient_candidate(
    candidate: np.ndarray, target: np.ndarray, spec: Mapping[str, str]
) -> np.ndarray:
    value = candidate
    if spec["axis"] == "flatten_or_split_heads":
        if value.ndim == 3 and target.ndim == 2 and value.shape[0] == target.shape[0]:
            value = value[:, 0, :]
        elif value.ndim == 2 and target.ndim == 3 and target.shape[0] == value.shape[0]:
            if value.shape[1] == target.shape[1] * target.shape[2]:
                value = value.reshape(target.shape)
    if spec["orientation"] == "transpose_last_two" and value.ndim >= 2:
        value = np.swapaxes(value, -1, -2)
    return value


def _candidate_name(spec: Mapping[str, str]) -> str:
    return (
        f"{spec['orientation']}__{spec['sign']}__{spec['activation']}__"
        f"{spec['base_term']}__{spec['dtype']}__{spec['axis']}"
    )


def _record_key(row: LogWRecord) -> tuple[Any, ...]:
    return (row.case, row.layer, row.head, row.token_index)


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        keys,
        key=lambda key: (
            str(key[0]),
            -1 if key[1] is None else int(key[1]),
            -1 if key[2] is None else int(key[2]),
            -1 if key[3] is None else int(key[3]),
        ),
    )


def _missing_row(key: tuple[Any, ...], *, missing: str) -> dict[str, Any]:
    return {
        "case": key[0],
        "layer": key[1],
        "head": key[2],
        "token_index": key[3],
        "status": f"missing_{missing}",
        "shape_match": False,
        "dtype_match": False,
        "finite_both": False,
        "max_abs_error": None,
        "mean_abs_error": None,
        "max_relative_error": None,
        "allclose": False,
        "reason": "log_w row exists on only one side",
    }


def _shape_or_none(value: Any) -> list[int] | None:
    if value is None:
        return None
    if jax is not None:
        value = jax.device_get(value)
    return [int(dim) for dim in np.asarray(value).shape]


def _mean_existing(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(np.mean(numeric))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _parity_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P57 log_w Decay Parity",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Diagnostic only: `{report.get('diagnostic_only')}`",
        f"- Row count: `{report.get('counts', {}).get('rows')}`",
        f"- Pass/fail: `{report.get('counts', {}).get('pass')}` / "
        f"`{report.get('counts', {}).get('fail')}`",
        f"- All finite: `{report.get('all_finite')}`",
        "",
        "P57 is a source-audit/caliper phase. No model patch is made by this report.",
        "",
        "## First mismatch",
        "",
        f"`{report.get('first_mismatch')}`",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("rows", [])[:160]:
        lines.append(
            "- `{case}` L`{layer}` H`{head}` T`{token}`: `{status}` "
            "max_abs=`{max_abs}`".format(
                case=row.get("case"),
                layer=row.get("layer"),
                head=row.get("head"),
                token=row.get("token_index"),
                status=row.get("status"),
                max_abs=row.get("max_abs_error"),
            )
        )
    return "\n".join(lines) + "\n"


def _candidate_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P57 log_w Candidate Formula Calipers",
        "",
        f"- Best candidate: `{report.get('best_candidate')}`",
        f"- Best status: `{report.get('best_candidate_status')}`",
        f"- Best max abs error: `{report.get('best_candidate_max_abs_error')}`",
        "",
        "These candidates evaluate orientation, sign, activation, base-term, "
        "dtype, and axis variants against captured RADLADS log_w rows.",
        "",
        "## Candidates",
        "",
    ]
    for row in report.get("rows", [])[:80]:
        lines.append(
            f"- `{row['candidate_name']}` status=`{row['status']}` "
            f"max_abs=`{row['max_abs_error']}` pass=`{row['pass_count']}`/"
            f"`{row['row_count']}`"
        )
    return "\n".join(lines) + "\n"


def _write_npz(
    path: Path,
    *,
    radlads_rows: list[LogWRecord],
    qrwkv_rows: list[LogWRecord],
    qrwkv_w_rows: list[LogWRecord],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for prefix, rows in (
        ("radlads_log_w", radlads_rows),
        ("qrwkv_log_w", qrwkv_rows),
        ("qrwkv_w_source", qrwkv_w_rows),
    ):
        for index, row in enumerate(rows):
            key = (
                f"{prefix}_{index}_case_{row.case}_layer_{row.layer}_head_"
                f"{row.head}_token_{row.token_index}"
            )
            arrays[key] = np.asarray(row.array)
    np.savez(path, **arrays)
