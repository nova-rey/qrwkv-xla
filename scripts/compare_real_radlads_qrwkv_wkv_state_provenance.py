#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_wkv_state_provenance import (
    compare_provenance_records,
    load_provenance_jsonl,
)

DEFAULT_OUT = Path("artifacts/p60_real_wkv_state_provenance/comparison")
DEFAULT_RADLADS_TRACE = Path(
    "artifacts/p60_real_wkv_state_provenance/real_wkv_state_provenance_radlads.jsonl"
)
DEFAULT_QRWKV_TRACE = Path(
    "artifacts/p60_real_wkv_state_provenance/real_wkv_state_provenance_qrwkv.jsonl"
)

COMPARISON_ORDER = {
    "initial_state": 0,
    "initial_state_handoff": 1,
    "token_carry": 2,
    "full_vs_stepwise": 3,
    "mask_behavior": 4,
}
STATE_ORDER = {
    "wkv_matrix_state": 0,
    "shift_state": 1,
    "next_position": 2,
    "hidden_states": 3,
    "logits": 4,
}


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("case")),
        COMPARISON_ORDER.get(str(row.get("comparison")), 999),
        STATE_ORDER.get(str(row.get("state_name")), 999),
        -1 if row.get("layer") is None else int(row["layer"]),
        -1 if row.get("token_index") is None else int(row["token_index"]),
    )


def _first_divergence(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    divergent = [row for row in rows if row.get("status") != "pass"]
    if not divergent:
        return None
    return sorted(divergent, key=_row_sort_key)[0]


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _case_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        case = str(row.get("case"))
        counts[case] = counts.get(case, 0) + 1
    return counts


def _all_pass(
    rows: list[dict[str, Any]],
    *,
    comparison: str | None = None,
    state_name: str | None = None,
) -> bool:
    filtered = rows
    if comparison is not None:
        filtered = [row for row in filtered if row.get("comparison") == comparison]
    if state_name is not None:
        filtered = [row for row in filtered if row.get("state_name") == state_name]
    return bool(filtered) and all(row.get("status") == "pass" for row in filtered)


def _first_status(
    rows: list[dict[str, Any]],
    *,
    comparison: str | None = None,
    state_name: str | None = None,
) -> str | None:
    filtered = rows
    if comparison is not None:
        filtered = [row for row in filtered if row.get("comparison") == comparison]
    if state_name is not None:
        filtered = [row for row in filtered if row.get("state_name") == state_name]
    if not filtered:
        return None
    for row in sorted(filtered, key=_row_sort_key):
        if row.get("status") != "pass":
            return str(row.get("status"))
    return "pass"


def _filtered(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict]:
    rows = records
    if args.case:
        rows = [row for row in rows if row.get("case") in set(args.case)]
    if args.mode:
        rows = [row for row in rows if row.get("comparison") in set(args.mode)]
    return rows


def _augment_report(
    report: dict[str, Any],
    *,
    radlads_trace: Path,
    qrwkv_trace: Path,
    radlads_records: list[dict[str, Any]],
    qrwkv_records: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = sorted(report["rows"], key=_row_sort_key)
    first = _first_divergence(rows)
    cases = sorted(_case_counts(rows).keys())
    first_real_divergence = first or {}
    initial_state_match_real = _all_pass(rows, comparison="initial_state")
    token_carry_match_real = _all_pass(rows, comparison="token_carry")
    full_vs_stepwise_match_real = _all_pass(rows, comparison="full_vs_stepwise")
    mask_behavior_match_real = _all_pass(rows, comparison="mask_behavior")
    state_slot_match_real = _all_pass(
        rows, state_name="wkv_matrix_state"
    ) and _all_pass(rows, state_name="shift_state")
    state_export_match_real = _all_pass(
        rows, comparison="initial_state_handoff", state_name="wkv_matrix_state"
    ) and _all_pass(rows, comparison="full_vs_stepwise", state_name="wkv_matrix_state")
    hidden_status = _first_status(
        rows, comparison="full_vs_stepwise", state_name="hidden_states"
    )
    hidden_states_explained = (
        "comparison_only"
        if hidden_status == "shape_mismatch"
        else ("yes" if hidden_status == "pass" else "unknown")
    )
    suspected_root_cause = (
        "cached WKV matrix-state export/slot convention mismatch, with hidden-state "
        "shape comparison noise downstream"
        if not state_export_match_real
        else "unknown"
    )
    report = {
        **report,
        "schema": "radlads_qrwkv_p60_real_wkv_state_provenance_comparison.v1",
        "phase": "P60",
        "overall_status": "pass" if rows and first is None else "fail",
        "status": "pass" if rows and first is None else "fail",
        "cases_compared": cases,
        "first_real_divergence_case": first_real_divergence.get("case"),
        "first_real_divergence_mode": first_real_divergence.get("comparison"),
        "first_real_divergence_layer": first_real_divergence.get("layer"),
        "first_real_divergence_token": first_real_divergence.get("token_index"),
        "first_real_divergence_head": first_real_divergence.get("head"),
        "first_real_divergence_stage": first_real_divergence.get("comparison"),
        "first_real_divergence_max_abs_error": first_real_divergence.get(
            "max_abs_error"
        ),
        "initial_state_match_real": initial_state_match_real,
        "token_carry_match_real": token_carry_match_real,
        "full_vs_stepwise_match_real": full_vs_stepwise_match_real,
        "mask_behavior_match_real": mask_behavior_match_real,
        "state_slot_match_real": state_slot_match_real,
        "state_export_match_real": state_export_match_real,
        "hidden_states_explained": hidden_states_explained,
        "suspected_root_cause": suspected_root_cause,
        "fix_recommended": (
            "P61: normalize the real WKV matrix-state export/slot convention, then "
            "recheck hidden-state shape comparison and preserve the P58 log_w fix"
        ),
        "kernel_ready": False,
        "rows": rows,
        "row_count": len(rows),
        "first_divergence": first,
        "first_mismatch": first,
        "status_counts": _status_counts(rows),
        "case_counts": _case_counts(rows),
        "inputs": {
            "radlads_trace": str(radlads_trace),
            "qrwkv_trace": str(qrwkv_trace),
            "radlads_records": len(radlads_records),
            "qrwkv_records": len(qrwkv_records),
        },
        "metadata_labels": {
            "real_artifact_trace": True,
            "synthetic_trace": False,
            "derived_from_cached_outputs": True,
            "regenerated_live_outputs": False,
        },
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    first = report.get("first_divergence")
    lines = [
        "# P60 Real WKV State Provenance Comparison",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Rows: `{report.get('row_count')}`",
        f"- First divergence: `{None if first is None else first.get('case')}`",
        f"- Suspected root cause: `{report.get('suspected_root_cause')}`",
    ]
    if first is not None:
        lines.extend(
            [
                f"- First comparison: `{first.get('comparison')}`",
                f"- First state: `{first.get('state_name')}`",
                f"- First layer: `{first.get('layer')}`",
                f"- First token: `{first.get('token_index')}`",
                f"- First status: `{first.get('status')}`",
                f"- First max abs error: `{first.get('max_abs_error')}`",
            ]
        )
    lines.extend(["", "## Status Counts", ""])
    for status, count in sorted(report.get("status_counts", {}).items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Sample Rows", ""])
    for row in report.get("rows", [])[:160]:
        lines.append(
            f"- `{row['case']}` `{row['comparison']}` `{row['state_name']}` "
            f"L`{row['layer']}` T`{row['token_index']}`: `{row['status']}` "
            f"max_abs=`{row.get('max_abs_error')}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P60 compare real RADLADS and QRWKV WKV state provenance traces."
    )
    parser.add_argument("--radlads-trace", type=Path, default=DEFAULT_RADLADS_TRACE)
    parser.add_argument("--qrwkv-trace", type=Path, default=DEFAULT_QRWKV_TRACE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--mode",
        action="append",
        choices=(
            "initial_state",
            "initial_state_handoff",
            "token_carry",
            "full_vs_stepwise",
            "mask_behavior",
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.radlads_trace.is_file():
        raise SystemExit(f"missing RADLADS trace: {args.radlads_trace}")
    if not args.qrwkv_trace.is_file():
        raise SystemExit(f"missing QRWKV trace: {args.qrwkv_trace}")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    radlads_records = _filtered(load_provenance_jsonl(args.radlads_trace), args)
    qrwkv_records = _filtered(load_provenance_jsonl(args.qrwkv_trace), args)
    if not radlads_records or not qrwkv_records:
        raise SystemExit("case/mode filters produced no paired provenance rows")

    report = compare_provenance_records(
        radlads_records,
        qrwkv_records,
        atol=args.atol,
        rtol=args.rtol,
    )
    report = _augment_report(
        report,
        radlads_trace=args.radlads_trace,
        qrwkv_trace=args.qrwkv_trace,
        radlads_records=radlads_records,
        qrwkv_records=qrwkv_records,
    )
    report_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.out / "p60_real_wkv_state_provenance_report.json").write_text(
        report_json,
        encoding="utf-8",
    )
    (args.out / "P60_REAL_WKV_STATE_PROVENANCE.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(f"wrote P60 real WKV state provenance comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
