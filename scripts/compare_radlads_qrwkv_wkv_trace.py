#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qrwkv_xla.parity.radlads_wkv_trace import (
    compare_trace_entries,
    load_trace_jsonl,
    write_trace_comparison_reports,
)

STAGE_ORDER = {
    "input_to_attention": 0,
    "pre_attention_norm": 1,
    "q": 2,
    "k": 3,
    "v": 4,
    "receptance_or_r": 5,
    "decay_raw": 6,
    "log_w": 7,
    "decay_after_transform": 8,
    "a_or_iclr_raw": 9,
    "a_or_iclr_after_transform": 10,
    "g_or_gate_raw": 11,
    "g_or_gate_after_activation": 12,
    "value_before_v_first_mix": 13,
    "value_after_v_first_mix": 14,
    "wkv_state_before": 15,
    "wkv_update_outer_or_term": 16,
    "wkv_decay_applied": 17,
    "wkv_state_after": 18,
    "wkv_output_before_o_proj": 19,
    "o_proj_output": 20,
    "attention_output": 21,
    "layer_output": 22,
    "logits": 23,
}


def _growth_by_token(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    token_map: dict[int | None, float] = {}
    for row in rows:
        if row.get("stage") != "wkv_state_after" or row.get("max_abs_error") is None:
            continue
        token = row.get("token_index")
        current = token_map.get(token)
        value = float(row["max_abs_error"])
        if current is None or value > current:
            token_map[token] = value
    return [
        {"token_index": token, "max_abs_error": value}
        for token, value in sorted(
            token_map.items(), key=lambda item: -1 if item[0] is None else item[0]
        )
    ]


def _first_divergent_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    divergent = [row for row in rows if row["status"] != "pass"]
    if not divergent:
        return None
    divergent.sort(
        key=lambda row: (
            STAGE_ORDER.get(row["stage"], 999),
            -1 if row["layer"] is None else int(row["layer"]),
            -1 if row["head"] is None else int(row["head"]),
            -1 if row["token_index"] is None else int(row["token_index"]),
        )
    )
    return divergent[0]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P56 WKV Trace Comparison",
        "",
        f"- First divergent stage: `{report.get('first_divergent_stage')}`",
        f"- First divergent layer: `{report.get('first_divergent_layer')}`",
        f"- First divergent head: `{report.get('first_divergent_head')}`",
        f"- First divergent token: `{report.get('first_divergent_token')}`",
        "- First divergent max abs error: "
        f"`{report.get('first_divergent_max_abs_error')}`",
        "",
        "## Growth by token",
        "",
    ]
    for row in report.get("divergence_growth_over_tokens", []):
        lines.append(f"- token `{row['token_index']}`: `{row['max_abs_error']}`")
    lines.extend(["", "## Sample rows", ""])
    for row in report.get("rows", [])[:160]:
        lines.append(
            f"- {row['case']} / L{row['layer']} / H{row['head']} / "
            f"T{row['token_index']} / {row['stage']}: {row['status']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RADLADS and QRWKV WKV traces."
    )
    parser.add_argument("--radlads-trace", type=Path, required=True)
    parser.add_argument("--qrwkv-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    radlads = load_trace_jsonl(args.radlads_trace)
    qrwkv = load_trace_jsonl(args.qrwkv_trace)
    report = compare_trace_entries(radlads, qrwkv, atol=args.atol, rtol=args.rtol)
    first = _first_divergent_row(report["rows"])
    if first is not None:
        report["first_divergent_stage"] = first["stage"]
        report["first_divergent_layer"] = first["layer"]
        report["first_divergent_head"] = first["head"]
        report["first_divergent_token"] = first["token_index"]
        report["first_divergent_max_abs_error"] = first["max_abs_error"]
    report["cases_traced"] = len(
        {entry["case"] for entry in radlads} | {entry["case"] for entry in qrwkv}
    )
    report["surface_status_counts"] = {
        "pass": sum(1 for row in report["rows"] if row["status"] == "pass"),
        "fail": sum(1 for row in report["rows"] if row["status"] == "fail"),
        "shape_mismatch": sum(
            1 for row in report["rows"] if row["status"] == "shape_mismatch"
        ),
        "non_finite": sum(1 for row in report["rows"] if row["status"] == "non_finite"),
    }
    report["divergence_growth_over_tokens"] = _growth_by_token(report["rows"])
    write_trace_comparison_reports(report, args.out)
    (args.out / "P56_WKV_TRACE_COMPARISON.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(f"wrote WKV trace comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
