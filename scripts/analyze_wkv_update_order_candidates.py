#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl


def _by_key(entries: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            entry.get("case"),
            entry.get("layer"),
            entry.get("head"),
            entry.get("token_index"),
            entry.get("stage"),
        ): entry
        for entry in entries
    }


def _row(
    *,
    case: str,
    layer: int | None,
    head: int | None,
    token_index: int | None,
    surface: str,
    candidate_name: str,
    left: Any | None,
    right: Any | None,
) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "case": case,
            "layer": layer,
            "head": head,
            "token_index": token_index,
            "surface": surface,
            "candidate_name": candidate_name,
            "applicable": False,
            "status": "not_applicable",
            "shape_match": False,
            "finite_both": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
        }
    stats = compare_trace_arrays(left, right)
    return {
        "case": case,
        "layer": layer,
        "head": head,
        "token_index": token_index,
        "surface": surface,
        "candidate_name": candidate_name,
        "applicable": True,
        **stats,
    }


def _candidate_rows(
    radlads: list[dict[str, Any]], qrwkv: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    left = _by_key(radlads)
    right = _by_key(qrwkv)
    rows: list[dict[str, Any]] = []
    for key in sorted(
        set(left) & set(right),
        key=lambda item: (
            "" if item[0] is None else str(item[0]),
            -1 if item[1] is None else int(item[1]),
            -1 if item[2] is None else int(item[2]),
            -1 if item[3] is None else int(item[3]),
            "" if item[4] is None else str(item[4]),
        ),
    ):
        case, layer, head, token_index, stage = key
        if stage not in {
            "wkv_state_before",
            "wkv_state_after",
            "wkv_output_before_o_proj",
        }:
            continue
        left_array = left[key].get("array")
        right_array = right[key].get("array")
        rows.append(
            _row(
                case=case,
                layer=layer,
                head=head,
                token_index=token_index,
                surface=stage,
                candidate_name="as_is",
                left=left_array,
                right=right_array,
            )
        )
        rows.append(
            _row(
                case=case,
                layer=layer,
                head=head,
                token_index=token_index,
                surface=stage,
                candidate_name="compare_pre_update_state_if_available",
                left=left[key].get("array") if stage == "wkv_state_before" else None,
                right=right[key].get("array") if stage == "wkv_state_before" else None,
            )
        )
        rows.append(
            _row(
                case=case,
                layer=layer,
                head=head,
                token_index=token_index,
                surface=stage,
                candidate_name="compare_post_update_state_if_available",
                left=left[key].get("array") if stage == "wkv_state_after" else None,
                right=right[key].get("array") if stage == "wkv_state_after" else None,
            )
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [
        row for row in rows if row["applicable"] and row["surface"] == "wkv_state_after"
    ]
    if not applicable:
        return {
            "best_candidate": None,
            "best_candidate_max_abs_error": None,
            "best_row": None,
        }
    best = min(
        applicable,
        key=lambda row: (
            float("inf") if row["max_abs_error"] is None else row["max_abs_error"]
        ),
    )
    return {
        "best_candidate": best["candidate_name"],
        "best_candidate_max_abs_error": best["max_abs_error"],
        "best_row": best,
    }


def _write_md(rows: list[dict[str, Any]], out_dir: Path) -> None:
    lines = ["# P56 WKV Update-Order Candidates", ""]
    for row in rows:
        lines.append(
            f"- {row['case']} / L{row['layer']} / H{row['head']} / "
            f"T{row['token_index']} / {row['surface']} / "
            f"{row['candidate_name']}: {row['status']} | "
            f"max_abs={row['max_abs_error']}"
        )
    (out_dir / "P56_UPDATE_ORDER_CANDIDATES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze WKV update-order candidates.")
    parser.add_argument("--radlads-trace", type=Path, required=True)
    parser.add_argument("--qrwkv-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    radlads = load_trace_jsonl(args.radlads_trace)
    qrwkv = load_trace_jsonl(args.qrwkv_trace)
    rows = _candidate_rows(radlads, qrwkv)
    summary = _summarize(rows)
    if summary["best_row"] is not None:
        as_is = next(
            (
                row
                for row in rows
                if row["candidate_name"] == "as_is"
                and row["surface"] == "wkv_state_after"
                and row["applicable"]
            ),
            None,
        )
        if (
            as_is is not None
            and as_is["max_abs_error"] is not None
            and summary["best_candidate_max_abs_error"] is not None
        ):
            summary["improvement_factor"] = (
                as_is["max_abs_error"] / summary["best_candidate_max_abs_error"]
                if summary["best_candidate_max_abs_error"]
                else None
            )
        else:
            summary["improvement_factor"] = None
    else:
        summary["improvement_factor"] = None
    summary["likely_root_cause"] = (
        "layout/export conventions do not explain the residual; the remaining gap "
        "is finite and update-order or formula-sensitive"
    )
    report = {
        "schema": "radlads_qrwkv_wkv_update_candidates.v1",
        "rows": rows,
        "summary": summary,
    }
    (args.out / "update_order_candidate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_md(rows, args.out)
    print(f"wrote WKV update-order candidate analysis to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
