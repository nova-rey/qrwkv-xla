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

from qrwkv_xla.parity.radlads_wkv_state_provenance import (
    compare_provenance_records,
    load_provenance_jsonl,
)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P59 WKV State Provenance Comparison",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Rows: `{report.get('row_count')}`",
        f"- Diagnostic only: `{report.get('diagnostic_only')}`",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("rows", [])[:200]:
        lines.append(
            f"- {row['case']} / {row['comparison']} / {row['state_name']} / "
            f"T{row['token_index']}: {row['status']} "
            f"(max_abs={row.get('max_abs_error')})"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two P59 WKV state provenance JSONL traces."
    )
    parser.add_argument(
        "--radlads-trace", "--left", dest="left", type=Path, required=True
    )
    parser.add_argument(
        "--qrwkv-trace", "--right", dest="right", type=Path, required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    report = compare_provenance_records(
        load_provenance_jsonl(args.left),
        load_provenance_jsonl(args.right),
        atol=args.atol,
        rtol=args.rtol,
    )
    report_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.out / "wkv_state_provenance_report.json").write_text(
        report_json,
        encoding="utf-8",
    )
    (args.out / "wkv_state_provenance_comparison_report.json").write_text(
        report_json,
        encoding="utf-8",
    )
    markdown = _markdown(report)
    (args.out / "P59_WKV_STATE_PROVENANCE.md").write_text(
        markdown,
        encoding="utf-8",
    )
    (args.out / "P59_WKV_STATE_PROVENANCE_COMPARISON.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(f"wrote P59 WKV state provenance comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
