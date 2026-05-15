#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_same_run_update_ingredients import (
    compare_same_run_update_ingredients,
    load_same_run_update_ingredient_jsonl,
    write_same_run_update_ingredient_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P67 compare same-run WKV update ingredient traces."
    )
    parser.add_argument("--radlads-trace", type=Path, required=True)
    parser.add_argument("--qrwkv-off-trace", type=Path, required=True)
    parser.add_argument("--qrwkv-experimental-trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out-dir", "--out", type=Path, required=True)
    parser.add_argument("--strict-same-run", action="store_true", default=True)
    parser.add_argument(
        "--no-strict-same-run", dest="strict_same_run", action="store_false"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out_dir} is not empty; pass --overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = (
        json.loads(args.metadata.read_text(encoding="utf-8"))
        if args.metadata is not None and args.metadata.is_file()
        else None
    )
    radlads_entries = load_same_run_update_ingredient_jsonl(args.radlads_trace)
    qrwkv_off_entries = load_same_run_update_ingredient_jsonl(args.qrwkv_off_trace)
    qrwkv_experimental_entries = load_same_run_update_ingredient_jsonl(
        args.qrwkv_experimental_trace
    )
    report = compare_same_run_update_ingredients(
        radlads_entries=radlads_entries,
        qrwkv_off_entries=qrwkv_off_entries,
        qrwkv_experimental_entries=qrwkv_experimental_entries,
        metadata=metadata,
        strict_same_run=args.strict_same_run,
        atol=args.atol,
        rtol=args.rtol,
    )
    write_same_run_update_ingredient_reports(
        radlads_entries=radlads_entries,
        qrwkv_off_entries=qrwkv_off_entries,
        qrwkv_experimental_entries=qrwkv_experimental_entries,
        comparison_report=report,
        out_dir=args.out_dir,
    )
    print(f"wrote P67 same-run update ingredient comparison to {args.out_dir}")
    print(f"same_run_valid={report['same_run_valid']}")
    print(f"overall_status={report['overall_status']}")
    if report.get("recommended_next_phase") is not None:
        print(f"recommended_next_phase={report['recommended_next_phase']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
