#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_wkv_update_residual import (
    compare_update_residual_traces,
    load_update_residual_jsonl,
    write_update_residual_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P62 compare RADLADS/QRWKV WKV update residual traces."
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

    radlads_entries = load_update_residual_jsonl(args.radlads_trace)
    qrwkv_entries = load_update_residual_jsonl(args.qrwkv_trace)
    report = compare_update_residual_traces(
        radlads_entries,
        qrwkv_entries,
        atol=args.atol,
        rtol=args.rtol,
    )
    write_update_residual_reports(
        radlads_entries=radlads_entries,
        qrwkv_entries=qrwkv_entries,
        comparison_report=report,
        out_dir=args.out,
    )
    print(f"wrote P62 WKV update residual comparison to {args.out}")
    print(f"kernel_ready={report['kernel_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
