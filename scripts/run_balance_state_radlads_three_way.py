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

from qrwkv_xla.parity.radlads_balance_state_three_way import (
    DEFAULT_OUT,
    DEFAULT_QRWKV_EXPERIMENTAL_MODE,
    DEFAULT_QRWKV_OFF_MODE,
    DEFAULT_RADLADS_TRACE,
    run_balance_state_radlads_three_way,
)

DEFAULT_FIXTURE_MANIFEST = Path("artifacts/p54_confirmation/fixtures/manifest.json")
DEFAULT_RADLADS_OUTPUTS = Path(
    "artifacts/p54_confirmation/radlads_outputs/manifest.json"
)
DEFAULT_QRWKV_OUTPUTS = Path("artifacts/p54_confirmation/qrwkv_outputs/manifest.json")
DEFAULT_P65_REPORT = Path(
    "artifacts/p65_balance_state_experiment/balance_state_experiment_report.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P66 three-way compare RADLADS vs QRWKV off vs experimental "
            "balance-state parity."
        )
    )
    parser.add_argument(
        "--fixture-manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST
    )
    parser.add_argument("--radlads-outputs", type=Path, default=DEFAULT_RADLADS_OUTPUTS)
    parser.add_argument("--qrwkv-outputs", type=Path, default=DEFAULT_QRWKV_OUTPUTS)
    parser.add_argument("--p65-report", type=Path, default=DEFAULT_P65_REPORT)
    parser.add_argument("--radlads-trace", type=Path, default=DEFAULT_RADLADS_TRACE)
    parser.add_argument("--p65-off", type=Path, default=DEFAULT_QRWKV_OFF_MODE)
    parser.add_argument(
        "--p65-experimental", type=Path, default=DEFAULT_QRWKV_EXPERIMENTAL_MODE
    )
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--mode", choices=("full", "stepwise", "both"), default="both")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--out", "--out-dir", dest="out", type=Path, default=DEFAULT_OUT
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-real-artifacts", action="store_true")
    parser.add_argument("--radlads-repo", type=Path, default=None)
    parser.add_argument("--rerun-radlads", action="store_true")
    parser.add_argument("--rerun-qrwkv-off", action="store_true")
    parser.add_argument("--rerun-qrwkv-experimental", action="store_true")
    parser.add_argument("--balance-state-mode-name", type=str, default="experimental")
    args = parser.parse_args()

    case_list = (
        None if args.cases is None else [item for item in args.cases.split(",") if item]
    )

    report = run_balance_state_radlads_three_way(
        radlads_trace=args.radlads_trace,
        qrwkv_off_mode=args.p65_off,
        qrwkv_experimental_mode=args.p65_experimental,
        cases=case_list,
        mode=args.mode,
        layer=args.layer,
        head=args.head,
        max_tokens=args.max_tokens,
        strict_real_artifacts=args.strict_real_artifacts,
        balance_state_mode_name=args.balance_state_mode_name,
        radlads_repo=args.radlads_repo,
        rerun_radlads=args.rerun_radlads,
        rerun_qrwkv_off=args.rerun_qrwkv_off,
        rerun_qrwkv_experimental=args.rerun_qrwkv_experimental,
        out_dir=args.out,
        overwrite=args.overwrite,
    )
    print(f"wrote P66 three-way comparison to {args.out}")
    print(f"status={report['overall_status']}")
    print(f"recommendation={report['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
