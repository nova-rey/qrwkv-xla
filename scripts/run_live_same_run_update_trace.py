#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_RADLADS_REPO = ROOT.parent / "_refs" / "RADLADS"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_live_same_run_trace import (
    DEFAULT_OUT,
    run_live_same_run_trace,
)


def _optional_int(value: str) -> int | None:
    return None if value == "all" else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P68 generate strict-live same-run WKV update ingredient traces. "
            "No P66/P67 rows are used as trace source of truth."
        )
    )
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict-live", action="store_true", required=True)
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--parameter-manifest", type=Path, default=None)
    parser.add_argument("--fixture-parameter-key", default=None)
    parser.add_argument("--radlads-repo", type=Path, default=DEFAULT_RADLADS_REPO)
    parser.add_argument("--cases", default="")
    parser.add_argument("--mode", default="both")
    parser.add_argument("--layer", default="all")
    parser.add_argument("--head", default="all")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--fail-on-missing-critical-stage", action="store_true")
    args = parser.parse_args()

    if not (args.parameters or args.parameter_manifest or args.fixture_parameter_key):
        parser.error(
            "one of --parameters, --parameter-manifest, or "
            "--fixture-parameter-key is required"
        )

    cases = [item for item in args.cases.split(",") if item] or None
    report = run_live_same_run_trace(
        fixture_manifest=args.fixture_manifest,
        out_dir=args.out,
        parameters=args.parameters,
        parameter_manifest=args.parameter_manifest,
        fixture_parameter_key=args.fixture_parameter_key,
        radlads_repo=args.radlads_repo,
        cases=cases,
        mode=args.mode,
        layer=_optional_int(args.layer),
        head=_optional_int(args.head),
        max_tokens=args.max_tokens,
        strict_live=args.strict_live,
        overwrite=args.overwrite,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(f"wrote P68 live same-run trace to {args.out}")
    print(f"same_run_valid={report['same_run_valid']}")
    print(f"first_divergent_stage={report['first_divergent_stage']}")
    print(f"live_rows_captured_radlads={report['live_rows_captured_radlads']}")
    print(f"live_rows_captured_qrwkv_off={report['live_rows_captured_qrwkv_off']}")
    print(
        "live_rows_captured_qrwkv_experimental="
        f"{report['live_rows_captured_qrwkv_experimental']}"
    )
    print(f"unavailable_minimum_stages={len(report['unavailable_minimum_stages'])}")
    print(f"recommended_next_phase={report['recommended_next_phase']}")
    if args.fail_on_missing_critical_stage and report["unavailable_rows"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
