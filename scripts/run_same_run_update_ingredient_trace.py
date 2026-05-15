#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_FIXTURE_MANIFEST = ROOT / "artifacts/p54_confirmation/fixtures/manifest.json"
DEFAULT_PARAMETERS = ROOT / "artifacts/p54_confirmation/fixtures/radlads_parameters.npz"
DEFAULT_RADLADS_REPO = ROOT.parent / "_refs" / "RADLADS"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_same_run_update_ingredients import (
    DEFAULT_METADATA,
    DEFAULT_OUT,
    DEFAULT_QRWKV_EXPERIMENTAL_TRACE,
    DEFAULT_QRWKV_OFF_TRACE,
    DEFAULT_RADLADS_TRACE,
    run_same_run_update_ingredient_trace,
)


def _optional_int(value: str) -> int | None:
    return None if value == "all" else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P67 trace same-run WKV update ingredients from existing artifacts."
    )
    parser.add_argument("--out-dir", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--radlads-trace", type=Path, default=DEFAULT_RADLADS_TRACE)
    parser.add_argument("--qrwkv-off-trace", type=Path, default=DEFAULT_QRWKV_OFF_TRACE)
    parser.add_argument(
        "--qrwkv-experimental-trace",
        type=Path,
        default=DEFAULT_QRWKV_EXPERIMENTAL_TRACE,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--fixture-manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST
    )
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--parameter-manifest", type=Path, default=None)
    parser.add_argument("--radlads-repo", type=Path, default=DEFAULT_RADLADS_REPO)
    parser.add_argument("--qrwkv-root", type=Path, default=ROOT)
    parser.add_argument("--reuse-existing-if-same-run", action="store_true")
    parser.add_argument("--fail-on-unavailable-critical-stage", action="store_true")
    parser.add_argument("--cases", default="")
    parser.add_argument("--mode", choices=["full", "stepwise", "both"], default="both")
    parser.add_argument("--layer", default="all")
    parser.add_argument("--head", default="all")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--strict-same-run", action="store_true", default=True)
    parser.add_argument(
        "--no-strict-same-run", dest="strict_same_run", action="store_false"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    del args.mode

    cases = [item for item in args.cases.split(",") if item] or None
    report = run_same_run_update_ingredient_trace(
        out_dir=args.out_dir,
        radlads_trace=args.radlads_trace,
        qrwkv_off_trace=args.qrwkv_off_trace,
        qrwkv_experimental_trace=args.qrwkv_experimental_trace,
        metadata_path=args.metadata,
        fixture_manifest_path=args.fixture_manifest,
        parameter_manifest_or_npz_path=args.parameters or args.parameter_manifest,
        radlads_repo_path=args.radlads_repo,
        qrwkv_root_path=args.qrwkv_root,
        cases=cases,
        layer=_optional_int(args.layer),
        head=_optional_int(args.head),
        max_tokens=args.max_tokens,
        strict_same_run=args.strict_same_run,
        overwrite=args.overwrite,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(f"wrote P67 same-run update ingredient trace to {args.out_dir}")
    print(f"same_run_valid={report['same_run_valid']}")
    print(f"first_divergent_stage={report['first_divergent_stage']}")
    print(f"recommended_next_phase={report['recommended_next_phase']}")
    if args.fail_on_unavailable_critical_stage and report["unavailable_rows"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
