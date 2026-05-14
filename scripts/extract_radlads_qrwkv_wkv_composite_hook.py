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

from qrwkv_xla.parity.radlads_wkv_composite_hook import (
    WKV_COMPOSITE_BALANCE_HOOK_SCHEMA,
    build_composite_hook_trace,
    load_composite_hook_jsonl,
    write_composite_hook_reports,
    write_composite_hook_trace,
)

DEFAULT_OUT = Path("artifacts/p64_composite_balance_hook")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P64 extract RADLADS/QRWKV WKV composite balance-state hooks."
    )
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--radlads-outputs", type=Path, required=True)
    parser.add_argument("--qrwkv-outputs", type=Path, required=True)
    parser.add_argument("--p63-hooks", type=Path, required=True)
    parser.add_argument("--source-locator", type=Path, required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--mode", choices=["full", "stepwise", "both"], default="both")
    parser.add_argument("--layer", default="all")
    parser.add_argument("--head", default="all")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-real-artifacts", action="store_true")
    parser.add_argument("--radlads-repo", type=Path)
    parser.add_argument("--rerun-radlads", action="store_true")
    parser.add_argument("--rerun-qrwkv", action="store_true")
    parser.add_argument("--allow-exact-reconstruction", action="store_true")
    parser.add_argument("--allow-partial-reconstruction", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    p63_trace_dir = args.p63_hooks.parent
    radlads_trace_path = p63_trace_dir / "live_update_hooks_radlads.jsonl"
    qrwkv_trace_path = p63_trace_dir / "live_update_hooks_qrwkv.jsonl"
    if args.strict_real_artifacts and (
        not radlads_trace_path.is_file() or not qrwkv_trace_path.is_file()
    ):
        raise SystemExit(
            "strict real-artifact mode requires sibling live_update_hooks JSONL traces"
        )
    if not radlads_trace_path.is_file() or not qrwkv_trace_path.is_file():
        raise SystemExit("P63 live update hook traces not found next to metadata file")
    if not args.source_locator.is_file():
        raise SystemExit(f"source locator not found: {args.source_locator}")

    cases = [case for case in args.cases.split(",") if case]
    p63_meta = json.loads(args.p63_hooks.read_text(encoding="utf-8"))
    source_locator = json.loads(args.source_locator.read_text(encoding="utf-8"))
    radlads_source = load_composite_hook_jsonl(radlads_trace_path)
    qrwkv_source = load_composite_hook_jsonl(qrwkv_trace_path)

    radlads_entries = build_composite_hook_trace(
        radlads_source,
        side="radlads",
        mode=args.mode,
        allow_exact_reconstruction=args.allow_exact_reconstruction,
        allow_partial_reconstruction=args.allow_partial_reconstruction,
    )
    qrwkv_entries = build_composite_hook_trace(
        qrwkv_source,
        side="qrwkv",
        mode=args.mode,
        allow_exact_reconstruction=args.allow_exact_reconstruction,
        allow_partial_reconstruction=args.allow_partial_reconstruction,
    )
    if cases:
        radlads_entries = [entry for entry in radlads_entries if entry["case"] in cases]
        qrwkv_entries = [entry for entry in qrwkv_entries if entry["case"] in cases]

    radlads_out = args.out / "composite_hook_radlads.jsonl"
    qrwkv_out = args.out / "composite_hook_qrwkv.jsonl"
    write_composite_hook_trace(radlads_entries, radlads_out)
    write_composite_hook_trace(qrwkv_entries, qrwkv_out)

    metadata = {
        "schema": WKV_COMPOSITE_BALANCE_HOOK_SCHEMA,
        "phase": "P64",
        "fixture_manifest": str(args.fixture_manifest),
        "radlads_outputs": str(args.radlads_outputs),
        "qrwkv_outputs": str(args.qrwkv_outputs),
        "p63_hooks": str(args.p63_hooks),
        "source_locator": str(args.source_locator),
        "cases": cases,
        "mode": args.mode,
        "layer": args.layer,
        "head": args.head,
        "max_tokens": args.max_tokens,
        "strict_real_artifacts": args.strict_real_artifacts,
        "radlads_repo": None if args.radlads_repo is None else str(args.radlads_repo),
        "rerun_radlads": args.rerun_radlads,
        "rerun_qrwkv": args.rerun_qrwkv,
        "allow_exact_reconstruction": args.allow_exact_reconstruction,
        "allow_partial_reconstruction": args.allow_partial_reconstruction,
        "real_artifacts_used": True,
        "synthetic_fallback_used": False,
        "capture_kind_by_stage": {
            "radlads": sorted({entry["capture_kind"] for entry in radlads_entries}),
            "qrwkv": sorted({entry["capture_kind"] for entry in qrwkv_entries}),
        },
        "unavailable_hooks": sum(
            1
            for entry in radlads_entries + qrwkv_entries
            if entry["capture_kind"] == "unavailable"
        ),
        "reconstructed_hooks": sum(
            1
            for entry in radlads_entries + qrwkv_entries
            if entry["capture_kind"] == "exact_reconstruction"
        ),
        "partial_reconstructions": sum(
            1
            for entry in radlads_entries + qrwkv_entries
            if entry["capture_kind"] == "partial_reconstruction"
        ),
        "p63_metadata": p63_meta,
        "source_locator_summary": source_locator,
    }
    (args.out / "composite_hook_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison = {
        "schema": "radlads_qrwkv_wkv_composite_balance_hook_comparison.v1",
        "overall_status": "pending",
        "hook_extraction_status": "pending",
        "radlads_capture_kind": metadata["capture_kind_by_stage"]["radlads"],
        "qrwkv_capture_kind": metadata["capture_kind_by_stage"]["qrwkv"],
        "first_divergent_case": None,
        "first_divergent_mode": None,
        "first_divergent_layer": None,
        "first_divergent_token": None,
        "first_divergent_head": None,
        "first_divergent_stage": None,
        "first_divergent_max_abs_error": None,
        "composite_balance_update_term_match": False,
        "state_after_from_full_source_formula_match": False,
        "residual_explained_by_composite_term": "partial",
        "residual_remaining_after_composite_term": None,
        "source_backed_fix_available": False,
        "fix_recommended": "pending",
        "kernel_ready": "no",
        "next_recommended_phase": "pending",
        "rows": [],
    }
    write_composite_hook_reports(
        radlads_entries=radlads_entries,
        qrwkv_entries=qrwkv_entries,
        comparison_report=comparison,
        out_dir=args.out,
    )
    print(f"wrote P64 WKV composite balance hook extraction to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
