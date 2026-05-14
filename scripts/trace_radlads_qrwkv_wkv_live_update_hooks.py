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

from qrwkv_xla.parity.radlads_wkv_live_update_hooks import (
    WKV_LIVE_UPDATE_HOOK_SCHEMA,
    build_live_update_hook_trace,
    load_live_update_hook_jsonl,
    write_live_update_hook_reports,
    write_live_update_hook_trace,
)

DEFAULT_RADLADS_TRACE = Path(
    "artifacts/p58_log_w_decay_fix/post_fix_trace/wkv_trace_radlads.jsonl"
)
DEFAULT_QRWKV_TRACE = Path(
    "artifacts/p58_log_w_decay_fix/post_fix_trace/wkv_trace_qrwkv.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P63 trace RADLADS/QRWKV WKV live update hooks."
    )
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--radlads-outputs", type=Path)
    parser.add_argument("--qrwkv-outputs", type=Path)
    parser.add_argument("--p62-report", type=Path, required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--mode", choices=["full", "stepwise", "both"], default="both")
    parser.add_argument("--layer", default="all")
    parser.add_argument("--head", default="all")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--radlads-trace", type=Path, default=DEFAULT_RADLADS_TRACE)
    parser.add_argument("--qrwkv-trace", type=Path, default=DEFAULT_QRWKV_TRACE)
    parser.add_argument("--radlads-repo", type=Path)
    parser.add_argument("--strict-real-artifacts", action="store_true")
    parser.add_argument("--rerun-radlads", action="store_true")
    parser.add_argument("--rerun-qrwkv", action="store_true")
    parser.add_argument("--allow-reconstructed", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/p63_wkv_live_update_hooks")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.p62_report.is_file():
        raise SystemExit(f"P62 report not found: {args.p62_report}")
    if not args.radlads_trace.is_file():
        raise SystemExit(f"RADLADS trace not found: {args.radlads_trace}")
    if not args.qrwkv_trace.is_file():
        raise SystemExit(f"QRWKV trace not found: {args.qrwkv_trace}")

    p62_report = json.loads(args.p62_report.read_text(encoding="utf-8"))
    cases = [case for case in args.cases.split(",") if case] or [
        p62_report.get("first_divergent_case", "tiny_no_mask")
    ]
    radlads_source = load_live_update_hook_jsonl(args.radlads_trace)
    qrwkv_source = load_live_update_hook_jsonl(args.qrwkv_trace)

    radlads_entries = build_live_update_hook_trace(
        radlads_source,
        side="radlads",
        mode=args.mode,
        allow_reconstructed=args.allow_reconstructed,
    )
    qrwkv_entries = build_live_update_hook_trace(
        qrwkv_source,
        side="qrwkv",
        mode=args.mode,
        allow_reconstructed=args.allow_reconstructed,
    )

    if cases:
        radlads_entries = [entry for entry in radlads_entries if entry["case"] in cases]
        qrwkv_entries = [entry for entry in qrwkv_entries if entry["case"] in cases]

    radlads_out = args.out / "live_update_hooks_radlads.jsonl"
    qrwkv_out = args.out / "live_update_hooks_qrwkv.jsonl"
    write_live_update_hook_trace(radlads_entries, radlads_out)
    write_live_update_hook_trace(qrwkv_entries, qrwkv_out)

    report = {
        "schema": WKV_LIVE_UPDATE_HOOK_SCHEMA,
        "phase": "P63",
        "p62_report": str(args.p62_report),
        "source_traces": {
            "radlads": str(args.radlads_trace),
            "qrwkv": str(args.qrwkv_trace),
        },
        "cases": cases,
        "mode": args.mode,
        "layer": args.layer,
        "head": args.head,
        "max_tokens": args.max_tokens,
        "strict_real_artifacts": args.strict_real_artifacts,
        "radlads_repo": None if args.radlads_repo is None else str(args.radlads_repo),
        "rerun_radlads": args.rerun_radlads,
        "rerun_qrwkv": args.rerun_qrwkv,
        "allow_reconstructed": args.allow_reconstructed,
        "radlads_rows": len(radlads_entries),
        "qrwkv_rows": len(qrwkv_entries),
        "diagnostic_only": True,
    }
    (args.out / "live_update_hooks_metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison = {
        "schema": "radlads_qrwkv_wkv_live_update_hooks_comparison.v1",
        "overall_status": "pending",
        "live_hooks_complete": False,
        "missing_hooks": 0,
        "reconstructed_hooks": 0,
        "first_divergent_case": None,
        "first_divergent_mode": None,
        "first_divergent_layer": None,
        "first_divergent_token": None,
        "first_divergent_head": None,
        "first_divergent_stage": None,
        "first_divergent_capture_kind": None,
        "first_divergent_max_abs_error": None,
        "decayed_state_match": False,
        "update_outer_product_match": False,
        "balance_state_matmul_match": False,
        "composite_update_term_match": False,
        "update_term_match": False,
        "state_after_match": False,
        "suspected_root_cause": "pending",
        "fix_recommended": "pending",
        "kernel_ready": "no",
        "rows": [],
    }
    write_live_update_hook_reports(
        radlads_entries=radlads_entries,
        qrwkv_entries=qrwkv_entries,
        comparison_report=comparison,
        out_dir=args.out,
    )
    print(f"wrote P63 WKV live update hook trace to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
