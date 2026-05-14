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

from qrwkv_xla.parity.radlads_wkv_trace import load_trace_jsonl
from qrwkv_xla.parity.radlads_wkv_update_residual import (
    WKV_UPDATE_RESIDUAL_SCHEMA,
    build_update_residual_trace,
    compare_update_residual_traces,
    write_update_residual_reports,
    write_update_residual_trace,
)

DEFAULT_RADLADS_TRACE = Path(
    "artifacts/p58_log_w_decay_fix/post_fix_trace/wkv_trace_radlads.jsonl"
)
DEFAULT_QRWKV_TRACE = Path(
    "artifacts/p58_log_w_decay_fix/post_fix_trace/wkv_trace_qrwkv.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P62 trace RADLADS/QRWKV WKV update residual stages."
    )
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--radlads-outputs", type=Path)
    parser.add_argument("--qrwkv-outputs", type=Path)
    parser.add_argument("--p61-slot-audit", type=Path)
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
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/p62_wkv_update_residual")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.radlads_trace.is_file():
        raise SystemExit(f"RADLADS trace not found: {args.radlads_trace}")
    if not args.qrwkv_trace.is_file():
        raise SystemExit(f"QRWKV trace not found: {args.qrwkv_trace}")

    radlads_source = load_trace_jsonl(args.radlads_trace)
    qrwkv_source = load_trace_jsonl(args.qrwkv_trace)
    radlads_entries = build_update_residual_trace(radlads_source, side="radlads")
    qrwkv_entries = build_update_residual_trace(qrwkv_source, side="qrwkv")

    radlads_out = args.out / "wkv_update_residual_radlads.jsonl"
    qrwkv_out = args.out / "wkv_update_residual_qrwkv.jsonl"
    write_update_residual_trace(radlads_entries, radlads_out)
    write_update_residual_trace(qrwkv_entries, qrwkv_out)

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

    manifest = {
        "schema": WKV_UPDATE_RESIDUAL_SCHEMA,
        "phase": "P62",
        "fixture_manifest": None
        if args.fixture_manifest is None
        else str(args.fixture_manifest),
        "radlads_outputs": None
        if args.radlads_outputs is None
        else str(args.radlads_outputs),
        "qrwkv_outputs": None
        if args.qrwkv_outputs is None
        else str(args.qrwkv_outputs),
        "p61_slot_audit": None
        if args.p61_slot_audit is None
        else str(args.p61_slot_audit),
        "cases": [case for case in args.cases.split(",") if case],
        "mode": args.mode,
        "layer": args.layer,
        "head": args.head,
        "max_tokens": args.max_tokens,
        "source_traces": {
            "radlads": str(args.radlads_trace),
            "qrwkv": str(args.qrwkv_trace),
        },
        "radlads_repo": None if args.radlads_repo is None else str(args.radlads_repo),
        "strict_real_artifacts": args.strict_real_artifacts,
        "rerun_radlads": args.rerun_radlads,
        "rerun_qrwkv": args.rerun_qrwkv,
        "traces": {
            "radlads": str(radlads_out),
            "qrwkv": str(qrwkv_out),
        },
        "comparison_report": str(
            args.out / "wkv_update_residual_comparison_report.json"
        ),
        "kernel_ready": report["kernel_ready"],
        "diagnostic_only": True,
    }
    (args.out / "wkv_update_residual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "P62_RESULTS.md").write_text(
        "\n".join(
            [
                "# P62 Results",
                "",
                f"- status: `{report['status']}`",
                f"- kernel_ready: `{report['kernel_ready']}`",
                f"- first divergent stage: `{report['first_divergent_stage']}`",
                "- first divergent max abs error: "
                f"`{report['first_divergent_max_abs_error']}`",
                "- fix applied: `none`",
                f"- recommendation: {report['next_phase_recommendation']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote P62 WKV update residual trace to {args.out}")
    print(f"kernel_ready={report['kernel_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
