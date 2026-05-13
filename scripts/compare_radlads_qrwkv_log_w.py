#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_log_w_parity import (
    LogWRecord,
    capture_qrwkv_log_w_from_current_run,
    compare_log_w_records,
    evaluate_log_w_candidate_variants,
    load_radlads_log_w_from_jsonl,
    write_log_w_reports,
)
from qrwkv_xla.parity.radlads_numerical_fixtures import load_numerical_case_arrays
from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
)
from qrwkv_xla.parity.radlads_replay import (
    replay_profile_for_case,
    student_for_replay_profile,
)

DEFAULT_MANIFEST = Path("artifacts/p54_confirmation/fixtures/manifest.json")
DEFAULT_RADLADS_TRACE = Path(
    "artifacts/p56_wkv_state_residual_trace/wkv_trace_radlads.jsonl"
)
DEFAULT_OUT = Path("artifacts/p57_log_w_decay_parity")
DEFAULT_CASE = "tiny_no_mask"


def _select_case(manifest: dict[str, object], name: str) -> dict[str, object]:
    for case in manifest["cases"]:  # type: ignore[index]
        if case["name"] == name:
            return dict(case)
    raise SystemExit(f"case not found in manifest: {name}")


def _current_qrwkv_records(
    *,
    manifest_path: Path,
    case_name: str,
    parameter_npz: Path | None,
    seed: int,
) -> tuple[list[LogWRecord], list[LogWRecord], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = _select_case(manifest, case_name)
    parameter_path = parameter_npz or manifest_path.parent / str(
        manifest.get("parameter_payload", "radlads_parameters.npz")
    )
    import_result = import_radlads_parameters_for_replay(
        parameter_path,
        allow_defaults=True,
        seed=seed,
    )
    profile = replay_profile_for_case(case)
    student = student_for_replay_profile(import_result.qrwkv_config, profile)
    arrays = load_numerical_case_arrays(manifest_path, case)
    input_ids = np.asarray(arrays["input_ids"], dtype=np.int32)
    attention_mask = None
    if "attention_mask" in arrays:
        attention_mask = np.asarray(arrays["attention_mask"], dtype=np.int32)
    capture = capture_qrwkv_log_w_from_current_run(
        student,
        import_result.params,
        input_ids,
        attention_mask=attention_mask,
        case=case_name,
    )
    log_w = [LogWRecord(**row) for row in capture["log_w"]]  # type: ignore[arg-type]
    w_source = [
        LogWRecord(**row)
        for row in capture["w_source"]  # type: ignore[arg-type]
    ]
    summary = {
        "manifest": str(manifest_path),
        "parameter_npz": str(parameter_path),
        "case": case_name,
        "profile": profile.reason,
        "diagnostic_entry_count": capture["diagnostic_entry_count"],
        "qrwkv_log_w_rows": len(log_w),
        "qrwkv_w_source_rows": len(w_source),
    }
    return log_w, w_source, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P57 compare RADLADS log_w trace rows against current QRWKV."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--radlads-trace", type=Path, default=DEFAULT_RADLADS_TRACE)
    parser.add_argument("--parameter-npz", type=Path)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=5353)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")

    radlads_rows = [
        row
        for row in load_radlads_log_w_from_jsonl(args.radlads_trace)
        if row.case == args.case
    ]
    if not radlads_rows:
        raise SystemExit(f"RADLADS trace has no log_w rows for case {args.case}")

    qrwkv_rows, qrwkv_w_rows, run_summary = _current_qrwkv_records(
        manifest_path=args.manifest,
        case_name=args.case,
        parameter_npz=args.parameter_npz,
        seed=args.seed,
    )
    parity_report = compare_log_w_records(
        radlads_rows,
        qrwkv_rows,
        atol=args.atol,
        rtol=args.rtol,
    )
    parity_report["inputs"] = {"radlads_trace": str(args.radlads_trace), **run_summary}
    candidate_report = evaluate_log_w_candidate_variants(
        radlads_rows=radlads_rows,
        qrwkv_w_rows=qrwkv_w_rows,
        atol=args.atol,
        rtol=args.rtol,
    )
    write_log_w_reports(
        parity_report=parity_report,
        candidate_report=candidate_report,
        radlads_rows=radlads_rows,
        qrwkv_rows=qrwkv_rows,
        qrwkv_w_rows=qrwkv_w_rows,
        out_dir=args.out,
    )
    print(f"wrote P57 log_w parity artifacts to {args.out}")
    print(
        f"diagnostic-only status={parity_report['status']} "
        f"first_mismatch={parity_report.get('first_mismatch')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
