from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.kernels.wkv7_candidates import UnsupportedCandidate, run_wkv7_candidate
from qrwkv_xla.kernels.wkv7_fixtures import (
    TOLERANCES,
    WKV7Tolerance,
    compare_arrays,
    load_wkv7_case,
    validate_wkv7_manifest,
)

REPORT_SCHEMA = "qrwkv_xla.wkv7_correctness_report.v1"
REPORT_STATUSES = {
    "pass",
    "fail",
    "unsupported",
    "missing_fixture",
    "shape_mismatch",
    "dtype_mismatch",
    "non_finite",
    "candidate_error",
}


def compare_wkv7_manifest(manifest_path: Path, *, candidate: str) -> dict[str, Any]:
    manifest = validate_wkv7_manifest(manifest_path, verify_payloads=False)
    cases: list[dict[str, Any]] = []
    counts = {status: 0 for status in sorted(REPORT_STATUSES)}

    for case in manifest["cases"]:
        try:
            inputs, expected = load_wkv7_case(manifest_path, case)
        except FileNotFoundError as exc:
            result = {
                "case_id": case.get("case_id", "<unknown>"),
                "status": "missing_fixture",
                "reason": str(exc),
            }
            cases.append(result)
            counts["missing_fixture"] += 1
            continue

        tolerance = _case_tolerance(case)
        try:
            actual = run_wkv7_candidate(candidate, inputs)
        except UnsupportedCandidate as exc:
            result = {
                "case_id": case["case_id"],
                "status": "unsupported",
                "reason": exc.reason,
                "candidate": exc.candidate,
            }
            cases.append(result)
            counts["unsupported"] += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive report path
            result = {
                "case_id": case["case_id"],
                "status": "candidate_error",
                "reason": f"{type(exc).__name__}: {exc}",
                "candidate": candidate,
            }
            cases.append(result)
            counts["candidate_error"] += 1
            continue

        output = _compare_named(actual, expected, "output", tolerance)
        next_state = _compare_named(actual, expected, "next_state", tolerance)
        status = _case_status(output["status"], next_state["status"])
        result = {
            "case_id": case["case_id"],
            "status": status,
            "candidate": candidate,
            "output": output,
            "next_state": next_state,
            "full_scan_vs_stepwise": case["full_scan_vs_stepwise"],
        }
        cases.append(result)
        counts[status] += 1

    overall = _overall_status(counts)
    return {
        "schema": REPORT_SCHEMA,
        "phase": "P43",
        "fixture_set": manifest.get("fixture_set"),
        "manifest": str(_manifest_path(manifest_path)),
        "candidate": candidate,
        "overall_status": overall,
        "counts": counts,
        "num_cases": len(cases),
        "status_vocabulary": sorted(REPORT_STATUSES),
        "cases": cases,
    }


def write_wkv7_comparison_reports(
    manifest_path: Path, out_dir: Path, *, candidate: str, overwrite: bool = False
) -> dict[str, Any]:
    report = compare_wkv7_manifest(manifest_path, candidate=candidate)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_report.json"
    markdown_path = out_dir / "P43_WKV7_COMPARISON_REPORT.md"
    if not overwrite:
        for path in (json_path, markdown_path):
            if path.exists():
                raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_comparison_markdown(report), encoding="utf-8")
    return report


def _compare_named(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    name: str,
    tolerance: WKV7Tolerance,
) -> dict[str, Any]:
    if name not in actual or name not in expected:
        return {
            "status": "missing_fixture",
            "array": name,
            "actual_present": name in actual,
            "expected_present": name in expected,
        }
    result = compare_arrays(actual[name], expected[name], tolerance=tolerance)
    return {"array": name, **result}


def _case_status(output_status: str, state_status: str) -> str:
    if output_status == "pass" and state_status == "pass":
        return "pass"
    for status in (
        "missing_fixture",
        "shape_mismatch",
        "dtype_mismatch",
        "non_finite",
        "candidate_error",
    ):
        if output_status == status or state_status == status:
            return status
    return "fail"


def _overall_status(counts: dict[str, int]) -> str:
    for status in (
        "candidate_error",
        "missing_fixture",
        "shape_mismatch",
        "dtype_mismatch",
        "non_finite",
        "fail",
    ):
        if counts[status]:
            return status
    if counts["pass"]:
        return "pass"
    if counts["unsupported"]:
        return "unsupported"
    return "missing_fixture"


def _case_tolerance(case: dict[str, Any]) -> WKV7Tolerance:
    dtype = str(case.get("tolerance", {}).get("dtype", "float32"))
    return TOLERANCES.get(dtype, TOLERANCES["float32"])


def _comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P43 WKV7 Correctness Comparison Report",
        "",
        f"- Candidate: `{report['candidate']}`",
        f"- Overall status: `{report['overall_status']}`",
        f"- Pass: {report['counts']['pass']}",
        f"- Fail: {report['counts']['fail']}",
        f"- Unsupported: {report['counts']['unsupported']}",
        "",
        "| Case | Status | Output | Next state | Full scan vs stepwise |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        output = _metric_cell(case.get("output"))
        state = _metric_cell(case.get("next_state"))
        equivalence = case.get("full_scan_vs_stepwise", {}).get("status", "")
        notes = case.get("reason")
        if notes:
            output = notes
        lines.append(
            "| {name} | `{status}` | {output} | {state} | `{equivalence}` |".format(
                name=case["case_id"],
                status=case["status"],
                output=output,
                state=state,
                equivalence=equivalence,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _metric_cell(metric: dict[str, Any] | None) -> str:
    if not metric:
        return ""
    if metric.get("status") != "pass":
        return f"`{metric.get('status')}`"
    return (
        f"`pass` max_abs={metric['max_abs_error']:.3g} "
        f"mean_abs={metric['mean_abs_error']:.3g} "
        f"max_rel={metric['max_relative_error']:.3g}"
    )


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path
