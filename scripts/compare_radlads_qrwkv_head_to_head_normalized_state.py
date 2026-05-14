from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_clean_loader import load_case_output_arrays
from qrwkv_xla.parity.radlads_head_to_head import compare_surface_arrays
from qrwkv_xla.parity.radlads_wkv_state_convention import (
    compare_wkv_matrix_state_conventions,
    write_wkv_state_convention_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RADLADS and QRWKV outputs using the P61 WKV state "
            "convention audit."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radlads-outputs", type=Path, required=True)
    parser.add_argument("--qrwkv-outputs", type=Path, required=True)
    parser.add_argument("--slot-audit", type=Path, required=True)
    parser.add_argument("--p60-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_cases = manifest.get("cases", [])
    slot_audit = json.loads(args.slot_audit.read_text(encoding="utf-8"))
    p60_report = _load_optional_json(args.p60_report)
    radlads = load_case_output_arrays(args.radlads_outputs)
    qrwkv = load_case_output_arrays(args.qrwkv_outputs)

    case_rows: list[dict[str, Any]] = []
    wkv_raw_errors: list[float] = []
    wkv_normalized_errors: list[float] = []
    logits_preserved = True
    shift_state_preserved = True
    hidden_state_issues: list[str] = []

    for case_name in sorted(set(radlads) & set(qrwkv)):
        rad = radlads[case_name]
        qrw = qrwkv[case_name]

        raw_logits = compare_surface_arrays(
            "logits",
            rad.get("radlads_logits"),
            qrw.get("qrwkv_logits"),
        )
        raw_shift = compare_surface_arrays(
            "shift_state",
            rad.get("radlads_shift_state"),
            qrw.get("qrwkv_shift_state"),
        )
        raw_hidden = compare_surface_arrays(
            "hidden_states",
            rad.get("radlads_hidden_states"),
            qrw.get("qrwkv_hidden_states"),
        )
        wkv = compare_wkv_matrix_state_conventions(
            rad.get("radlads_wkv_matrix_state"),
            qrw.get("qrwkv_wkv_matrix_state"),
            slot_audit=slot_audit,
            normalization=str(slot_audit.get("recommended_normalization", "as_is")),
        )
        wkv_raw_errors.append(
            float(wkv["raw_wkv_matrix_state_error"]["max_abs_error"] or 0.0)
        )
        wkv_normalized_errors.append(
            float(wkv["normalized_wkv_matrix_state_error"]["max_abs_error"] or 0.0)
        )
        logits_preserved = logits_preserved and raw_logits["status"] == "pass"
        shift_state_preserved = shift_state_preserved and raw_shift["status"] == "pass"
        hidden_state_issues.append(_classify_hidden_state_issue(raw_hidden, p60_report))
        case_rows.append(
            {
                "case": case_name,
                "logits": raw_logits,
                "shift_state": raw_shift,
                "hidden_states": raw_hidden,
                "wkv_matrix_state": wkv,
            }
        )

    raw_error = max(wkv_raw_errors) if wkv_raw_errors else None
    normalized_error = max(wkv_normalized_errors) if wkv_normalized_errors else None
    report = {
        "schema": "radlads_qrwkv_wkv_state_convention_report.v1",
        "phase": "P61",
        "manifest": str(args.manifest),
        "manifest_cases": manifest_cases,
        "slot_audit": str(args.slot_audit),
        "overall_status": "pass"
        if normalized_error is not None
        and normalized_error <= 1e-5
        and logits_preserved
        and shift_state_preserved
        else "fail",
        "logits_preserved": logits_preserved,
        "shift_state_preserved": shift_state_preserved,
        "raw_wkv_matrix_state_error": raw_error,
        "normalized_wkv_matrix_state_error": normalized_error,
        "normalization_applied": slot_audit.get("recommended_normalization", "as_is"),
        "normalization_source_backed": bool(slot_audit.get("source_backed", False)),
        "normalization_changed_values": bool(
            slot_audit.get("comparison", {}).get("normalization_changed_values", False)
        ),
        "wkv_matrix_state_status": "pass"
        if normalized_error is not None and normalized_error <= 1e-5
        else "fail",
        "hidden_states_status": _summarize_hidden_state_issue(hidden_state_issues),
        "candidate_normalizations": slot_audit.get("candidate_normalizations", []),
        "radlads_state_slots": slot_audit.get("radlads_state_slots", []),
        "qrwkv_state_slots": slot_audit.get("qrwkv_state_slots", []),
        "slot_count_match": slot_audit.get("slot_count_match", False),
        "shift_state_slot_match": slot_audit.get("shift_state_slot_match", False),
        "wkv_matrix_state_slot_match": slot_audit.get(
            "wkv_matrix_state_slot_match", False
        ),
        "pre_post_update_match": slot_audit.get("pre_post_update_match", False),
        "full_stepwise_export_match": slot_audit.get(
            "full_stepwise_export_match", False
        ),
        "cached_live_export_match": slot_audit.get("cached_live_export_match", False),
        "hidden_states_issue_type": _summarize_hidden_state_issue(hidden_state_issues),
        "cases": case_rows,
        "p60_report": p60_report,
    }

    (out / "head_to_head_normalized_state_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "P61_SURFACE_COMPARISON.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    write_wkv_state_convention_report(report, out)
    root = out.parent
    (root / "p61_wkv_state_export_convention_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "P61_RESULTS.md").write_text(_results_markdown(report), encoding="utf-8")
    (root / "HIDDEN_STATES_CONVENTION_AUDIT.md").write_text(
        _hidden_state_audit_markdown(report), encoding="utf-8"
    )
    print(f"P61 normalized WKV state comparison written to {out}")
    return 0


def _classify_hidden_state_issue(
    raw_hidden: dict[str, Any], p60_report: dict[str, Any] | None
) -> str:
    if p60_report and p60_report.get("hidden_states_explained") == "comparison_only":
        return "comparison_normalization_issue"
    if raw_hidden.get("status") == "shape_mismatch":
        return "output_shape_convention"
    return "unknown"


def _summarize_hidden_state_issue(issues: list[str]) -> str:
    if not issues:
        return "unknown"
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue] = counts.get(issue, 0) + 1
    return max(counts, key=counts.get)


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P61 Surface Comparison",
        "",
        f"- logits preserved: `{report['logits_preserved']}`",
        f"- shift_state preserved: `{report['shift_state_preserved']}`",
        f"- raw wkv error: `{report['raw_wkv_matrix_state_error']}`",
        f"- normalized wkv error: `{report['normalized_wkv_matrix_state_error']}`",
        f"- normalization: `{report['normalization_applied']}`",
        f"- hidden_states issue: `{report['hidden_states_issue_type']}`",
    ]
    return "\n".join(lines) + "\n"


def _results_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P61 Results",
        "",
        f"- overall_status: `{report['overall_status']}`",
        f"- logits preserved: `{report['logits_preserved']}`",
        f"- shift_state preserved: `{report['shift_state_preserved']}`",
        f"- raw wkv error: `{report['raw_wkv_matrix_state_error']}`",
        f"- normalized wkv error: `{report['normalized_wkv_matrix_state_error']}`",
        f"- normalization: `{report['normalization_applied']}`",
        f"- hidden_states issue: `{report['hidden_states_issue_type']}`",
        "- kernel-ready: `no`",
        "",
        "Known caveats:",
        "- P61 does not implement Pallas.",
        "- P61 does not prove training throughput.",
        "- P61 does not prove model quality.",
        (
            "- P61 only audits/fixes tiny local CPU WKV matrix-state "
            "export/slot convention."
        ),
        "- If WKV matrix state remains unexplained, Pallas remains blocked.",
    ]
    return "\n".join(lines) + "\n"


def _hidden_state_audit_markdown(report: Mapping[str, Any]) -> str:
    issue = str(report.get("hidden_states_issue_type", "unknown"))
    recommendation = {
        "comparison_normalization_issue": (
            "Normalize the hidden-state surface by selecting the final layer "
            "before compare."
        ),
        "output_shape_convention": (
            "Normalize the hidden-state shape/axis convention before comparing."
        ),
        "downstream_of_wkv": (
            "Re-run once the WKV surface is explained; the hidden-state issue "
            "may clear."
        ),
        "separate_math_residual": (
            "Investigate hidden-state math separately from the WKV surface."
        ),
    }.get(issue, "Investigate the hidden-state surface separately.")
    lines = [
        "# HIDDEN_STATES_CONVENTION_AUDIT",
        "",
        f"- hidden_states issue type: `{issue}`",
        f"- recommended next action: {recommendation}",
    ]
    return "\n".join(lines) + "\n"


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
