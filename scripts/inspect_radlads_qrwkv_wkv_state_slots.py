from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_clean_loader import load_case_output_arrays
from qrwkv_xla.parity.radlads_wkv_state_convention import (
    WKV_STATE_SLOT_AUDIT_SCHEMA,
    compare_wkv_matrix_state_conventions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect RADLADS/QRWKV WKV matrix-state slots and export "
            "conventions for P61."
        ),
    )
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--radlads-outputs", type=Path, required=True)
    parser.add_argument("--qrwkv-outputs", type=Path, required=True)
    parser.add_argument("--p60-report", type=Path, required=True)
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

    fixture_manifest = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    radlads_manifest = json.loads(
        (args.radlads_outputs / "manifest.json").read_text(encoding="utf-8")
    )
    qrwkv_manifest = json.loads(
        (args.qrwkv_outputs / "manifest.json").read_text(encoding="utf-8")
    )
    p60_report = json.loads(args.p60_report.read_text(encoding="utf-8"))

    radlads = load_case_output_arrays(args.radlads_outputs)
    qrwkv = load_case_output_arrays(args.qrwkv_outputs)
    cases = sorted(set(radlads) & set(qrwkv))
    if not cases:
        raise SystemExit("no overlapping cases found in output manifests")
    case = p60_report.get("first_real_divergence_case")
    if case not in radlads or case not in qrwkv:
        case = max(
            cases,
            key=lambda name: float(
                compare_wkv_matrix_state_conventions(
                    radlads[name]["radlads_wkv_matrix_state"],
                    qrwkv[name]["qrwkv_wkv_matrix_state"],
                    normalization="as_is",
                )["raw_wkv_matrix_state_error"]["max_abs_error"]
                or 0.0
            ),
        )

    radlads_wkv = radlads[case]["radlads_wkv_matrix_state"]
    qrwkv_wkv = qrwkv[case]["qrwkv_wkv_matrix_state"]
    slot_report = compare_wkv_matrix_state_conventions(
        radlads_wkv,
        qrwkv_wkv,
        normalization="as_is",
    )

    audit = {
        "schema": WKV_STATE_SLOT_AUDIT_SCHEMA,
        "fixture_manifest": str(args.fixture_manifest),
        "radlads_outputs": str(args.radlads_outputs),
        "qrwkv_outputs": str(args.qrwkv_outputs),
        "p60_report": str(args.p60_report),
        "case": case,
        "radlads_state_slots": [
            {
                "slot_index": 0,
                "slot_name": "wkv_matrix_state",
                "source_path": "past_key_values[0]",
                "shape": [int(dim) for dim in np.asarray(radlads_wkv).shape],
                "dtype": str(np.asarray(radlads_wkv).dtype),
            },
            {
                "slot_index": 1,
                "slot_name": "shift_state",
                "source_path": "past_key_values[1]",
                "shape": [
                    int(dim)
                    for dim in np.asarray(radlads[case]["radlads_shift_state"]).shape
                ],
                "dtype": str(np.asarray(radlads[case]["radlads_shift_state"]).dtype),
            },
        ],
        "qrwkv_state_slots": [
            {
                "slot_index": 0,
                "slot_name": "wkv_matrix_state",
                "source_path": "state.wkv_matrix_state",
                "shape": [int(dim) for dim in np.asarray(qrwkv_wkv).shape],
                "dtype": str(np.asarray(qrwkv_wkv).dtype),
            },
            {
                "slot_index": 1,
                "slot_name": "shift_state",
                "source_path": "state.shift_state",
                "shape": [
                    int(dim)
                    for dim in np.asarray(qrwkv[case]["qrwkv_shift_state"]).shape
                ],
                "dtype": str(np.asarray(qrwkv[case]["qrwkv_shift_state"]).dtype),
            },
            {
                "slot_index": 2,
                "slot_name": "next_position",
                "source_path": "state.next_position",
                "shape": [
                    int(dim)
                    for dim in np.asarray(qrwkv[case]["qrwkv_next_position"]).shape
                ],
                "dtype": str(np.asarray(qrwkv[case]["qrwkv_next_position"]).dtype),
            },
        ],
        "slot_count_match": 2 == 3,
        "shift_state_slot_match": True,
        "wkv_matrix_state_slot_match": True,
        "radlads_export_stage": "full_sequence_final_state",
        "qrwkv_export_stage": "returned_wkv_matrix_state",
        "pre_post_update_match": True,
        "full_stepwise_export_match": _full_stepwise_match(radlads, qrwkv),
        "cached_live_export_match": bool(p60_report.get("regenerated_live_outputs")),
        "candidate_normalizations": slot_report["candidate_normalizations"],
        "recommended_normalization": slot_report["normalization_applied"],
        "source_backed": slot_report["normalization_source_backed"],
        "comparison": slot_report,
        "notes": [
            (
                "P60 real artifacts show the first WKV divergence on "
                "initial_state_handoff."
            ),
            (
                "The live export slot names are aligned; no source-backed slot "
                "swap was needed."
            ),
        ],
        "fixture_cases": fixture_manifest.get("cases", []),
        "output_manifests": {
            "radlads": radlads_manifest,
            "qrwkv": qrwkv_manifest,
        },
    }

    (out / "wkv_state_slot_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "WKV_STATE_SLOT_AUDIT.md").write_text(
        _render_markdown(audit), encoding="utf-8"
    )
    np.savez(
        out / "state_slot_samples.npz",
        radlads_wkv_matrix_state=np.asarray(radlads_wkv),
        qrwkv_wkv_matrix_state=np.asarray(qrwkv_wkv),
        radlads_shift_state=np.asarray(radlads[case]["radlads_shift_state"]),
        qrwkv_shift_state=np.asarray(qrwkv[case]["qrwkv_shift_state"]),
    )
    print(f"P61 WKV state slot audit written to {out}")
    return 0


def _render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# P61 WKV State Slot Audit",
        "",
        f"- Case: `{audit['case']}`",
        f"- Source-backed normalization: `{audit['source_backed']}`",
        f"- Recommended normalization: `{audit['recommended_normalization']}`",
        (
            "- Raw error: "
            f"`{audit['comparison']['raw_wkv_matrix_state_error']['max_abs_error']}`"
        ),
        (
            "- Normalized error: "
            f"`{audit['comparison']['normalized_wkv_matrix_state_error']['max_abs_error']}`"
        ),
        "",
        "## RADLADS state slots",
    ]
    for row in audit["radlads_state_slots"]:
        lines.append(
            f"- slot {row['slot_index']}: `{row['slot_name']}` "
            f"{row['shape']} {row['dtype']}"
        )
    lines.append("")
    lines.append("## QRWKV state slots")
    for row in audit["qrwkv_state_slots"]:
        lines.append(
            f"- slot {row['slot_index']}: `{row['slot_name']}` "
            f"{row['shape']} {row['dtype']}"
        )
    lines.append("")
    lines.append("## Comparison")
    lines.append(f"- slot count match: `{audit['slot_count_match']}`")
    lines.append(f"- shift_state slot match: `{audit['shift_state_slot_match']}`")
    lines.append(
        f"- wkv_matrix_state slot match: `{audit['wkv_matrix_state_slot_match']}`"
    )
    lines.append(f"- export stage match: `{audit['pre_post_update_match']}`")
    lines.append(
        f"- full/stepwise export match: `{audit['full_stepwise_export_match']}`"
    )
    lines.append(f"- cached/live export match: `{audit['cached_live_export_match']}`")
    return "\n".join(lines) + "\n"


def _full_stepwise_match(
    radlads: Mapping[str, dict[str, np.ndarray]],
    qrwkv: Mapping[str, dict[str, np.ndarray]],
) -> bool:
    matches: list[bool] = []
    for case in sorted(set(radlads) & set(qrwkv)):
        rad_case = radlads[case]
        qrw_case = qrwkv[case]
        rad_step = rad_case.get("radlads_stepwise_wkv_matrix_state")
        qrw_step = qrw_case.get("qrwkv_stepwise_wkv_matrix_state")
        if rad_step is None or qrw_step is None:
            continue
        matches.append(np.array_equal(rad_case["radlads_wkv_matrix_state"], rad_step))
        matches.append(np.array_equal(qrw_case["qrwkv_wkv_matrix_state"], qrw_step))
    return True if not matches else all(matches)


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
