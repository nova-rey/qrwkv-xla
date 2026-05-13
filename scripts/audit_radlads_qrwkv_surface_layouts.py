#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qrwkv_xla.parity.radlads_state_layout import (
    HIDDEN_STATES_CONVENTION,
    STEPWISE_CONVENTION,
    WKV_STATE_CONVENTION,
    load_output_pairs,
    stats_for_surface_pair,
)

SURFACES = (
    "hidden_states",
    "wkv_matrix_state",
    "shift_state",
    "logits",
    "stepwise_hidden_states",
    "stepwise_wkv_matrix_state",
    "stepwise_shift_state",
    "stepwise_logits",
)


def _resolve_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "radlads_outputs" in payload and "qrwkv_outputs" in payload:
        return {
            "radlads_outputs": Path(path.parent, payload["radlads_outputs"]),
            "qrwkv_outputs": Path(path.parent, payload["qrwkv_outputs"]),
        }
    raise SystemExit("manifest must contain radlads_outputs and qrwkv_outputs")


def _output_dir(path: Path) -> Path:
    return path if path.name != "manifest.json" else path.parent


def _load_pair(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if args.manifest:
        resolved = _resolve_manifest(Path(args.manifest))
        rad = (
            _output_dir(Path(args.radlads_outputs))
            if args.radlads_outputs
            else resolved["radlads_outputs"]
        )
        qrw = (
            _output_dir(Path(args.qrwkv_outputs))
            if args.qrwkv_outputs
            else resolved["qrwkv_outputs"]
        )
    else:
        if not args.radlads_outputs or not args.qrwkv_outputs:
            raise SystemExit(
                "provide --radlads-outputs and --qrwkv-outputs or --manifest"
            )
        rad = _output_dir(Path(args.radlads_outputs))
        qrw = _output_dir(Path(args.qrwkv_outputs))
    return load_output_pairs(rad, qrw)


def _shape_relation_summary(row: dict[str, Any]) -> str:
    return f"{row['radlads_shape']} vs {row['qrwkv_shape']} ({row['shape_relation']})"


def _write_md(rows: list[dict[str, Any]], out_dir: Path) -> None:
    lines = ["# P55 Surface Layout Audit", ""]
    lines.append("## Conventions")
    lines.append("")
    lines.append(f"- hidden_states: {HIDDEN_STATES_CONVENTION}")
    lines.append(f"- wkv_matrix_state: {WKV_STATE_CONVENTION}")
    lines.append(f"- stepwise: {STEPWISE_CONVENTION}")
    lines.append("")
    for surface in SURFACES:
        surf_rows = [row for row in rows if row["surface"] == surface]
        if not surf_rows:
            continue
        lines.append(f"## {surface}")
        lines.append("")
        for row in surf_rows:
            lines.append(
                f"- {row['case']}: {row['shape_relation']} | "
                f"{row['radlads_shape']} vs {row['qrwkv_shape']} | "
                f"{row['suspected_axis_meaning']}"
            )
        lines.append("")
    (out_dir / "P55_SURFACE_LAYOUT_AUDIT.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit P55 RADLADS/QRWKV surface layouts."
    )
    parser.add_argument("--radlads-outputs")
    parser.add_argument("--qrwkv-outputs")
    parser.add_argument("--manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    radlads, qrwkv = _load_pair(args)
    rows: list[dict[str, Any]] = []
    for case in sorted(set(radlads) | set(qrwkv)):
        rcase = radlads.get(case, {})
        qcase = qrwkv.get(case, {})
        for surface in SURFACES:
            rows.append(
                stats_for_surface_pair(
                    case,
                    surface,
                    rcase.get(f"radlads_{surface}"),
                    qcase.get(f"qrwkv_{surface}"),
                )
            )
    summary = {
        "cases": sorted(set(radlads) | set(qrwkv)),
        "surface_count": len(SURFACES),
        "rows": len(rows),
        "conventions": {
            "hidden_states": HIDDEN_STATES_CONVENTION,
            "wkv_matrix_state": WKV_STATE_CONVENTION,
            "stepwise": STEPWISE_CONVENTION,
        },
    }
    payload = {"summary": summary, "rows": rows}
    (out_dir / "surface_layout_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_md(rows, out_dir)
    print(f"wrote surface layout audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
