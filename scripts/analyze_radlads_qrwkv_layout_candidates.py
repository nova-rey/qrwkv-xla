#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qrwkv_xla.parity.radlads_state_layout import (
    HIDDEN_CANDIDATES,
    STEPWISE_CANDIDATES,
    WKV_CANDIDATES,
    evaluate_candidates,
    load_output_pairs,
    summarize_candidate_rows,
)

SURFACE_TO_CANDIDATES = {
    "hidden_states": HIDDEN_CANDIDATES,
    "wkv_matrix_state": WKV_CANDIDATES,
    "stepwise_hidden_states": STEPWISE_CANDIDATES,
    "stepwise_wkv_matrix_state": STEPWISE_CANDIDATES,
    "stepwise_shift_state": STEPWISE_CANDIDATES,
    "stepwise_logits": STEPWISE_CANDIDATES,
}


def _resolve_manifest(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "radlads_outputs" in payload and "qrwkv_outputs" in payload:
        return {
            "radlads_outputs": Path(path.parent, payload["radlads_outputs"]),
            "qrwkv_outputs": Path(path.parent, payload["qrwkv_outputs"]),
        }
    raise SystemExit("manifest must contain radlads_outputs and qrwkv_outputs")


def _load_pair(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if args.manifest:
        resolved = _resolve_manifest(Path(args.manifest))
        rad = (
            Path(args.radlads_outputs)
            if args.radlads_outputs
            else resolved["radlads_outputs"]
        )
        qrw = (
            Path(args.qrwkv_outputs)
            if args.qrwkv_outputs
            else resolved["qrwkv_outputs"]
        )
    else:
        if not args.radlads_outputs or not args.qrwkv_outputs:
            raise SystemExit(
                "provide --radlads-outputs and --qrwkv-outputs or --manifest"
            )
        rad = Path(args.radlads_outputs)
        qrw = Path(args.qrwkv_outputs)
    return load_output_pairs(rad, qrw)


def _write_md(rows: list[dict[str, Any]], out_dir: Path) -> None:
    lines = ["# P55 Layout Candidate Analysis", ""]
    best = {}
    for row in rows:
        key = (row["surface"], row["case"])
        if row["status"] != "not_applicable":
            current = best.get(key)
            if current is None or (
                row.get("max_abs_error") is not None
                and (
                    current.get("max_abs_error") is None
                    or row["max_abs_error"] < current["max_abs_error"]
                )
            ):
                best[key] = row
    for surface in sorted({row["surface"] for row in rows}):
        lines.append(f"## {surface}")
        lines.append("")
        surf_rows = [row for row in rows if row["surface"] == surface]
        for row in surf_rows:
            lines.append(
                "- "
                f"{row['case']} / {row['candidate_name']}: {row['status']} | "
                f"shape_match={row['shape_match']} | max_abs={row['max_abs_error']} | "
                f"improvement={row['rank_improvement_vs_as_is']}"
            )
        lines.append("")
    (out_dir / "P55_LAYOUT_CANDIDATES.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze P55 RADLADS/QRWKV layout candidates."
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
        for surface, candidates in SURFACE_TO_CANDIDATES.items():
            if surface.startswith("stepwise_"):
                rad = rcase.get(f"radlads_{surface}")
                qrv = qcase.get(f"qrwkv_{surface}")
            else:
                rad = rcase.get(f"radlads_{surface}")
                qrv = qcase.get(f"qrwkv_{surface}")
            rows.extend(evaluate_candidates(case, surface, rad, qrv, candidates))
    summary = {
        "cases": sorted(set(radlads) | set(qrwkv)),
        "rows": len(rows),
        "best_by_surface": {},
    }
    for surface in SURFACE_TO_CANDIDATES:
        surf_rows = [
            row
            for row in rows
            if row["surface"] == surface and row["status"] != "not_applicable"
        ]
        summary["best_by_surface"][surface] = summarize_candidate_rows(surf_rows)
    payload = {"summary": summary, "rows": rows}
    (out_dir / "layout_candidate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_md(rows, out_dir)
    print(f"wrote layout candidate analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
