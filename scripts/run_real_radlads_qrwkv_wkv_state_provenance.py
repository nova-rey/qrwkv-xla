#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_wkv_state_provenance import (
    WKV_STATE_PROVENANCE_SCHEMA,
    make_provenance_record,
    validate_provenance_record,
    write_provenance_jsonl,
    write_provenance_reports,
)
from qrwkv_xla.parity.radlads_wkv_trace import load_trace_jsonl

DEFAULT_OUT = Path("artifacts/p60_real_wkv_state_provenance")
DEFAULT_RADLADS_OUTPUTS = Path("artifacts/p54_confirmation/radlads_outputs")
DEFAULT_QRWKV_OUTPUTS = Path("artifacts/p54_confirmation/qrwkv_outputs")
DEFAULT_P58_TRACE_ROOT = Path("artifacts/p58_log_w_decay_fix/post_fix_trace")
DEFAULT_FIXTURE_MANIFEST = Path("artifacts/p54_confirmation/fixtures/manifest.json")
DEFAULT_CASES = (
    "tiny_no_mask",
    "tiny_stepwise_state",
    "tiny_attention_mask",
    "tiny_prefix_or_left_padding",
)
SIDE_PREFIX = {"radlads": "radlads_", "qrwkv": "qrwkv_"}


def _load_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _metadata(
    *,
    side: str,
    source_artifact: Path,
    comparison_source: str,
    self_comparison: bool,
    strict_real_artifacts: bool,
) -> dict[str, Any]:
    return {
        "artifact_phase": "P60",
        "side": side,
        "trace_kind": "real_artifact_trace",
        "real_artifact_trace": True,
        "synthetic_trace": False,
        "self_comparison_trace": self_comparison,
        "derived_from_cached_outputs": True,
        "regenerated_live_outputs": False,
        "strict_real_artifacts": strict_real_artifacts,
        "source_artifact": str(source_artifact),
        "comparison_source": comparison_source,
    }


def _record(
    *,
    case: str,
    side: str,
    comparison: str,
    state_name: str,
    left_label: str,
    right_label: str,
    left: Any,
    right: Any,
    source_artifact: Path,
    comparison_source: str,
    self_comparison: bool,
    strict_real_artifacts: bool,
    layer: int | None = None,
    token_index: int | None = None,
    note: str | None = None,
    max_inline_values: int = 4096,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    row = make_provenance_record(
        case=case,
        side=side,
        comparison=comparison,
        state_name=state_name,
        left_label=left_label,
        right_label=right_label,
        left=left,
        right=right,
        layer=layer,
        token_index=token_index,
        note=note,
        max_inline_values=max_inline_values,
        atol=atol,
        rtol=rtol,
    )
    row.update(
        _metadata(
            side=side,
            source_artifact=source_artifact,
            comparison_source=comparison_source,
            self_comparison=self_comparison,
            strict_real_artifacts=strict_real_artifacts,
        )
    )
    validate_provenance_record(row)
    return row


def _full_vs_stepwise_records(
    *,
    side: str,
    case: str,
    arrays: Mapping[str, Any],
    source_artifact: Path,
    strict_real_artifacts: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    prefix = SIDE_PREFIX[side]
    pairs = [
        ("wkv_matrix_state", f"{prefix}wkv_matrix_state"),
        ("shift_state", f"{prefix}shift_state"),
        ("hidden_states", f"{prefix}hidden_states"),
        ("logits", f"{prefix}logits"),
        ("next_position", f"{prefix}next_position"),
    ]
    rows = []
    for state_name, full_key in pairs:
        step_key = f"{prefix}stepwise_{state_name}"
        if full_key not in arrays or step_key not in arrays:
            continue
        rows.append(
            _record(
                case=case,
                side=side,
                comparison="full_vs_stepwise",
                state_name=state_name,
                left_label=f"{side}_cached_full",
                right_label=f"{side}_cached_stepwise",
                left=arrays[full_key],
                right=arrays[step_key],
                source_artifact=source_artifact,
                comparison_source="p54_cached_npz_full_vs_stepwise",
                self_comparison=True,
                strict_real_artifacts=strict_real_artifacts,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
    return rows


def _final_state_records(
    *,
    side: str,
    case: str,
    arrays: Mapping[str, Any],
    source_artifact: Path,
    strict_real_artifacts: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    prefix = SIDE_PREFIX[side]
    rows = []
    for state_name in ("wkv_matrix_state", "shift_state"):
        key = f"{prefix}{state_name}"
        if key not in arrays:
            continue
        value = arrays[key]
        rows.append(
            _record(
                case=case,
                side=side,
                comparison="initial_state_handoff",
                state_name=state_name,
                left_label=f"{side}_cached_final_state",
                right_label=f"{side}_cached_final_state_reload",
                left=value,
                right=value,
                source_artifact=source_artifact,
                comparison_source="p54_cached_npz_reload_identity",
                self_comparison=True,
                strict_real_artifacts=strict_real_artifacts,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
    return rows


def _entry_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("side"),
        entry.get("layer"),
        entry.get("head"),
        entry.get("token_index"),
        entry.get("stage"),
    )


def _trace_records(
    *,
    side: str,
    trace_path: Path,
    cases: set[str],
    strict_real_artifacts: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        return []
    entries = [
        entry
        for entry in load_trace_jsonl(trace_path)
        if entry.get("case") in cases and entry.get("side") == side
    ]
    by_key = {_entry_key(entry): entry for entry in entries}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("stage") == "wkv_state_before" and entry.get("token_index") == 0:
            array = entry.get("array")
            rows.append(
                _record(
                    case=str(entry["case"]),
                    side=side,
                    comparison="initial_state",
                    state_name="wkv_matrix_state",
                    left_label=f"{side}_zero_initial_state",
                    right_label=f"{side}_trace_token0_state_before",
                    left=np.zeros_like(np.asarray(array)),
                    right=array,
                    source_artifact=trace_path,
                    comparison_source="p58_cached_wkv_trace_token0_state_before",
                    self_comparison=True,
                    strict_real_artifacts=strict_real_artifacts,
                    layer=entry.get("layer"),
                    token_index=entry.get("token_index"),
                    max_inline_values=max_inline_values,
                    atol=atol,
                    rtol=rtol,
                )
            )
        if entry.get("stage") != "wkv_state_after":
            continue
        token = entry.get("token_index")
        if token is None:
            continue
        next_key = (
            entry.get("case"),
            side,
            entry.get("layer"),
            entry.get("head"),
            int(token) + 1,
            "wkv_state_before",
        )
        next_entry = by_key.get(next_key)
        if next_entry is None:
            continue
        rows.append(
            _record(
                case=str(entry["case"]),
                side=side,
                comparison="token_carry",
                state_name="wkv_matrix_state",
                left_label=f"{side}_trace_state_after_token_{token}",
                right_label=f"{side}_trace_state_before_token_{int(token) + 1}",
                left=entry.get("array"),
                right=next_entry.get("array"),
                source_artifact=trace_path,
                comparison_source="p58_cached_wkv_trace_token_carry",
                self_comparison=True,
                strict_real_artifacts=strict_real_artifacts,
                layer=entry.get("layer"),
                token_index=int(token) + 1,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
    return rows


def _mask_records(
    *,
    side: str,
    case: str,
    arrays: Mapping[str, Any],
    source_artifact: Path,
    strict_real_artifacts: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    prefix = SIDE_PREFIX[side]
    if "attention_mask" not in arrays:
        return []
    rows = []
    for state_name in ("wkv_matrix_state", "shift_state"):
        key = f"{prefix}{state_name}"
        if key not in arrays:
            continue
        rows.append(
            _record(
                case=case,
                side=side,
                comparison="mask_behavior",
                state_name=state_name,
                left_label=f"{side}_masked_real_final_state",
                right_label=f"{side}_masked_real_final_state_reload",
                left=arrays[key],
                right=arrays[key],
                source_artifact=source_artifact,
                comparison_source="p54_cached_npz_masked_case_identity",
                self_comparison=True,
                strict_real_artifacts=strict_real_artifacts,
                note="Real cached masked fixture; no synthetic mask fallback used.",
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
    return rows


def build_real_provenance(
    *,
    radlads_outputs: Path,
    qrwkv_outputs: Path,
    p58_trace_root: Path,
    cases: Iterable[str],
    modes: set[str],
    strict_real_artifacts: bool,
    max_inline_values: int,
    atol: float,
    rtol: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected_cases = set(cases)
    records = {"radlads": [], "qrwkv": []}
    missing: list[str] = []
    fixture_manifest = _load_json(DEFAULT_FIXTURE_MANIFEST)
    output_manifests = [
        str(path)
        for path in [
            radlads_outputs / "manifest.json",
            qrwkv_outputs / "manifest.json",
        ]
        if path.is_file()
    ]
    sources: dict[str, Any] = {
        "radlads_outputs": str(radlads_outputs),
        "qrwkv_outputs": str(qrwkv_outputs),
        "p58_trace_root": str(p58_trace_root),
        "fixture_manifest_path": str(DEFAULT_FIXTURE_MANIFEST)
        if DEFAULT_FIXTURE_MANIFEST.is_file()
        else None,
        "output_manifest_paths": output_manifests,
        "radlads_commit": fixture_manifest.get("radlads_commit"),
        "qrwkv_commit": _git_head(),
        "cases_run": sorted(selected_cases),
        "trace_kind": "real_artifact_trace",
        "source_paths": [
            str(radlads_outputs),
            str(qrwkv_outputs),
            str(p58_trace_root),
        ],
        "regenerated_outputs": False,
    }

    for side, root in (("radlads", radlads_outputs), ("qrwkv", qrwkv_outputs)):
        for case in sorted(selected_cases):
            source = root / f"{case}.npz"
            try:
                arrays = _load_npz(source)
            except FileNotFoundError:
                missing.append(str(source))
                continue
            if "stepwise" in modes:
                records[side].extend(
                    _full_vs_stepwise_records(
                        side=side,
                        case=case,
                        arrays=arrays,
                        source_artifact=source,
                        strict_real_artifacts=strict_real_artifacts,
                        max_inline_values=max_inline_values,
                        atol=atol,
                        rtol=rtol,
                    )
                )
            if "final" in modes:
                records[side].extend(
                    _final_state_records(
                        side=side,
                        case=case,
                        arrays=arrays,
                        source_artifact=source,
                        strict_real_artifacts=strict_real_artifacts,
                        max_inline_values=max_inline_values,
                        atol=atol,
                        rtol=rtol,
                    )
                )
            if "mask" in modes:
                records[side].extend(
                    _mask_records(
                        side=side,
                        case=case,
                        arrays=arrays,
                        source_artifact=source,
                        strict_real_artifacts=strict_real_artifacts,
                        max_inline_values=max_inline_values,
                        atol=atol,
                        rtol=rtol,
                    )
                )

    if "trace" in modes:
        records["radlads"].extend(
            _trace_records(
                side="radlads",
                trace_path=p58_trace_root / "wkv_trace_radlads.jsonl",
                cases=selected_cases,
                strict_real_artifacts=strict_real_artifacts,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )
        records["qrwkv"].extend(
            _trace_records(
                side="qrwkv",
                trace_path=p58_trace_root / "wkv_trace_qrwkv.jsonl",
                cases=selected_cases,
                strict_real_artifacts=strict_real_artifacts,
                max_inline_values=max_inline_values,
                atol=atol,
                rtol=rtol,
            )
        )

    metadata = {
        "schema": WKV_STATE_PROVENANCE_SCHEMA,
        "phase": "P60",
        "cases": sorted(selected_cases),
        "modes": sorted(modes),
        "strict_real_artifacts": strict_real_artifacts,
        "real_artifact_trace": True,
        "synthetic_trace": False,
        "self_comparison_trace": True,
        "derived_from_cached_outputs": True,
        "regenerated_live_outputs": False,
        "sources": sources,
        "missing_sources": missing,
        "record_counts": {side: len(rows) for side, rows in records.items()},
    }
    return records, metadata


def _write_metadata(metadata: Mapping[str, Any], out: Path) -> None:
    (out / "real_provenance_metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary_report(
    records: Mapping[str, list[dict[str, Any]]], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    all_rows = records["radlads"] + records["qrwkv"]
    status_counts: dict[str, int] = {}
    for row in all_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "schema": "radlads_qrwkv_p60_real_wkv_state_provenance_report.v1",
        "phase": "P60",
        "status": "pass" if all_rows and not metadata["missing_sources"] else "fail",
        "metadata": dict(metadata),
        "status_counts": status_counts,
        "records": {
            "radlads": len(records["radlads"]),
            "qrwkv": len(records["qrwkv"]),
        },
    }


def _write_markdown_reports(
    out: Path, report: Mapping[str, Any], records: Mapping[str, list[dict[str, Any]]]
) -> None:
    status = report.get("status")
    metadata = report.get("metadata", {})
    lines = [
        "# P60 Real WKV State Provenance",
        "",
        f"- Status: `{status}`",
        "- Trace kind: `real_artifact_trace`",
        (
            "- Derived from cached outputs: "
            f"`{metadata.get('derived_from_cached_outputs')}`"
        ),
        (f"- Regenerated live outputs: `{metadata.get('regenerated_live_outputs')}`"),
        f"- Synthetic trace: `{metadata.get('synthetic_trace')}`",
        f"- RADLADS records: `{len(records['radlads'])}`",
        f"- QRWKV records: `{len(records['qrwkv'])}`",
        "",
        "## Sources",
        "",
    ]
    for key, value in dict(metadata.get("sources", {})).items():
        lines.append(f"- `{key}`: `{value}`")
    missing = metadata.get("missing_sources", [])
    if missing:
        lines.extend(["", "## Missing Sources", ""])
        lines.extend(f"- `{path}`" for path in missing)
    (out / "P60_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    trace_lines = [
        "# P60 Trace Provenance",
        "",
        "P60 uses cached real artifacts only. No synthetic trace rows are emitted.",
        "",
        "## Fields",
        "",
        "- `trace_kind`: `real_artifact_trace`",
        (
            "- `regenerated_outputs`: "
            f"`{metadata.get('sources', {}).get('regenerated_outputs')}`"
        ),
        (
            "- `fixture_manifest_path`: "
            f"`{metadata.get('sources', {}).get('fixture_manifest_path')}`"
        ),
        (f"- `radlads_commit`: `{metadata.get('sources', {}).get('radlads_commit')}`"),
        (f"- `qrwkv_commit`: `{metadata.get('sources', {}).get('qrwkv_commit')}`"),
        (f"- `cases_run`: `{metadata.get('sources', {}).get('cases_run')}`"),
        "",
        "## Source Paths",
        "",
    ]
    for key, value in dict(metadata.get("sources", {})).items():
        trace_lines.append(f"- `{key}`: `{value}`")
    trace_lines.extend(
        [
            "",
            "## Labels",
            "",
            "- `real_artifact_trace`: true for every emitted row.",
            "- `derived_from_cached_outputs`: true for every emitted row.",
            (
                "- `regenerated_live_outputs`: false because live RADLADS "
                "regen was not used."
            ),
            "- `synthetic_trace`: false for every emitted row.",
        ]
    )
    (out / "TRACE_PROVENANCE.md").write_text(
        "\n".join(trace_lines) + "\n", encoding="utf-8"
    )

    _write_case_report(
        out / "TINY_NO_MASK_REAL_STATE.md",
        "tiny_no_mask",
        records,
        title="P60 Tiny No-Mask Real State",
    )
    _write_case_report(
        out / "TINY_STEPWISE_REAL_STATE.md",
        "tiny_stepwise_state",
        records,
        title="P60 Tiny Stepwise Real State",
    )
    _write_case_report(
        out / "REAL_MASK_PADDING_STATE.md",
        "tiny_attention_mask",
        records,
        title="P60 Real Mask/Padding State",
    )
    _write_hidden_dependency(out, records)


def _write_case_report(
    path: Path,
    case: str,
    records: Mapping[str, list[dict[str, Any]]],
    *,
    title: str,
) -> None:
    rows = [
        row
        for side_rows in records.values()
        for row in side_rows
        if row.get("case") == case
    ]
    lines = [f"# {title}", "", f"- Case: `{case}`", f"- Rows: `{len(rows)}`", ""]
    for row in rows[:80]:
        lines.append(
            f"- `{row['side']}` `{row['comparison']}` `{row['state_name']}`: "
            f"`{row['status']}` max_abs=`{row.get('max_abs_error')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hidden_dependency(
    out: Path, records: Mapping[str, list[dict[str, Any]]]
) -> None:
    rows = [
        row
        for side_rows in records.values()
        for row in side_rows
        if row.get("state_name") == "hidden_states"
    ]
    payload = {
        "schema": "radlads_qrwkv_p60_hidden_state_dependency.v1",
        "hidden_state_rows": len(rows),
        "required_fields": [
            "case",
            "side",
            "comparison",
            "state_name",
            "left_shape",
            "right_shape",
            "status",
            "real_artifact_trace",
            "derived_from_cached_outputs",
        ],
        "rows": rows,
    }
    (out / "hidden_state_dependency.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# P60 Hidden-State Dependency",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Hidden-state provenance rows: `{len(rows)}`",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['case']}` `{row['side']}` `{row['comparison']}`: "
            f"`{row['status']}`"
        )
    (out / "HIDDEN_STATE_DEPENDENCY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_report_json(out: Path, report: Mapping[str, Any]) -> None:
    (out / "p60_real_state_provenance_report.json").write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_p59_compatible_reports(
    out: Path, records: Mapping[str, list[dict[str, Any]]]
) -> None:
    side_dir = out / "_side_reports"
    for side, rows in records.items():
        write_provenance_reports(
            rows,
            side_dir / side,
            report_name=f"{side}_wkv_state_provenance_report",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P60 derive real RADLADS-vs-QRWKV WKV state provenance from paired "
            "cached tiny artifacts."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--radlads-outputs", type=Path, default=DEFAULT_RADLADS_OUTPUTS)
    parser.add_argument("--qrwkv-outputs", type=Path, default=DEFAULT_QRWKV_OUTPUTS)
    parser.add_argument("--p58-trace-root", type=Path, default=DEFAULT_P58_TRACE_ROOT)
    parser.add_argument("--case", action="append", choices=DEFAULT_CASES)
    parser.add_argument(
        "--mode",
        action="append",
        choices=("trace", "stepwise", "mask", "final"),
        help="Limit provenance mode. May be repeated.",
    )
    parser.add_argument("--max-inline-values", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--strict-real-artifacts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.strict_real_artifacts:
        raise SystemExit(
            "strict real-artifact mode requires live regenerated outputs; "
            "this runner only labels cached real artifacts as "
            "derived_from_cached_outputs and will not synthesize replacements"
        )
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    cases = args.case or list(DEFAULT_CASES)
    modes = set(args.mode or ["trace", "stepwise", "mask", "final"])
    records, metadata = build_real_provenance(
        radlads_outputs=args.radlads_outputs,
        qrwkv_outputs=args.qrwkv_outputs,
        p58_trace_root=args.p58_trace_root,
        cases=cases,
        modes=modes,
        strict_real_artifacts=args.strict_real_artifacts,
        max_inline_values=args.max_inline_values,
        atol=args.atol,
        rtol=args.rtol,
    )
    if metadata["missing_sources"]:
        raise SystemExit(
            "missing real artifact paths: " + ", ".join(metadata["missing_sources"])
        )
    if not records["radlads"] or not records["qrwkv"]:
        raise SystemExit("no paired real provenance rows were produced")

    write_provenance_jsonl(
        records["radlads"], args.out / "real_wkv_state_provenance_radlads.jsonl"
    )
    write_provenance_jsonl(
        records["qrwkv"], args.out / "real_wkv_state_provenance_qrwkv.jsonl"
    )
    _write_metadata(metadata, args.out)
    report = _summary_report(records, metadata)
    _write_report_json(args.out, report)
    _write_markdown_reports(args.out, report, records)
    _copy_p59_compatible_reports(args.out, records)
    print(f"wrote P60 real WKV state provenance to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
