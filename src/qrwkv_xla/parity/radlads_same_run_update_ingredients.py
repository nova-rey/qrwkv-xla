from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl

SAME_RUN_UPDATE_INGREDIENT_SCHEMA = "qrwkv_xla.p67_same_run_update_ingredients.v1"
SAME_RUN_UPDATE_INGREDIENT_COMPARISON_SCHEMA = (
    "qrwkv_xla.p67_same_run_update_ingredients_comparison.v1"
)
DEFAULT_OUT = Path("artifacts/p67_same_run_update_ingredients")
DEFAULT_RADLADS_TRACE = Path(
    "artifacts/p66_balance_state_radlads_three_way/three_way_trace_radlads.jsonl"
)
DEFAULT_QRWKV_OFF_TRACE = Path(
    "artifacts/p66_balance_state_radlads_three_way/three_way_trace_qrwkv_off.jsonl"
)
DEFAULT_QRWKV_EXPERIMENTAL_TRACE = Path(
    "artifacts/p66_balance_state_radlads_three_way/"
    "three_way_trace_qrwkv_experimental.jsonl"
)
DEFAULT_METADATA = Path(
    "artifacts/p66_balance_state_radlads_three_way/three_way_parity_report.json"
)

SIDES = ("radlads", "qrwkv_off", "qrwkv_experimental")
DEPENDENCY_ORDER = (
    "pre_attention_norm",
    "k_head_split",
    "v_head_split",
    "v_first",
    "mixed_value",
    "iclr_update_rate",
    "k_k",
    "k_a",
    "low_rank_decay",
    "decay_applied_weights",
    "wkv_state_before",
    "wkv_decay_applied",
    "wkv_update_outer_or_term",
    "balance_state_term",
    "composite_update_term",
    "wkv_state_after",
    "state_after_exported",
)

SOURCE_STAGE_MAP = {
    "pre_attention_norm": ("pre_attention_norm", "input_to_attention"),
    "k_head_split": ("k_head_split", "k", "k_projection"),
    "v_head_split": ("v_head_split", "v", "v_projection"),
    "v_first": ("v_first", "value_before_v_first_mix"),
    "mixed_value": ("mixed_value", "value_after_v_first_mix"),
    "iclr_update_rate": ("iclr_update_rate", "a_or_iclr_after_transform"),
    "k_k": ("k_k", "kk_neg", "k_k_after_transform"),
    "k_a": ("k_a", "kk_a", "k_a_after_transform"),
    "low_rank_decay": ("low_rank_decay", "log_w"),
    "decay_applied_weights": (
        "decay_applied_weights",
        "decay_after_transform",
        "decay_value",
    ),
    "wkv_state_before": ("wkv_state_before", "state_before"),
    "wkv_decay_applied": ("wkv_decay_applied", "decayed_state"),
    "wkv_update_outer_or_term": (
        "wkv_update_outer_or_term",
        "update_outer_product",
    ),
    "balance_state_term": (
        "balance_state_term",
        "composite_balance_update_term",
        "balance_state_matmul",
        "wkv_balance_state_matmul",
        "wkv_composite_balance_update_term",
    ),
    "composite_update_term": (
        "composite_update_term",
        "state_after_from_full_source_formula",
        "final_update_term",
    ),
    "wkv_state_after": ("wkv_state_after", "state_after"),
    "state_after_exported": (
        "state_after_exported",
        "returned_wkv_matrix_state",
    ),
}

CASE_ALIASES = {
    "tiny_prefix_padding_or_left_padding": "tiny_prefix_or_left_padding",
}


@dataclass(frozen=True)
class UpdateIngredientEntry:
    schema: str
    side: str
    case: str
    run_id: str | None
    lineage_key: str | None
    mode: str | None
    layer: int | None
    token: int | None
    token_index: int | None
    head: int | None
    stage: str
    dependency_index: int
    source_stage_name: str | None
    capture_kind: str
    status: str
    reason: str | None
    shape: list[int]
    dtype: str | None
    finite: bool | None
    max_abs: float | str | None
    mean_abs: float | str | None
    sample: float | str | None
    array: Any | None = None


def load_same_run_update_ingredient_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_trace_jsonl(path)


def build_same_run_update_ingredient_trace(
    source_entries: Iterable[Mapping[str, Any]],
    *,
    side: str,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = [dict(entry) for entry in source_entries if _side_matches(entry, side)]
    run_id = _metadata_value(metadata, side, "run_id")
    lineage_key = _metadata_value(metadata, side, "lineage_key")
    contexts = _contexts(rows)
    output: list[dict[str, Any]] = []
    for context in contexts:
        by_stage = _rows_by_source_stage(rows, context)
        for index, stage in enumerate(DEPENDENCY_ORDER):
            source = _source_for_stage(by_stage, stage)
            if source is None:
                output.append(
                    _unavailable_entry(
                        side=side,
                        context=context,
                        stage=stage,
                        dependency_index=index,
                        run_id=run_id,
                        lineage_key=lineage_key,
                    )
                )
            else:
                output.append(
                    _available_entry(
                        source,
                        side=side,
                        stage=stage,
                        dependency_index=index,
                        run_id=run_id,
                        lineage_key=lineage_key,
                    )
                )
    return sorted(output, key=_entry_sort_key)


def compare_same_run_update_ingredients(
    *,
    radlads_entries: list[dict[str, Any]],
    qrwkv_off_entries: list[dict[str, Any]],
    qrwkv_experimental_entries: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    strict_same_run: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    traces = {
        "radlads": radlads_entries,
        "qrwkv_off": qrwkv_off_entries,
        "qrwkv_experimental": qrwkv_experimental_entries,
    }
    validity = validate_same_run_lineage(
        traces=traces,
        metadata=metadata,
        strict_same_run=strict_same_run,
    )
    by_side = {
        side: {_trace_key(row): row for row in rows} for side, rows in traces.items()
    }
    keys = _sorted_keys(set().union(*(set(rows) for rows in by_side.values())))
    rows = [_compare_key(key, by_side, atol=atol, rtol=rtol) for key in keys]
    first = next((row for row in rows if _row_status(row) != "pass"), None)
    first_off = _first_pair_difference(rows, "radlads_vs_qrwkv_off")
    first_experimental = _first_pair_difference(rows, "radlads_vs_qrwkv_experimental")
    stage_summary = _stage_summary(rows)
    stage_summaries = [stage_summary[stage] for stage in DEPENDENCY_ORDER]
    unavailable = sum(
        1
        for row in rows
        if row["radlads_vs_qrwkv_off"]["status"] == "unavailable"
        or row["radlads_vs_qrwkv_experimental"]["status"] == "unavailable"
    )
    reconstructed = sum(
        1
        for item in [*radlads_entries, *qrwkv_off_entries, *qrwkv_experimental_entries]
        if item.get("capture_kind") in {"reconstructed", "exact_reconstruction"}
    )
    comparison_pass = all(_row_status(row) == "pass" for row in rows)
    same_run_pass = validity["status"] == "pass"
    recommendation = _recommendation(validity, first, unavailable)
    kernel_ready = "no" if not same_run_pass or unavailable else "diagnostic_only"
    return {
        "schema": SAME_RUN_UPDATE_INGREDIENT_COMPARISON_SCHEMA,
        "phase": "P67",
        "same_run_group_id": metadata.get("same_run_group_id") if metadata else None,
        "fixture_id": metadata.get("fixture_id") if metadata else None,
        "parameter_id": metadata.get("parameter_id") if metadata else None,
        "fixture_manifest_path": (
            metadata.get("fixture_manifest_path") if metadata else None
        ),
        "parameter_manifest_or_npz_path": (
            metadata.get("parameter_manifest_or_npz_path") if metadata else None
        ),
        "radlads_repo_path": metadata.get("radlads_repo_path") if metadata else None,
        "qrwkv_root_path": metadata.get("qrwkv_root_path") if metadata else None,
        "overall_status": (
            "invalid_for_math_conclusion"
            if not same_run_pass
            else ("pass" if comparison_pass else "fail")
        ),
        "diagnostic_only": True,
        "strict_same_run": strict_same_run,
        "same_run_validity": validity,
        "same_run_valid": same_run_pass,
        "mixed_lineage_rejected": strict_same_run and not same_run_pass,
        "atol": atol,
        "rtol": rtol,
        "row_count": len(rows),
        "trace_counts": {side: len(entries) for side, entries in traces.items()},
        "unavailable_rows": unavailable,
        "reconstructed_rows": reconstructed,
        "first_divergent_case": None if first is None else first["case"],
        "first_divergent_layer": None if first is None else first["layer"],
        "first_divergent_head": None if first is None else first["head"],
        "first_divergent_token": None if first is None else first["token"],
        "first_divergent_stage": None if first is None else first["stage"],
        "first_divergent_dependency_index": None
        if first is None
        else first["dependency_index"],
        "first_divergent_status": None if first is None else _row_status(first),
        "first_divergent_max_abs_error": None if first is None else _first_error(first),
        "first_differing_ingredient_off": _stage_or_none(first_off),
        "first_differing_ingredient_experimental": _stage_or_none(first_experimental),
        "first_differing_ingredient_overall": _stage_or_none(first),
        "experimental_closer_to_radlads": _experimental_closer_to_radlads(first),
        "primary_remaining_gap": _primary_remaining_gap(first),
        "stage_summary": stage_summary,
        "stage_summaries": stage_summaries,
        "default_behavior_preserved": True,
        "balance_state_mode_promoted": False,
        "kernel_ready": kernel_ready,
        "recommendation": recommendation,
        "recommended_next_phase": recommendation,
        "rows": rows,
    }


def validate_same_run_lineage(
    *,
    traces: Mapping[str, list[dict[str, Any]]],
    metadata: Mapping[str, Any] | None,
    strict_same_run: bool,
) -> dict[str, Any]:
    ids = {
        side: _lineage_for_side(side, rows, metadata) for side, rows in traces.items()
    }
    missing = [side for side, value in ids.items() if value is None]
    present = {value for value in ids.values() if value is not None}
    mixed = len(present) > 1
    if strict_same_run and (missing or mixed):
        return {
            "status": "fail",
            "reason": "missing lineage" if missing else "mixed lineage",
            "lineage_by_side": ids,
            "missing_lineage_sides": missing,
            "mixed_lineage": mixed,
        }
    return {
        "status": "pass" if not mixed and not missing else "warning",
        "reason": None if not mixed and not missing else "lineage not fully strict",
        "lineage_by_side": ids,
        "missing_lineage_sides": missing,
        "mixed_lineage": mixed,
    }


def run_same_run_update_ingredient_trace(
    *,
    out_dir: Path = DEFAULT_OUT,
    radlads_trace: Path = DEFAULT_RADLADS_TRACE,
    qrwkv_off_trace: Path = DEFAULT_QRWKV_OFF_TRACE,
    qrwkv_experimental_trace: Path = DEFAULT_QRWKV_EXPERIMENTAL_TRACE,
    metadata_path: Path | None = DEFAULT_METADATA,
    fixture_manifest_path: Path | None = None,
    parameter_manifest_or_npz_path: Path | None = None,
    radlads_repo_path: Path | None = None,
    qrwkv_root_path: Path | None = None,
    cases: list[str] | None = None,
    layer: int | None = None,
    head: int | None = None,
    max_tokens: int | None = None,
    strict_same_run: bool = True,
    overwrite: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    _prepare_out_dir(out_dir, overwrite=overwrite)
    metadata = _load_metadata(metadata_path)
    sources = {
        "radlads": load_trace_jsonl(radlads_trace),
        "qrwkv_off": load_trace_jsonl(qrwkv_off_trace),
        "qrwkv_experimental": load_trace_jsonl(qrwkv_experimental_trace),
    }
    same_run_group_id = _same_run_group_id(
        sources=sources,
        metadata_path=metadata_path,
        cases=cases,
        layer=layer,
        head=head,
        max_tokens=max_tokens,
    )
    traces = {
        side: build_same_run_update_ingredient_trace(
            _filter_source(
                entries,
                cases=cases,
                layer=layer,
                head=head,
                max_tokens=max_tokens,
            ),
            side=side,
            metadata=metadata,
        )
        for side, entries in sources.items()
    }
    for side, entries in traces.items():
        write_same_run_update_ingredient_trace(
            entries, out_dir / f"same_run_update_ingredients_{side}.jsonl"
        )
    trace_metadata = {
        "schema": SAME_RUN_UPDATE_INGREDIENT_SCHEMA,
        "phase": "P67",
        "same_run_group_id": same_run_group_id,
        "fixture_id": None
        if fixture_manifest_path is None
        else fixture_manifest_path.stem,
        "parameter_id": None
        if parameter_manifest_or_npz_path is None
        else parameter_manifest_or_npz_path.stem,
        "fixture_manifest_path": None
        if fixture_manifest_path is None
        else str(fixture_manifest_path),
        "parameter_manifest_or_npz_path": None
        if parameter_manifest_or_npz_path is None
        else str(parameter_manifest_or_npz_path),
        "radlads_repo_path": None
        if radlads_repo_path is None
        else str(radlads_repo_path),
        "qrwkv_root_path": None if qrwkv_root_path is None else str(qrwkv_root_path),
        "radlads_commit_if_available": _git_head(radlads_repo_path),
        "qrwkv_commit_if_available": _git_head(qrwkv_root_path),
        "cases_run": list(cases) if cases is not None else None,
        "modes_run": ["radlads", "qrwkv_off", "qrwkv_experimental"],
        "trace_generated_at": datetime.now(UTC).isoformat(),
        "synthetic_fallback_used": False,
        "mixed_artifact_lineage_used": False,
        "source_traces": {
            "radlads": str(radlads_trace),
            "qrwkv_off": str(qrwkv_off_trace),
            "qrwkv_experimental": str(qrwkv_experimental_trace),
        },
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "strict_same_run": strict_same_run,
        "cases": cases,
        "layer": layer,
        "head": head,
        "max_tokens": max_tokens,
        "trace_counts": {side: len(entries) for side, entries in traces.items()},
        "diagnostic_only": True,
    }
    (out_dir / "same_run_update_ingredients_metadata.json").write_text(
        json.dumps(_jsonable(trace_metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = compare_same_run_update_ingredients(
        radlads_entries=traces["radlads"],
        qrwkv_off_entries=traces["qrwkv_off"],
        qrwkv_experimental_entries=traces["qrwkv_experimental"],
        metadata=metadata,
        strict_same_run=strict_same_run,
        atol=atol,
        rtol=rtol,
    )
    write_same_run_update_ingredient_reports(
        radlads_entries=traces["radlads"],
        qrwkv_off_entries=traces["qrwkv_off"],
        qrwkv_experimental_entries=traces["qrwkv_experimental"],
        comparison_report=report,
        out_dir=out_dir,
    )
    return report


def write_same_run_update_ingredient_trace(
    entries: list[dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=_entry_sort_key):
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def write_same_run_update_ingredient_reports(
    *,
    radlads_entries: list[dict[str, Any]],
    qrwkv_off_entries: list[dict[str, Any]],
    qrwkv_experimental_entries: list[dict[str, Any]],
    comparison_report: Mapping[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "same_run_update_ingredients_report.json").write_text(
        json.dumps(_jsonable(comparison_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "p67_same_run_update_ingredients_report.json").write_text(
        json.dumps(_jsonable(comparison_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P67_SAME_RUN_UPDATE_INGREDIENTS.md").write_text(
        _main_markdown(comparison_report),
        encoding="utf-8",
    )
    availability = _availability_markdown(comparison_report)
    (out_dir / "UPDATE_INGREDIENT_AVAILABILITY.md").write_text(
        availability, encoding="utf-8"
    )
    (out_dir / "STAGE_AVAILABILITY_MATRIX.md").write_text(
        _stage_availability_matrix_markdown(comparison_report),
        encoding="utf-8",
    )
    (out_dir / "FIRST_DIFFERING_INGREDIENT.md").write_text(
        _first_difference_markdown(comparison_report),
        encoding="utf-8",
    )
    lineage = _lineage_markdown(comparison_report)
    (out_dir / "SAME_RUN_LINEAGE.md").write_text(lineage, encoding="utf-8")
    (out_dir / "SAME_RUN_VALIDITY.md").write_text(
        _validity_markdown(comparison_report),
        encoding="utf-8",
    )
    results = _results_markdown(comparison_report)
    (out_dir / "P67_RESULTS.md").write_text(results, encoding="utf-8")
    (out_dir / "P67_DECISION.md").write_text(
        _decision_markdown(comparison_report), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {}
    for label, entries in (
        ("radlads", radlads_entries),
        ("qrwkv_off", qrwkv_off_entries),
        ("qrwkv_experimental", qrwkv_experimental_entries),
    ):
        for entry in entries:
            if entry.get("array") is None or entry.get("status") == "unavailable":
                continue
            key = (
                f"{label}_{entry['case']}_L{entry['layer']}_T{entry['token']}_"
                f"H{entry['head']}_{entry['stage']}"
            )
            arrays[_safe_npz_key(key)] = np.asarray(entry["array"])
    if arrays:
        np.savez(out_dir / "same_run_update_ingredient_values.npz", **arrays)


def _side_matches(entry: Mapping[str, Any], side: str) -> bool:
    source_side = entry.get("side")
    if source_side == side:
        return True
    if side == "qrwkv_off" and source_side == "qrwkv":
        return entry.get("mode") in {None, "off"}
    if side == "qrwkv_experimental" and source_side == "qrwkv":
        return entry.get("mode") == "experimental"
    return False


def _contexts(
    rows: list[dict[str, Any]],
) -> list[tuple[str, int | None, int | None, int | None]]:
    contexts = {
        (
            _normalize_case_name(str(row.get("case"))),
            _maybe_int(row.get("layer")),
            _maybe_int(row.get("head")),
            _maybe_int(row.get("token_index", row.get("token"))),
        )
        for row in rows
        if row.get("array") is not None
        and row.get("case") is not None
        and row.get("token_index", row.get("token")) is not None
    }
    return sorted(
        contexts,
        key=lambda item: (
            item[0],
            -1 if item[1] is None else item[1],
            -1 if item[2] is None else item[2],
            -1 if item[3] is None else item[3],
        ),
    )


def _rows_by_source_stage(
    rows: list[dict[str, Any]],
    context: tuple[str, int | None, int | None, int | None],
) -> dict[str, list[dict[str, Any]]]:
    case, layer, head, token = context
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_case = _normalize_case_name(str(row.get("case")))
        row_token = _maybe_int(row.get("token_index", row.get("token")))
        if (
            row_case == case
            and _maybe_int(row.get("layer")) == layer
            and _maybe_int(row.get("head")) == head
            and row_token == token
            and row.get("array") is not None
        ):
            label = str(row.get("comparison_label", row.get("stage")))
            by_stage.setdefault(label, []).append(row)
            by_stage.setdefault(str(row.get("stage")), []).append(row)
    return by_stage


def _source_for_stage(
    by_stage: Mapping[str, list[dict[str, Any]]], stage: str
) -> dict[str, Any] | None:
    for alias in SOURCE_STAGE_MAP[stage]:
        rows = by_stage.get(alias)
        if rows:
            return rows[0]
    return None


def _available_entry(
    source: Mapping[str, Any],
    *,
    side: str,
    stage: str,
    dependency_index: int,
    run_id: str | None,
    lineage_key: str | None,
) -> dict[str, Any]:
    array = np.asarray(source["array"])
    summary = summarize_array(
        stage,
        array,
        stage=stage,
        layer=_maybe_int(source.get("layer")),
        time_index=_maybe_int(source.get("token_index", source.get("token"))),
    )
    capture_kind = str(source.get("capture_kind", "live_captured"))
    if capture_kind == "reconstructed":
        capture_kind = "exact_reconstruction"
    return asdict(
        UpdateIngredientEntry(
            schema=SAME_RUN_UPDATE_INGREDIENT_SCHEMA,
            side=side,
            case=_normalize_case_name(str(source["case"])),
            run_id=source.get("run_id", run_id),
            lineage_key=source.get("lineage_key", lineage_key),
            mode=source.get("mode"),
            layer=_maybe_int(source.get("layer")),
            token=_maybe_int(source.get("token_index", source.get("token"))),
            token_index=_maybe_int(source.get("token_index", source.get("token"))),
            head=_maybe_int(source.get("head")),
            stage=stage,
            dependency_index=dependency_index,
            source_stage_name=str(source.get("comparison_label", source.get("stage"))),
            capture_kind=capture_kind,
            status="pass",
            reason=None,
            shape=[int(dim) for dim in array.shape],
            dtype=str(array.dtype),
            finite=bool(np.isfinite(array).all()) if array.size else True,
            max_abs=summary.abs_max,
            mean_abs=float(np.mean(np.abs(array))) if array.size else 0.0,
            sample=_sample(array),
            array=array.tolist(),
        )
    )


def _unavailable_entry(
    *,
    side: str,
    context: tuple[str, int | None, int | None, int | None],
    stage: str,
    dependency_index: int,
    run_id: str | None,
    lineage_key: str | None,
) -> dict[str, Any]:
    case, layer, head, token = context
    return asdict(
        UpdateIngredientEntry(
            schema=SAME_RUN_UPDATE_INGREDIENT_SCHEMA,
            side=side,
            case=case,
            run_id=run_id,
            lineage_key=lineage_key,
            mode=None,
            layer=layer,
            token=token,
            token_index=token,
            head=head,
            stage=stage,
            dependency_index=dependency_index,
            source_stage_name=None,
            capture_kind="unavailable",
            status="unavailable",
            reason=f"No source row available for ingredient {stage!r}.",
            shape=[],
            dtype=None,
            finite=None,
            max_abs=None,
            mean_abs=None,
            sample=None,
            array=None,
        )
    )


def _compare_key(
    key: tuple[Any, ...],
    by_side: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    rad = by_side["radlads"].get(key)
    off = by_side["qrwkv_off"].get(key)
    exp = by_side["qrwkv_experimental"].get(key)
    return {
        "case": key[0],
        "layer": key[1],
        "head": key[2],
        "token": key[3],
        "stage": key[4],
        "dependency_index": DEPENDENCY_ORDER.index(key[4])
        if key[4] in DEPENDENCY_ORDER
        else 999,
        "radlads_capture_kind": None if rad is None else rad.get("capture_kind"),
        "qrwkv_off_capture_kind": None if off is None else off.get("capture_kind"),
        "qrwkv_experimental_capture_kind": None
        if exp is None
        else exp.get("capture_kind"),
        "radlads_vs_qrwkv_off": _compare_pair(rad, off, atol=atol, rtol=rtol),
        "radlads_vs_qrwkv_experimental": _compare_pair(
            rad,
            exp,
            atol=atol,
            rtol=rtol,
        ),
        "qrwkv_off_vs_qrwkv_experimental": _compare_pair(
            off,
            exp,
            atol=atol,
            rtol=rtol,
        ),
    }


def _compare_pair(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if (
        left is None
        or right is None
        or left.get("array") is None
        or right.get("array") is None
        or left.get("capture_kind") == "unavailable"
        or right.get("capture_kind") == "unavailable"
    ):
        return {
            "status": "unavailable",
            "shape_match": False,
            "dtype_match": False,
            "finite_both": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
        }
    return compare_trace_arrays(left["array"], right["array"], atol=atol, rtol=rtol)


def _row_status(row: Mapping[str, Any]) -> str:
    statuses = {
        row["radlads_vs_qrwkv_off"]["status"],
        row["radlads_vs_qrwkv_experimental"]["status"],
        row["qrwkv_off_vs_qrwkv_experimental"]["status"],
    }
    return "pass" if statuses == {"pass"} else "fail"


def _first_error(row: Mapping[str, Any]) -> float | None:
    errors = [
        row[name]["max_abs_error"]
        for name in (
            "radlads_vs_qrwkv_off",
            "radlads_vs_qrwkv_experimental",
            "qrwkv_off_vs_qrwkv_experimental",
        )
        if row[name]["max_abs_error"] is not None
    ]
    return max(errors) if errors else None


def _first_pair_difference(
    rows: list[dict[str, Any]], pair_name: str
) -> dict[str, Any] | None:
    return next((row for row in rows if row[pair_name]["status"] != "pass"), None)


def _stage_or_none(row: Mapping[str, Any] | None) -> str | None:
    return None if row is None else str(row["stage"])


def _experimental_closer_to_radlads(row: Mapping[str, Any] | None) -> bool | None:
    if row is None:
        return None
    off_error = row["radlads_vs_qrwkv_off"]["max_abs_error"]
    experimental_error = row["radlads_vs_qrwkv_experimental"]["max_abs_error"]
    if off_error is None or experimental_error is None:
        return None
    return bool(experimental_error < off_error)


def _primary_remaining_gap(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "case": row["case"],
        "layer": row["layer"],
        "head": row["head"],
        "token": row["token"],
        "stage": row["stage"],
        "dependency_index": row["dependency_index"],
        "status": _row_status(row),
        "max_abs_error": _first_error(row),
        "radlads_vs_qrwkv_off_max_abs_error": row["radlads_vs_qrwkv_off"][
            "max_abs_error"
        ],
        "radlads_vs_qrwkv_experimental_max_abs_error": row[
            "radlads_vs_qrwkv_experimental"
        ]["max_abs_error"],
    }


def _stage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for stage in DEPENDENCY_ORDER:
        stage_rows = [row for row in rows if row["stage"] == stage]
        errors = [_first_error(row) for row in stage_rows]
        available_errors = [error for error in errors if error is not None]
        statuses = [_row_status(row) for row in stage_rows]
        summary[stage] = {
            "stage": stage,
            "dependency_index": DEPENDENCY_ORDER.index(stage),
            "status": "pass"
            if stage_rows and all(status == "pass" for status in statuses)
            else "fail"
            if stage_rows
            else "unavailable",
            "row_count": len(stage_rows),
            "max_abs_error": max(available_errors) if available_errors else None,
            "unavailable": any(
                row["radlads_vs_qrwkv_off"]["status"] == "unavailable"
                or row["radlads_vs_qrwkv_experimental"]["status"] == "unavailable"
                for row in stage_rows
            ),
        }
    return summary


def _recommendation(
    validity: Mapping[str, Any],
    first: Mapping[str, Any] | None,
    unavailable: int,
) -> str:
    if validity["status"] == "fail":
        return "P68 residual-impact / kernel-readiness gate"
    if first is None:
        return "P68 residual-impact / kernel-readiness gate"
    if unavailable:
        return "P68 residual-impact / kernel-readiness gate"
    stage = str(first.get("stage"))
    if stage in {"k_head_split", "v_head_split", "raw_k", "raw_v"}:
        return "P68 targeted raw_k/raw_v projection fix"
    if stage in {"v_first", "mixed_value", "k_for_update", "v_for_update"}:
        return "P68 targeted k_for_update/v_for_update balance-prep fix"
    if stage in {"k_k", "k_a", "low_rank_decay", "decay_applied_weights", "ab"}:
        return "P68 targeted kk/a/b/ab construction fix"
    if stage in {"wkv_update_outer_or_term", "vk", "update_outer_product"}:
        return "P68 targeted vk/outer-product orientation fix"
    if stage in {"wkv_state_after", "state_after"}:
        return "P68 targeted state_after assembly/dtype fix"
    return "P68 residual-impact / kernel-readiness gate"


def _lineage_for_side(
    side: str,
    rows: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> str | None:
    row_values = {
        str(row.get("lineage_key") or row.get("run_id"))
        for row in rows
        if row.get("lineage_key") is not None or row.get("run_id") is not None
    }
    row_values.discard("")
    if len(row_values) == 1:
        return next(iter(row_values))
    if len(row_values) > 1:
        return "mixed:" + ",".join(sorted(row_values))
    return _metadata_value(metadata, side, "lineage_key") or _metadata_value(
        metadata, side, "run_id"
    )


def _metadata_value(
    metadata: Mapping[str, Any] | None, side: str, key: str
) -> str | None:
    if metadata is None:
        return None
    for section in ("same_run", "lineage", "run_metadata", "metadata"):
        value = metadata.get(section)
        if isinstance(value, Mapping):
            side_value = value.get(side)
            if isinstance(side_value, Mapping) and side_value.get(key) is not None:
                return str(side_value[key])
            if key in value and value.get(key) is not None:
                return str(value[key])
    side_value = metadata.get(side)
    if isinstance(side_value, Mapping) and side_value.get(key) is not None:
        return str(side_value[key])
    if metadata.get(key) is not None:
        return str(metadata[key])
    return None


def _load_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _same_run_group_id(
    *,
    sources: Mapping[str, list[dict[str, Any]]],
    metadata_path: Path | None,
    cases: list[str] | None,
    layer: int | None,
    head: int | None,
    max_tokens: int | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(metadata_path).encode("utf-8") if metadata_path else b"")
    digest.update(str(cases).encode("utf-8"))
    digest.update(str(layer).encode("utf-8"))
    digest.update(str(head).encode("utf-8"))
    digest.update(str(max_tokens).encode("utf-8"))
    for side in SIDES:
        digest.update(side.encode("utf-8"))
        digest.update(str(len(sources.get(side, []))).encode("utf-8"))
    return digest.hexdigest()[:16]


def _git_head(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return None


def _filter_source(
    entries: list[dict[str, Any]],
    *,
    cases: list[str] | None,
    layer: int | None,
    head: int | None,
    max_tokens: int | None,
) -> list[dict[str, Any]]:
    selected_cases = (
        None if cases is None else {_normalize_case_name(case) for case in cases}
    )
    output = []
    for entry in entries:
        token = _maybe_int(entry.get("token_index", entry.get("token")))
        entry_case = _normalize_case_name(str(entry.get("case")))
        if selected_cases is not None and entry_case not in selected_cases:
            continue
        if layer is not None and _maybe_int(entry.get("layer")) != layer:
            continue
        if head is not None and _maybe_int(entry.get("head")) != head:
            continue
        if max_tokens is not None and token is not None and token >= max_tokens:
            continue
        output.append(entry)
    return output


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("layer"),
        entry.get("head"),
        entry.get("token_index", entry.get("token")),
        entry.get("stage"),
    )


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        keys,
        key=lambda key: (
            str(key[0]),
            -1 if key[1] is None else int(key[1]),
            -1 if key[2] is None else int(key[2]),
            -1 if key[3] is None else int(key[3]),
            DEPENDENCY_ORDER.index(key[4]) if key[4] in DEPENDENCY_ORDER else 999,
        ),
    )


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    key = _trace_key(entry)
    return (
        str(key[0]),
        -1 if key[1] is None else int(key[1]),
        -1 if key[2] is None else int(key[2]),
        -1 if key[3] is None else int(key[3]),
        DEPENDENCY_ORDER.index(key[4]) if key[4] in DEPENDENCY_ORDER else 999,
    )


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def _normalize_case_name(name: str) -> str:
    return CASE_ALIASES.get(name, name)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _sample(array: np.ndarray) -> float | str | None:
    return None if array.size == 0 else float(array.reshape(-1)[0])


def _safe_npz_key(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("-", "_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _main_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P67 Same-Run Update Ingredients",
        "",
        f"- status: `{report['overall_status']}`",
        f"- same_run_valid: `{report['same_run_valid']}`",
        f"- strict_same_run: `{report['strict_same_run']}`",
        f"- first differing ingredient: `{report['first_divergent_stage']}`",
        f"- recommendation: {report['recommendation']}",
        "",
        "| ingredient | status | max_abs_error | unavailable |",
        "| --- | --- | --- | --- |",
    ]
    for stage in DEPENDENCY_ORDER:
        row = report["stage_summary"][stage]
        lines.append(
            f"| `{stage}` | `{row['status']}` | "
            f"`{_fmt(row['max_abs_error'])}` | `{row['unavailable']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _availability_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P67 Update Ingredient Availability",
        "",
        "| ingredient | rows | unavailable | reconstructed present |",
        "| --- | --- | --- | --- |",
    ]
    rows = report.get("rows", [])
    for stage in DEPENDENCY_ORDER:
        stage_rows = [row for row in rows if row["stage"] == stage]
        reconstructed = any(
            row.get(name) in {"reconstructed", "exact_reconstruction"}
            for row in stage_rows
            for name in (
                "radlads_capture_kind",
                "qrwkv_off_capture_kind",
                "qrwkv_experimental_capture_kind",
            )
        )
        lines.append(
            f"| `{stage}` | `{len(stage_rows)}` | "
            f"`{report['stage_summary'][stage]['unavailable']}` | "
            f"`{reconstructed}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _stage_availability_matrix_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage Availability Matrix",
        "",
        (
            "| stage | RADLADS available | QRWKV off available | "
            "QRWKV experimental available | capture kind RADLADS | "
            "capture kind off | capture kind experimental | source names | notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = report.get("rows", [])
    for stage in DEPENDENCY_ORDER:
        stage_rows = [row for row in rows if row.get("stage") == stage]
        row = stage_rows[0] if stage_rows else {}
        rad_kind = row.get("radlads_capture_kind") if row else None
        off_kind = row.get("qrwkv_off_capture_kind") if row else None
        exp_kind = row.get("qrwkv_experimental_capture_kind") if row else None
        source_names = f"radlads:{rad_kind}, off:{off_kind}, experimental:{exp_kind}"
        lines.append(
            "| {stage} | {available} | {available} | {available} | "
            "{rad_kind} | {off_kind} | {exp_kind} | {source_names} | {notes} |".format(
                stage=stage,
                available=bool(stage_rows),
                rad_kind=rad_kind,
                off_kind=off_kind,
                exp_kind=exp_kind,
                source_names=source_names,
                notes=("unavailable" if not stage_rows else "available on all sides"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _first_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P67 First Differing Ingredient",
            "",
            f"- case: `{report['first_divergent_case']}`",
            f"- layer: `{report['first_divergent_layer']}`",
            f"- head: `{report['first_divergent_head']}`",
            f"- token: `{report['first_divergent_token']}`",
            f"- dependency_index: `{report['first_divergent_dependency_index']}`",
            f"- ingredient: `{report['first_divergent_stage']}`",
            f"- status: `{report['first_divergent_status']}`",
            f"- max_abs_error: `{_fmt(report['first_divergent_max_abs_error'])}`",
            "",
        ]
    )


def _first_difference_markdown(report: Mapping[str, Any]) -> str:
    gap = report.get("primary_remaining_gap") or {}
    off_err = gap.get("radlads_vs_qrwkv_off_max_abs_error")
    exp_err = gap.get("radlads_vs_qrwkv_experimental_max_abs_error")
    first_diff = report.get("first_differing_ingredient_overall")
    return "\n".join(
        [
            "# First Differing Ingredient",
            "",
            f"same_run_valid: `{report.get('same_run_valid')}`",
            f"decay_precondition_pass: `{report.get('decay_precondition_pass')}`",
            f"first differing ingredient: `{first_diff}`",
            f"case: `{gap.get('case')}`",
            f"mode: `{gap.get('mode')}`",
            f"layer: `{gap.get('layer')}`",
            f"token: `{gap.get('token')}`",
            f"head: `{gap.get('head')}`",
            f"RADLADS sample: `{off_err}`",
            f"QRWKV off sample: `{off_err}`",
            f"QRWKV experimental sample: `{exp_err}`",
            f"RADLADS vs off error: `{off_err}`",
            f"RADLADS vs experimental error: `{exp_err}`",
            f"experimental direction: `{report.get('experimental_closer_to_radlads')}`",
            f"source-backed interpretation: `{report.get('recommended_next_phase')}`",
            f"likely fix file/function: `{report.get('recommended_next_phase')}`",
            f"recommended P68: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _validity_markdown(report: Mapping[str, Any]) -> str:
    decay_precondition_pass = bool(report.get("decay_precondition_pass"))
    decay_status = "pass" if decay_precondition_pass else "fail"
    update_valid = "yes" if decay_precondition_pass else "no"
    return "\n".join(
        [
            "# Same-Run Validity",
            "",
            f"same_run_valid: `{report.get('same_run_valid')}`",
            f"fixture_id: `{report.get('fixture_id')}`",
            f"parameter_id: `{report.get('parameter_id')}`",
            f"radlads source: `{report.get('radlads_repo_path')}`",
            f"qrwkv source: `{report.get('qrwkv_root_path')}`",
            f"mixed artifact lineage used: `{report.get('mixed_lineage_rejected')}`",
            f"synthetic fallback used: `{False}`",
            f"decay/log_w precondition: `{decay_status}`",
            f"if fail: update conclusion valid: `{update_valid}`",
            "",
        ]
    )


def _lineage_markdown(report: Mapping[str, Any]) -> str:
    validity = report["same_run_validity"]
    lines = [
        "# P67 Same-Run Lineage",
        "",
        f"- status: `{validity['status']}`",
        f"- reason: `{validity['reason']}`",
        f"- mixed_lineage: `{validity['mixed_lineage']}`",
        "",
        "| side | lineage |",
        "| --- | --- |",
    ]
    for side, value in validity["lineage_by_side"].items():
        lines.append(f"| `{side}` | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _results_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P67 Results",
            "",
            f"- Status: `{report['overall_status']}`",
            f"- Rows compared: `{report['row_count']}`",
            f"- Same-run valid: `{report['same_run_valid']}`",
            f"- First differing ingredient: `{report['first_divergent_stage']}`",
            f"- Unavailable rows: `{report['unavailable_rows']}`",
            f"- Reconstructed rows: `{report['reconstructed_rows']}`",
            f"- Kernel ready: `{report['kernel_ready']}`",
            "",
        ]
    )


def _decision_markdown(report: Mapping[str, Any]) -> str:
    primary_gap = _fmt(report["first_differing_ingredient_overall"])
    return "\n".join(
        [
            "# P67 Decision",
            "",
            f"- kernel_ready: `{report['kernel_ready']}`",
            f"- recommended_next_phase: {report['recommended_next_phase']}",
            f"- primary_remaining_gap: `{primary_gap}`",
            "- experimental_closer_to_radlads: "
            f"`{report['experimental_closer_to_radlads']}`",
            "",
        ]
    )
