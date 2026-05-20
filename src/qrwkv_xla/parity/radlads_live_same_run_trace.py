from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_numerical_fixtures import (
    load_numerical_case_arrays,
    load_numerical_manifest,
)
from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
)
from qrwkv_xla.parity.radlads_replay import (
    replay_profile_for_case,
    student_for_replay_profile,
)
from qrwkv_xla.parity.radlads_replay_diagnostics import summarize_array
from qrwkv_xla.parity.radlads_same_run_update_ingredients import (
    DEPENDENCY_ORDER as P67_DEPENDENCY_ORDER,
)
from qrwkv_xla.parity.radlads_same_run_update_ingredients import (
    SOURCE_STAGE_MAP as P67_SOURCE_STAGE_MAP,
)
from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl

LIVE_SAME_RUN_TRACE_SCHEMA = "qrwkv_xla.p68_live_same_run_trace.v1"
LIVE_SAME_RUN_REPORT_SCHEMA = "qrwkv_xla.p68_live_same_run_trace_report.v1"
DEFAULT_OUT = Path("artifacts/p68_live_same_run_trace")
SIDES = ("radlads", "qrwkv_off", "qrwkv_experimental")
MINIMUM_STAGE_NORMALIZATION = {
    "pre_attention_norm": "pre_attention_norm",
    "k_head_split": "raw_k",
    "v_head_split": "raw_v",
    "low_rank_decay": "decay_log_w",
    "decay_applied_weights": "decay_value",
    "wkv_state_before": "prev_state",
    "wkv_update_outer_or_term": "vk",
    "wkv_state_after": "state_after_live",
}
NORMALIZED_TO_SOURCE_STAGE = {
    normalized: source for source, normalized in MINIMUM_STAGE_NORMALIZATION.items()
}
STRETCH_STAGE_NORMALIZATION = {
    stage: stage
    for stage in P67_DEPENDENCY_ORDER
    if stage not in MINIMUM_STAGE_NORMALIZATION
}
STAGE_NORMALIZATION = MINIMUM_STAGE_NORMALIZATION | STRETCH_STAGE_NORMALIZATION
DEPENDENCY_ORDER = tuple(
    STAGE_NORMALIZATION.get(stage, stage) for stage in P67_DEPENDENCY_ORDER
)
SOURCE_STAGE_MAP = {
    STAGE_NORMALIZATION.get(stage, stage): aliases
    for stage, aliases in P67_SOURCE_STAGE_MAP.items()
}
SOURCE_STAGE_MAP["prev_state"] = (
    *SOURCE_STAGE_MAP["prev_state"],
    "initial_matrix_state",
)
SOURCE_STAGE_MAP["vk"] = (*SOURCE_STAGE_MAP["vk"], "update_term")
MINIMUM_STAGES = tuple(MINIMUM_STAGE_NORMALIZATION.values())
CRITICAL_STAGES = MINIMUM_STAGES
BALANCE_STATE_CONFIG_KEYS = {
    "radlads_balance_state",
    "radlads_balance_state_terms",
    "use_radlads_balance_state_terms",
}
P70_RADLADS_HOOK_COMPLETION = (
    "P70 targeted live RADLADS pre_attention/k/v hook completion"
)
P70_QRWKV_HOOK_COMPLETION = "P70 targeted live QRWKV pre_attention/k/v hook completion"
P70_DECAY_REPAIR = "P70 targeted live decay/log_w hook repair"
P70_WKV_REPAIR = "P70 targeted live WKV state/update hook repair"


class LiveTraceCollector:
    def __init__(
        self,
        *,
        same_run_group_id: str,
        fixture_id: str,
        parameter_id: str,
        case: str,
        side: str,
        mode: str | None = None,
        live_config: Mapping[str, Any] | None = None,
        max_inline_values: int = 1_000_000,
    ) -> None:
        self.same_run_group_id = same_run_group_id
        self.fixture_id = fixture_id
        self.parameter_id = parameter_id
        self.case = case
        self.side = side
        self.mode = mode
        self.live_config = dict(live_config or {})
        self.max_inline_values = max_inline_values
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        name: str | None = None,
        value: Any | None = None,
        *,
        layer: int | None,
        token: int | None = None,
        head: int | None = None,
        stage: str,
        source_stage_name: str | None = None,
        time_index: int | None = None,
        token_index: int | None = None,
        capture_kind: str = "live_captured",
        source_file: str | None = None,
        source_function: str | None = None,
    ) -> None:
        del source_file, source_function
        if value is None:
            return
        source_stage = _canonical_source_stage(source_stage_name or stage)
        normalized_stage = _normalize_stage(source_stage)
        index = token if token is not None else token_index
        index = time_index if index is None else index
        array = np.array(value, copy=True)
        if (
            head is None
            and index is None
            and normalized_stage == "pre_attention_norm"
            and array.ndim == 3
            and array.shape[1] > 0
        ):
            for token_offset in range(int(array.shape[1])):
                self._append(
                    name=name,
                    value=array[:, token_offset, :],
                    layer=layer,
                    token=token_offset,
                    head=0,
                    stage=normalized_stage,
                    source_stage_name=source_stage,
                    capture_kind=capture_kind,
                )
            return
        if (
            head is None
            and normalized_stage in MINIMUM_STAGES
            and array.ndim >= 3
            and array.shape[1] > 0
        ):
            for head_index in range(int(array.shape[1])):
                self._append(
                    name=name,
                    value=np.take(array, head_index, axis=1),
                    layer=layer,
                    token=index,
                    head=head_index,
                    stage=normalized_stage,
                    source_stage_name=source_stage,
                    capture_kind=capture_kind,
                )
            return
        self._append(
            name=name,
            value=array,
            layer=layer,
            token=index,
            head=head,
            stage=normalized_stage,
            source_stage_name=source_stage,
            capture_kind=capture_kind,
        )

    def _append(
        self,
        *,
        name: str | None,
        value: np.ndarray,
        layer: int | None,
        token: int | None,
        head: int | None,
        stage: str,
        source_stage_name: str,
        capture_kind: str,
    ) -> None:
        summary = summarize_array(
            name or stage,
            value,
            stage=stage,
            layer=layer,
            time_index=token,
        )
        self.entries.append(
            {
                "same_run_group_id": self.same_run_group_id,
                "fixture_id": self.fixture_id,
                "parameter_id": self.parameter_id,
                "case": self.case,
                "side": self.side,
                "mode": self.mode,
                "layer": layer,
                "head": head,
                "token": token,
                "token_index": token,
                "stage": stage,
                "source_stage_name": source_stage_name,
                "capture_kind": capture_kind,
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "array": (
                    value.tolist() if value.size <= self.max_inline_values else None
                ),
                "summary": {
                    "finite": bool(np.isfinite(value).all()) if value.size else True,
                    "max_abs": summary.abs_max,
                    "mean_abs": float(np.mean(np.abs(value))) if value.size else 0.0,
                    "sample": None if value.size == 0 else float(value.reshape(-1)[0]),
                },
                "live_config": self.live_config,
            }
        )


def deterministic_fixture_id(path: Path) -> str:
    return "fixture-" + _path_digest(path)[:16]


def deterministic_parameter_id(
    *,
    parameters: Path | None = None,
    parameter_manifest: Path | None = None,
    fixture_parameter_key: str | None = None,
) -> str:
    if fixture_parameter_key:
        return (
            "parameter-"
            + hashlib.sha256(
                f"fixture-key:{fixture_parameter_key}".encode()
            ).hexdigest()[:16]
        )
    path = parameter_manifest or parameters
    if path is None:
        raise ValueError(
            "one of parameters, parameter_manifest, or "
            "fixture_parameter_key is required"
        )
    return "parameter-" + _path_digest(path)[:16]


def new_same_run_group_id(
    *,
    fixture_id: str | None = None,
    parameter_id: str | None = None,
    cases: list[str] | None = None,
    mode: str | None = None,
    layer: int | None = None,
    head: int | None = None,
    max_tokens: int | None = None,
    strict_live: bool | None = None,
) -> str:
    digest = hashlib.sha256()
    for value in (
        fixture_id,
        parameter_id,
        cases,
        mode,
        layer,
        head,
        max_tokens,
        strict_live,
    ):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return "p68-" + digest.hexdigest()[:16]


def load_live_same_run_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_trace_jsonl(path)


def build_live_same_run_trace(
    source_entries: Iterable[Mapping[str, Any]],
    *,
    side: str,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
    contexts: Iterable[tuple[str, str | None, int | None, int | None, int | None]],
) -> list[dict[str, Any]]:
    rows = [dict(entry) for entry in source_entries if _side_matches(entry, side)]
    output: list[dict[str, Any]] = []
    for context in sorted(contexts, key=_context_sort_key):
        by_stage = _rows_by_source_stage(rows, context)
        for dependency_index, stage in enumerate(DEPENDENCY_ORDER):
            source = _source_for_stage(by_stage, stage)
            if source is None:
                output.append(
                    _unavailable_row(
                        side=side,
                        context=context,
                        stage=stage,
                        dependency_index=dependency_index,
                        same_run_group_id=same_run_group_id,
                        fixture_id=fixture_id,
                        parameter_id=parameter_id,
                        reason=f"missing_live_hook:{side}:{stage}",
                    )
                )
            else:
                output.append(
                    _available_row(
                        source,
                        side=side,
                        stage=stage,
                        dependency_index=dependency_index,
                        same_run_group_id=same_run_group_id,
                        fixture_id=fixture_id,
                        parameter_id=parameter_id,
                    )
                )
    return sorted(output, key=_entry_sort_key)


def compare_live_same_run_traces(
    *,
    traces: Mapping[str, list[dict[str, Any]]],
    metadata: Mapping[str, Any],
    strict_live: bool,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    by_side = {
        side: {_trace_key(row): row for row in traces.get(side, [])} for side in SIDES
    }
    keys = _sorted_keys(set().union(*(set(rows) for rows in by_side.values())))
    rows = [_compare_key(key, by_side, atol=atol, rtol=rtol) for key in keys]
    identity = _validate_identity(traces=traces, metadata=metadata)
    config = _validate_live_config(traces)
    unavailable = _validate_critical_availability(traces)
    decay = _validate_decay_log_w_precondition(rows)
    first = next((row for row in rows if _row_status(row) != "pass"), None)
    stage_summary = _stage_summary(rows)
    live_counts = _live_row_counts(traces)
    minimum_availability = _minimum_stage_availability(traces)
    unavailable_minimum = _unavailable_minimum_stages(traces)
    same_run_valid = (
        (not strict_live or identity["status"] == "pass")
        and config["status"] == "pass"
        and unavailable["status"] == "pass"
        and decay["status"] == "pass"
    )
    recommendation = _recommendation(
        same_run_valid=same_run_valid,
        identity=identity,
        config=config,
        unavailable=unavailable,
        decay=decay,
        first=first,
    )
    return {
        "schema": LIVE_SAME_RUN_REPORT_SCHEMA,
        "phase": "P68",
        "same_run_group_id": metadata.get("same_run_group_id"),
        "fixture_id": metadata.get("fixture_id"),
        "parameter_id": metadata.get("parameter_id"),
        "fixture_manifest_path": metadata.get("fixture_manifest_path"),
        "parameter_manifest_or_npz_path": metadata.get(
            "parameter_manifest_or_npz_path"
        ),
        "radlads_repo_path": metadata.get("radlads_repo_path"),
        "qrwkv_root_path": metadata.get("qrwkv_root_path"),
        "strict_live": strict_live,
        "same_run_valid": same_run_valid,
        "same_run_validity": {
            "status": "pass" if same_run_valid else "fail",
            "identity": identity,
            "live_config": config,
            "critical_availability": unavailable,
            "decay_log_w_precondition": decay,
        },
        "decay_precondition_pass": decay["status"] == "pass",
        "overall_status": "pass"
        if same_run_valid and first is None
        else ("invalid_for_math_conclusion" if not same_run_valid else "fail"),
        "diagnostic_only": True,
        "default_behavior_preserved": True,
        "synthetic_fallback_used": False,
        "mixed_artifact_lineage_used": False,
        "row_count": len(rows),
        "trace_counts": {side: len(traces.get(side, [])) for side in SIDES},
        "live_rows_captured": live_counts,
        "live_rows_captured_radlads": live_counts["radlads"],
        "live_rows_captured_qrwkv_off": live_counts["qrwkv_off"],
        "live_rows_captured_qrwkv_experimental": live_counts["qrwkv_experimental"],
        "minimum_stage_availability": minimum_availability,
        "unavailable_minimum_stages": unavailable_minimum,
        "unavailable_rows": sum(
            1
            for side in SIDES
            for row in traces.get(side, [])
            if row.get("capture_kind") == "unavailable"
        ),
        "atol": atol,
        "rtol": rtol,
        "first_divergent_case": None if first is None else first["case"],
        "first_divergent_mode": None if first is None else first["mode"],
        "first_divergent_layer": None if first is None else first["layer"],
        "first_divergent_head": None if first is None else first["head"],
        "first_divergent_token": None if first is None else first["token"],
        "first_divergent_stage": None if first is None else first["stage"],
        "first_divergent_dependency_index": None
        if first is None
        else first["dependency_index"],
        "first_divergent_status": None if first is None else _row_status(first),
        "first_divergent_max_abs_error": None if first is None else _first_error(first),
        "first_differing_ingredient_overall": None if first is None else first["stage"],
        "primary_remaining_gap": None if first is None else _primary_gap(first),
        "stage_summary": stage_summary,
        "stage_summaries": [stage_summary[stage] for stage in DEPENDENCY_ORDER],
        "kernel_ready": "no",
        "recommended_next_phase": recommendation,
        "recommendation": recommendation,
        "rows": rows,
    }


def run_live_same_run_trace(
    *,
    fixture_manifest: Path,
    out_dir: Path = DEFAULT_OUT,
    parameters: Path | None = None,
    parameter_manifest: Path | None = None,
    fixture_parameter_key: str | None = None,
    radlads_repo: Path | None = None,
    cases: list[str] | None = None,
    mode: str = "both",
    layer: int | None = None,
    head: int | None = None,
    max_tokens: int | None = None,
    strict_live: bool = True,
    overwrite: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    del radlads_repo
    _prepare_out_dir(out_dir, overwrite=overwrite)
    fixture_manifest_data = load_numerical_manifest(fixture_manifest)
    fixture_id = deterministic_fixture_id(fixture_manifest)
    parameter_id = deterministic_parameter_id(
        parameters=parameters,
        parameter_manifest=parameter_manifest,
        fixture_parameter_key=fixture_parameter_key,
    )
    same_run_group_id = new_same_run_group_id(
        fixture_id=fixture_id,
        parameter_id=parameter_id,
        cases=cases,
        mode=mode,
        layer=layer,
        head=head,
        max_tokens=max_tokens,
        strict_live=strict_live,
    )
    contexts = _contexts_from_manifest(
        fixture_manifest,
        cases=cases,
        mode=mode,
        layer=layer,
        head=head,
        max_tokens=max_tokens,
    )
    live_sources, hook_status, config_snapshots = _capture_live_sources(
        fixture_manifest=fixture_manifest,
        fixture_manifest_data=fixture_manifest_data,
        parameters=parameters,
        parameter_manifest=parameter_manifest,
        same_run_group_id=same_run_group_id,
        fixture_id=fixture_id,
        parameter_id=parameter_id,
        cases=cases,
        mode=mode,
        max_tokens=max_tokens,
    )
    traces = {
        side: build_live_same_run_trace(
            live_sources.get(side, []),
            side=side,
            same_run_group_id=same_run_group_id,
            fixture_id=fixture_id,
            parameter_id=parameter_id,
            contexts=contexts,
        )
        for side in SIDES
    }
    for side, entries in traces.items():
        write_live_same_run_trace(entries, out_dir / f"live_trace_{side}.jsonl")
    combined = [row for side in SIDES for row in traces[side]]
    write_live_same_run_trace(combined, out_dir / "live_trace_combined.jsonl")
    trace_metadata = {
        "schema": LIVE_SAME_RUN_TRACE_SCHEMA,
        "phase": "P68",
        "same_run_group_id": same_run_group_id,
        "fixture_id": fixture_id,
        "parameter_id": parameter_id,
        "fixture_manifest_path": str(fixture_manifest),
        "parameter_manifest_or_npz_path": str(parameter_manifest or parameters)
        if parameter_manifest or parameters
        else None,
        "fixture_parameter_key": fixture_parameter_key,
        "radlads_repo_path": None,
        "qrwkv_root_path": str(Path.cwd()),
        "strict_live": strict_live,
        "cases": cases,
        "mode": mode,
        "layer": layer,
        "head": head,
        "max_tokens": max_tokens,
        "trace_generated_at": datetime.now(UTC).isoformat(),
        "synthetic_fallback_used": False,
        "live_hook_status": hook_status,
        "qrwkv_off_config": config_snapshots.get("qrwkv_off"),
        "qrwkv_experimental_config": config_snapshots.get("qrwkv_experimental"),
        "config_delta": _config_delta(
            config_snapshots.get("qrwkv_off"),
            config_snapshots.get("qrwkv_experimental"),
        ),
    }
    (out_dir / "live_same_run_trace_metadata.json").write_text(
        json.dumps(_jsonable(trace_metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = compare_live_same_run_traces(
        traces=traces,
        metadata=trace_metadata,
        strict_live=strict_live,
        atol=atol,
        rtol=rtol,
    )
    write_live_same_run_reports(report, out_dir)
    return report


def write_live_same_run_trace(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=_entry_sort_key):
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def write_live_same_run_reports(report: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_same_run_update_ingredients_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P68_RESULTS.md").write_text(_results_markdown(report), encoding="utf-8")
    (out_dir / "LIVE_SAME_RUN_VALIDITY.md").write_text(
        _validity_markdown(report), encoding="utf-8"
    )
    (out_dir / "STAGE_AVAILABILITY_MATRIX.md").write_text(
        _availability_markdown(report), encoding="utf-8"
    )
    (out_dir / "FIRST_DIFFERING_INGREDIENT.md").write_text(
        _first_markdown(report), encoding="utf-8"
    )
    (out_dir / "P68_DECISION.md").write_text(
        _decision_markdown(report), encoding="utf-8"
    )
    if not report.get("same_run_valid"):
        (out_dir / "P68_FIX_NOTE.md").write_text(
            "# P68 Fix Note\n\n"
            "P68 is invalid for a math conclusion until strict-live RADLADS "
            "update-ingredient rows are captured in the same invocation.\n",
            encoding="utf-8",
        )


def _capture_live_sources(
    *,
    fixture_manifest: Path,
    fixture_manifest_data: Mapping[str, Any],
    parameters: Path | None,
    parameter_manifest: Path | None,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
    cases: list[str] | None,
    mode: str,
    max_tokens: int | None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    sources = {side: [] for side in SIDES}
    status = {
        "radlads": {
            "status": "missing",
            "reason": "missing_live_hook:radlads:pre_attention_norm",
        },
        "qrwkv_off": {"status": "missing", "reason": None},
        "qrwkv_experimental": {"status": "missing", "reason": None},
    }
    config_snapshots: dict[str, dict[str, Any]] = {}
    parameter_path = parameters or parameter_manifest
    if parameter_path is None or not parameter_path.exists():
        reason = "parameter payload unavailable for QRWKV live capture"
        status["qrwkv_off"]["reason"] = reason
        status["qrwkv_experimental"]["reason"] = reason
        return sources, status, config_snapshots
    try:
        import_result = import_radlads_parameters_for_replay(
            parameter_path,
            manifest_path=fixture_manifest,
            allow_defaults=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        reason = f"QRWKV live capture import failed: {type(exc).__name__}: {exc}"
        status["qrwkv_off"]["reason"] = reason
        status["qrwkv_experimental"]["reason"] = reason
        return sources, status, config_snapshots
    selected_cases = _selected_case_dicts(fixture_manifest_data, cases=cases)
    for case in selected_cases:
        profile = replay_profile_for_case(case)
        base_student = student_for_replay_profile(import_result.qrwkv_config, profile)
        off_config = replace(
            base_student.config,
            radlads_balance_state=False,
        )
        exp_config = replace(
            off_config,
            radlads_balance_state_terms=True,
            radlads_balance_state=True,
        )
        for side, config in (
            ("qrwkv_off", off_config),
            ("qrwkv_experimental", exp_config),
        ):
            config_snapshots.setdefault(side, _config_snapshot(config))
            collector = LiveTraceCollector(
                same_run_group_id=same_run_group_id,
                fixture_id=fixture_id,
                parameter_id=parameter_id,
                case=str(case["name"]),
                side=side,
                mode=None if mode in {"both", "full", "stepwise"} else mode,
                live_config=_config_snapshot(config),
            )
            try:
                _capture_qrwkv_case(
                    fixture_manifest=fixture_manifest,
                    case=case,
                    params=import_result.params,
                    config=config,
                    collector=collector,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                status[side]["reason"] = (
                    f"QRWKV live capture failed: {type(exc).__name__}: {exc}"
                )
                continue
            sources[side].extend(collector.entries)
    for side in ("qrwkv_off", "qrwkv_experimental"):
        if sources[side]:
            status[side] = {"status": "captured", "reason": None}
        elif status[side]["reason"] is None:
            status[side]["reason"] = f"missing_live_hook:{side}:pre_attention_norm"
    return sources, status, config_snapshots


def _capture_qrwkv_case(
    *,
    fixture_manifest: Path,
    case: Mapping[str, Any],
    params: Mapping[str, Any],
    config: Any,
    collector: LiveTraceCollector,
    max_tokens: int | None,
) -> None:
    from qrwkv_xla.students import RWKV7QwenReferenceStudent

    arrays = load_numerical_case_arrays(fixture_manifest, dict(case))
    input_ids = np.asarray(arrays["input_ids"], dtype=np.int32)
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    attention_mask = None
    if "attention_mask" in arrays:
        attention_mask = np.asarray(arrays["attention_mask"], dtype=np.int32)
        if max_tokens is not None:
            attention_mask = attention_mask[:, :max_tokens]
    student = RWKV7QwenReferenceStudent(config)
    student.apply_with_state(
        dict(params),
        input_ids,
        attention_mask=attention_mask,
        diagnostics=collector,
    )


def _selected_case_dicts(
    manifest: Mapping[str, Any], *, cases: list[str] | None
) -> list[dict[str, Any]]:
    raw_cases = [
        dict(item) for item in manifest.get("cases", []) if isinstance(item, Mapping)
    ]
    if cases:
        selected = set(cases)
        return [case for case in raw_cases if case.get("name") in selected]
    return raw_cases


def _side_matches(entry: Mapping[str, Any], side: str) -> bool:
    source_side = entry.get("side")
    if source_side == side:
        return True
    if side == "qrwkv_off" and source_side == "qrwkv":
        return entry.get("mode") in {None, "off"}
    if side == "qrwkv_experimental" and source_side == "qrwkv":
        return entry.get("mode") == "experimental"
    return False


def _normalize_stage(stage: str) -> str:
    if stage in STAGE_NORMALIZATION:
        return STAGE_NORMALIZATION[stage]
    if stage in DEPENDENCY_ORDER:
        return stage
    for normalized, aliases in SOURCE_STAGE_MAP.items():
        if stage in aliases:
            return normalized
    return stage


def _canonical_source_stage(stage: str) -> str:
    for source, normalized in MINIMUM_STAGE_NORMALIZATION.items():
        aliases = SOURCE_STAGE_MAP.get(normalized, ())
        if stage == source or stage in aliases:
            return source
    return stage


def _config_snapshot(config: Any) -> dict[str, Any]:
    try:
        return asdict(config)
    except TypeError:
        payload = getattr(config, "__dict__", {})
        return {str(key): _jsonable(value) for key, value in payload.items()}


def _config_delta(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, Any]:
    if left is None or right is None:
        return {"status": "unavailable", "differences": {}}
    keys = sorted(set(left) | set(right))
    differences = {
        key: {"qrwkv_off": left.get(key), "qrwkv_experimental": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }
    unrelated = sorted(set(differences) - BALANCE_STATE_CONFIG_KEYS)
    return {
        "status": "pass" if not unrelated else "fail",
        "differences": differences,
        "unrelated_differences": unrelated,
    }


def _rows_by_source_stage(
    rows: list[dict[str, Any]],
    context: tuple[str, str | None, int | None, int | None, int | None],
) -> dict[str, list[dict[str, Any]]]:
    case, mode, layer, token, head = context
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (
            str(row.get("case")) == case
            and _mode_matches(row.get("mode"), mode)
            and _maybe_int(row.get("layer")) == layer
            and _maybe_int(row.get("token_index", row.get("token"))) == token
            and _maybe_int(row.get("head")) == head
            and row.get("array") is not None
        ):
            label = str(
                row.get(
                    "source_stage_name",
                    row.get("comparison_label", row.get("stage")),
                )
            )
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


def _available_row(
    source: Mapping[str, Any],
    *,
    side: str,
    stage: str,
    dependency_index: int,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
) -> dict[str, Any]:
    array = np.asarray(source["array"])
    summary = summarize_array(
        stage,
        array,
        stage=stage,
        layer=_maybe_int(source.get("layer")),
        time_index=_maybe_int(source.get("token_index", source.get("token"))),
    )
    row_summary = {
        "finite": bool(np.isfinite(array).all()) if array.size else True,
        "max_abs": summary.abs_max,
        "mean_abs": float(np.mean(np.abs(array))) if array.size else 0.0,
        "sample": None if array.size == 0 else float(array.reshape(-1)[0]),
    }
    return {
        "schema": LIVE_SAME_RUN_TRACE_SCHEMA,
        "phase": "P68",
        "same_run_group_id": str(source.get("same_run_group_id", same_run_group_id)),
        "fixture_id": str(source.get("fixture_id", fixture_id)),
        "parameter_id": str(source.get("parameter_id", parameter_id)),
        "side": side,
        "case": str(source["case"]),
        "mode": source.get("mode"),
        "layer": _maybe_int(source.get("layer")),
        "token": _maybe_int(source.get("token_index", source.get("token"))),
        "head": _maybe_int(source.get("head")),
        "stage": stage,
        "dependency_index": dependency_index,
        "source_stage_name": _canonical_source_stage(
            str(
                source.get(
                    "source_stage_name",
                    source.get("comparison_label", source.get("stage")),
                )
            )
        ),
        "capture_kind": str(source.get("capture_kind", "live_captured")),
        "status": "pass",
        "reason": None,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "array": array.tolist(),
        "summary": row_summary,
        "live_config": source.get("live_config", source.get("config")),
    }


def _unavailable_row(
    *,
    side: str,
    context: tuple[str, str | None, int | None, int | None, int | None],
    stage: str,
    dependency_index: int,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
    reason: str,
) -> dict[str, Any]:
    case, mode, layer, token, head = context
    return {
        "schema": LIVE_SAME_RUN_TRACE_SCHEMA,
        "phase": "P68",
        "same_run_group_id": same_run_group_id,
        "fixture_id": fixture_id,
        "parameter_id": parameter_id,
        "side": side,
        "case": case,
        "mode": mode,
        "layer": layer,
        "token": token,
        "head": head,
        "stage": stage,
        "dependency_index": dependency_index,
        "source_stage_name": None,
        "capture_kind": "unavailable",
        "status": "unavailable",
        "reason": reason,
        "shape": [],
        "dtype": None,
        "array": None,
        "summary": {"finite": None, "max_abs": None, "mean_abs": None, "sample": None},
        "live_config": None,
    }


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
        "mode": key[1],
        "layer": key[2],
        "token": key[3],
        "head": key[4],
        "stage": key[5],
        "dependency_index": DEPENDENCY_ORDER.index(key[5])
        if key[5] in DEPENDENCY_ORDER
        else 999,
        "radlads_capture_kind": None if rad is None else rad.get("capture_kind"),
        "qrwkv_off_capture_kind": None if off is None else off.get("capture_kind"),
        "qrwkv_experimental_capture_kind": None
        if exp is None
        else exp.get("capture_kind"),
        "radlads_vs_qrwkv_off": _compare_pair(rad, off, atol=atol, rtol=rtol),
        "radlads_vs_qrwkv_experimental": _compare_pair(rad, exp, atol=atol, rtol=rtol),
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
        or left.get("capture_kind") != "live_captured"
        or right.get("capture_kind") != "live_captured"
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


def _validate_identity(
    *,
    traces: Mapping[str, list[dict[str, Any]]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    fields = ("same_run_group_id", "fixture_id", "parameter_id")
    values = {
        field: sorted(
            {
                str(row.get(field))
                for side in SIDES
                for row in traces.get(side, [])
                if row.get(field) is not None
            }
        )
        for field in fields
    }
    expected = {field: metadata.get(field) for field in fields}
    failures = [
        field
        for field, present in values.items()
        if len(present) != 1
        or (expected[field] is not None and present[0] != expected[field])
    ]
    return {
        "status": "pass" if not failures else "fail",
        "reason": None if not failures else "mixed or mismatched strict-live ids",
        "values": values,
        "expected": expected,
        "failed_fields": failures,
    }


def _validate_live_config(
    traces: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    configs: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for side in SIDES:
        for row in traces.get(side, []):
            config = row.get("live_config")
            if config is None or row.get("capture_kind") == "unavailable":
                continue
            configs.setdefault(_trace_key(row), {})[side] = dict(config)
    mismatches = {}
    for key, value in configs.items():
        off = value.get("qrwkv_off")
        exp = value.get("qrwkv_experimental")
        if off is not None and exp is not None:
            delta = _config_delta(off, exp)
            if delta["status"] != "pass":
                mismatches[key] = delta
    return {
        "status": "pass" if not mismatches else "fail",
        "reason": None if not mismatches else "unrelated strict-live config delta",
        "mismatches": {str(key): value for key, value in mismatches.items()},
    }


def _validate_critical_availability(
    traces: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    missing = [
        {
            "side": side,
            "case": row.get("case"),
            "mode": row.get("mode"),
            "layer": row.get("layer"),
            "token": row.get("token"),
            "head": row.get("head"),
            "stage": row.get("stage"),
            "reason": row.get("reason"),
        }
        for side in SIDES
        for row in traces.get(side, [])
        if row.get("stage") in CRITICAL_STAGES
        and row.get("capture_kind") != "live_captured"
    ]
    return {
        "status": "pass" if not missing else "fail",
        "reason": None if not missing else "missing unavailable critical live stage",
        "missing": missing,
    }


def _live_row_counts(traces: Mapping[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {
        side: sum(
            1
            for row in traces.get(side, [])
            if row.get("capture_kind") == "live_captured"
        )
        for side in SIDES
    }


def _minimum_stage_availability(
    traces: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, bool]]:
    return {
        stage: {
            side: any(
                row.get("stage") == stage and row.get("capture_kind") == "live_captured"
                for row in traces.get(side, [])
            )
            for side in SIDES
        }
        for stage in MINIMUM_STAGES
    }


def _unavailable_minimum_stages(
    traces: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "side": side,
            "stage": row.get("stage"),
            "case": row.get("case"),
            "mode": row.get("mode"),
            "layer": row.get("layer"),
            "token": row.get("token"),
            "head": row.get("head"),
            "reason": row.get("reason"),
        }
        for side in SIDES
        for row in traces.get(side, [])
        if row.get("stage") in MINIMUM_STAGES
        and row.get("capture_kind") != "live_captured"
    ]


def _validate_decay_log_w_precondition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [
        row
        for row in rows
        if row.get("stage") in {"decay_log_w", "decay_value"}
        and _row_status(row) != "pass"
    ]
    return {
        "status": "pass" if not failures else "fail",
        "reason": None if not failures else "decay/log_w precondition failed",
        "failed_rows": [
            {
                "case": row["case"],
                "mode": row["mode"],
                "layer": row["layer"],
                "token": row["token"],
                "head": row["head"],
                "stage": row["stage"],
                "status": _row_status(row),
            }
            for row in failures
        ],
    }


def _recommendation(
    *,
    same_run_valid: bool,
    identity: Mapping[str, Any],
    config: Mapping[str, Any],
    unavailable: Mapping[str, Any],
    decay: Mapping[str, Any],
    first: Mapping[str, Any] | None,
) -> str:
    if same_run_valid:
        return "P70 residual-impact / kernel-readiness gate"
    if identity["status"] != "pass":
        return P70_QRWKV_HOOK_COMPLETION
    if config["status"] != "pass":
        return "P70 harden/promote balance-state compatibility path"
    if unavailable["status"] != "pass":
        missing = unavailable.get("missing", [])
        sides = {row.get("side") for row in missing}
        stages = {row.get("stage") for row in missing}
        if "radlads" in sides:
            return P70_RADLADS_HOOK_COMPLETION
        if not sides.isdisjoint({"qrwkv_off", "qrwkv_experimental"}):
            if stages & {"decay_log_w", "decay_value"}:
                return P70_DECAY_REPAIR
            if stages & {"prev_state", "vk", "state_after_live"}:
                return P70_WKV_REPAIR
            return P70_QRWKV_HOOK_COMPLETION
        return P70_RADLADS_HOOK_COMPLETION
    if decay["status"] != "pass":
        return P70_DECAY_REPAIR
    if first is not None:
        stage = first.get("stage")
        if stage in {"raw_k", "raw_v"}:
            return "P70 targeted raw_k/raw_v projection fix"
        if stage == "vk":
            return "P70 targeted vk/outer-product orientation fix"
        if stage == "state_after_live":
            return "P70 targeted state_after assembly/dtype fix"
    return P70_RADLADS_HOOK_COMPLETION


def _contexts_from_manifest(
    fixture_manifest: Path,
    *,
    cases: list[str] | None,
    mode: str,
    layer: int | None,
    head: int | None,
    max_tokens: int | None,
) -> list[tuple[str, str | None, int | None, int | None, int | None]]:
    manifest = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    case_names = _manifest_case_names(manifest)
    if cases:
        selected = set(cases)
        case_names = [case for case in case_names if case in selected]
    if not case_names:
        case_names = cases or ["tiny_no_mask"]
    modes = [None] if mode in {"both", "full", "stepwise"} else [mode]
    token_count = (
        max_tokens if max_tokens is not None else _manifest_token_count(manifest)
    )
    token_count = max(1, token_count)
    return [
        (
            case,
            item_mode,
            layer if layer is not None else 0,
            token,
            head if head is not None else 0,
        )
        for case in case_names
        for item_mode in modes
        for token in range(token_count)
    ]


def _manifest_case_names(manifest: Mapping[str, Any]) -> list[str]:
    raw = manifest.get("cases", [])
    names = []
    if isinstance(raw, Mapping):
        raw = raw.values()
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping):
            value = item.get("case") or item.get("name") or item.get("id")
            if value is not None:
                names.append(str(value))
    return names


def _manifest_token_count(manifest: Mapping[str, Any]) -> int:
    for key in ("max_tokens", "tokens", "seq_len", "sequence_length"):
        value = manifest.get(key)
        if isinstance(value, int):
            return value
    return 1


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


def _primary_gap(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case": row["case"],
        "mode": row["mode"],
        "layer": row["layer"],
        "token": row["token"],
        "head": row["head"],
        "stage": row["stage"],
        "dependency_index": row["dependency_index"],
        "status": _row_status(row),
        "max_abs_error": _first_error(row),
    }


def _stage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for stage in DEPENDENCY_ORDER:
        stage_rows = [row for row in rows if row["stage"] == stage]
        errors = [_first_error(row) for row in stage_rows]
        present = [error for error in errors if error is not None]
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
            "max_abs_error": max(present) if present else None,
            "unavailable": any(
                row["radlads_vs_qrwkv_off"]["status"] == "unavailable"
                or row["radlads_vs_qrwkv_experimental"]["status"] == "unavailable"
                for row in stage_rows
            ),
        }
    return summary


def _trace_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("case"),
        entry.get("mode"),
        entry.get("layer"),
        entry.get("token"),
        entry.get("head"),
        entry.get("stage"),
    )


def _sorted_keys(keys: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(keys, key=_key_sort_key)


def _key_sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return (
        str(key[0]),
        "" if key[1] is None else str(key[1]),
        -1 if key[2] is None else int(key[2]),
        -1 if key[3] is None else int(key[3]),
        -1 if key[4] is None else int(key[4]),
        DEPENDENCY_ORDER.index(key[5]) if key[5] in DEPENDENCY_ORDER else 999,
    )


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return _key_sort_key(_trace_key(entry))


def _context_sort_key(
    context: tuple[str, str | None, int | None, int | None, int | None],
) -> tuple[Any, ...]:
    case, mode, layer, token, head = context
    return (
        case,
        "" if mode is None else mode,
        -1 if layer is None else layer,
        -1 if token is None else token,
        -1 if head is None else head,
    )


def _mode_matches(row_mode: Any, context_mode: str | None) -> bool:
    return context_mode is None or row_mode in {None, context_mode}


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    if path.is_file():
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


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


def _results_markdown(report: Mapping[str, Any]) -> str:
    availability = json.dumps(
        report.get("minimum_stage_availability", {}),
        sort_keys=True,
    )
    return "\n".join(
        [
            "# P68 Results",
            "",
            f"- Status: `{report['overall_status']}`",
            f"- Rows compared: `{report['row_count']}`",
            f"- Same-run valid: `{report['same_run_valid']}`",
            f"- First differing ingredient: `{report['first_divergent_stage']}`",
            f"- Unavailable rows: `{report['unavailable_rows']}`",
            f"- Kernel ready: `{report['kernel_ready']}`",
            f"- Recommendation: {report['recommended_next_phase']}",
            "",
            "## P69 hook wiring",
            "",
            "- live_rows_captured_radlads: `"
            f"{report.get('live_rows_captured_radlads', 0)}`",
            "- live_rows_captured_qrwkv_off: `"
            f"{report.get('live_rows_captured_qrwkv_off', 0)}`",
            "- live_rows_captured_qrwkv_experimental: `"
            f"{report.get('live_rows_captured_qrwkv_experimental', 0)}`",
            f"- minimum_stage_availability: `{availability}`",
            "- unavailable_minimum_stages: `"
            f"{len(report.get('unavailable_minimum_stages', []))}`",
            "",
        ]
    )


def _validity_markdown(report: Mapping[str, Any]) -> str:
    validity = report["same_run_validity"]
    return "\n".join(
        [
            "# Live Same-Run Validity",
            "",
            f"same_run_valid: `{report.get('same_run_valid')}`",
            f"fixture_id: `{report.get('fixture_id')}`",
            f"parameter_id: `{report.get('parameter_id')}`",
            f"same_run_group_id: `{report.get('same_run_group_id')}`",
            f"identity: `{validity['identity']['status']}`",
            f"live_config: `{validity['live_config']['status']}`",
            f"critical_availability: `{validity['critical_availability']['status']}`",
            "decay/log_w precondition: `"
            f"{validity['decay_log_w_precondition']['status']}`",
            "synthetic fallback used: `False`",
            "if fail: update conclusion valid: `no`",
            "",
        ]
    )


def _availability_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage Availability Matrix",
        "",
        "| stage | RADLADS | QRWKV off | QRWKV experimental | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage in DEPENDENCY_ORDER:
        stage_rows = [row for row in report.get("rows", []) if row["stage"] == stage]
        kinds = {
            "radlads": None,
            "qrwkv_off": None,
            "qrwkv_experimental": None,
        }
        for row in stage_rows:
            kinds["radlads"] = row.get("radlads_capture_kind")
            kinds["qrwkv_off"] = row.get("qrwkv_off_capture_kind")
            kinds["qrwkv_experimental"] = row.get("qrwkv_experimental_capture_kind")
            break
        lines.append(
            f"| `{stage}` | `{kinds['radlads'] or 'unavailable'}` | "
            f"`{kinds['qrwkv_off'] or 'unavailable'}` | "
            f"`{kinds['qrwkv_experimental'] or 'unavailable'}` | "
            f"`{report['stage_summary'][stage]['status']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _first_markdown(report: Mapping[str, Any]) -> str:
    first_missing = None
    unavailable = report.get("unavailable_minimum_stages", [])
    if unavailable:
        first_missing = unavailable[0]
    math_valid = (
        report.get("same_run_valid")
        and report.get("decay_precondition_pass")
        and first_missing is None
    )
    return "\n".join(
        [
            "# First Differing Ingredient",
            "",
            f"math_conclusion_valid: `{bool(math_valid)}`",
            f"same_run_valid: `{report.get('same_run_valid')}`",
            f"decay_precondition_pass: `{report.get('decay_precondition_pass')}`",
            f"first_missing_live_hook: `{first_missing}`",
            "first differing ingredient: `"
            f"{report.get('first_differing_ingredient_overall')}`",
            f"case: `{report.get('first_divergent_case')}`",
            f"mode: `{report.get('first_divergent_mode')}`",
            f"layer: `{report.get('first_divergent_layer')}`",
            f"token: `{report.get('first_divergent_token')}`",
            f"head: `{report.get('first_divergent_head')}`",
            f"max_abs_error: `{_fmt(report.get('first_divergent_max_abs_error'))}`",
            f"recommended next phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _decision_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P68 Decision",
            "",
            f"- same_run_valid: `{report['same_run_valid']}`",
            f"- kernel_ready: `{report['kernel_ready']}`",
            f"- recommended_next_phase: {report['recommended_next_phase']}",
            "- math_fix_recommended: `False`",
            "- pallas_gate_recommended: `False`",
            "- residual_impact_gate_recommended: `False`",
            "",
        ]
    )
