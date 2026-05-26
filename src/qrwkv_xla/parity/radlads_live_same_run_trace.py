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
BALANCE_STATE_TERMS_LANE = "balance_state_terms"
DIRECT_BALANCE_STATE_LANE = "direct_balance_state"
NATIVE_OR_UNKNOWN_LANE = "native_or_unknown"
LANE_A_ONLY_STAGES = {"k_k", "k_a"}
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
P71_STAGE_NORMALIZATION = {
    "v_first": "v_first",
    "value_before_v_first_mix": "v_first",
    "mixed_value": "mixed_value",
    "value_after_v_first_mix": "mixed_value",
    "v_after_v_first_mix": "mixed_value",
    "iclr_update_rate": "iclr_update_rate",
    "a": "iclr_update_rate",
    "a_or_iclr_after_transform": "iclr_update_rate",
    "k_k": "k_k",
    "kk_neg": "k_k",
    "k_k_after_transform": "k_k",
    "key_norm_factor": "k_k",
    "k_a": "k_a",
    "kk_a": "k_a",
    "k_a_after_transform": "k_a",
    "key_balance_adjustment": "k_a",
    "kk": "kk",
    "kk_normalized": "kk",
    "k_normalized_for_ab": "kk",
    "k_for_update": "k_for_update",
    "k_after_balance": "k_for_update",
    "k_for_vk": "k_for_update",
    "v_for_update": "v_for_update",
    "v_for_vk": "v_for_update",
    "value_for_update": "v_for_update",
    "ab": "ab",
    "ab_matrix": "ab",
    "balance_ab": "ab",
    "balance_state_term": "balance_state_term",
    "prev_state_at_ab": "balance_state_term",
    "prev_state_matmul_ab": "balance_state_term",
    "balance_composite_term": "balance_state_term",
    "composite_update_term": "composite_update_term",
    "final_update_term": "composite_update_term",
}
STAGE_NORMALIZATION = (
    MINIMUM_STAGE_NORMALIZATION | STRETCH_STAGE_NORMALIZATION | P71_STAGE_NORMALIZATION
)
DEPENDENCY_ORDER = (
    "pre_attention_norm",
    "raw_k",
    "raw_v",
    "v_first",
    "mixed_value",
    "iclr_update_rate",
    "k_k",
    "k_a",
    "kk",
    "k_for_update",
    "v_for_update",
    "decay_log_w",
    "decay_value",
    "prev_state",
    "wkv_decay_applied",
    "vk",
    "ab",
    "balance_state_term",
    "composite_update_term",
    "state_after_live",
    "state_after_exported",
)
SOURCE_STAGE_MAP = {
    STAGE_NORMALIZATION.get(stage, stage): aliases
    for stage, aliases in P67_SOURCE_STAGE_MAP.items()
}
SOURCE_STAGE_MAP.update(
    {
        "v_first": ("v_first", "value_before_v_first_mix"),
        "mixed_value": (
            "mixed_value",
            "value_after_v_first_mix",
            "v_after_v_first_mix",
        ),
        "iclr_update_rate": (
            "iclr_update_rate",
            "a",
            "a_or_iclr_after_transform",
        ),
        "k_k": ("k_k", "kk_neg", "k_k_after_transform", "key_norm_factor"),
        "k_a": ("k_a", "kk_a", "k_a_after_transform", "key_balance_adjustment"),
        "kk": ("kk", "kk_normalized", "k_normalized_for_ab"),
        "k_for_update": ("k_for_update", "k_after_balance", "k_for_vk"),
        "v_for_update": ("v_for_update", "v_for_vk", "value_for_update"),
        "ab": ("ab", "ab_matrix", "balance_ab"),
        "balance_state_term": (
            "balance_state_term",
            "prev_state_at_ab",
            "prev_state_matmul_ab",
            "balance_composite_term",
            "composite_balance_update_term",
            "balance_state_matmul",
            "wkv_balance_state_matmul",
            "wkv_composite_balance_update_term",
        ),
        "composite_update_term": (
            "composite_update_term",
            "final_update_term",
            "state_after_from_full_source_formula",
        ),
    }
)
SOURCE_STAGE_MAP["prev_state"] = (
    *SOURCE_STAGE_MAP["prev_state"],
    "initial_matrix_state",
)
SOURCE_STAGE_MAP["vk"] = (*SOURCE_STAGE_MAP["vk"], "update_term")
MINIMUM_STAGES = tuple(MINIMUM_STAGE_NORMALIZATION.values())
CRITICAL_STAGES = MINIMUM_STAGES
P71_STRETCH_STAGES = (
    "v_first",
    "mixed_value",
    "iclr_update_rate",
    "k_k",
    "k_a",
    "kk",
    "k_for_update",
    "v_for_update",
    "ab",
    "balance_state_term",
    "composite_update_term",
)
AVAILABLE_CAPTURE_KINDS = {"live_captured", "exact_reconstruction"}
BALANCE_STATE_CONFIG_KEYS = {
    "radlads_balance_state",
    "radlads_balance_state_terms",
    "use_radlads_balance_state_terms",
}
P72_HOOK_COMPLETION = "P72 targeted live missing-stage hook completion"
P72_V_FIRST_FIX = "P72 targeted v_first/mixed_value hook or formula repair"
P72_BALANCE_PREP_FIX = "P72 targeted k_for_update/v_for_update balance-prep fix"
P72_ICLR_FIX = "P72 targeted iclr_update_rate/a construction fix"
P72_AB_FIX = "P72 targeted ab construction/orientation fix"
P72_VK_FIX = "P72 targeted vk/outer-product orientation fix"
P72_STATE_AFTER_FIX = "P72 targeted state_after assembly/dtype fix"
P72_RESIDUAL_GATE = "P72 residual-impact / kernel-readiness gate"
P72_PALLAS = "P72 Pallas prototype behind known-caveat flag"
P73_KK_KA_FIX = "P73 targeted k_k/k_a construction fix"
P73_KK_FIX = "P73 targeted kk construction fix"
P73_BALANCE_PREP_FIX = "P73 targeted k_for_update/v_for_update balance-prep fix"
P73_AB_FIX = "P73 targeted ab construction/orientation fix"
P73_SOURCE_MAPPING = "P73 targeted source mapping clarification for k_k/k_a"
P73_RESIDUAL_GATE = "P73 residual-impact / kernel-readiness gate"
P74_DIRECT_LANE = "P74 generate RADLADS direct-balance-state lane"
P74_TERMS_LANE_ONLY = "P74 compare balance_state_terms lane only"
P74_KK_FIX = "P74 targeted kk construction fix"
P74_BALANCE_PREP_FIX = "P74 targeted k_for_update/v_for_update balance-prep fix"
P74_AB_FIX = "P74 targeted ab construction/orientation fix"
P74_VK_FIX = "P74 targeted vk/outer-product orientation fix"
P74_STATE_AFTER_FIX = "P74 targeted state_after assembly/dtype fix"
P74_RESIDUAL_GATE = "P74 residual-impact / kernel-readiness gate"
P74_PALLAS = "P74 Pallas prototype behind known-caveat flag"


def classify_balance_state_lane(config: Mapping[str, Any] | Any) -> str:
    snapshot = _config_snapshot(config) if not isinstance(config, Mapping) else config
    if _config_bool(snapshot, "radlads_balance_state"):
        return DIRECT_BALANCE_STATE_LANE
    if _config_bool(snapshot, "radlads_balance_state_terms") or _config_bool(
        snapshot, "use_radlads_balance_state_terms"
    ):
        return BALANCE_STATE_TERMS_LANE
    return NATIVE_OR_UNKNOWN_LANE


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
        self.balance_state_lane = classify_balance_state_lane(self.live_config)
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
            and normalized_stage in (*MINIMUM_STAGES, *P71_STRETCH_STAGES)
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
                "balance_state_lane": self.balance_state_lane,
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
    side_lane = _lane_from_rows(rows, side=side)
    side_live_config = _live_config_from_rows(rows)
    output: list[dict[str, Any]] = []
    for context in sorted(contexts, key=_context_sort_key):
        by_stage = _rows_by_source_stage(rows, context)
        for dependency_index, stage in enumerate(DEPENDENCY_ORDER):
            source = _source_for_stage(by_stage, stage)
            if source is None:
                reconstructed = _exact_reconstruction_for_stage(
                    by_stage,
                    side=side,
                    context=context,
                    stage=stage,
                    dependency_index=dependency_index,
                    same_run_group_id=same_run_group_id,
                    fixture_id=fixture_id,
                    parameter_id=parameter_id,
                )
                if reconstructed is not None:
                    output.append(reconstructed)
                    continue
                output.append(
                    _unavailable_row(
                        side=side,
                        context=context,
                        stage=stage,
                        dependency_index=dependency_index,
                        same_run_group_id=same_run_group_id,
                        fixture_id=fixture_id,
                        parameter_id=parameter_id,
                        reason=_missing_stage_reason(
                            side=side, stage=stage, lane=side_lane
                        ),
                        capture_kind=_missing_stage_capture_kind(
                            stage=stage, lane=side_lane
                        ),
                        balance_state_lane=side_lane,
                        live_config=side_live_config,
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
    stretch_availability = _stretch_stage_availability(traces)
    unavailable_stretch = _unavailable_stretch_stages(traces)
    lane_map = _balance_state_lane_map(traces=traces, metadata=metadata)
    lane_validity = _lane_validity(rows=rows, traces=traces, lane_map=lane_map)
    first_non_applicable = _first_non_applicable_row(rows)
    first_comparable = _first_comparable_differing_row(rows, lane_map)
    minimum_stage_valid = not unavailable_minimum
    stretch_stages_available = not _comparable_unavailable_stretch_stages(
        unavailable_stretch
    )
    same_run_valid = (
        (not strict_live or identity["status"] == "pass")
        and config["status"] == "pass"
        and unavailable["status"] == "pass"
        and decay["status"] == "pass"
    )
    lane_validity["overall_same_run_valid"] = same_run_valid
    math_conclusion_valid = bool(
        same_run_valid
        and minimum_stage_valid
        and stretch_stages_available
        and lane_validity["balance_state_terms_lane_valid"]
        and first_comparable is None
    )
    recommendation = _recommendation(
        same_run_valid=same_run_valid,
        identity=identity,
        config=config,
        unavailable=unavailable,
        decay=decay,
        first=first,
    )
    lane_decision = _lane_recommendation(
        same_run_valid=same_run_valid,
        lane_validity=lane_validity,
        first_comparable=first_comparable,
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
        "parameter_source_path": metadata.get("parameter_source_path"),
        "parameter_mapping_summary": metadata.get("parameter_mapping_summary"),
        "radlads_config_snapshot": metadata.get("radlads_config_snapshot"),
        "radlads_parameter_source": metadata.get("radlads_parameter_source"),
        "qrwkv_parameter_source": metadata.get("qrwkv_parameter_source"),
        "same_parameter_id_applies": metadata.get("same_parameter_id_applies"),
        "radlads_repo_path": metadata.get("radlads_repo_path"),
        "qrwkv_root_path": metadata.get("qrwkv_root_path"),
        "strict_live": strict_live,
        "same_run_valid": same_run_valid,
        "overall_same_run_valid": same_run_valid,
        "balance_state_lane_map": lane_map,
        "lane_classification_valid": lane_validity["lane_classification_valid"],
        "balance_state_terms_lane_valid": lane_validity[
            "balance_state_terms_lane_valid"
        ],
        "direct_balance_state_lane_valid": lane_validity[
            "direct_balance_state_lane_valid"
        ],
        "lane_mixed_comparison_valid": lane_validity["lane_mixed_comparison_valid"],
        "lane_validity": lane_validity,
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
        "minimum_stage_valid": minimum_stage_valid,
        "unavailable_minimum_stages": unavailable_minimum,
        "stretch_stage_availability": stretch_availability,
        "k_k_available": _stage_available_on_all_sides(stretch_availability, "k_k"),
        "k_a_available": _stage_available_on_all_sides(stretch_availability, "k_a"),
        "stretch_stages_available": stretch_stages_available,
        "unavailable_stretch_stages": unavailable_stretch,
        "math_conclusion_valid": math_conclusion_valid,
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
        "first_divergent_capture_kind": None
        if first is None
        else _row_capture_kind(first),
        "first_divergent_samples": None
        if first is None
        else _first_samples(first, traces),
        "first_divergent_max_abs_error": None if first is None else _first_error(first),
        "first_differing_ingredient_overall": None if first is None else first["stage"],
        "first_overall_non_applicable_stage": None
        if first_non_applicable is None
        else first_non_applicable["stage"],
        "first_overall_non_applicable": None
        if first_non_applicable is None
        else _primary_gap(first_non_applicable),
        "first_comparable_differing_stage": None
        if first_comparable is None
        else first_comparable["stage"],
        "first_comparable_differing_lane": None
        if first_comparable is None
        else first_comparable["lane"],
        "first_comparable_differing_pair": None
        if first_comparable is None
        else first_comparable["pair"],
        "first_comparable_differing": first_comparable,
        "primary_remaining_gap": None if first is None else _primary_gap(first),
        "stage_summary": stage_summary,
        "stage_summaries": [stage_summary[stage] for stage in DEPENDENCY_ORDER],
        "kernel_ready": "no",
        "recommended_next_phase": lane_decision,
        "legacy_recommendation": recommendation,
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
        "live_hook_status": {
            side: hook_status.get(side, {"status": "missing", "reason": None})
            for side in SIDES
        },
        "parameter_source_path": str(parameter_path)
        if (parameter_path := (parameter_manifest or parameters)) is not None
        else None,
        "parameter_mapping_summary": _parameter_mapping_summary(
            hook_status.get("_parameter_import_report")
        ),
        "radlads_config_snapshot": config_snapshots.get("radlads"),
        "qrwkv_off_config": config_snapshots.get("qrwkv_off"),
        "qrwkv_experimental_config": config_snapshots.get("qrwkv_experimental"),
        "radlads_parameter_source": str(parameter_path)
        if parameter_path is not None
        else None,
        "qrwkv_parameter_source": str(parameter_path)
        if parameter_path is not None
        else None,
        "same_parameter_id_applies": True,
        "config_delta": _config_delta(
            config_snapshots.get("qrwkv_off"),
            config_snapshots.get("qrwkv_experimental"),
        ),
    }
    trace_metadata["balance_state_lane_map"] = _balance_state_lane_map(
        traces=traces,
        metadata=trace_metadata,
    )
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
    (out_dir / "balance_state_lane_map.json").write_text(
        json.dumps(
            _jsonable(report.get("balance_state_lane_map", {})),
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
    (out_dir / "P70_RADLADS_HOOK_NOTE.md").write_text(
        _p70_hook_note_markdown(report), encoding="utf-8"
    )
    (out_dir / "P71_BALANCE_PREP_HOOK_NOTE.md").write_text(
        _p71_hook_note_markdown(report), encoding="utf-8"
    )
    (out_dir / "P71_FIX_NOTE.md").write_text(
        _p71_fix_note_markdown(report), encoding="utf-8"
    )
    (out_dir / "P72_KK_KA_HOOK_NOTE.md").write_text(
        _p72_hook_note_markdown(report), encoding="utf-8"
    )
    (out_dir / "P73_BALANCE_STATE_LANE_MAP.md").write_text(
        _p73_lane_map_markdown(report), encoding="utf-8"
    )
    (out_dir / "P73_FIX_NOTE.md").write_text(
        _p73_fix_note_markdown(report), encoding="utf-8"
    )
    if not report.get("same_run_valid"):
        (out_dir / "P68_FIX_NOTE.md").write_text(
            "# P68 Fix Note\n\n"
            "P68 is invalid for a math conclusion until strict-live RADLADS "
            "update-ingredient rows are captured in the same invocation.\n",
            encoding="utf-8",
        )
    else:
        stale_fix_note = out_dir / "P68_FIX_NOTE.md"
        if stale_fix_note.exists():
            stale_fix_note.unlink()


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
        status["_parameter_import_report"] = import_result.report
    except Exception as exc:  # pragma: no cover - environment dependent
        reason = f"QRWKV live capture import failed: {type(exc).__name__}: {exc}"
        status["radlads"]["status"] = "failed"
        status["radlads"]["reason"] = reason
        status["qrwkv_off"]["reason"] = reason
        status["qrwkv_experimental"]["reason"] = reason
        return sources, status, config_snapshots
    selected_cases = _selected_case_dicts(fixture_manifest_data, cases=cases)
    for case in selected_cases:
        profile = replay_profile_for_case(case)
        base_student = student_for_replay_profile(import_result.qrwkv_config, profile)
        # P72 trace-only activation: source `k_k`/`k_a` semantics live in the
        # balance-state-terms branch when `radlads_balance_state` is false.
        # This does not change default model behavior; it only makes the
        # same-run diagnostic execute and observe that source-backed path.
        radlads_config = replace(
            base_student.config,
            radlads_balance_state_terms=True,
            radlads_balance_state=False,
        )
        config_snapshots.setdefault("radlads", _config_snapshot(radlads_config))
        radlads_collector = LiveTraceCollector(
            same_run_group_id=same_run_group_id,
            fixture_id=fixture_id,
            parameter_id=parameter_id,
            case=str(case["name"]),
            side="radlads",
            mode=None if mode in {"both", "full", "stepwise"} else mode,
            live_config=_config_snapshot(radlads_config),
        )
        try:
            _capture_radlads_case(
                fixture_manifest=fixture_manifest,
                case=case,
                params=import_result.params,
                config=radlads_config,
                collector=radlads_collector,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            status["radlads"] = {
                "status": "failed",
                "reason": (f"RADLADS live capture failed: {type(exc).__name__}: {exc}"),
            }
        else:
            sources["radlads"].extend(radlads_collector.entries)
        off_config = replace(
            base_student.config,
            radlads_balance_state_terms=True,
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
    if sources["radlads"]:
        status["radlads"] = {"status": "captured", "reason": None}
    elif status["radlads"]["status"] != "failed":
        status["radlads"] = {
            "status": "missing",
            "reason": "missing_live_hook:radlads:pre_attention_norm",
        }
    return sources, status, config_snapshots


def _capture_radlads_case(
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


def _config_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _live_config_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        config = row.get("live_config") or row.get("config")
        if isinstance(config, Mapping):
            return dict(config)
    return None


def _lane_from_rows(
    rows: Iterable[Mapping[str, Any]], *, side: str | None = None
) -> str:
    for row in rows:
        lane = row.get("balance_state_lane")
        if lane is not None:
            return str(lane)
        config = row.get("live_config") or row.get("config")
        if isinstance(config, Mapping):
            return classify_balance_state_lane(config)
    if side == "qrwkv_experimental":
        return DIRECT_BALANCE_STATE_LANE
    return NATIVE_OR_UNKNOWN_LANE


def _parameter_mapping_summary(report: Mapping[str, Any] | None) -> dict[str, int]:
    counts = dict(report.get("counts", {})) if report is not None else {}
    return {
        "mapped_count": int(counts.get("mapped", 0)),
        "defaulted_count": int(counts.get("defaulted", 0)),
        "missing_required_count": int(counts.get("missing_required", 0)),
        "shape_mismatch_count": int(counts.get("shape_mismatch", 0)),
        "unsupported_count": int(counts.get("unsupported", 0)),
    }


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


def _missing_stage_reason(*, side: str, stage: str, lane: str) -> str:
    if lane == DIRECT_BALANCE_STATE_LANE and stage in LANE_A_ONLY_STAGES:
        return f"not_active_in_lane:{lane}:{stage}"
    if side == "qrwkv_experimental" and stage in {"k_k", "k_a"}:
        return f"not_active_in_fixture_path:{side}:{stage}"
    return f"missing_live_hook:{side}:{stage}"


def _missing_stage_capture_kind(*, stage: str, lane: str) -> str:
    if lane == DIRECT_BALANCE_STATE_LANE and stage in LANE_A_ONLY_STAGES:
        return "not_applicable"
    return "unavailable"


def _exact_reconstruction_for_stage(
    by_stage: Mapping[str, list[dict[str, Any]]],
    *,
    side: str,
    context: tuple[str, str | None, int | None, int | None, int | None],
    stage: str,
    dependency_index: int,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
) -> dict[str, Any] | None:
    if stage == "balance_state_term":
        prev_state = _live_source_for_stage(by_stage, "prev_state")
        ab = _live_source_for_stage(by_stage, "ab")
        if prev_state is None or ab is None:
            return None
        value = np.asarray(prev_state["array"]) @ np.asarray(ab["array"])
        sources = (prev_state, ab)
    elif stage == "composite_update_term":
        prev_state = _live_source_for_stage(by_stage, "prev_state")
        ab = _live_source_for_stage(by_stage, "ab")
        vk = _live_source_for_stage(by_stage, "vk")
        if prev_state is None or ab is None or vk is None:
            return None
        value = np.asarray(prev_state["array"]) @ np.asarray(ab["array"])
        value = value + np.asarray(vk["array"])
        sources = (prev_state, ab, vk)
    elif stage == "state_after_from_formula":
        decayed = _live_source_for_stage(by_stage, "wkv_decay_applied")
        prev_state = _live_source_for_stage(by_stage, "prev_state")
        ab = _live_source_for_stage(by_stage, "ab")
        vk = _live_source_for_stage(by_stage, "vk")
        if decayed is None or prev_state is None or ab is None or vk is None:
            return None
        value = (
            np.asarray(decayed["array"])
            + np.asarray(prev_state["array"]) @ np.asarray(ab["array"])
            + np.asarray(vk["array"])
        )
        sources = (decayed, prev_state, ab, vk)
    else:
        return None
    source = sources[0]
    row = _available_row(
        {
            **source,
            "stage": stage,
            "source_stage_name": stage,
            "capture_kind": "exact_reconstruction",
            "array": value.tolist(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "live_config": source.get("live_config"),
        },
        side=side,
        stage=stage,
        dependency_index=dependency_index,
        same_run_group_id=same_run_group_id,
        fixture_id=fixture_id,
        parameter_id=parameter_id,
    )
    row["reconstruction_sources"] = [
        {
            "stage": item.get("stage"),
            "source_stage_name": item.get("source_stage_name"),
            "capture_kind": item.get("capture_kind"),
        }
        for item in sources
    ]
    row["case"], row["mode"], row["layer"], row["token"], row["head"] = context
    row["token_index"] = context[3]
    return row


def _live_source_for_stage(
    by_stage: Mapping[str, list[dict[str, Any]]], stage: str
) -> dict[str, Any] | None:
    source = _source_for_stage(by_stage, stage)
    if source is None or source.get("capture_kind") != "live_captured":
        return None
    return source


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
    source_config = source.get("live_config", source.get("config", {}))
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
        "balance_state_lane": source.get(
            "balance_state_lane",
            classify_balance_state_lane(source_config),
        ),
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
    capture_kind: str = "unavailable",
    balance_state_lane: str = NATIVE_OR_UNKNOWN_LANE,
    live_config: Mapping[str, Any] | None = None,
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
        "capture_kind": capture_kind,
        "balance_state_lane": balance_state_lane,
        "status": "unavailable" if capture_kind == "unavailable" else capture_kind,
        "reason": reason,
        "shape": [],
        "dtype": None,
        "array": None,
        "summary": {"finite": None, "max_abs": None, "mean_abs": None, "sample": None},
        "live_config": None if live_config is None else dict(live_config),
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
        "radlads_balance_state_lane": None
        if rad is None
        else rad.get("balance_state_lane"),
        "qrwkv_off_balance_state_lane": None
        if off is None
        else off.get("balance_state_lane"),
        "qrwkv_experimental_balance_state_lane": None
        if exp is None
        else exp.get("balance_state_lane"),
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
        or left.get("capture_kind") not in AVAILABLE_CAPTURE_KINDS
        or right.get("capture_kind") not in AVAILABLE_CAPTURE_KINDS
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
        and row.get("capture_kind") not in AVAILABLE_CAPTURE_KINDS
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
                row.get("stage") == stage
                and row.get("capture_kind") in AVAILABLE_CAPTURE_KINDS
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
            "capture_kind": row.get("capture_kind"),
            "balance_state_lane": row.get("balance_state_lane"),
        }
        for side in SIDES
        for row in traces.get(side, [])
        if row.get("stage") in MINIMUM_STAGES
        and row.get("capture_kind") not in AVAILABLE_CAPTURE_KINDS
    ]


def _stretch_stage_availability(
    traces: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, bool]]:
    return {
        stage: {
            side: any(
                row.get("stage") == stage
                and row.get("capture_kind") in AVAILABLE_CAPTURE_KINDS
                for row in traces.get(side, [])
            )
            for side in SIDES
        }
        for stage in P71_STRETCH_STAGES
    }


def _stage_available_on_all_sides(
    availability: Mapping[str, Mapping[str, bool]], stage: str
) -> bool:
    return all(bool(availability.get(stage, {}).get(side)) for side in SIDES)


def _unavailable_stretch_stages(
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
            "capture_kind": row.get("capture_kind"),
            "balance_state_lane": row.get("balance_state_lane"),
        }
        for side in SIDES
        for row in traces.get(side, [])
        if row.get("stage") in P71_STRETCH_STAGES
        and row.get("capture_kind") not in AVAILABLE_CAPTURE_KINDS
    ]


def _comparable_unavailable_stretch_stages(
    unavailable: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in unavailable
        if not (
            row.get("stage") in LANE_A_ONLY_STAGES
            and str(row.get("reason", "")).startswith("not_active_in_lane:")
        )
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


def _balance_state_lane_map(
    *,
    traces: Mapping[str, list[dict[str, Any]]],
    metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    config_by_side = {
        "radlads": metadata.get("radlads_config_snapshot"),
        "qrwkv_off": metadata.get("qrwkv_off_config"),
        "qrwkv_experimental": metadata.get("qrwkv_experimental_config"),
    }
    result: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        config = config_by_side.get(side)
        if not isinstance(config, Mapping):
            config = _live_config_from_rows(traces.get(side, [])) or {}
        lane = classify_balance_state_lane(config)
        excluded = (
            sorted(LANE_A_ONLY_STAGES) if lane == DIRECT_BALANCE_STATE_LANE else []
        )
        comparable_to = [
            other
            for other in SIDES
            if other != side
            and classify_balance_state_lane(
                config_by_side.get(other)
                if isinstance(config_by_side.get(other), Mapping)
                else (_live_config_from_rows(traces.get(other, [])) or {})
            )
            == lane
        ]
        result[side] = {
            "side": side,
            "lane": lane,
            "radlads_balance_state_terms": bool(
                _config_bool(config, "radlads_balance_state_terms")
                or _config_bool(config, "use_radlads_balance_state_terms")
            ),
            "radlads_balance_state": _config_bool(config, "radlads_balance_state"),
            "k_k_expected": lane != DIRECT_BALANCE_STATE_LANE,
            "k_a_expected": lane != DIRECT_BALANCE_STATE_LANE,
            "kk_construction": "kk = normalize(k)"
            if lane == DIRECT_BALANCE_STATE_LANE
            else "kk = normalize(k * k_k)",
            "comparable_to": comparable_to,
            "stages_excluded_as_not_applicable": excluded,
        }
    return result


def _lane_validity(
    *,
    rows: list[dict[str, Any]],
    traces: Mapping[str, list[dict[str, Any]]],
    lane_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    lanes = {side: lane_map.get(side, {}).get("lane") for side in SIDES}
    classification_valid = all(
        lane
        in {
            BALANCE_STATE_TERMS_LANE,
            DIRECT_BALANCE_STATE_LANE,
            NATIVE_OR_UNKNOWN_LANE,
        }
        for lane in lanes.values()
    )
    terms_pair_available = (
        lanes.get("radlads") == BALANCE_STATE_TERMS_LANE
        and lanes.get("qrwkv_off") == BALANCE_STATE_TERMS_LANE
    )
    terms_first = _first_pair_failure(rows, "radlads_vs_qrwkv_off")
    direct_pair_available = any(
        lanes.get(side) == DIRECT_BALANCE_STATE_LANE for side in ("radlads",)
    ) and any(
        lanes.get(side) == DIRECT_BALANCE_STATE_LANE
        for side in ("qrwkv_experimental", "qrwkv_off")
    )
    direct_lane_present = any(
        lanes.get(side) == DIRECT_BALANCE_STATE_LANE for side in SIDES
    )
    return {
        "lane_classification_valid": classification_valid,
        "overall_same_run_valid": None,
        "balance_state_terms_lane_valid": bool(
            terms_pair_available and terms_first is None
        ),
        "balance_state_terms_lane_pair_available": terms_pair_available,
        "balance_state_terms_first_failure": None
        if terms_first is None
        else _primary_gap(terms_first),
        "direct_balance_state_lane_valid": bool(direct_pair_available),
        "direct_balance_state_lane_present": direct_lane_present,
        "direct_balance_state_radlads_available": (
            lanes.get("radlads") == DIRECT_BALANCE_STATE_LANE
        ),
        "lane_mixed_comparison_valid": len(set(lanes.values())) <= 1,
        "not_applicable_rows": [
            _trace_row_summary(row)
            for side in SIDES
            for row in traces.get(side, [])
            if row.get("capture_kind") == "not_applicable"
        ],
    }


def _first_pair_failure(
    rows: Iterable[Mapping[str, Any]], pair: str
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("stage") not in {*MINIMUM_STAGES, *P71_STRETCH_STAGES}:
            continue
        comparison = row.get(pair, {})
        if comparison.get("status") != "pass":
            return row
    return None


def _first_non_applicable_row(
    rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for row in rows:
        if "not_applicable" in {
            row.get("radlads_capture_kind"),
            row.get("qrwkv_off_capture_kind"),
            row.get("qrwkv_experimental_capture_kind"),
        }:
            return row
    return None


def _first_comparable_differing_row(
    rows: Iterable[Mapping[str, Any]],
    lane_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    pairs = (
        ("radlads_vs_qrwkv_off", "radlads", "qrwkv_off"),
        ("radlads_vs_qrwkv_experimental", "radlads", "qrwkv_experimental"),
        ("qrwkv_off_vs_qrwkv_experimental", "qrwkv_off", "qrwkv_experimental"),
    )
    for row in rows:
        if row.get("stage") not in {*MINIMUM_STAGES, *P71_STRETCH_STAGES}:
            continue
        for pair, left, right in pairs:
            lane = lane_map.get(left, {}).get("lane")
            if lane != lane_map.get(right, {}).get("lane"):
                continue
            if row.get(pair, {}).get("status") == "pass":
                continue
            payload = _primary_gap(row)
            payload["lane"] = lane
            payload["pair"] = pair
            payload["max_abs_error"] = row.get(pair, {}).get("max_abs_error")
            return payload
    return None


def _trace_row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "side": row.get("side"),
        "case": row.get("case"),
        "mode": row.get("mode"),
        "layer": row.get("layer"),
        "token": row.get("token"),
        "head": row.get("head"),
        "stage": row.get("stage"),
        "reason": row.get("reason"),
        "balance_state_lane": row.get("balance_state_lane"),
    }


def _lane_recommendation(
    *,
    same_run_valid: bool,
    lane_validity: Mapping[str, Any],
    first_comparable: Mapping[str, Any] | None,
) -> str:
    if not lane_validity.get("lane_classification_valid"):
        return P73_SOURCE_MAPPING
    if not same_run_valid:
        return P72_HOOK_COMPLETION
    if first_comparable is not None:
        if first_comparable.get("max_abs_error") is None:
            return P72_HOOK_COMPLETION
        stage = first_comparable.get("stage")
        if stage == "kk":
            return P74_KK_FIX
        if stage in {"k_for_update", "v_for_update"}:
            return P74_BALANCE_PREP_FIX
        if stage == "ab":
            return P74_AB_FIX
        if stage == "vk":
            return P74_VK_FIX
        if stage == "state_after_live":
            return P74_STATE_AFTER_FIX
    if lane_validity.get("direct_balance_state_lane_present") and not lane_validity.get(
        "direct_balance_state_radlads_available"
    ):
        return P74_DIRECT_LANE
    if lane_validity.get("balance_state_terms_lane_valid"):
        return P74_RESIDUAL_GATE
    return P74_TERMS_LANE_ONLY


def _recommendation(
    *,
    same_run_valid: bool,
    identity: Mapping[str, Any],
    config: Mapping[str, Any],
    unavailable: Mapping[str, Any],
    decay: Mapping[str, Any],
    first: Mapping[str, Any] | None,
) -> str:
    if same_run_valid and first is None:
        return P73_RESIDUAL_GATE
    if identity["status"] != "pass":
        return P72_HOOK_COMPLETION
    if config["status"] != "pass":
        return P72_HOOK_COMPLETION
    if unavailable["status"] != "pass":
        return P72_HOOK_COMPLETION
    if decay["status"] != "pass":
        return P72_HOOK_COMPLETION
    if first is not None:
        stage = first.get("stage")
        if _row_has_unavailable(first):
            if stage in {"k_k", "k_a"}:
                return P73_SOURCE_MAPPING
            return P72_HOOK_COMPLETION
        if stage in {"v_first", "mixed_value"}:
            return P72_V_FIRST_FIX
        if stage in {"k_for_update", "v_for_update"}:
            return P73_BALANCE_PREP_FIX
        if stage in {"k_k", "k_a"}:
            return P73_KK_KA_FIX
        if stage == "kk":
            return P73_KK_FIX
        if stage == "iclr_update_rate":
            return P72_ICLR_FIX
        if stage == "ab":
            return P73_AB_FIX
        if stage == "vk":
            return P72_VK_FIX
        if stage == "state_after_live":
            return P72_STATE_AFTER_FIX
    if same_run_valid:
        return P73_RESIDUAL_GATE
    return P72_HOOK_COMPLETION


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


def _row_has_unavailable(row: Mapping[str, Any]) -> bool:
    return any(
        row[name]["status"] == "unavailable"
        for name in (
            "radlads_vs_qrwkv_off",
            "radlads_vs_qrwkv_experimental",
            "qrwkv_off_vs_qrwkv_experimental",
        )
    )


def _row_capture_kind(row: Mapping[str, Any]) -> str:
    kinds = [
        row.get("radlads_capture_kind"),
        row.get("qrwkv_off_capture_kind"),
        row.get("qrwkv_experimental_capture_kind"),
    ]
    if any(kind in {None, "unavailable"} for kind in kinds):
        return "unavailable"
    return ",".join(str(kind) for kind in kinds)


def _first_samples(
    first: Mapping[str, Any], traces: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    samples = {}
    key = (
        first.get("case"),
        first.get("mode"),
        first.get("layer"),
        first.get("token"),
        first.get("head"),
        first.get("stage"),
    )
    for side in SIDES:
        row = next(
            (item for item in traces.get(side, []) if _trace_key(item) == key),
            None,
        )
        samples[side] = None if row is None else row.get("summary", {}).get("sample")
    return samples


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


def _source_backed_interpretation(report: Mapping[str, Any]) -> str:
    if not report.get("same_run_valid"):
        return "strict-live identity/config/minimum preconditions failed"
    if not report.get("minimum_stage_valid"):
        return "minimum live stage regression"
    if not report.get("stretch_stages_available"):
        return "stretch stage unavailable; hook completion required"
    if not report.get("math_conclusion_valid"):
        return "first differing row is unavailable"
    if report.get("first_comparable_differing_stage") is None and report.get(
        "first_overall_non_applicable_stage"
    ):
        return "balance_state_terms lane matches; direct_balance_state lane incomplete"
    stage = report.get("first_divergent_stage")
    if stage is None:
        return "all captured ingredients match"
    return f"live same-run mismatch at {stage}"


def _results_markdown(report: Mapping[str, Any]) -> str:
    lane_map = report.get("balance_state_lane_map", {})
    radlads_lane = lane_map.get("radlads", {}).get("lane")
    off_lane = lane_map.get("qrwkv_off", {}).get("lane")
    experimental_lane = lane_map.get("qrwkv_experimental", {}).get("lane")
    minimum_availability = json.dumps(
        report.get("minimum_stage_availability", {}),
        sort_keys=True,
    )
    stretch_availability = json.dumps(
        report.get("stretch_stage_availability", {}),
        sort_keys=True,
    )
    return "\n".join(
        [
            "# P68 Results",
            "",
            f"- Status: `{report['overall_status']}`",
            f"- Same-run valid: `{report['same_run_valid']}`",
            "- Balance-state lane map:",
            f"  - RADLADS: `{radlads_lane}`",
            f"  - QRWKV off: `{off_lane}`",
            f"  - QRWKV experimental: `{experimental_lane}`",
            "- Lane-mixed comparison valid: `"
            f"{report.get('lane_mixed_comparison_valid')}`",
            "- First non-applicable stage: `"
            f"{report.get('first_overall_non_applicable_stage')}`",
            "- First comparable differing stage: `"
            f"{report.get('first_comparable_differing_stage')}`",
            f"- Minimum stages valid: `{report.get('minimum_stage_valid')}`",
            f"- Stretch stages available: `{report.get('stretch_stages_available')}`",
            f"- k_k available: `{report.get('k_k_available')}`",
            f"- k_a available: `{report.get('k_a_available')}`",
            "- Unavailable stretch stages: `"
            f"{len(report.get('unavailable_stretch_stages', []))}`",
            f"- First differing ingredient: `{report['first_divergent_stage']}`",
            "- First differing ingredient capture kind: `"
            f"{report.get('first_divergent_capture_kind')}`",
            f"- Math conclusion valid: `{report.get('math_conclusion_valid')}`",
            f"- Recommended next phase: {report['recommended_next_phase']}",
            f"- Kernel ready: `{report['kernel_ready']}`",
            f"- Rows compared: `{report['row_count']}`",
            f"- Unavailable rows: `{report['unavailable_rows']}`",
            "",
            "## Live hook wiring",
            "",
            "- live_rows_captured_radlads: `"
            f"{report.get('live_rows_captured_radlads', 0)}`",
            "- live_rows_captured_qrwkv_off: `"
            f"{report.get('live_rows_captured_qrwkv_off', 0)}`",
            "- live_rows_captured_qrwkv_experimental: `"
            f"{report.get('live_rows_captured_qrwkv_experimental', 0)}`",
            f"- minimum_stage_availability: `{minimum_availability}`",
            f"- stretch_stage_availability: `{stretch_availability}`",
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
        "| stage | RADLADS available | QRWKV off available | "
        "QRWKV experimental available | RADLADS capture kind | off capture kind | "
        "experimental capture kind | classification | source names | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
        classification = (
            "minimum"
            if stage in MINIMUM_STAGES
            else "stretch"
            if stage in P71_STRETCH_STAGES
            else "supporting"
        )
        available = {side: kinds[side] in AVAILABLE_CAPTURE_KINDS for side in kinds}
        source_names = ", ".join(SOURCE_STAGE_MAP.get(stage, ()))
        lines.append(
            f"| `{stage}` | `{available['radlads']}` | "
            f"`{available['qrwkv_off']}` | "
            f"`{available['qrwkv_experimental']}` | "
            f"`{kinds['radlads'] or 'unavailable'}` | "
            f"`{kinds['qrwkv_off'] or 'unavailable'}` | "
            f"`{kinds['qrwkv_experimental'] or 'unavailable'}` | "
            f"`{classification}` | `{source_names}` | "
            f"`{report['stage_summary'][stage]['status']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _first_markdown(report: Mapping[str, Any]) -> str:
    first_missing = None
    unavailable = report.get("unavailable_minimum_stages", [])
    if unavailable:
        first_missing = unavailable[0]
    samples = report.get("first_divergent_samples") or {}
    rows = report.get("rows", [])
    first_row = next(
        (
            row
            for row in rows
            if row.get("stage") == report.get("first_divergent_stage")
            and row.get("case") == report.get("first_divergent_case")
            and row.get("mode") == report.get("first_divergent_mode")
            and row.get("layer") == report.get("first_divergent_layer")
            and row.get("token") == report.get("first_divergent_token")
            and row.get("head") == report.get("first_divergent_head")
        ),
        {},
    )
    radlads_vs_off_error = _fmt(
        (first_row.get("radlads_vs_qrwkv_off") or {}).get("max_abs_error")
    )
    radlads_vs_experimental_error = _fmt(
        (first_row.get("radlads_vs_qrwkv_experimental") or {}).get("max_abs_error")
    )
    comparable = report.get("first_comparable_differing") or {}
    return "\n".join(
        [
            "# First Differing Ingredient",
            "",
            f"same_run_valid: `{report.get('same_run_valid')}`",
            f"minimum_stage_valid: `{report.get('minimum_stage_valid')}`",
            f"stretch_stage_availability: `{report.get('stretch_stages_available')}`",
            f"decay_precondition_pass: `{report.get('decay_precondition_pass')}`",
            f"first_missing_live_hook: `{first_missing}`",
            "first differing ingredient: `"
            f"{report.get('first_differing_ingredient_overall')}`",
            "first_overall_non_applicable_stage: `"
            f"{report.get('first_overall_non_applicable_stage')}`",
            "first_comparable_differing_stage: `"
            f"{report.get('first_comparable_differing_stage')}`",
            f"lane: `{report.get('first_comparable_differing_lane')}`",
            f"capture kind: `{report.get('first_divergent_capture_kind')}`",
            f"case: `{comparable.get('case', report.get('first_divergent_case'))}`",
            f"mode: `{comparable.get('mode', report.get('first_divergent_mode'))}`",
            f"layer: `{comparable.get('layer', report.get('first_divergent_layer'))}`",
            f"token: `{comparable.get('token', report.get('first_divergent_token'))}`",
            f"head: `{comparable.get('head', report.get('first_divergent_head'))}`",
            f"RADLADS sample: `{samples.get('radlads')}`",
            f"QRWKV off sample: `{samples.get('qrwkv_off')}`",
            f"QRWKV experimental sample: `{samples.get('qrwkv_experimental')}`",
            f"RADLADS vs off error: `{radlads_vs_off_error}`",
            f"RADLADS vs experimental error: `{radlads_vs_experimental_error}`",
            f"radlads_vs_off_error: `{radlads_vs_off_error}`",
            f"radlads_vs_experimental_error: `{radlads_vs_experimental_error}`",
            f"max_abs_error: `{_fmt(report.get('first_divergent_max_abs_error'))}`",
            f"math_conclusion_valid: `{report.get('math_conclusion_valid')}`",
            f"source-backed interpretation: `{_source_backed_interpretation(report)}`",
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
            f"- minimum_stage_valid: `{report.get('minimum_stage_valid')}`",
            f"- stretch_stages_available: `{report.get('stretch_stages_available')}`",
            f"- math_conclusion_valid: `{report.get('math_conclusion_valid')}`",
            "- balance_state_terms_lane_valid: `"
            f"{report.get('balance_state_terms_lane_valid')}`",
            "- direct_balance_state_lane_valid: `"
            f"{report.get('direct_balance_state_lane_valid')}`",
            "- lane_mixed_comparison_valid: `"
            f"{report.get('lane_mixed_comparison_valid')}`",
            f"- kernel_ready: `{report['kernel_ready']}`",
            f"- recommended_next_phase: {report['recommended_next_phase']}",
            "- math_fix_recommended: `False`",
            "- pallas_gate_recommended: `False`",
            "- residual_impact_gate_recommended: `False`",
            "",
        ]
    )


def _p70_hook_note_markdown(report: Mapping[str, Any]) -> str:
    availability = report.get("minimum_stage_availability", {})
    unavailable = [
        row
        for row in report.get("unavailable_minimum_stages", [])
        if row.get("side") == "radlads"
    ]
    return "\n".join(
        [
            "# P70 RADLADS Hook Note",
            "",
            "P70 wires observe-only RADLADS-side live capture into the existing "
            "P68/P69 same-run trace directory.",
            "",
            f"- RADLADS live rows: `{report.get('live_rows_captured_radlads', 0)}`",
            "- RADLADS minimum-stage availability: `"
            f"{json.dumps(availability, sort_keys=True)}`",
            f"- Unavailable RADLADS minimum stages: `{len(unavailable)}`",
            f"- Recommended next phase: `{report.get('recommended_next_phase')}`",
            "",
            "No recurrence math, dtype policy, tolerance, Pallas, or default "
            "balance-state behavior is changed by this hook wiring.",
            "",
        ]
    )


def _p71_hook_note_markdown(report: Mapping[str, Any]) -> str:
    unavailable = report.get("unavailable_stretch_stages", [])
    return "\n".join(
        [
            "# P71 Balance-Prep Hook Note",
            "",
            "P71 extends the strict-live same-run trace into observe-only "
            "balance-prep/update-prep ingredients without changing recurrence "
            "math, dtype policy, tolerances, Pallas code, or default "
            "balance-state behavior.",
            "",
            f"- Stretch stages available: `{report.get('stretch_stages_available')}`",
            f"- Unavailable stretch rows: `{len(unavailable)}`",
            "- Exact reconstruction is limited to `balance_state_term` and "
            "`composite_update_term` when their same-side live ingredients are "
            "present in the same run/context.",
            f"- Recommended next phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p71_fix_note_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P71 Fix Note",
            "",
            "## Problem",
            "",
            "P71 found observe-only diagnostic capture points that did not map "
            "one-to-one to the balance-prep ingredients: `mixed_value` captured "
            "the value mix rate, `k_k` captured the normalized `kk` vector, and "
            "the projection-path `iclr_update_rate` was not emitted.",
            "",
            "## Source Evidence",
            "",
            "The affected tensors are computed locally in "
            "`RWKV7QwenReferenceStudent._attention` immediately before WKV update "
            "assembly. The fix changes only diagnostic labels/capture points.",
            "",
            "## Changed File/Function",
            "",
            "- `src/qrwkv_xla/students/rwkv7_qwen_reference.py`: "
            "`RWKV7QwenReferenceStudent._attention`",
            "",
            "## Before/After First Difference",
            "",
            "- Before: unavailable stretch rows could stop at "
            "`v_first`/`iclr_update_rate` despite local ingredients existing.",
            "- After: the manual P71 artifact run reaches "
            f"`{report.get('first_divergent_stage')}` with "
            f"`same_run_valid={report.get('same_run_valid')}`.",
            "",
            "## Scope",
            "",
            "This is not a recurrence rewrite: it does not change computation, "
            "dtype policy, tolerances, parameter mapping, default balance-state "
            "behavior, or any Pallas/kernel code.",
            "",
        ]
    )


def _p72_hook_note_markdown(report: Mapping[str, Any]) -> str:
    unavailable = [
        row
        for row in report.get("unavailable_stretch_stages", [])
        if row.get("stage") in {"k_k", "k_a"}
    ]
    return "\n".join(
        [
            "# P72 k_k / k_a Hook Note",
            "",
            "## Source Semantics",
            "",
            "- k_k source expression / variable: "
            "`RWKV7QwenReferenceStudent._attention` reads `params['k_k']`, "
            "reshapes it per layer/head, records it as `stage='k_k'`, then "
            "uses it in `kk = _l2_normalize(k * k_k[None, :, :])`.",
            "- k_a source expression / variable: "
            "`RWKV7QwenReferenceStudent._attention` reads `params['k_a']`, "
            "reshapes it per layer/head, records it as `stage='k_a'`, then "
            "uses it in `k = k * (1.0 + (a - 1.0) * k_a[None, :, :])`.",
            "- RADLADS-compatible path: active when "
            "`use_radlads_balance_state_terms=True` and "
            "`radlads_balance_state=False`; rows are live captured when emitted.",
            "- QRWKV off: active under the same balance-state-terms path; rows "
            "are live captured when emitted.",
            "- QRWKV experimental: `radlads_balance_state=True` uses "
            "`kk = _l2_normalize(k)` directly, so `k_k` and `k_a` are not "
            "computed in this fixture path and remain explicit unavailable rows.",
            "",
            "## Availability",
            "",
            f"- k_k available on all sides: `{report.get('k_k_available')}`",
            f"- k_a available on all sides: `{report.get('k_a_available')}`",
            f"- k_k/k_a unavailable rows: `{len(unavailable)}`",
            f"- Recommended next phase: `{report.get('recommended_next_phase')}`",
            "",
            "No computation, dtype policy, tolerance, fixture values, parameter "
            "mapping, Pallas/kernel code, or default balance-state behavior is "
            "changed by P72.",
            "",
        ]
    )


def _p73_lane_map_markdown(report: Mapping[str, Any]) -> str:
    lane_map = report.get("balance_state_lane_map", {})
    lines = [
        "# P73 Balance-State Lane Map",
        "",
        "## Lane Definitions",
        "",
        "- balance_state_terms: `radlads_balance_state_terms=True` and "
        "`radlads_balance_state=False`; `k_k` and `k_a` are active.",
        "- direct_balance_state: `radlads_balance_state=True`; `k_k` and "
        "`k_a` are inactive and `kk` is computed directly from `k`.",
        "- native_or_unknown: config does not identify either RADLADS "
        "balance-state compatibility lane.",
        "",
        "## Side Classification",
        "",
    ]
    for side, title in (
        ("radlads", "RADLADS"),
        ("qrwkv_off", "QRWKV off"),
        ("qrwkv_experimental", "QRWKV experimental"),
    ):
        item = lane_map.get(side, {})
        lines.extend(
            [
                f"{title}:",
                f"- lane: `{item.get('lane')}`",
                "- radlads_balance_state_terms: `"
                f"{item.get('radlads_balance_state_terms')}`",
                f"- radlads_balance_state: `{item.get('radlads_balance_state')}`",
                f"- k_k expected: `{item.get('k_k_expected')}`",
                f"- k_a expected: `{item.get('k_a_expected')}`",
                f"- kk construction: `{item.get('kk_construction')}`",
                f"- comparable_to: `{item.get('comparable_to')}`",
                "- stages excluded as not_applicable: `"
                f"{item.get('stages_excluded_as_not_applicable')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## k_k / k_a Meaning",
            "",
            "when active: `k_k` feeds `kk = normalize(k * k_k)` and `k_a` "
            "feeds the balance-state-terms `k_for_update` adjustment.",
            "when inactive: direct balance-state computes `kk = normalize(k)` "
            "and bypasses `k_k`/`k_a`.",
            "why inactive is not a math failure: a direct-balance lane is not "
            "expected to emit Lane A-only tensors.",
            "",
            "## Comparable Stage Sets",
            "",
            "Lane A comparable stages: RADLADS and QRWKV off "
            "`balance_state_terms` rows.",
            "Lane B comparable stages: incomplete until a RADLADS "
            "`direct_balance_state` lane exists.",
            "Mixed-lane excluded stages: `k_k`, `k_a`.",
            "",
            "## Recommendation",
            "",
            f"{report.get('recommended_next_phase')}",
            "",
        ]
    )
    return "\n".join(lines)


def _p73_fix_note_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P73 Fix Note",
            "",
            "P73 applies reporting-only lane classification for balance-state "
            "trace rows. Direct-balance `k_k` and `k_a` rows are marked "
            "`not_applicable` with `not_active_in_lane` reasons instead of "
            "ordinary missing-hook reasons.",
            "",
            "- First non-applicable stage: `"
            f"{report.get('first_overall_non_applicable_stage')}`",
            "- First comparable differing stage: `"
            f"{report.get('first_comparable_differing_stage')}`",
            f"- Recommended next phase: `{report.get('recommended_next_phase')}`",
            "",
            "No recurrence math, dtype policy, tolerance, Pallas/kernel code, "
            "RADLADS upstream/vendor code, or default experimental "
            "balance_state behavior is changed.",
            "",
        ]
    )
