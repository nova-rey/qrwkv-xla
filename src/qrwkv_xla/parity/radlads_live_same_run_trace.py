from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
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
from qrwkv_xla.parity.radlads_wkv_state_convention import (
    REFERENCE_STATE_EXPORT_PATH,
    REFERENCE_STATE_IMPORT_PATH,
    export_reference_state_object,
    extract_state_slot,
    import_reference_state_object,
)
from qrwkv_xla.parity.radlads_wkv_trace import compare_trace_arrays, load_trace_jsonl
from qrwkv_xla.students.wkv_runtime import (
    WKVRuntime,
    build_pallas_runtime_probe,
    normalize_wkv_runtime,
)

LIVE_SAME_RUN_TRACE_SCHEMA = "qrwkv_xla.p68_live_same_run_trace.v1"
LIVE_SAME_RUN_REPORT_SCHEMA = "qrwkv_xla.p68_live_same_run_trace_report.v1"
P75_RESIDUAL_IMPACT_GATE_SCHEMA = "qrwkv_xla.p75_residual_impact_gate.v1"
P76_STATE_EXPORT_IMPORT_RESIDUAL_SCHEMA = (
    "qrwkv_xla.p76_state_export_import_residual.v1"
)
P77_FULL_VS_STEPWISE_RESIDUAL_SCHEMA = "qrwkv_xla.p77_full_vs_stepwise_residual.v1"
P78_LOGITS_OUTPUT_RESIDUAL_SCHEMA = "qrwkv_xla.p78_logits_output_residual.v1"
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
        "state_after_exported": (
            "state_after_exported",
            "returned_wkv_matrix_state",
            "exported_wkv_matrix_state",
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
AVAILABLE_CAPTURE_KINDS = {"live_captured", "exact_reconstruction", "exported_state"}
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
P75_DIRECT_LANE_REPAIR = "P75 targeted RADLADS direct-lane generation repair"
P75_DIRECT_KK_FIX = "P75 targeted direct-lane kk construction fix"
P75_DIRECT_BALANCE_PREP_FIX = (
    "P75 targeted direct-lane k_for_update/v_for_update balance-prep fix"
)
P75_DIRECT_AB_FIX = "P75 targeted direct-lane ab construction/orientation fix"
P75_DIRECT_VK_FIX = "P75 targeted direct-lane vk/outer-product orientation fix"
P75_DIRECT_STATE_AFTER_FIX = "P75 targeted direct-lane state_after assembly/dtype fix"
P75_TERMS_KK_FIX = "P75 targeted terms-lane kk construction fix"
P75_TERMS_BALANCE_PREP_FIX = (
    "P75 targeted terms-lane k_for_update/v_for_update balance-prep fix"
)
P75_TERMS_AB_FIX = "P75 targeted terms-lane ab construction/orientation fix"
P75_TERMS_VK_FIX = "P75 targeted terms-lane vk/outer-product orientation fix"
P75_RESIDUAL_GATE = "P75 residual-impact / kernel-readiness gate"
P75_PALLAS = "P75 Pallas prototype behind known-caveat flag"
P77_BROADER_FIXTURE_VALIDATION = "P77 broader fixture residual-impact validation"
P77_STATE_AFTER_FIX = "P77 targeted lane-aware state layout fix"
P77_LOGITS_OUTPUT_FIX = "P78 targeted logits/output residual fix"
P78_BROADER_FIXTURE_VALIDATION = "P79 broader fixture residual-impact validation"
P78_LOGITS_OUTPUT_HOOK_COMPLETION = "P79 targeted logits/output hook completion"
P78_LOGITS_OUTPUT_RESIDUAL_FIX = "P79 targeted logits/output residual fix"
P78_OUTPUT_HEAD_LAYOUT_FIX = "P79 targeted output-head/layout fix"
P78_LANE_AWARE_OUTPUT_COMPARISON_FIX = "P79 targeted lane-aware output comparison fix"
P78_KERNEL_GATE_HARDENING = "P79 kernel-readiness hardening gate"
P78_PALLAS_PROTOTYPE = "P79 Pallas prototype behind known-caveat flag"
P79_BROADER_FIXTURE_RESIDUAL_MATRIX_SCHEMA = (
    "qrwkv_xla.p79_broader_fixture_residual_matrix.v1"
)
P80_FIXTURE_LINEAGE_RESOLUTION_SCHEMA = "qrwkv_xla.p80_fixture_lineage_resolution.v1"
P79_ACTIVE_EXPECTED_CASES = (
    "tiny_no_mask",
    "tiny_attention_mask",
    "tiny_stepwise_state",
    "tiny_prefix_or_left_padding",
    "tiny_all_radlads_math_enabled",
)
P79_ACCEPTED_ALIASES = {
    "tiny_prefix_padding_or_left_padding": "tiny_prefix_or_left_padding",
}
P79_DEPRECATED_CASES: tuple[str, ...] = ()
P79_OPTIONAL_CASES: tuple[str, ...] = ()
P79_EXPECTED_CASES = (
    *P79_ACTIVE_EXPECTED_CASES,
    *P79_ACCEPTED_ALIASES.keys(),
)
P81_PALLAS_PROTOTYPE = "P81 Pallas prototype behind known-caveat flag"
P81_KERNEL_REFERENCE_PARITY = "P81 kernel/reference parity scaffold"
P80_PALLAS_PROTOTYPE = P81_PALLAS_PROTOTYPE
P80_KERNEL_REFERENCE_PARITY = P81_KERNEL_REFERENCE_PARITY
P80_STATE_AFTER_FIX = "P80 targeted broader-fixture state_after residual fix"
P80_EXPORTED_STATE_FIX = "P80 targeted broader-fixture exported_state residual fix"
P80_FULL_VS_STEPWISE_FIX = "P80 targeted broader-fixture full-vs-stepwise residual fix"
P80_LOGITS_OUTPUT_FIX = "P80 targeted broader-fixture logits/output residual fix"
P80_LINEAGE_REPAIR = "P80 targeted fixture lineage/harness repair"
P80_KERNEL_GATE_HARDENING = "P80 kernel-readiness hardening gate"
P82_REFERENCE_VS_PALLAS_PARITY_GATE = "P82 reference-vs-Pallas parity gate"
P82_PALLAS_RUNTIME_SCAFFOLD_COMPLETION = (
    "P82 targeted Pallas runtime scaffold completion"
)
P82_PALLAS_DEPENDENCY_BACKEND_FIX = (
    "P82 targeted Pallas dependency/backend availability fix"
)
P83_REFERENCE_VS_PALLAS_PARITY_GATE = "P83 reference-vs-Pallas parity gate"
P83_PALLAS_RUNTIME_SCAFFOLD_COMPLETION = (
    "P83 targeted Pallas runtime scaffold completion"
)
P84_BROADER_PALLAS_SHAPE_DTYPE_PARITY = "P84 broader Pallas WKV shape/dtype parity"
P77_STATE_EXPORT_FIX = "P77 targeted state export/import convention fix"
P77_FULL_VS_STEPWISE_FIX = "P77 targeted full-vs-stepwise residual fix"
P77_LANE_LAYOUT_FIX = "P77 targeted lane-aware state layout fix"
P77_KERNEL_GATE_HARDENING = "P77 kernel-readiness hardening gate"
P75_RESIDUAL_STAGES = (
    "state_after_live",
    "vk",
    "balance_state_term",
    "composite_update_term",
    "decay_value",
    "decay_log_w",
    "state_after_exported",
    "logits",
    "full_vs_stepwise",
)
P75_REQUIRED_OUTPUT_GATES = (
    "state_after",
    "exported_state",
    "full_vs_stepwise",
    "logits_output",
)


@dataclass(frozen=True)
class FixtureExpectationMetadata:
    active_expected_cases: tuple[str, ...] = P79_ACTIVE_EXPECTED_CASES
    accepted_aliases: Mapping[str, str] | None = None
    deprecated_cases: tuple[str, ...] = P79_DEPRECATED_CASES
    optional_cases: tuple[str, ...] = P79_OPTIONAL_CASES

    def alias_map(self) -> dict[str, str]:
        if self.accepted_aliases is None:
            return dict(P79_ACCEPTED_ALIASES)
        return dict(self.accepted_aliases)

    def requested_cases(self) -> list[str]:
        return [
            *self.active_expected_cases,
            *self.alias_map().keys(),
            *self.deprecated_cases,
            *self.optional_cases,
        ]


@dataclass(frozen=True)
class FixtureCaseResolution:
    requested_case: str
    canonical_case: str | None
    resolved_case: str | None
    resolution: str
    category: str


P75_WARNING_MAX_ABS = 1e-5


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
        export_path: str | None = None,
        import_path: str | None = None,
        import_roundtrip_status: str | None = None,
        import_roundtrip_reason: str | None = None,
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
                    extra_metadata=_compact_metadata(
                        export_path=export_path,
                        import_path=import_path,
                        import_roundtrip_status=import_roundtrip_status,
                        import_roundtrip_reason=import_roundtrip_reason,
                    ),
                )
            return
        if (
            head is None
            and normalized_stage
            in (*MINIMUM_STAGES, *P71_STRETCH_STAGES, "state_after_exported")
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
                    extra_metadata=_compact_metadata(
                        export_path=export_path,
                        import_path=import_path,
                        import_roundtrip_status=import_roundtrip_status,
                        import_roundtrip_reason=import_roundtrip_reason,
                    ),
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
            extra_metadata=_compact_metadata(
                export_path=export_path,
                import_path=import_path,
                import_roundtrip_status=import_roundtrip_status,
                import_roundtrip_reason=import_roundtrip_reason,
            ),
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
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        summary = summarize_array(
            name or stage,
            value,
            stage=stage,
            layer=layer,
            time_index=token,
        )
        row = {
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
            "array": (value.tolist() if value.size <= self.max_inline_values else None),
            "summary": {
                "finite": bool(np.isfinite(value).all()) if value.size else True,
                "max_abs": summary.abs_max,
                "mean_abs": float(np.mean(np.abs(value))) if value.size else 0.0,
                "sample": None if value.size == 0 else float(value.reshape(-1)[0]),
            },
            "live_config": self.live_config,
        }
        if extra_metadata:
            row.update(dict(extra_metadata))
        self.entries.append(row)


def _compact_metadata(**items: Any) -> dict[str, Any]:
    return {key: value for key, value in items.items() if value is not None}


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
    lanes = _lanes_from_rows(rows, side=side)
    output: list[dict[str, Any]] = []
    for side_lane in lanes:
        lane_rows = [
            row for row in rows if _row_balance_state_lane(row, side=side) == side_lane
        ]
        side_live_config = _live_config_from_rows(lane_rows)
        for context in sorted(contexts, key=_context_sort_key):
            by_stage = _rows_by_source_stage(lane_rows, context)
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
        and lane_validity["direct_balance_state_lane_valid"]
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
    report = {
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
        "balance_state_terms_first_differing_stage": None
        if lane_validity.get("balance_state_terms_first_failure") is None
        else lane_validity["balance_state_terms_first_failure"].get("stage"),
        "balance_state_terms_first_differing_capture_kind": None
        if lane_validity.get("balance_state_terms_first_failure") is None
        else lane_validity["balance_state_terms_first_failure"].get("status"),
        "balance_state_terms_math_conclusion_valid": lane_validity[
            "balance_state_terms_math_conclusion_valid"
        ],
        "balance_state_terms_recommended_next_phase": _lane_specific_recommendation(
            lane=BALANCE_STATE_TERMS_LANE,
            first=lane_validity.get("balance_state_terms_first_failure"),
            valid=lane_validity["balance_state_terms_lane_valid"],
        ),
        "direct_balance_state_first_differing_stage": None
        if lane_validity.get("direct_balance_state_first_failure") is None
        else lane_validity["direct_balance_state_first_failure"].get("stage"),
        "direct_balance_state_first_differing_capture_kind": None
        if lane_validity.get("direct_balance_state_first_failure") is None
        else lane_validity["direct_balance_state_first_failure"].get("status"),
        "direct_balance_state_math_conclusion_valid": lane_validity[
            "direct_balance_state_math_conclusion_valid"
        ],
        "direct_balance_state_recommended_next_phase": _lane_specific_recommendation(
            lane=DIRECT_BALANCE_STATE_LANE,
            first=lane_validity.get("direct_balance_state_first_failure"),
            valid=lane_validity["direct_balance_state_lane_valid"],
        ),
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
        "live_rows_captured_radlads_terms": live_counts["radlads_terms"],
        "live_rows_captured_radlads_direct": live_counts["radlads_direct"],
        "live_rows_captured_qrwkv_off": live_counts["qrwkv_off"],
        "live_rows_captured_qrwkv_off_terms": live_counts["qrwkv_off_terms"],
        "live_rows_captured_qrwkv_experimental": live_counts["qrwkv_experimental"],
        "live_rows_captured_qrwkv_experimental_direct": live_counts[
            "qrwkv_experimental_direct"
        ],
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
    report["p77_full_vs_stepwise_residual"] = build_p77_full_vs_stepwise_residual(
        evidence=metadata.get("p77_full_vs_stepwise_evidence", []),
        report=report,
        atol=atol,
        rtol=rtol,
    )
    report["p78_logits_output_residual"] = build_p78_logits_output_residual(
        evidence=metadata.get(
            "p78_logits_output_evidence",
            metadata.get("p77_full_vs_stepwise_evidence", []),
        ),
        report=report,
        atol=atol,
        rtol=rtol,
    )
    p75_gate = build_p75_residual_impact_gate(report)
    report["p75_residual_impact_gate"] = p75_gate
    report["p76_state_export_import_residual"] = build_p76_state_export_import_residual(
        traces=traces,
        comparison_rows=rows,
        report=report,
        p75_gate=p75_gate,
        atol=atol,
        rtol=rtol,
    )
    report["kernel_ready"] = p75_gate["kernel_ready"]
    report["kernel_readiness_reason"] = p75_gate["kernel_readiness"]["reason"]
    report["blocking_gates"] = p75_gate["blocking_gates"]
    report["warning_gates"] = p75_gate["warning_gates"]
    report["recommended_next_phase"] = (
        p75_gate["recommended_next_phase"]
        if _p75_should_replace_recommendation(report, p75_gate, lane_decision)
        else lane_decision
    )
    return report


def _p75_should_replace_recommendation(
    report: Mapping[str, Any], p75_gate: Mapping[str, Any], lane_decision: str
) -> bool:
    return bool(
        lane_decision == P75_RESIDUAL_GATE and p75_gate.get("lane_comparisons_valid")
    )


def build_p75_residual_impact_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    terms_valid = bool(report.get("balance_state_terms_lane_valid"))
    direct_valid = bool(report.get("direct_balance_state_lane_valid"))
    lane_comparisons_valid = bool(
        terms_valid
        and direct_valid
        and report.get("balance_state_terms_first_differing_stage") is None
        and report.get("direct_balance_state_first_differing_stage") is None
    )
    residuals = {
        BALANCE_STATE_TERMS_LANE: _p75_lane_residuals(
            report,
            lane=BALANCE_STATE_TERMS_LANE,
            pair="radlads_vs_qrwkv_off",
        ),
        DIRECT_BALANCE_STATE_LANE: _p75_lane_residuals(
            report,
            lane=DIRECT_BALANCE_STATE_LANE,
            pair="radlads_vs_qrwkv_experimental",
        ),
    }
    for lane_report in residuals.values():
        for measurement in lane_report.get("measurements", {}).values():
            measurement["allclose_atol"] = report.get("atol")
            measurement["allclose_rtol"] = report.get("rtol")
    output_gates = _p75_output_gates(report, residuals=residuals)
    blocking_gates = _p75_blocking_gates(
        same_run_valid=bool(report.get("same_run_valid")),
        lane_comparisons_valid=lane_comparisons_valid,
        residuals=residuals,
        output_gates=output_gates,
    )
    warning_gates = _p75_warning_gates(residuals=residuals, output_gates=output_gates)
    kernel_ready = "yes" if not blocking_gates and not warning_gates else "no"
    reason = "all_required_gates_pass" if kernel_ready == "yes" else blocking_gates[0]
    return {
        "schema": P75_RESIDUAL_IMPACT_GATE_SCHEMA,
        "same_run_valid": bool(report.get("same_run_valid")),
        "fixture_id": report.get("fixture_id"),
        "parameter_id": report.get("parameter_id"),
        "same_run_group_id": report.get("same_run_group_id"),
        "lane_aware_keys": True,
        "lane_comparisons_valid": lane_comparisons_valid,
        "tolerances": {
            "atol": report.get("atol"),
            "rtol": report.get("rtol"),
            "warning_threshold_max_abs": P75_WARNING_MAX_ABS,
            "dtype": "trace row dtype",
            "comparison_policy": (
                "lane-primary pair allclose using existing report tolerance; "
                "shape mismatch and non-finite values are blocking"
            ),
        },
        "lane_comparisons": {
            BALANCE_STATE_TERMS_LANE: {
                "left": "RADLADS terms",
                "right": "QRWKV off terms",
                "valid": terms_valid,
                "first_differing_stage": report.get(
                    "balance_state_terms_first_differing_stage"
                ),
                "math_conclusion_valid": bool(
                    report.get("balance_state_terms_math_conclusion_valid")
                ),
            },
            DIRECT_BALANCE_STATE_LANE: {
                "left": "RADLADS direct",
                "right": "QRWKV experimental direct",
                "valid": direct_valid,
                "first_differing_stage": report.get(
                    "direct_balance_state_first_differing_stage"
                ),
                "math_conclusion_valid": bool(
                    report.get("direct_balance_state_math_conclusion_valid")
                ),
            },
        },
        "residuals": residuals,
        "output_gates": output_gates,
        "kernel_readiness": {
            "kernel_ready": kernel_ready,
            "reason": reason,
            "policy": (
                "yes only when same-run validity, both P74 lane comparisons, "
                "critical residual gates, state/export/full-vs-stepwise, and "
                "logits/output evidence are present and passing"
            ),
        },
        "kernel_ready": kernel_ready,
        "blocking_gates": blocking_gates,
        "warning_gates": warning_gates,
        "recommended_next_phase": _p75_recommended_next_phase(
            blocking_gates=blocking_gates,
            warning_gates=warning_gates,
            report=report,
        ),
    }


def build_p76_state_export_import_residual(
    *,
    traces: Mapping[str, list[dict[str, Any]]],
    comparison_rows: list[dict[str, Any]],
    report: Mapping[str, Any],
    p75_gate: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    intra_rows = _p76_intra_side_rows(traces=traces, atol=atol, rtol=rtol)
    import_rows = _p76_import_roundtrip_rows(traces)
    inter_rows = _p76_inter_side_rows(comparison_rows)
    required_surfaces = {
        "radlads_terms": ("radlads", BALANCE_STATE_TERMS_LANE),
        "qrwkv_off_terms": ("qrwkv_off", BALANCE_STATE_TERMS_LANE),
        "radlads_direct": ("radlads", DIRECT_BALANCE_STATE_LANE),
        "qrwkv_experimental_direct": (
            "qrwkv_experimental",
            DIRECT_BALANCE_STATE_LANE,
        ),
    }
    surface_status = {}
    for surface, (side, lane) in required_surfaces.items():
        surface_import = [
            row for row in import_rows if row["side"] == side and row["lane"] == lane
        ]
        surface_intra = [
            row for row in intra_rows if row["side"] == side and row["lane"] == lane
        ]
        if not surface_import or not surface_intra:
            surface_status[surface] = {
                "status": "unavailable",
                "reason": "missing_export_path",
            }
        elif all(row["status"] == "pass" for row in surface_import + surface_intra):
            surface_status[surface] = {"status": "pass", "reason": "allclose"}
        else:
            surface_status[surface] = {
                "status": "fail",
                "reason": "live_export_or_import_roundtrip_mismatch",
            }
    lane_pairs = {
        BALANCE_STATE_TERMS_LANE: "radlads_vs_qrwkv_off",
        DIRECT_BALANCE_STATE_LANE: "radlads_vs_qrwkv_experimental",
    }
    lane_status = {}
    for lane, pair in lane_pairs.items():
        rows = [row for row in inter_rows if row["lane"] == lane]
        available = [row for row in rows if row.get("status") != "unavailable"]
        if not available:
            lane_status[lane] = {
                "status": "unavailable",
                "reason": "missing_exported_state_rows",
                "pair": pair,
            }
        elif all(row["status"] == "pass" for row in available):
            lane_status[lane] = {"status": "pass", "reason": "allclose", "pair": pair}
        else:
            lane_status[lane] = {
                "status": "fail",
                "reason": "exported_state_residual",
                "pair": pair,
            }
    statuses = [
        item["status"] for item in [*surface_status.values(), *lane_status.values()]
    ]
    overall = (
        "pass"
        if statuses and all(status == "pass" for status in statuses)
        else "fail"
        if any(status == "fail" for status in statuses)
        else "unavailable"
    )
    blocking_gates = _p76_blocking_gates(
        surface_status=surface_status,
        lane_status=lane_status,
    )
    return {
        "schema": P76_STATE_EXPORT_IMPORT_RESIDUAL_SCHEMA,
        "phase": "P76",
        "same_run_valid": bool(report.get("same_run_valid")),
        "lane_aware_keys": True,
        "same_run_group_id": report.get("same_run_group_id"),
        "fixture_id": report.get("fixture_id"),
        "parameter_id": report.get("parameter_id"),
        "export_path": REFERENCE_STATE_EXPORT_PATH,
        "import_path": REFERENCE_STATE_IMPORT_PATH,
        "state_export_semantics": {
            "meaning": "reference_state_slots",
            "exported_state_slot": "wkv_matrix_state",
            "trace_stage": "state_after_exported",
            "capture_kind": "exported_state",
            "export_path": REFERENCE_STATE_EXPORT_PATH,
            "import_path": REFERENCE_STATE_IMPORT_PATH,
        },
        "status": overall,
        "p75_exported_state_gate": p75_gate.get("output_gates", {}).get(
            "exported_state"
        ),
        "kernel_ready": p75_gate.get("kernel_ready"),
        "blocking_gates": blocking_gates,
        "p75_recommended_next_phase": p75_gate.get("recommended_next_phase"),
        "recommended_next_phase": p75_gate.get("recommended_next_phase")
        if overall == "pass"
        else _p76_recommended_next_phase(
            status=overall,
            surface_status=surface_status,
            lane_status=lane_status,
        ),
        "lanes": {
            BALANCE_STATE_TERMS_LANE: {
                "fair_pair": lane_pairs[BALANCE_STATE_TERMS_LANE],
                "status": lane_status[BALANCE_STATE_TERMS_LANE]["status"],
            },
            DIRECT_BALANCE_STATE_LANE: {
                "fair_pair": lane_pairs[DIRECT_BALANCE_STATE_LANE],
                "status": lane_status[DIRECT_BALANCE_STATE_LANE]["status"],
            },
        },
        "intra_side_consistency": surface_status,
        "inter_side_parity": lane_status,
        "round_trip": surface_status,
        "surface_status": surface_status,
        "lane_pair_status": lane_status,
        "intra_side_live_vs_exported": intra_rows,
        "import_roundtrip": import_rows,
        "inter_side_exported_state": inter_rows,
    }


def build_p77_full_vs_stepwise_residual(
    *,
    evidence: Iterable[Mapping[str, Any]],
    report: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    rows = [dict(row) for row in evidence]
    required_surfaces = {
        "radlads_terms": ("radlads", BALANCE_STATE_TERMS_LANE),
        "qrwkv_off_terms": ("qrwkv_off", BALANCE_STATE_TERMS_LANE),
        "radlads_direct": ("radlads", DIRECT_BALANCE_STATE_LANE),
        "qrwkv_experimental_direct": (
            "qrwkv_experimental",
            DIRECT_BALANCE_STATE_LANE,
        ),
    }
    surface_status = {}
    for surface, (side, lane) in required_surfaces.items():
        surface_rows = [
            row
            for row in rows
            if row.get("side") == side
            and row.get("balance_state_lane") == lane
            and row.get("stage") in {"state_after_live", "state_after_exported"}
        ]
        available = [row for row in surface_rows if row.get("status") != "unavailable"]
        if not available:
            surface_status[surface] = {
                "status": "unavailable",
                "reason": "missing_full_vs_stepwise_rows",
            }
        elif all(row.get("status") == "pass" for row in available) and {
            row.get("stage") for row in available
        } >= {"state_after_live", "state_after_exported"}:
            surface_status[surface] = {"status": "pass", "reason": "allclose"}
        else:
            first = next(
                (row for row in available if row.get("status") != "pass"), None
            )
            surface_status[surface] = {
                "status": "fail",
                "reason": "full_vs_stepwise_residual",
                "first_failure": _p77_failure_summary(first),
            }
    lane_status = {}
    for lane, surfaces in {
        BALANCE_STATE_TERMS_LANE: ("radlads_terms", "qrwkv_off_terms"),
        DIRECT_BALANCE_STATE_LANE: ("radlads_direct", "qrwkv_experimental_direct"),
    }.items():
        statuses = [surface_status[surface]["status"] for surface in surfaces]
        if all(status == "pass" for status in statuses):
            lane_status[lane] = {"status": "pass", "reason": "all_surfaces_pass"}
        elif any(status == "fail" for status in statuses):
            lane_status[lane] = {"status": "fail", "reason": "surface_residual"}
        else:
            lane_status[lane] = {"status": "unavailable", "reason": "missing_surface"}
    statuses = [item["status"] for item in surface_status.values()]
    overall = (
        "pass"
        if statuses and all(status == "pass" for status in statuses)
        else "fail"
        if any(status == "fail" for status in statuses)
        else "unavailable"
    )
    final_state = _p77_stage_summary(rows, "state_after_live")
    exported_state = _p77_stage_summary(rows, "state_after_exported")
    outputs = _p77_stage_summary(rows, "logits_output")
    blocking_gates = []
    if overall == "fail":
        blocking_gates.append("full_vs_stepwise_residual")
    elif overall == "unavailable":
        blocking_gates.append("missing_evidence:full_vs_stepwise")
    return {
        "schema": P77_FULL_VS_STEPWISE_RESIDUAL_SCHEMA,
        "phase": "P77",
        "status": overall,
        "same_run_valid": bool(report.get("same_run_valid")),
        "lane_aware_keys": True,
        "full_path": "RWKV7QwenReferenceStudent.apply_with_state",
        "stepwise_path": "RWKV7QwenReferenceStudent.step",
        "same_run_group_id": report.get("same_run_group_id"),
        "fixture_id": report.get("fixture_id"),
        "parameter_id": report.get("parameter_id"),
        "tolerances": {"atol": atol, "rtol": rtol},
        "paths": {
            "full": "RWKV7QwenReferenceStudent.apply_with_state",
            "stepwise": "RWKV7QwenReferenceStudent.step",
            "initial_state": "student.init_state(batch_size)",
            "state_carry": "explicit returned state from each token step",
            "state_slots": ["wkv_matrix_state", "shift_state", "next_position"],
        },
        "lanes": {
            BALANCE_STATE_TERMS_LANE: {
                "left": "RADLADS terms",
                "right": "QRWKV off terms",
                "status": lane_status[BALANCE_STATE_TERMS_LANE]["status"],
            },
            DIRECT_BALANCE_STATE_LANE: {
                "left": "RADLADS direct",
                "right": "QRWKV experimental direct",
                "status": lane_status[DIRECT_BALANCE_STATE_LANE]["status"],
            },
        },
        "surface_status": surface_status,
        "lane_status": lane_status,
        "final_state": final_state,
        "exported_state": exported_state,
        "outputs": outputs,
        "blocking_gates": blocking_gates,
        "row_count": len(rows),
        "rows": rows,
        "p75_full_vs_stepwise_gate": _p77_gate_from_status(overall),
        "recommended_next_phase": _p77_recommended_next_phase(overall),
    }


def _p77_stage_summary(rows: Iterable[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    stage_rows = [dict(row) for row in rows if row.get("stage") == stage]
    if not stage_rows:
        return {
            "status": "unavailable",
            "reason": f"missing_evidence:{stage}",
            "row_count": 0,
        }
    statuses = [str(row.get("status", "unavailable")) for row in stage_rows]
    if all(status == "pass" for status in statuses):
        status = "pass"
        reason = "allclose"
    elif any(status == "fail" for status in statuses) or any(
        status.endswith("mismatch") for status in statuses
    ):
        status = "fail"
        reason = "residual_or_shape_mismatch"
    else:
        status = "unavailable"
        reason = next(
            (
                str(row.get("reason"))
                for row in stage_rows
                if row.get("status") == "unavailable"
            ),
            f"missing_evidence:{stage}",
        )
    numeric_rows = [row for row in stage_rows if row.get("max_abs_error") is not None]
    max_abs = max((float(row["max_abs_error"]) for row in numeric_rows), default=None)
    mean_abs = max((float(row["mean_abs_error"]) for row in numeric_rows), default=None)
    relative_max = max(
        (float(row["max_relative_error"]) for row in numeric_rows), default=None
    )
    rms = max((float(row.get("rms_error", 0.0)) for row in numeric_rows), default=None)
    return {
        "status": status,
        "reason": reason,
        "row_count": len(stage_rows),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "relative_max": relative_max,
        "shape_match": all(bool(row.get("shape_match", False)) for row in stage_rows),
        "dtype_match": all(bool(row.get("dtype_match", False)) for row in stage_rows),
        "finite": all(bool(row.get("finite_both", False)) for row in stage_rows),
    }


def _p77_gate_from_status(status: str) -> dict[str, Any]:
    if status == "pass":
        return {"status": "pass", "required": True, "reason": "all_lanes_pass"}
    if status == "fail":
        return {
            "status": "fail",
            "required": True,
            "reason": "full_vs_stepwise_residual",
        }
    return {
        "status": "unavailable",
        "required": True,
        "reason": "missing_evidence:full_vs_stepwise",
    }


def _p77_recommended_next_phase(status: str) -> str:
    if status == "pass":
        return P77_LOGITS_OUTPUT_FIX
    if status == "fail":
        return P77_FULL_VS_STEPWISE_FIX
    return P77_FULL_VS_STEPWISE_FIX


def _p77_failure_summary(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "case": row.get("case"),
        "side": row.get("side"),
        "lane": row.get("balance_state_lane"),
        "stage": row.get("stage"),
        "layer": row.get("layer"),
        "head": row.get("head"),
        "final_token": row.get("final_token"),
        "status": row.get("status"),
        "max_abs_error": row.get("max_abs_error"),
    }


def build_p78_logits_output_residual(
    *,
    evidence: Iterable[Mapping[str, Any]],
    report: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in evidence
        if row.get("comparison") == "full_vs_stepwise_output"
        or row.get("stage")
        in {
            "post_block_hidden_output",
            "final_normalized_hidden",
            "final_lm_head_logits",
            "selected_token_logits",
        }
    ]
    full_vs_stepwise = _p78_full_vs_stepwise_summary(rows)
    inter_rows = _p78_inter_side_rows(rows, atol=atol, rtol=rtol)
    inter_side = _p78_inter_side_summary(inter_rows)
    hidden_path = _p78_stage_path(rows, "post_block_hidden_output")
    output_path = _p78_stage_path(rows, "final_normalized_hidden")
    logits_path = _p78_logits_path(rows)
    status = _p78_overall_status(
        hidden_path=hidden_path,
        output_path=output_path,
        logits_path=logits_path,
        inter_side=inter_side,
        full_vs_stepwise=full_vs_stepwise,
    )
    blocking_gates = []
    if status == "fail":
        blocking_gates.append("logits_output_residual")
    elif status == "unavailable":
        if logits_path.get("status") == "unavailable":
            blocking_gates.append(
                str(logits_path.get("reason", "missing_lm_head_logits_path"))
            )
        else:
            blocking_gates.append("missing_stepwise_output_capture")
    return {
        "schema": P78_LOGITS_OUTPUT_RESIDUAL_SCHEMA,
        "phase": "P78",
        "status": status,
        "same_run_valid": bool(report.get("same_run_valid")),
        "lane_aware_keys": True,
        "same_run_group_id": report.get("same_run_group_id"),
        "fixture_id": report.get("fixture_id"),
        "parameter_id": report.get("parameter_id"),
        "tolerances": {"atol": atol, "rtol": rtol},
        "hidden_path": hidden_path,
        "output_path": output_path,
        "logits_path": logits_path,
        "lanes": {
            BALANCE_STATE_TERMS_LANE: {
                "left": "RADLADS terms",
                "right": "QRWKV off terms",
                "status": inter_side[BALANCE_STATE_TERMS_LANE]["status"],
            },
            DIRECT_BALANCE_STATE_LANE: {
                "left": "RADLADS direct",
                "right": "QRWKV experimental direct",
                "status": inter_side[DIRECT_BALANCE_STATE_LANE]["status"],
            },
        },
        "inter_side_parity": inter_side,
        "inter_side_rows": inter_rows,
        "full_vs_stepwise_output": full_vs_stepwise,
        "blocking_gates": blocking_gates,
        "p75_logits_output_gate": _p78_gate_from_status(status, logits_path),
        "row_count": len(rows),
        "rows": rows,
        "recommended_next_phase": _p78_recommended_next_phase(status),
    }


def _p78_full_vs_stepwise_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    surface_status = {}
    required_surfaces = {
        "radlads_terms": ("radlads", BALANCE_STATE_TERMS_LANE),
        "qrwkv_off_terms": ("qrwkv_off", BALANCE_STATE_TERMS_LANE),
        "radlads_direct": ("radlads", DIRECT_BALANCE_STATE_LANE),
        "qrwkv_experimental_direct": (
            "qrwkv_experimental",
            DIRECT_BALANCE_STATE_LANE,
        ),
    }
    for surface, (side, lane) in required_surfaces.items():
        surface_rows = [
            row
            for row in rows
            if row.get("side") == side
            and row.get("balance_state_lane") == lane
            and row.get("stage") in {"post_block_hidden_output", "final_lm_head_logits"}
        ]
        hidden_rows = [
            row
            for row in surface_rows
            if row.get("stage") == "post_block_hidden_output"
        ]
        comparable = [row for row in surface_rows if row.get("status") != "unavailable"]
        if not hidden_rows:
            surface_status[surface] = {
                "status": "unavailable",
                "reason": "missing_stepwise_output_capture",
            }
        elif any(
            row.get("status") not in {"pass", "unavailable"} for row in surface_rows
        ):
            surface_status[surface] = {
                "status": "fail",
                "reason": "full_vs_stepwise_output_residual",
            }
        elif comparable and all(row.get("status") == "pass" for row in comparable):
            surface_status[surface] = {
                "status": "pass",
                "reason": "hidden_output_allclose",
            }
        else:
            surface_status[surface] = {
                "status": "unavailable",
                "reason": "missing_stepwise_output_capture",
            }
    statuses = [item["status"] for item in surface_status.values()]
    overall = (
        "pass"
        if statuses and all(status == "pass" for status in statuses)
        else "fail"
        if any(status == "fail" for status in statuses)
        else "unavailable"
    )
    return {
        "status": overall,
        "reason": "all_surfaces_pass"
        if overall == "pass"
        else "surface_residual"
        if overall == "fail"
        else "missing_stepwise_output_capture",
        "surface_status": surface_status,
    }


def _p78_inter_side_rows(
    rows: list[dict[str, Any]], *, atol: float, rtol: float
) -> list[dict[str, Any]]:
    result = []
    pairs = {
        BALANCE_STATE_TERMS_LANE: ("radlads", "qrwkv_off", "radlads_vs_qrwkv_off"),
        DIRECT_BALANCE_STATE_LANE: (
            "radlads",
            "qrwkv_experimental",
            "radlads_vs_qrwkv_experimental",
        ),
    }
    keys = sorted(
        {
            (
                row.get("case"),
                row.get("balance_state_lane"),
                row.get("stage"),
                row.get("final_token"),
            )
            for row in rows
            if row.get("full_array") is not None
        }
    )
    for case, lane, stage, final_token in keys:
        if lane not in pairs:
            continue
        left_side, right_side, pair = pairs[str(lane)]
        left = _p78_find_output_row(
            rows,
            case=case,
            side=left_side,
            lane=str(lane),
            stage=stage,
            final_token=final_token,
        )
        right = _p78_find_output_row(
            rows,
            case=case,
            side=right_side,
            lane=str(lane),
            stage=stage,
            final_token=final_token,
        )
        if left is None or right is None:
            comparison = {
                "status": "unavailable",
                "shape_match": False,
                "dtype_match": False,
                "finite_both": False,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_relative_error": None,
                "allclose": False,
            }
            reason = "missing_inter_side_output_capture"
        else:
            comparison = compare_trace_arrays(
                left["full_array"], right["full_array"], atol=atol, rtol=rtol
            )
            reason = (
                "allclose" if comparison["status"] == "pass" else comparison["status"]
            )
        result.append(
            {
                "case": case,
                "lane": lane,
                "stage": stage,
                "final_token": final_token,
                "pair": pair,
                "left_side": left_side,
                "right_side": right_side,
                "status": comparison["status"],
                "reason": reason,
                **comparison,
            }
        )
    return result


def _p78_find_output_row(
    rows: list[dict[str, Any]],
    *,
    case: Any,
    side: str,
    lane: str,
    stage: Any,
    final_token: Any,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("case") == case
            and row.get("side") == side
            and row.get("balance_state_lane") == lane
            and row.get("stage") == stage
            and row.get("final_token") == final_token
            and row.get("full_array") is not None
        ),
        None,
    )


def _p78_inter_side_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for lane in (BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE):
        lane_rows = [row for row in rows if row.get("lane") == lane]
        hidden_rows = [
            row for row in lane_rows if row.get("stage") == "post_block_hidden_output"
        ]
        available = [row for row in lane_rows if row.get("status") != "unavailable"]
        if not hidden_rows:
            result[lane] = {
                "status": "unavailable",
                "reason": "missing_inter_side_output_capture",
                "row_count": len(lane_rows),
            }
        elif any(row.get("status") != "pass" for row in available):
            result[lane] = {
                "status": "fail",
                "reason": "inter_side_output_residual",
                "row_count": len(lane_rows),
            }
        elif available:
            result[lane] = {
                "status": "pass",
                "reason": "hidden_output_allclose",
                "row_count": len(lane_rows),
            }
        else:
            result[lane] = {
                "status": "unavailable",
                "reason": "missing_inter_side_output_capture",
                "row_count": len(lane_rows),
            }
    return result


def _p78_stage_path(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    stage_rows = [row for row in rows if row.get("stage") == stage]
    if not stage_rows:
        reason = (
            "missing_final_normalized_hidden_capture"
            if stage == "final_normalized_hidden"
            else "missing_stepwise_output_capture"
        )
        return {"status": "unavailable", "reason": reason, "row_count": 0}
    statuses = [str(row.get("status", "unavailable")) for row in stage_rows]
    if any(status not in {"pass", "unavailable"} for status in statuses):
        status = "fail"
        reason = "output_residual_or_shape_mismatch"
    elif any(status == "pass" for status in statuses):
        status = "pass"
        reason = "allclose"
    else:
        status = "unavailable"
        reason = next(
            (
                str(row.get("reason"))
                for row in stage_rows
                if row.get("reason") is not None
            ),
            "missing_stepwise_output_capture",
        )
    return {"status": status, "reason": reason, "row_count": len(stage_rows)}


def _p78_logits_path(rows: list[dict[str, Any]]) -> dict[str, Any]:
    logits_rows = [
        row
        for row in rows
        if row.get("stage") in {"final_lm_head_logits", "selected_token_logits"}
    ]
    if not logits_rows:
        return {
            "status": "unavailable",
            "reason": "missing_stepwise_output_capture",
            "row_count": 0,
        }
    statuses = [str(row.get("status", "unavailable")) for row in logits_rows]
    if any(status not in {"pass", "unavailable"} for status in statuses):
        status = "fail"
        reason = "logits_residual_or_shape_mismatch"
    elif any(status == "pass" for status in statuses):
        status = "pass"
        reason = "true_lm_head_logits_allclose"
    else:
        status = "unavailable"
        reason = next(
            (
                str(row.get("reason"))
                for row in logits_rows
                if row.get("reason") is not None
            ),
            "missing_lm_head_logits_path",
        )
    return {"status": status, "reason": reason, "row_count": len(logits_rows)}


def _p78_overall_status(
    *,
    hidden_path: Mapping[str, Any],
    output_path: Mapping[str, Any],
    logits_path: Mapping[str, Any],
    inter_side: Mapping[str, Mapping[str, Any]],
    full_vs_stepwise: Mapping[str, Any],
) -> str:
    statuses = [
        str(hidden_path.get("status")),
        str(full_vs_stepwise.get("status")),
        *(str(item.get("status")) for item in inter_side.values()),
    ]
    optional_statuses = [str(output_path.get("status")), str(logits_path.get("status"))]
    if any(status == "fail" for status in [*statuses, *optional_statuses]):
        return "fail"
    if str(logits_path.get("status")) == "unavailable":
        return "unavailable"
    if statuses and all(status == "pass" for status in statuses):
        return "pass"
    return "unavailable"


def _p78_gate_from_report(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _p78_gate_from_status(
            str(value.get("status", "unavailable")),
            value.get("logits_path"),
        )
    return {
        "status": "unavailable",
        "required": True,
        "reason": "missing_evidence:logits_output",
    }


def _p78_gate_from_status(status: str, logits_path: Any = None) -> dict[str, Any]:
    if status == "pass":
        reason = "all_lanes_pass"
        if (
            isinstance(logits_path, Mapping)
            and logits_path.get("status") == "unavailable"
        ):
            reason = (
                f"hidden_output_pass_logits_unavailable:{logits_path.get('reason')}"
            )
        return {"status": "pass", "required": True, "reason": reason}
    if status == "fail":
        return {
            "status": "fail",
            "required": True,
            "reason": "logits_output_residual",
        }
    reason = "missing_evidence:logits_output"
    if isinstance(logits_path, Mapping) and logits_path.get("status") == "unavailable":
        reason = str(logits_path.get("reason", "missing_lm_head_logits_path"))
    return {
        "status": "unavailable",
        "required": True,
        "reason": reason,
    }


def _p78_recommended_next_phase(status: str) -> str:
    if status == "pass":
        return P78_BROADER_FIXTURE_VALIDATION
    if status == "unavailable":
        return P78_LOGITS_OUTPUT_HOOK_COMPLETION
    return P78_LOGITS_OUTPUT_RESIDUAL_FIX


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
    broader_fixture_report: bool = False,
    wkv_runtime: str | WKVRuntime = WKVRuntime.REFERENCE,
) -> dict[str, Any]:
    del radlads_repo
    selected_wkv_runtime = normalize_wkv_runtime(wkv_runtime)
    if cases == ["all"]:
        cases = None
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
    if selected_wkv_runtime is WKVRuntime.PALLAS:
        probe = build_pallas_runtime_probe(
            requested=selected_wkv_runtime,
            reference_default_preserved=True,
        )
        report = _p82_pallas_runtime_only_report(
            fixture_manifest=fixture_manifest,
            parameter_path=parameter_manifest or parameters,
            fixture_parameter_key=fixture_parameter_key,
            same_run_group_id=same_run_group_id,
            fixture_id=fixture_id,
            parameter_id=parameter_id,
            cases=cases,
            mode=mode,
            layer=layer,
            head=head,
            max_tokens=max_tokens,
            strict_live=strict_live,
            probe=probe,
        )
        write_pallas_runtime_reports(report, out_dir)
        return report
    contexts = _contexts_from_manifest(
        fixture_manifest,
        cases=cases,
        mode=mode,
        layer=layer,
        head=head,
        max_tokens=max_tokens,
    )
    (
        live_sources,
        hook_status,
        config_snapshots,
        full_vs_stepwise_evidence,
    ) = _capture_live_sources(
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
        "radlads_terms_config": config_snapshots.get("radlads_terms"),
        "radlads_direct_config": config_snapshots.get("radlads_direct"),
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
        "p77_full_vs_stepwise_evidence": full_vs_stepwise_evidence,
        "p78_logits_output_evidence": [
            row
            for row in full_vs_stepwise_evidence
            if row.get("comparison") == "full_vs_stepwise_output"
        ],
        "default_wkv_runtime": WKVRuntime.REFERENCE.value,
        "allowed_wkv_runtimes": [runtime.value for runtime in WKVRuntime],
        "wkv_runtime_requested": selected_wkv_runtime.value,
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
    if broader_fixture_report:
        report["p79_broader_fixture_residual_matrix"] = (
            build_p79_broader_fixture_residual_matrix(
                fixture_manifest_data=fixture_manifest_data,
                traces=traces,
                metadata=trace_metadata,
                strict_live=strict_live,
                atol=atol,
                rtol=rtol,
                fixture_manifest_path=fixture_manifest,
                parameters_path=parameter_manifest or parameters,
            )
        )
        report["recommended_next_phase"] = report[
            "p79_broader_fixture_residual_matrix"
        ]["recommended_action"]
    pallas_probe = build_pallas_runtime_probe(
        requested=selected_wkv_runtime,
        reference_default_preserved=True,
    )
    report["p83_pallas_wkv_parity_probe"] = pallas_probe
    report["p82_pallas_runtime_probe"] = pallas_probe
    report["p81_pallas_runtime_probe"] = pallas_probe
    write_live_same_run_reports(report, out_dir)
    return report


def build_p79_broader_fixture_residual_matrix(
    *,
    fixture_manifest_data: Mapping[str, Any],
    traces: Mapping[str, list[dict[str, Any]]],
    metadata: Mapping[str, Any],
    strict_live: bool,
    atol: float,
    rtol: float,
    fixture_manifest_path: Path | str | None = None,
    parameters_path: Path | str | None = None,
    expected_cases: tuple[str, ...] | None = None,
    fixture_expectations: FixtureExpectationMetadata | None = None,
) -> dict[str, Any]:
    expectations = _p79_fixture_expectations(
        expected_cases=expected_cases,
        fixture_expectations=fixture_expectations,
    )
    manifest_cases = set(_manifest_case_names(fixture_manifest_data))
    rows = []
    case_resolutions = _p79_case_resolutions(expectations, manifest_cases)
    for resolution in case_resolutions:
        if resolution.resolution in {"missing", "optional_absent", "deprecated"}:
            rows.append(_p79_unavailable_case_row(resolution, metadata=metadata))
            continue
        case = resolution.resolved_case or resolution.requested_case
        case_traces = {
            side: [row for row in side_rows if row.get("case") == case]
            for side, side_rows in traces.items()
        }
        case_metadata = dict(metadata)
        case_metadata["p77_full_vs_stepwise_evidence"] = [
            row
            for row in metadata.get("p77_full_vs_stepwise_evidence", [])
            if isinstance(row, Mapping) and row.get("case") == case
        ]
        case_metadata["p78_logits_output_evidence"] = [
            row
            for row in metadata.get("p78_logits_output_evidence", [])
            if isinstance(row, Mapping) and row.get("case") == case
        ]
        case_report = compare_live_same_run_traces(
            traces=case_traces,
            metadata=case_metadata,
            strict_live=strict_live,
            atol=atol,
            rtol=rtol,
        )
        rows.append(_p79_case_row(resolution, case_report))
    active_rows = [row for row in rows if row.get("expectation_category") == "active"]
    blocking_rows = [
        row
        for row in rows
        if row.get("expectation_category") in {"active", "alias"}
        and row.get("resolution") == "missing"
    ]
    required_rows = [
        row for row in rows if row.get("expectation_category") in {"active", "alias"}
    ]
    all_found = not blocking_rows
    all_pass = all(
        row["kernel_ready_for_case"] == "yes"
        for row in required_rows
        if row.get("resolution") != "missing"
    )
    recommended = (
        P80_PALLAS_PROTOTYPE
        if all_found and all_pass
        else rows[0]["recommended_action"]
    )
    for row in rows:
        if row.get("expectation_category") in {"deprecated", "optional"}:
            continue
        if row["recommended_action"] != P80_PALLAS_PROTOTYPE:
            recommended = row["recommended_action"]
            break
    cases_found = [
        row["requested_case"]
        for row in rows
        if row.get("resolution") in {"direct", "alias"}
    ]
    cases_missing = [
        row["requested_case"] for row in rows if row.get("resolution") == "missing"
    ]
    cases_pass = sum(row["kernel_ready_for_case"] == "yes" for row in rows)
    cases_unavailable = sum(
        row["kernel_ready_for_case"] == "unavailable" for row in rows
    )
    cases_fail = sum(
        row["kernel_ready_for_case"] not in {"yes", "unavailable", "not_applicable"}
        for row in rows
    )
    cases_by_name = {row["requested_case"]: row for row in rows}
    active_expected_cases = list(expectations.active_expected_cases)
    accepted_aliases = expectations.alias_map()
    deprecated_cases = list(expectations.deprecated_cases)
    optional_cases = list(expectations.optional_cases)
    remaining_missing_cases = [
        row["requested_case"]
        for row in rows
        if row.get("expectation_category") == "active"
        and row.get("resolution") == "missing"
    ]
    return {
        "schema": P79_BROADER_FIXTURE_RESIDUAL_MATRIX_SCHEMA,
        "phase": "P79",
        "fixture_manifest": str(fixture_manifest_path)
        if fixture_manifest_path is not None
        else metadata.get("fixture_manifest_path"),
        "parameters": str(parameters_path)
        if parameters_path is not None
        else metadata.get("parameter_manifest_or_npz_path"),
        "same_run_policy": "strict" if strict_live else "non_strict",
        "tolerances": {"atol": atol, "rtol": rtol},
        "active_expected_cases": active_expected_cases,
        "accepted_aliases": accepted_aliases,
        "deprecated_cases": deprecated_cases,
        "optional_cases": optional_cases,
        "missing_cases": remaining_missing_cases,
        "expected_cases": active_expected_cases,
        "cases_requested": expectations.requested_cases(),
        "cases_found": cases_found,
        "cases_missing": cases_missing,
        "remaining_missing_cases": remaining_missing_cases,
        "same_run_group_id": metadata.get("same_run_group_id"),
        "fixture_id": metadata.get("fixture_id"),
        "parameter_id": metadata.get("parameter_id"),
        "same_run_valid": all(
            row["same_run_valid"]
            for row in required_rows
            if row["fixture_id"] is not None
        ),
        "all_expected_cases_present": all_found,
        "all_active_expected_cases_pass": all_found
        and all(row["kernel_ready_for_case"] == "yes" for row in active_rows),
        "all_expected_cases_pass": all_found and all_pass,
        "kernel_ready": "yes" if all_found and all_pass else "no",
        "recommended_action": recommended,
        "recommended_next_phase": recommended,
        "kernel_ready_scope": "covered_fixture_family",
        "summary": {
            "cases_total": len(rows),
            "cases_pass": cases_pass,
            "cases_fail": cases_fail,
            "cases_unavailable": cases_unavailable,
            "all_expected_cases_pass": all_found and all_pass,
        },
        "case_rows": rows,
        "cases": cases_by_name,
    }


def _p79_fixture_expectations(
    *,
    expected_cases: tuple[str, ...] | None,
    fixture_expectations: FixtureExpectationMetadata | None,
) -> FixtureExpectationMetadata:
    if fixture_expectations is not None:
        return fixture_expectations
    if expected_cases is None:
        return FixtureExpectationMetadata(accepted_aliases=P79_ACCEPTED_ALIASES)
    aliases = {
        requested: canonical
        for requested, canonical in P79_ACCEPTED_ALIASES.items()
        if requested in expected_cases
    }
    active = tuple(case for case in expected_cases if case not in aliases)
    return FixtureExpectationMetadata(
        active_expected_cases=active,
        accepted_aliases=aliases,
    )


def _p79_case_resolutions(
    expectations: FixtureExpectationMetadata, manifest_cases: set[str]
) -> list[FixtureCaseResolution]:
    rows: list[FixtureCaseResolution] = []
    for case in expectations.active_expected_cases:
        rows.append(
            FixtureCaseResolution(
                requested_case=case,
                canonical_case=case,
                resolved_case=case if case in manifest_cases else None,
                resolution="direct" if case in manifest_cases else "missing",
                category="active",
            )
        )
    for requested, canonical in expectations.alias_map().items():
        rows.append(
            FixtureCaseResolution(
                requested_case=requested,
                canonical_case=canonical,
                resolved_case=canonical if canonical in manifest_cases else None,
                resolution="alias" if canonical in manifest_cases else "missing",
                category="alias",
            )
        )
    for case in expectations.deprecated_cases:
        rows.append(
            FixtureCaseResolution(
                requested_case=case,
                canonical_case=case,
                resolved_case=case if case in manifest_cases else None,
                resolution="deprecated",
                category="deprecated",
            )
        )
    for case in expectations.optional_cases:
        present = case in manifest_cases
        rows.append(
            FixtureCaseResolution(
                requested_case=case,
                canonical_case=case,
                resolved_case=case if present else None,
                resolution="direct" if present else "optional_absent",
                category="optional",
            )
        )
    return rows


def _p79_unavailable_case_row(
    resolution: FixtureCaseResolution, *, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    reason = (
        "optional_case_absent"
        if resolution.resolution == "optional_absent"
        else "deprecated_case"
        if resolution.resolution == "deprecated"
        else "fixture_case_not_found"
    )
    kernel_ready = (
        "not_applicable" if resolution.resolution != "missing" else "unavailable"
    )
    blocking = ["fixture_case_not_found"] if resolution.resolution == "missing" else []
    return {
        "case": resolution.requested_case,
        "requested_case": resolution.requested_case,
        "canonical_case": resolution.canonical_case,
        "resolved_case": resolution.resolved_case,
        "resolution": resolution.resolution,
        "expectation_category": resolution.category,
        "fixture_id": metadata.get("fixture_id"),
        "parameter_id": metadata.get("parameter_id"),
        "same_run_group_id": metadata.get("same_run_group_id"),
        "lanes_checked": [BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE],
        "same_run_valid": False,
        "mixed_artifact_lineage_used": False,
        "synthetic_fallback_used": False,
        "state_after": "unavailable",
        "exported_state": "unavailable",
        "full_vs_stepwise": "unavailable",
        "logits_output": "unavailable",
        "kernel_ready_for_case": kernel_ready,
        "blocking_gates": blocking,
        "warning_gates": [],
        "recommended_action": P80_LINEAGE_REPAIR
        if resolution.resolution == "missing"
        else P80_PALLAS_PROTOTYPE,
        "unavailable_reason": reason,
        "lane_details": _p79_unavailable_lane_details(),
    }


def _p79_case_row(
    resolution: FixtureCaseResolution, report: Mapping[str, Any]
) -> dict[str, Any]:
    gate = report.get("p75_residual_impact_gate", {})
    output_gates = gate.get("output_gates", {})
    blocking = list(gate.get("blocking_gates", []))
    row = {
        "case": resolution.requested_case,
        "requested_case": resolution.requested_case,
        "canonical_case": resolution.canonical_case,
        "resolved_case": resolution.resolved_case,
        "resolution": resolution.resolution,
        "expectation_category": resolution.category,
        "evidence_source_case": resolution.resolved_case,
        "resolved_by_alias": resolution.resolution == "alias",
        "duplicates_fixture_evidence": False,
        "fixture_id": report.get("fixture_id"),
        "parameter_id": report.get("parameter_id"),
        "same_run_group_id": report.get("same_run_group_id"),
        "lanes_checked": [BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE],
        "same_run_valid": bool(report.get("same_run_valid")),
        "mixed_artifact_lineage_used": bool(report.get("mixed_artifact_lineage_used")),
        "synthetic_fallback_used": bool(report.get("synthetic_fallback_used")),
        "state_after": _p79_gate_status(output_gates, "state_after"),
        "exported_state": _p79_gate_status(output_gates, "exported_state"),
        "full_vs_stepwise": _p79_gate_status(output_gates, "full_vs_stepwise"),
        "logits_output": _p79_gate_status(output_gates, "logits_output"),
        "kernel_ready_for_case": gate.get("kernel_ready"),
        "blocking_gates": blocking,
        "warning_gates": list(gate.get("warning_gates", [])),
        "lane_details": _p79_lane_details(report, gate),
    }
    row["recommended_action"] = _p79_recommended_action(row)
    return row


def _p79_gate_status(output_gates: Mapping[str, Any], gate: str) -> str:
    item = output_gates.get(gate, {})
    if isinstance(item, Mapping):
        return str(item.get("status", "unavailable"))
    return "unavailable"


def _p79_unavailable_lane_details() -> dict[str, dict[str, Any]]:
    return {
        lane: {
            "valid": False,
            "first_differing_stage": None,
            "state_after": "unavailable",
            "exported_state": "unavailable",
            "full_vs_stepwise": "unavailable",
            "logits_output": "unavailable",
        }
        for lane in (BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE)
    }


def _p79_lane_details(
    report: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    lane_details = {}
    p77 = report.get("p77_full_vs_stepwise_residual", {})
    p78 = report.get("p78_logits_output_residual", {})
    for lane in (BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE):
        lane_gate = gate.get("lane_comparisons", {}).get(lane, {})
        lane_details[lane] = {
            "valid": bool(lane_gate.get("valid")),
            "first_differing_stage": lane_gate.get("first_differing_stage"),
            "state_after": _p79_lane_measurement_status(gate, lane, "state_after_live"),
            "exported_state": _p79_lane_measurement_status(
                gate, lane, "state_after_exported"
            ),
            "full_vs_stepwise": _p79_nested_lane_status(p77, lane),
            "logits_output": _p79_nested_lane_status(p78, lane),
        }
    return lane_details


def _p79_lane_measurement_status(gate: Mapping[str, Any], lane: str, stage: str) -> str:
    item = (
        gate.get("residuals", {}).get(lane, {}).get("measurements", {}).get(stage, {})
    )
    if isinstance(item, Mapping):
        return str(item.get("status", "unavailable"))
    return "unavailable"


def _p79_nested_lane_status(report: Any, lane: str) -> str:
    if isinstance(report, Mapping):
        lane_status = report.get("lane_status", report.get("inter_side_parity", {}))
        item = lane_status.get(lane, {}) if isinstance(lane_status, Mapping) else {}
        if isinstance(item, Mapping):
            return str(item.get("status", "unavailable"))
    return "unavailable"


def _p79_recommended_action(row: Mapping[str, Any]) -> str:
    if (
        not row.get("same_run_valid")
        or row.get("mixed_artifact_lineage_used")
        or row.get("synthetic_fallback_used")
        or "fixture_case_not_found" in row.get("blocking_gates", [])
    ):
        return P80_LINEAGE_REPAIR
    if row.get("state_after") != "pass":
        return P80_STATE_AFTER_FIX
    if row.get("exported_state") != "pass":
        return P80_EXPORTED_STATE_FIX
    if row.get("full_vs_stepwise") != "pass":
        return P80_FULL_VS_STEPWISE_FIX
    if row.get("logits_output") != "pass":
        return P80_LOGITS_OUTPUT_FIX
    if row.get("kernel_ready_for_case") == "yes":
        return P80_PALLAS_PROTOTYPE
    return P80_KERNEL_GATE_HARDENING


def _p75_lane_residuals(
    report: Mapping[str, Any], *, lane: str, pair: str
) -> dict[str, Any]:
    rows_by_stage = {
        stage: [
            row
            for row in report.get("rows", [])
            if row.get("balance_state_lane") == lane and row.get("stage") == stage
        ]
        for stage in P75_RESIDUAL_STAGES
    }
    measurements = {
        stage: _p75_stage_measurement(stage_rows, pair=pair)
        for stage, stage_rows in rows_by_stage.items()
    }
    blocking = [
        stage for stage, item in measurements.items() if item["severity"] == "blocking"
    ]
    warnings = [
        stage for stage, item in measurements.items() if item["severity"] == "warning"
    ]
    comparable = [
        item
        for item in measurements.values()
        if item["status"] not in {"unavailable", "not_applicable"}
    ]
    max_abs_values = [
        item["max_abs"]
        for item in comparable
        if isinstance(item.get("max_abs"), int | float)
    ]
    return {
        "lane": lane,
        "pair": pair,
        "status": "fail" if blocking else "warning" if warnings else "pass",
        "blocking": blocking,
        "warnings": warnings,
        "max_abs_summary": {
            "max": max(max_abs_values) if max_abs_values else None,
            "measured_stage_count": len(comparable),
        },
        "measurements": measurements,
    }


def _p75_stage_measurement(
    stage_rows: list[Mapping[str, Any]], *, pair: str
) -> dict[str, Any]:
    if not stage_rows:
        return _p75_unavailable_measurement("missing_comparable_stage")
    comparisons = [row.get(pair, {}) for row in stage_rows]
    available = [
        (row, comparison)
        for row, comparison in zip(stage_rows, comparisons, strict=True)
        if comparison.get("status") != "unavailable"
    ]
    if not available:
        reasons = sorted(
            {
                str(row.get("reason") or "missing_lane_pair")
                for row in stage_rows
                if row.get("reason") is not None
            }
        )
        reason = ";".join(reasons) if reasons else "missing_lane_pair"
        if all(row.get("capture_kind") == "not_applicable" for row in stage_rows):
            return {
                **_p75_unavailable_measurement(reason),
                "status": "not_applicable",
                "severity": "non_blocking",
            }
        return _p75_unavailable_measurement(reason)
    statuses = [comparison.get("status") for _, comparison in available]
    finite = all(bool(comparison.get("finite_both")) for _, comparison in available)
    shape_match = all(
        bool(comparison.get("shape_match")) for _, comparison in available
    )
    dtype_values = sorted(
        {
            str(row.get("radlads_capture_kind"))
            for row, _ in available
            if row.get("radlads_capture_kind") is not None
        }
        | {
            str(row.get("qrwkv_off_capture_kind"))
            for row, _ in available
            if row.get("qrwkv_off_capture_kind") is not None
        }
        | {
            str(row.get("qrwkv_experimental_capture_kind"))
            for row, _ in available
            if row.get("qrwkv_experimental_capture_kind") is not None
        }
    )
    max_abs_values = [
        float(comparison["max_abs_error"])
        for _, comparison in available
        if comparison.get("max_abs_error") is not None
    ]
    mean_abs_values = [
        float(comparison["mean_abs_error"])
        for _, comparison in available
        if comparison.get("mean_abs_error") is not None
    ]
    relative_values = [
        float(comparison["max_relative_error"])
        for _, comparison in available
        if comparison.get("max_relative_error") is not None
    ]
    max_abs = max(max_abs_values) if max_abs_values else None
    mean_abs = max(mean_abs_values) if mean_abs_values else None
    rms = _p75_rms(stage_rows, pair=pair)
    if not shape_match:
        status = "fail"
        severity = "blocking"
        reason = "shape_mismatch"
    elif not finite or "non_finite" in statuses:
        status = "fail"
        severity = "blocking"
        reason = "non_finite"
    elif all(status == "pass" for status in statuses):
        status = "pass"
        severity = "non_blocking"
        reason = "allclose"
    elif max_abs is not None and max_abs <= P75_WARNING_MAX_ABS:
        status = "pass"
        severity = "non_blocking"
        reason = "bounded_tiny_residual"
    else:
        status = "fail"
        severity = "blocking"
        reason = "residual_exceeds_tolerance"
    return {
        "status": status,
        "severity": severity,
        "reason": reason,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "relative_max": max(relative_values) if relative_values else None,
        "allclose_atol": None,
        "allclose_rtol": None,
        "finite": finite,
        "shape_match": shape_match,
        "dtype": ",".join(dtype_values) if dtype_values else None,
        "row_count": len(stage_rows),
        "comparable_row_count": len(available),
    }


def _p75_unavailable_measurement(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "severity": "non_blocking",
        "reason": reason,
        "max_abs": None,
        "mean_abs": None,
        "rms": None,
        "relative_max": None,
        "allclose_atol": None,
        "allclose_rtol": None,
        "finite": None,
        "shape_match": None,
        "dtype": None,
        "row_count": 0,
        "comparable_row_count": 0,
    }


def _p75_rms(stage_rows: list[Mapping[str, Any]], *, pair: str) -> float | None:
    values = []
    for row in stage_rows:
        comparison = row.get(pair, {})
        mean_abs = comparison.get("mean_abs_error")
        if mean_abs is not None:
            values.append(float(mean_abs) ** 2)
    if not values:
        return None
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64))))


def _p75_output_gates(
    report: Mapping[str, Any], *, residuals: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    explicit = report.get("p75_output_gates")
    if isinstance(explicit, Mapping):
        return {
            gate: _p75_normalize_output_gate(explicit.get(gate), gate)
            for gate in P75_REQUIRED_OUTPUT_GATES
        }
    terms_state = residuals[BALANCE_STATE_TERMS_LANE]["measurements"][
        "state_after_live"
    ]
    direct_state = residuals[DIRECT_BALANCE_STATE_LANE]["measurements"][
        "state_after_live"
    ]
    exported_terms = residuals[BALANCE_STATE_TERMS_LANE]["measurements"][
        "state_after_exported"
    ]
    exported_direct = residuals[DIRECT_BALANCE_STATE_LANE]["measurements"][
        "state_after_exported"
    ]
    return {
        "state_after": _p75_join_lane_gate(
            "state_after", (terms_state, direct_state), required=True
        ),
        "exported_state": _p75_join_lane_gate(
            "exported_state", (exported_terms, exported_direct), required=True
        ),
        "full_vs_stepwise": _p77_gate_from_status(
            str(
                (report.get("p77_full_vs_stepwise_residual") or {}).get(
                    "status", "unavailable"
                )
            )
        ),
        "logits_output": _p78_gate_from_report(
            report.get("p78_logits_output_residual")
        ),
    }


def _p75_normalize_output_gate(value: Any, gate: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        status = str(value.get("status", "unavailable"))
        reason = value.get("reason")
        return {
            "status": status,
            "required": bool(value.get("required", True)),
            "reason": reason if reason is not None else status,
        }
    return {
        "status": "unavailable",
        "required": True,
        "reason": f"missing_evidence:{gate}",
    }


def _p75_join_lane_gate(
    gate: str, measurements: tuple[Mapping[str, Any], ...], *, required: bool
) -> dict[str, Any]:
    if any(item.get("severity") == "blocking" for item in measurements):
        return {
            "status": "fail",
            "required": required,
            "reason": f"{gate}:lane_residual_blocking",
        }
    if all(item.get("status") == "pass" for item in measurements):
        return {"status": "pass", "required": required, "reason": "all_lanes_pass"}
    return {
        "status": "unavailable",
        "required": required,
        "reason": f"missing_evidence:{gate}",
    }


def _p75_blocking_gates(
    *,
    same_run_valid: bool,
    lane_comparisons_valid: bool,
    residuals: Mapping[str, Mapping[str, Any]],
    output_gates: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers = []
    if not same_run_valid:
        blockers.append("same_run_valid:false")
    if not lane_comparisons_valid:
        blockers.append("lane_comparison_regression")
    for lane, lane_report in residuals.items():
        for stage in lane_report.get("blocking", []):
            blockers.append(f"residual_blocking:{lane}:{stage}")
    for gate, item in output_gates.items():
        if not item.get("required", True):
            continue
        if item.get("status") == "fail":
            blockers.append(f"output_gate_failed:{gate}")
        elif item.get("status") == "unavailable":
            reason = str(item.get("reason", "missing_evidence"))
            if reason.startswith("missing_evidence"):
                blockers.append(f"missing_evidence:{gate}")
            else:
                blockers.append(f"output_gate_unavailable:{gate}:{reason}")
    return blockers


def _p75_warning_gates(
    *,
    residuals: Mapping[str, Mapping[str, Any]],
    output_gates: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    del output_gates
    warnings = []
    for lane, lane_report in residuals.items():
        for stage in lane_report.get("warnings", []):
            warnings.append(f"residual_warning:{lane}:{stage}")
    return warnings


def _p75_recommended_next_phase(
    *,
    blocking_gates: list[str],
    warning_gates: list[str],
    report: Mapping[str, Any] | None = None,
) -> str:
    del warning_gates
    for blocker in blocking_gates:
        if "state_after" in blocker:
            return P77_STATE_AFTER_FIX
    for blocker in blocking_gates:
        if "lane_comparison_regression" in blocker:
            return P77_LANE_LAYOUT_FIX
    for blocker in blocking_gates:
        if "exported_state" in blocker:
            return P77_STATE_EXPORT_FIX
    for blocker in blocking_gates:
        if "full_vs_stepwise" in blocker:
            return P77_FULL_VS_STEPWISE_FIX
    for blocker in blocking_gates:
        if "logits_output" in blocker:
            return (
                P78_LOGITS_OUTPUT_HOOK_COMPLETION
                if "unavailable" in blocker or "missing" in blocker
                else P78_LOGITS_OUTPUT_RESIDUAL_FIX
            )
    if blocking_gates:
        return P78_KERNEL_GATE_HARDENING
    if isinstance(report, Mapping) and isinstance(
        report.get("p78_logits_output_residual"), Mapping
    ):
        return P78_BROADER_FIXTURE_VALIDATION
    return P77_BROADER_FIXTURE_VALIDATION


def _p76_blocking_gates(
    *,
    surface_status: Mapping[str, Mapping[str, Any]],
    lane_status: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers = []
    for surface, item in surface_status.items():
        if item.get("status") != "pass":
            blockers.append(f"surface:{surface}:{item.get('reason')}")
    for lane, item in lane_status.items():
        if item.get("status") != "pass":
            blockers.append(f"lane:{lane}:{item.get('reason')}")
    return blockers


def _p76_recommended_next_phase(
    *,
    status: str,
    surface_status: Mapping[str, Mapping[str, Any]],
    lane_status: Mapping[str, Mapping[str, Any]],
) -> str:
    if status == "pass":
        return P77_FULL_VS_STEPWISE_FIX
    for item in surface_status.values():
        if item.get("status") != "pass":
            return P77_STATE_EXPORT_FIX
    for item in lane_status.values():
        if item.get("status") != "pass":
            return P77_STATE_EXPORT_FIX
    return P77_BROADER_FIXTURE_VALIDATION


def _p76_intra_side_rows(
    *, traces: Mapping[str, list[dict[str, Any]]], atol: float, rtol: float
) -> list[dict[str, Any]]:
    rows = []
    for side in SIDES:
        by_key = {_trace_key(row): row for row in traces.get(side, [])}
        for exported in traces.get(side, []):
            if (
                exported.get("stage") != "state_after_exported"
                or exported.get("capture_kind") != "exported_state"
            ):
                continue
            live_key = (*_trace_key(exported)[:-1], "state_after_live")
            live = by_key.get(live_key)
            stats = _compare_pair(live, exported, atol=atol, rtol=rtol)
            rows.append(
                {
                    "side": side,
                    "lane": exported.get("balance_state_lane"),
                    "case": exported.get("case"),
                    "mode": exported.get("mode"),
                    "layer": exported.get("layer"),
                    "token": exported.get("token"),
                    "head": exported.get("head"),
                    "status": stats["status"],
                    "max_abs_error": stats["max_abs_error"],
                    "mean_abs_error": stats["mean_abs_error"],
                    "export_path": exported.get("export_path"),
                    "source_stage_name": exported.get("source_stage_name"),
                }
            )
    return sorted(rows, key=_p76_row_sort_key)


def _p76_import_roundtrip_rows(
    traces: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for side in SIDES:
        for exported in traces.get(side, []):
            if (
                exported.get("stage") != "state_after_exported"
                or exported.get("capture_kind") != "exported_state"
            ):
                continue
            status = exported.get("import_roundtrip_status")
            reason = exported.get("import_roundtrip_reason")
            if status is None:
                status = "unavailable"
                reason = "missing_import_path"
            rows.append(
                {
                    "side": side,
                    "lane": exported.get("balance_state_lane"),
                    "case": exported.get("case"),
                    "mode": exported.get("mode"),
                    "layer": exported.get("layer"),
                    "token": exported.get("token"),
                    "head": exported.get("head"),
                    "status": status,
                    "reason": reason,
                    "export_path": exported.get("export_path"),
                    "import_path": exported.get("import_path"),
                }
            )
    return sorted(rows, key=_p76_row_sort_key)


def _p76_inter_side_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in comparison_rows:
        if row.get("stage") != "state_after_exported":
            continue
        lane = row.get("balance_state_lane")
        pair = _lane_primary_pair(row)
        if pair is None:
            continue
        comparison = row.get(pair, {})
        rows.append(
            {
                "lane": lane,
                "pair": pair,
                "case": row.get("case"),
                "mode": row.get("mode"),
                "layer": row.get("layer"),
                "token": row.get("token"),
                "head": row.get("head"),
                "status": comparison.get("status"),
                "max_abs_error": comparison.get("max_abs_error"),
                "mean_abs_error": comparison.get("mean_abs_error"),
                "radlads_capture_kind": row.get("radlads_capture_kind"),
                "qrwkv_off_capture_kind": row.get("qrwkv_off_capture_kind"),
                "qrwkv_experimental_capture_kind": row.get(
                    "qrwkv_experimental_capture_kind"
                ),
            }
        )
    return sorted(rows, key=_p76_row_sort_key)


def _p76_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("side", "")),
        str(row.get("lane", "")),
        str(row.get("case", "")),
        "" if row.get("mode") is None else str(row.get("mode")),
        -1 if row.get("layer") is None else int(row.get("layer")),
        -1 if row.get("token") is None else int(row.get("token")),
        -1 if row.get("head") is None else int(row.get("head")),
    )


def write_live_same_run_trace(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=_entry_sort_key):
            handle.write(json.dumps(_jsonable(entry), sort_keys=True) + "\n")


def _p82_pallas_runtime_only_report(
    *,
    fixture_manifest: Path,
    parameter_path: Path | None,
    fixture_parameter_key: str | None,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
    cases: list[str] | None,
    mode: str,
    layer: int | None,
    head: int | None,
    max_tokens: int | None,
    strict_live: bool,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": LIVE_SAME_RUN_REPORT_SCHEMA,
        "phase": "P83",
        "same_run_group_id": same_run_group_id,
        "fixture_id": fixture_id,
        "parameter_id": parameter_id,
        "fixture_manifest_path": str(fixture_manifest),
        "parameter_manifest_or_npz_path": str(parameter_path)
        if parameter_path is not None
        else None,
        "fixture_parameter_key": fixture_parameter_key,
        "strict_live": strict_live,
        "cases": cases,
        "mode": mode,
        "layer": layer,
        "head": head,
        "max_tokens": max_tokens,
        "same_run_valid": False,
        "same_run_validity_reason": "pallas_runtime_probe_only_no_reference_capture",
        "first_divergent_stage": None,
        "live_rows_captured_radlads": 0,
        "live_rows_captured_qrwkv_off": 0,
        "live_rows_captured_qrwkv_experimental": 0,
        "unavailable_minimum_stages": [],
        "unavailable_rows": [],
        "pallas_requested_reference_trace_contamination": False,
        "fail_closed_before_capture": probe.get("prototype_status") != "pass",
        "reference_trace_capture_skipped": True,
        "default_wkv_runtime": WKVRuntime.REFERENCE.value,
        "allowed_wkv_runtimes": [runtime.value for runtime in WKVRuntime],
        "wkv_runtime_requested": WKVRuntime.PALLAS.value,
        "wkv_runtime_effective": probe.get("wkv_runtime_effective"),
        "p83_pallas_wkv_parity_probe": dict(probe),
        "p82_pallas_runtime_probe": dict(probe),
        "p81_pallas_runtime_probe": dict(probe),
        "pallas_runtime_status": _p82_pallas_runtime_status(probe),
        "kernel_parity_claimed": probe.get("kernel_parity_claimed", False),
        "recommended_next_phase": probe.get(
            "recommended_next_phase", P83_PALLAS_RUNTIME_SCAFFOLD_COMPLETION
        ),
    }


def _p82_pallas_runtime_status(probe: Mapping[str, Any]) -> str:
    status = probe.get("prototype_status")
    if status == "pass":
        return "prototype_pass"
    if status == "unavailable":
        return "unavailable"
    return "failed"


def write_pallas_runtime_reports(report: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_live_trace_capture_artifacts(out_dir)
    probe = report.get("p82_pallas_runtime_probe")
    if not isinstance(probe, Mapping):
        raise ValueError("P82 pallas runtime report requires p82_pallas_runtime_probe")
    (out_dir / "live_same_run_update_ingredients_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "pallas_runtime_probe.json").write_text(
        json.dumps(_jsonable(probe), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "pallas_reference_parity_probe.json").write_text(
        json.dumps(_jsonable(probe), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P83_PALLAS_REFERENCE_PARITY_REPORT.md").write_text(
        _p83_pallas_reference_parity_report_markdown(report, probe),
        encoding="utf-8",
    )
    (out_dir / "P82_PALLAS_RUNTIME_SCAFFOLD_COMPLETION_REPORT.md").write_text(
        _p82_pallas_runtime_report_markdown(report, probe),
        encoding="utf-8",
    )
    (out_dir / "P81_PALLAS_PROTOTYPE_REPORT.md").write_text(
        _p81_pallas_prototype_report_markdown(report, probe),
        encoding="utf-8",
    )
    (out_dir / "P82_FIX_NOTE.md").write_text(
        _p82_fix_note_markdown(probe),
        encoding="utf-8",
    )
    (out_dir / "P68_DECISION.md").write_text(
        "# P68 Decision\n\n"
        "P83 ran an opt-in Pallas reference parity probe and skipped reference live "
        "trace capture to avoid Pallas-requested reference-trace contamination.\n\n"
        f"- recommended_next_phase: `{report.get('recommended_next_phase')}`\n",
        encoding="utf-8",
    )
    (out_dir / "P68_RESULTS.md").write_text(
        "# P68 Results\n\n"
        "P83 Pallas-requested run is parity-probe-only. Reference live trace capture "
        "was skipped, so no reference live rows are reported for this run.\n\n"
        "- pallas_requested_reference_trace_contamination: "
        f"`{report.get('pallas_requested_reference_trace_contamination')}`\n"
        "- reference_trace_capture_skipped: "
        f"`{report.get('reference_trace_capture_skipped')}`\n",
        encoding="utf-8",
    )
    (out_dir / "LIVE_SAME_RUN_VALIDITY.md").write_text(
        "# Live Same-Run Validity\n\n"
        "Not applicable for the P83 Pallas parity-probe-only run; no reference live "
        "trace capture was performed.\n",
        encoding="utf-8",
    )
    (out_dir / "STAGE_AVAILABILITY_MATRIX.md").write_text(
        "# Stage Availability Matrix\n\n"
        "Not applicable for the P83 Pallas parity-probe-only run.\n",
        encoding="utf-8",
    )
    (out_dir / "FIRST_DIFFERING_INGREDIENT.md").write_text(
        "# First Differing Ingredient\n\n"
        "Not applicable for the P83 Pallas parity-probe-only run.\n",
        encoding="utf-8",
    )
    (out_dir / "P75_KERNEL_READINESS_DECISION.md").write_text(
        "# P75 Kernel Readiness Decision\n\n"
        "Reference kernel readiness is preserved from the previous covered "
        "fixture-family run. P83 does not mark full Pallas kernel-ready; it only "
        "records tiny one-step WKV update parity.\n\n"
        f"- pallas_runtime_status: `{report.get('pallas_runtime_status')}`\n"
        f"- kernel_parity_claimed: `{probe.get('kernel_parity_claimed')}`\n",
        encoding="utf-8",
    )
    blocker_path = out_dir / "P82_BLOCKER_REPORT.md"
    if probe.get("prototype_status") != "pass":
        blocker_path.write_text(
            _p82_blocker_report_markdown(probe),
            encoding="utf-8",
        )
    elif blocker_path.exists():
        blocker_path.unlink()
    p81_blocker = out_dir / "P81_BLOCKER_REPORT.md"
    if p81_blocker.exists():
        p81_blocker.unlink()


def _remove_stale_live_trace_capture_artifacts(out_dir: Path) -> None:
    for pattern in ("live_trace_*.jsonl", "live_same_run_trace_metadata.json"):
        for path in out_dir.glob(pattern):
            path.unlink()


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
    (out_dir / "direct_balance_lane_comparison.json").write_text(
        json.dumps(
            _jsonable(_p74_direct_lane_comparison(report)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    p75_gate = report.get("p75_residual_impact_gate") or build_p75_residual_impact_gate(
        report
    )
    (out_dir / "residual_impact_gate.json").write_text(
        json.dumps(_jsonable(p75_gate), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    p76_report = report.get("p76_state_export_import_residual")
    if isinstance(p76_report, Mapping):
        (out_dir / "state_export_import_residual.json").write_text(
            json.dumps(_jsonable(p76_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    p77_report = report.get("p77_full_vs_stepwise_residual")
    if isinstance(p77_report, Mapping):
        (out_dir / "full_vs_stepwise_residual.json").write_text(
            json.dumps(_jsonable(p77_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    p78_report = report.get("p78_logits_output_residual")
    if isinstance(p78_report, Mapping):
        (out_dir / "logits_output_residual.json").write_text(
            json.dumps(_jsonable(p78_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    p79_report = report.get("p79_broader_fixture_residual_matrix")
    if isinstance(p79_report, Mapping):
        (out_dir / "broader_fixture_residual_matrix.json").write_text(
            json.dumps(_jsonable(p79_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    p82_probe = report.get("p82_pallas_runtime_probe")
    p83_probe = report.get("p83_pallas_wkv_parity_probe")
    p81_probe = report.get("p81_pallas_runtime_probe")
    pallas_probe = (
        p83_probe
        if isinstance(p83_probe, Mapping)
        else p82_probe
        if isinstance(p82_probe, Mapping)
        else p81_probe
    )
    if isinstance(pallas_probe, Mapping):
        (out_dir / "pallas_runtime_probe.json").write_text(
            json.dumps(_jsonable(pallas_probe), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "pallas_reference_parity_probe.json").write_text(
            json.dumps(_jsonable(pallas_probe), indent=2, sort_keys=True) + "\n",
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
    (out_dir / "P74_DIRECT_BALANCE_LANE_REPORT.md").write_text(
        _p74_direct_lane_markdown(report), encoding="utf-8"
    )
    (out_dir / "P74_FIX_NOTE.md").write_text(
        _p74_fix_note_markdown(report), encoding="utf-8"
    )
    (out_dir / "P75_RESIDUAL_IMPACT_GATE.md").write_text(
        _p75_residual_gate_markdown(report, p75_gate), encoding="utf-8"
    )
    (out_dir / "P75_KERNEL_READINESS_DECISION.md").write_text(
        _p75_kernel_decision_markdown(report, p75_gate), encoding="utf-8"
    )
    if p75_gate.get("blocking_gates"):
        (out_dir / "P75_BLOCKER_REPORT.md").write_text(
            _p75_blocker_report_markdown(p75_gate), encoding="utf-8"
        )
    else:
        stale_blocker = out_dir / "P75_BLOCKER_REPORT.md"
        if stale_blocker.exists():
            stale_blocker.unlink()
    (out_dir / "P75_FIX_NOTE.md").write_text(
        _p75_fix_note_markdown(report, p75_gate), encoding="utf-8"
    )
    if isinstance(p76_report, Mapping):
        (out_dir / "P76_STATE_EXPORT_IMPORT_REPORT.md").write_text(
            _p76_state_export_import_markdown(p76_report, p75_gate),
            encoding="utf-8",
        )
        if p76_report.get("status") != "pass":
            (out_dir / "P76_BLOCKER_REPORT.md").write_text(
                _p76_blocker_report_markdown(p76_report),
                encoding="utf-8",
            )
        else:
            stale_blocker = out_dir / "P76_BLOCKER_REPORT.md"
            if stale_blocker.exists():
                stale_blocker.unlink()
    if isinstance(p77_report, Mapping):
        (out_dir / "P77_FULL_VS_STEPWISE_REPORT.md").write_text(
            _p77_full_vs_stepwise_markdown(p77_report, p75_gate),
            encoding="utf-8",
        )
        (out_dir / "P77_FIX_NOTE.md").write_text(
            _p77_fix_note_markdown(p77_report, p75_gate),
            encoding="utf-8",
        )
        if p77_report.get("status") != "pass":
            (out_dir / "P77_BLOCKER_REPORT.md").write_text(
                _p77_blocker_report_markdown(p77_report),
                encoding="utf-8",
            )
        else:
            stale_blocker = out_dir / "P77_BLOCKER_REPORT.md"
            if stale_blocker.exists():
                stale_blocker.unlink()
    if isinstance(p78_report, Mapping):
        (out_dir / "P78_LOGITS_OUTPUT_REPORT.md").write_text(
            _p78_logits_output_markdown(p78_report, p75_gate),
            encoding="utf-8",
        )
        (out_dir / "P78_FIX_NOTE.md").write_text(
            _p78_fix_note_markdown(p78_report, p75_gate),
            encoding="utf-8",
        )
    if isinstance(p79_report, Mapping):
        matrix_md = _p79_matrix_markdown(p79_report)
        (out_dir / "broader_fixture_residual_matrix.md").write_text(
            matrix_md,
            encoding="utf-8",
        )
        (out_dir / "P79_BROADER_FIXTURE_VALIDATION_REPORT.md").write_text(
            _p79_validation_report_markdown(p79_report),
            encoding="utf-8",
        )
        blockers = [
            row
            for row in _p79_case_rows(p79_report)
            if row.get("kernel_ready_for_case") != "yes"
        ]
        blocker_path = out_dir / "P79_BLOCKER_REPORT.md"
        if blockers:
            blocker_path.write_text(
                _p79_blocker_report_markdown(p79_report),
                encoding="utf-8",
            )
        elif blocker_path.exists():
            blocker_path.unlink()
        (out_dir / "P79_FIX_NOTE.md").write_text(
            _p79_fix_note_markdown(p79_report),
            encoding="utf-8",
        )
        p80_resolution = _p80_fixture_lineage_resolution(p79_report)
        (out_dir / "fixture_lineage_resolution.json").write_text(
            json.dumps(_jsonable(p80_resolution), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "P80_FIXTURE_LINEAGE_REPAIR_REPORT.md").write_text(
            _p80_fixture_lineage_report_markdown(p80_resolution, p79_report),
            encoding="utf-8",
        )
        (out_dir / "P80_FIX_NOTE.md").write_text(
            _p80_fix_note_markdown(p80_resolution),
            encoding="utf-8",
        )
        blocker_path = out_dir / "P80_BLOCKER_REPORT.md"
        if p80_resolution.get("remaining_missing_cases"):
            blocker_path.write_text(
                _p80_blocker_report_markdown(p80_resolution),
                encoding="utf-8",
            )
        elif blocker_path.exists():
            blocker_path.unlink()
    if isinstance(pallas_probe, Mapping):
        (out_dir / "P81_PALLAS_PROTOTYPE_REPORT.md").write_text(
            _p81_pallas_prototype_report_markdown(report, pallas_probe),
            encoding="utf-8",
        )
        (out_dir / "P82_PALLAS_RUNTIME_SCAFFOLD_COMPLETION_REPORT.md").write_text(
            _p82_pallas_runtime_report_markdown(report, pallas_probe),
            encoding="utf-8",
        )
        (out_dir / "P83_PALLAS_REFERENCE_PARITY_REPORT.md").write_text(
            _p83_pallas_reference_parity_report_markdown(report, pallas_probe),
            encoding="utf-8",
        )
        (out_dir / "P81_FIX_NOTE.md").write_text(
            _p81_fix_note_markdown(pallas_probe),
            encoding="utf-8",
        )
        blocker_path = out_dir / "P81_BLOCKER_REPORT.md"
        if pallas_probe.get("prototype_status") == "unavailable":
            blocker_path.write_text(
                _p81_blocker_report_markdown(pallas_probe),
                encoding="utf-8",
            )
        elif blocker_path.exists():
            blocker_path.unlink()
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


def _p81_pallas_prototype_report_markdown(
    report: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> str:
    requested = probe.get("wkv_runtime_requested")
    p79 = report.get("p79_broader_fixture_residual_matrix")
    p80_resolution = (
        _p80_fixture_lineage_resolution(p79) if isinstance(p79, Mapping) else {}
    )
    if isinstance(p79, Mapping) and p79.get("all_expected_cases_pass") is True:
        p80_status = "pass"
        alias_lineage = "pass"
    elif isinstance(p79, Mapping) and p79.get("same_run_valid") is True:
        p80_missing = p80_resolution.get("remaining_missing_cases", [])
        p80_status = "blocked"
        alias_lineage = f"remaining_missing_cases={p80_missing}"
    else:
        p80_status = "not_rerun_by_p81_probe"
        alias_lineage = "preserved_from_p80_reference_path"
    return "\n".join(
        [
            "# P81 Pallas Prototype Report",
            "",
            "## Runtime Selector",
            f"- default runtime: `{probe.get('default_runtime')}`",
            f"- allowed runtimes: `{probe.get('allowed_runtimes')}`",
            "- reference default preserved: "
            f"`{probe.get('reference_default_preserved')}`",
            "- CLI/config path: `--wkv-runtime reference|pallas`, `wkv_runtime`",
            "",
            "## Pallas Path",
            f"- pallas requested: `{requested == WKVRuntime.PALLAS.value}`",
            f"- pallas available: `{probe.get('pallas_available')}`",
            f"- pallas effective runtime: `{probe.get('wkv_runtime_effective')}`",
            f"- fallback used: `{probe.get('fallback_used')}`",
            f"- fallback reason: `{probe.get('fallback_reason')}`",
            "",
            "## Prototype Probe",
            f"- prototype_status: `{probe.get('prototype_status')}`",
            f"- prototype_scope: `{probe.get('prototype_scope')}`",
            f"- kernel_parity_claimed: `{probe.get('kernel_parity_claimed')}`",
            "- reason parity not claimed: `P81 only establishes the opt-in "
            "runtime/probe path; no reference-vs-Pallas numerical comparison ran.`",
            "",
            "## Previous Gate Preservation",
            f"- P78/P79/P80 readiness: `{p80_status}`",
            f"- fixture alias lineage: `{alias_lineage}`",
            "- covered fixture family: `preserved for reference path`",
            "",
            "## Decision",
            f"- recommended_next_phase: `{probe.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p82_pallas_runtime_report_markdown(
    report: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# P82 Pallas Runtime Scaffold Completion Report",
            "",
            "## Runtime Selector",
            f"- default runtime: `{probe.get('default_runtime')}`",
            f"- allowed runtimes: `{probe.get('allowed_runtimes')}`",
            "- reference default preserved: "
            f"`{probe.get('reference_default_preserved')}`",
            "- CLI/config path: `--wkv-runtime reference|pallas`, `wkv_runtime`",
            "",
            "## Pallas Probe",
            f"- pallas requested: `{probe.get('wkv_runtime_requested') == 'pallas'}`",
            f"- pallas available: `{probe.get('pallas_available')}`",
            f"- pallas effective runtime: `{probe.get('wkv_runtime_effective')}`",
            f"- fallback used: `{probe.get('fallback_used')}`",
            f"- prototype_status: `{probe.get('prototype_status')}`",
            f"- prototype_scope: `{probe.get('prototype_scope')}`",
            f"- probe_backend: `{probe.get('probe_backend')}`",
            f"- probe_shapes: `{probe.get('probe_shapes')}`",
            f"- finite: `{probe.get('finite')}`",
            f"- kernel_parity_claimed: `{probe.get('kernel_parity_claimed')}`",
            "",
            "## Capture Semantics",
            "- pallas_requested_reference_trace_contamination: "
            f"`{report.get('pallas_requested_reference_trace_contamination')}`",
            "- fail_closed_before_capture: "
            f"`{report.get('fail_closed_before_capture')}`",
            "- reference_trace_capture_skipped: "
            f"`{report.get('reference_trace_capture_skipped')}`",
            "",
            "## Previous Gate Preservation",
            "- P80 fixture alias resolution: `preserved_from_reference_path`",
            "- covered fixture readiness: `preserved_for_reference_path`",
            "",
            "## Decision",
            f"- recommended_next_phase: `{probe.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p83_pallas_reference_parity_report_markdown(
    report: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# P83 Pallas Reference Parity Report",
            "",
            "## Runtime Selector",
            f"- default runtime: `{probe.get('default_runtime')}`",
            f"- allowed runtimes: `{probe.get('allowed_runtimes')}`",
            "- reference default preserved: "
            f"`{probe.get('reference_default_preserved')}`",
            "- CLI/config path: `--wkv-runtime reference|pallas`, `wkv_runtime`",
            "",
            "## Tiny Reference-vs-Pallas Parity Gate",
            f"- pallas requested: `{probe.get('wkv_runtime_requested') == 'pallas'}`",
            f"- pallas available: `{probe.get('pallas_available')}`",
            f"- pallas effective runtime: `{probe.get('wkv_runtime_effective')}`",
            f"- prototype_status: `{probe.get('prototype_status')}`",
            f"- parity_status: `{probe.get('parity_status')}`",
            f"- parity_scope: `{probe.get('parity_scope')}`",
            f"- probe_backend: `{probe.get('probe_backend')}`",
            f"- probe_shapes: `{probe.get('probe_shapes')}`",
            f"- shape_match: `{probe.get('shape_match')}`",
            f"- finite: `{probe.get('finite')}`",
            f"- max_abs_error: `{probe.get('max_abs_error')}`",
            f"- max_rel_error: `{probe.get('max_rel_error')}`",
            f"- atol: `{probe.get('atol')}`",
            f"- rtol: `{probe.get('rtol')}`",
            f"- kernel_parity_claimed: `{probe.get('kernel_parity_claimed')}`",
            "",
            "## Capture Semantics",
            "- pallas_requested_reference_trace_contamination: "
            f"`{report.get('pallas_requested_reference_trace_contamination')}`",
            "- fail_closed_before_capture: "
            f"`{report.get('fail_closed_before_capture')}`",
            "- reference_trace_capture_skipped: "
            f"`{report.get('reference_trace_capture_skipped')}`",
            "",
            "## Scope",
            "- formula: `state * decay[..., None, :] + "
            "k[..., :, None] * v[..., None, :]`",
            "- full Pallas kernel readiness: `not_claimed_by_p83`",
            "- default runtime promotion: `not_performed`",
            "",
            "## Decision",
            f"- recommended_next_phase: `{probe.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p82_blocker_report_markdown(probe: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P82 Blocker Report",
            "",
            f"- pallas_requested: `{probe.get('wkv_runtime_requested') == 'pallas'}`",
            f"- pallas_available: `{probe.get('pallas_available')}`",
            f"- status: `{probe.get('prototype_status')}`",
            f"- reason: `{probe.get('reason')}`",
            f"- recommended_next_phase: `{probe.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p82_fix_note_markdown(probe: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P82 Fix Note",
            "",
            "- problem: P81 accepted `wkv_runtime=pallas` but could still run "
            "reference live trace capture while reporting Pallas unavailable.",
            "- exact file/function changed: `qrwkv_xla.students.pallas_wkv`, "
            "`build_pallas_runtime_probe`, `run_live_same_run_trace`, "
            "`write_pallas_runtime_reports`.",
            "- why this is runtime scaffold only: P82 runs a minimal Pallas "
            "execution probe and skips reference capture for Pallas-requested "
            "runs; the reference recurrence is unchanged.",
            "- before/after behavior: Pallas-requested runs now record a P82 "
            "probe-only report instead of ambiguous reference trace artifacts.",
            f"- prototype_status: `{probe.get('prototype_status')}`",
            "",
        ]
    )


def _p81_blocker_report_markdown(probe: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P81 Blocker Report",
            "",
            f"- pallas_requested: `{probe.get('wkv_runtime_requested') == 'pallas'}`",
            f"- pallas_available: `{probe.get('pallas_available')}`",
            f"- status: `{probe.get('prototype_status')}`",
            f"- reason: `{probe.get('reason')}`",
            f"- recommended_next_phase: `{probe.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p81_fix_note_markdown(probe: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P81 Fix Note",
            "",
            "- problem: Pallas work needed an explicit opt-in runtime selector "
            "without changing the trusted reference path.",
            "- exact file/function changed: "
            "`qrwkv_xla.students.wkv_runtime`, "
            "`RWKV7QwenReferenceConfig.__post_init__`, "
            "`RWKV7QwenReferenceStudent._attention`, "
            "`run_live_same_run_trace`, `write_live_same_run_reports`.",
            "- why this is runtime scaffold only: P81 validates the selector and "
            "writes an explicit Pallas probe report; recurrence math is unchanged.",
            "- before/after behavior: default calls use `reference`; explicit "
            "`pallas` requests report unavailable instead of silently falling back.",
            f"- prototype_status: `{probe.get('prototype_status')}`",
            "",
        ]
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
    list[dict[str, Any]],
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
    full_vs_stepwise_evidence: list[dict[str, Any]] = []
    parameter_path = parameters or parameter_manifest
    if parameter_path is None or not parameter_path.exists():
        reason = "parameter payload unavailable for QRWKV live capture"
        status["qrwkv_off"]["reason"] = reason
        status["qrwkv_experimental"]["reason"] = reason
        return sources, status, config_snapshots, full_vs_stepwise_evidence
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
        return sources, status, config_snapshots, full_vs_stepwise_evidence
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
        config_snapshots.setdefault("radlads_terms", _config_snapshot(radlads_config))
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
            full_vs_stepwise_evidence.extend(
                _capture_full_vs_stepwise_case(
                    fixture_manifest=fixture_manifest,
                    case=case,
                    params=import_result.params,
                    config=radlads_config,
                    side="radlads",
                    same_run_group_id=same_run_group_id,
                    fixture_id=fixture_id,
                    parameter_id=parameter_id,
                    max_tokens=max_tokens,
                )
            )
        radlads_direct_config = replace(
            base_student.config,
            radlads_balance_state_terms=True,
            radlads_balance_state=True,
        )
        config_snapshots.setdefault(
            "radlads_direct", _config_snapshot(radlads_direct_config)
        )
        radlads_direct_collector = LiveTraceCollector(
            same_run_group_id=same_run_group_id,
            fixture_id=fixture_id,
            parameter_id=parameter_id,
            case=str(case["name"]),
            side="radlads",
            mode=None if mode in {"both", "full", "stepwise"} else mode,
            live_config=_config_snapshot(radlads_direct_config),
        )
        try:
            _capture_radlads_case(
                fixture_manifest=fixture_manifest,
                case=case,
                params=import_result.params,
                config=radlads_direct_config,
                collector=radlads_direct_collector,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            status["radlads_direct"] = {
                "status": "failed",
                "reason": (
                    f"RADLADS direct live capture failed: {type(exc).__name__}: {exc}"
                ),
            }
        else:
            sources["radlads"].extend(radlads_direct_collector.entries)
            full_vs_stepwise_evidence.extend(
                _capture_full_vs_stepwise_case(
                    fixture_manifest=fixture_manifest,
                    case=case,
                    params=import_result.params,
                    config=radlads_direct_config,
                    side="radlads",
                    same_run_group_id=same_run_group_id,
                    fixture_id=fixture_id,
                    parameter_id=parameter_id,
                    max_tokens=max_tokens,
                )
            )
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
            full_vs_stepwise_evidence.extend(
                _capture_full_vs_stepwise_case(
                    fixture_manifest=fixture_manifest,
                    case=case,
                    params=import_result.params,
                    config=config,
                    side=side,
                    same_run_group_id=same_run_group_id,
                    fixture_id=fixture_id,
                    parameter_id=parameter_id,
                    max_tokens=max_tokens,
                )
            )
    for side in ("qrwkv_off", "qrwkv_experimental"):
        if sources[side]:
            status[side] = {"status": "captured", "reason": None}
        elif status[side]["reason"] is None:
            status[side]["reason"] = f"missing_live_hook:{side}:pre_attention_norm"
    if sources["radlads"]:
        status["radlads"] = {"status": "captured", "reason": None}
        if "radlads_direct" not in status:
            direct_rows = [
                row
                for row in sources["radlads"]
                if row.get("balance_state_lane") == DIRECT_BALANCE_STATE_LANE
            ]
            status["radlads_direct"] = {
                "status": "captured" if direct_rows else "missing",
                "reason": None
                if direct_rows
                else "missing_live_hook:radlads_direct:pre_attention_norm",
            }
    elif status["radlads"]["status"] != "failed":
        status["radlads"] = {
            "status": "missing",
            "reason": "missing_live_hook:radlads:pre_attention_norm",
        }
    return sources, status, config_snapshots, full_vs_stepwise_evidence


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
    _output, final_state = student.apply_with_state(
        dict(params),
        input_ids,
        attention_mask=attention_mask,
        diagnostics=collector,
    )
    _record_exported_state_rows(
        collector=collector,
        state=final_state,
        token_index=input_ids.shape[1] - 1,
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
    _output, final_state = student.apply_with_state(
        dict(params),
        input_ids,
        attention_mask=attention_mask,
        diagnostics=collector,
    )
    _record_exported_state_rows(
        collector=collector,
        state=final_state,
        token_index=input_ids.shape[1] - 1,
    )


def _capture_full_vs_stepwise_case(
    *,
    fixture_manifest: Path,
    case: Mapping[str, Any],
    params: Mapping[str, Any],
    config: Any,
    side: str,
    same_run_group_id: str,
    fixture_id: str,
    parameter_id: str,
    max_tokens: int | None,
) -> list[dict[str, Any]]:
    from qrwkv_xla.students import RWKV7QwenReferenceStudent

    lane = classify_balance_state_lane(config)
    case_name = str(case["name"])
    base = {
        "same_run_group_id": same_run_group_id,
        "fixture_id": fixture_id,
        "parameter_id": parameter_id,
        "case": case_name,
        "side": side,
        "balance_state_lane": lane,
        "comparison": "full_vs_stepwise",
        "full_path": "RWKV7QwenReferenceStudent.apply_with_state",
        "stepwise_path": "RWKV7QwenReferenceStudent.step",
        "initial_state": "student.init_state(batch_size)",
        "state_slot_structure": ["wkv_matrix_state", "shift_state", "next_position"],
        "state_carry": "explicit per-token returned state passed to next step",
    }
    try:
        arrays = load_numerical_case_arrays(fixture_manifest, dict(case))
        input_ids = np.asarray(arrays["input_ids"], dtype=np.int32)
        if max_tokens is not None:
            input_ids = input_ids[:, :max_tokens]
        attention_mask = None
        if "attention_mask" in arrays:
            attention_mask = np.asarray(arrays["attention_mask"], dtype=np.int32)
            if max_tokens is not None:
                attention_mask = attention_mask[:, :max_tokens]
        if input_ids.shape[1] <= 0:
            raise ValueError("input_ids must include at least one token")

        student = RWKV7QwenReferenceStudent(config)
        full_output, full_state = student.apply_with_state(
            dict(params),
            input_ids,
            attention_mask=attention_mask,
        )
        step_state = student.init_state(batch_size=int(input_ids.shape[0]))
        step_outputs = []
        for token_index in range(int(input_ids.shape[1])):
            token_mask = (
                None
                if attention_mask is None
                else attention_mask[:, token_index : token_index + 1]
            )
            step_output, step_state = student.step(
                dict(params),
                input_ids[:, token_index : token_index + 1],
                step_state,
                attention_mask=token_mask,
            )
            step_outputs.append(step_output)

        rows = _full_vs_stepwise_state_rows(
            base=base,
            full_state=full_state,
            step_state=step_state,
            final_token=int(input_ids.shape[1]) - 1,
        )
        rows.extend(
            _full_vs_stepwise_exported_rows(
                base=base,
                full_state=full_state,
                step_state=step_state,
                final_token=int(input_ids.shape[1]) - 1,
            )
        )
        rows.extend(
            _full_vs_stepwise_output_rows(
                base=base,
                full_output=full_output,
                step_outputs=step_outputs,
                final_token=int(input_ids.shape[1]) - 1,
            )
        )
        return rows
    except Exception as exc:  # pragma: no cover - defensive evidence path
        return [
            {
                **base,
                "layer": None,
                "head": None,
                "token": None,
                "final_token": None,
                "stage": "state_after_live",
                "state_slot": "wkv_matrix_state",
                "status": "unavailable",
                "reason": f"full_vs_stepwise_capture_failed:{type(exc).__name__}:{exc}",
                "mode": "full_vs_stepwise",
            }
        ]


def _full_vs_stepwise_output_rows(
    *,
    base: Mapping[str, Any],
    full_output: Any,
    step_outputs: list[Any],
    final_token: int,
) -> list[dict[str, Any]]:
    rows = []
    full_hidden = np.asarray(full_output.hidden_states)[:, -1, :, :]
    step_hidden_parts = [
        np.asarray(output.hidden_states)[:, -1, :, :] for output in step_outputs
    ]
    if step_hidden_parts:
        step_hidden = np.concatenate(step_hidden_parts, axis=1)
        rows.append(
            _full_vs_stepwise_output_row(
                base=base,
                stage="post_block_hidden_output",
                full_value=full_hidden,
                stepwise_value=step_hidden,
                final_token=final_token,
                reason_if_unavailable=None,
            )
        )
    else:
        rows.append(
            _full_vs_stepwise_unavailable_output_row(
                base=base,
                stage="post_block_hidden_output",
                final_token=final_token,
                reason="missing_stepwise_output_capture",
                full_shape=list(full_hidden.shape),
            )
        )
    rows.append(
        _full_vs_stepwise_unavailable_output_row(
            base=base,
            stage="final_normalized_hidden",
            final_token=final_token,
            reason="missing_stepwise_final_norm_capture",
            full_shape=None,
        )
    )
    full_logits = None if full_output.logits is None else np.asarray(full_output.logits)
    step_logits_parts = [
        np.asarray(output.logits)
        for output in step_outputs
        if getattr(output, "logits", None) is not None
    ]
    if full_logits is None:
        rows.append(
            _full_vs_stepwise_unavailable_output_row(
                base=base,
                stage="final_lm_head_logits",
                final_token=final_token,
                reason="missing_lm_head_logits_path",
                full_shape=None,
            )
        )
        rows.append(
            _full_vs_stepwise_unavailable_output_row(
                base=base,
                stage="selected_token_logits",
                final_token=final_token,
                reason="missing_lm_head_logits_path",
                full_shape=None,
            )
        )
        return rows
    if len(step_logits_parts) != len(step_outputs):
        rows.append(
            _full_vs_stepwise_unavailable_output_row(
                base=base,
                stage="final_lm_head_logits",
                final_token=final_token,
                reason="missing_stepwise_logits_capture",
                full_shape=list(full_logits.shape),
            )
        )
        rows.append(
            _full_vs_stepwise_unavailable_output_row(
                base=base,
                stage="selected_token_logits",
                final_token=final_token,
                reason="missing_stepwise_logits_capture",
                full_shape=list(full_logits[:, -1, :].shape),
            )
        )
        return rows
    step_logits = np.concatenate(step_logits_parts, axis=1)
    rows.append(
        _full_vs_stepwise_output_row(
            base=base,
            stage="final_lm_head_logits",
            full_value=full_logits,
            stepwise_value=step_logits,
            final_token=final_token,
            reason_if_unavailable=None,
        )
    )
    rows.append(
        _full_vs_stepwise_output_row(
            base=base,
            stage="selected_token_logits",
            full_value=full_logits[:, -1, :],
            stepwise_value=step_logits[:, -1, :],
            final_token=final_token,
            reason_if_unavailable=None,
        )
    )
    return rows


def _full_vs_stepwise_output_row(
    *,
    base: Mapping[str, Any],
    stage: str,
    full_value: Any,
    stepwise_value: Any,
    final_token: int,
    reason_if_unavailable: str | None,
) -> dict[str, Any]:
    full_array = np.asarray(full_value)
    step_array = np.asarray(stepwise_value)
    comparison = compare_trace_arrays(full_array, step_array)
    status = str(comparison["status"])
    return {
        **base,
        "comparison": "full_vs_stepwise_output",
        "layer": None,
        "head": None,
        "token": final_token,
        "final_token": final_token,
        "stage": stage,
        "state_slot": None,
        "status": status,
        "reason": "allclose"
        if status == "pass"
        else reason_if_unavailable
        if status == "unavailable"
        else status,
        "mode": "full_vs_stepwise",
        "full_mode": "full",
        "stepwise_mode": "stepwise",
        "full_shape": list(full_array.shape),
        "stepwise_shape": list(step_array.shape),
        "full_dtype": str(full_array.dtype),
        "stepwise_dtype": str(step_array.dtype),
        "full_array": full_array.tolist(),
        "stepwise_array": step_array.tolist(),
        **comparison,
    }


def _full_vs_stepwise_unavailable_output_row(
    *,
    base: Mapping[str, Any],
    stage: str,
    final_token: int,
    reason: str,
    full_shape: list[int] | None,
) -> dict[str, Any]:
    return {
        **base,
        "comparison": "full_vs_stepwise_output",
        "layer": None,
        "head": None,
        "token": final_token,
        "final_token": final_token,
        "stage": stage,
        "state_slot": None,
        "status": "unavailable",
        "reason": reason,
        "mode": "full_vs_stepwise",
        "full_mode": "full",
        "stepwise_mode": "stepwise",
        "full_shape": full_shape,
        "stepwise_shape": None,
        "full_array": None,
        "stepwise_array": None,
    }


def _full_vs_stepwise_state_rows(
    *,
    base: Mapping[str, Any],
    full_state: Any,
    step_state: Any,
    final_token: int,
) -> list[dict[str, Any]]:
    return _full_vs_stepwise_slot_rows(
        base=base,
        full_value=extract_state_slot(full_state, "wkv_matrix_state"),
        step_value=extract_state_slot(step_state, "wkv_matrix_state"),
        final_token=final_token,
        stage="state_after_live",
        state_slot="wkv_matrix_state",
    )


def _full_vs_stepwise_exported_rows(
    *,
    base: Mapping[str, Any],
    full_state: Any,
    step_state: Any,
    final_token: int,
) -> list[dict[str, Any]]:
    full_exported = export_reference_state_object(full_state)
    step_exported = export_reference_state_object(step_state)
    full_wkv = full_exported["state_slots"]["wkv_matrix_state"]
    step_wkv = step_exported["state_slots"]["wkv_matrix_state"]
    return _full_vs_stepwise_slot_rows(
        base={
            **base,
            "export_path": str(
                full_exported.get("export_path", REFERENCE_STATE_EXPORT_PATH)
            ),
            "import_path": REFERENCE_STATE_IMPORT_PATH,
        },
        full_value=full_wkv,
        step_value=step_wkv,
        final_token=final_token,
        stage="state_after_exported",
        state_slot="wkv_matrix_state",
    )


def _full_vs_stepwise_slot_rows(
    *,
    base: Mapping[str, Any],
    full_value: Any,
    step_value: Any,
    final_token: int,
    stage: str,
    state_slot: str,
) -> list[dict[str, Any]]:
    full_array = np.asarray(full_value)
    step_array = np.asarray(step_value)
    rows = []
    layer_count = int(full_array.shape[0]) if full_array.ndim >= 1 else 0
    head_count = int(full_array.shape[2]) if full_array.ndim >= 3 else 0
    for layer_index in range(layer_count):
        for head_index in range(head_count):
            full_head = full_array[layer_index, :, head_index]
            step_head = step_array[layer_index, :, head_index]
            stats = compare_trace_arrays(full_head, step_head)
            diff = full_head.astype(np.float64) - step_head.astype(np.float64)
            rows.append(
                {
                    **base,
                    "layer": layer_index,
                    "head": head_index,
                    "token": final_token,
                    "final_token": final_token,
                    "stage": stage,
                    "state_slot": state_slot,
                    "mode": "full_vs_stepwise",
                    "full_mode": "full",
                    "stepwise_mode": "stepwise",
                    "status": stats["status"],
                    "reason": "allclose"
                    if stats["status"] == "pass"
                    else stats["status"],
                    "shape_match": stats["shape_match"],
                    "dtype_match": stats["dtype_match"],
                    "finite_both": stats["finite_both"],
                    "max_abs_error": stats["max_abs_error"],
                    "mean_abs_error": stats["mean_abs_error"],
                    "rms_error": float(np.sqrt(np.mean(diff * diff)))
                    if diff.size
                    else 0.0,
                    "max_relative_error": stats["max_relative_error"],
                    "allclose": stats["allclose"],
                    "full_shape": list(full_head.shape),
                    "stepwise_shape": list(step_head.shape),
                    "full_dtype": str(full_head.dtype),
                    "stepwise_dtype": str(step_head.dtype),
                }
            )
    if not rows:
        rows.append(
            {
                **base,
                "layer": None,
                "head": None,
                "token": final_token,
                "final_token": final_token,
                "stage": stage,
                "state_slot": state_slot,
                "mode": "full_vs_stepwise",
                "status": "unavailable",
                "reason": "state_slot_shape_not_layer_head_addressable",
                "full_shape": list(full_array.shape),
                "stepwise_shape": list(step_array.shape),
            }
        )
    return rows


def _record_exported_state_rows(
    *,
    collector: LiveTraceCollector,
    state: Any,
    token_index: int,
) -> None:
    try:
        exported = export_reference_state_object(state)
        imported = import_reference_state_object(exported, template=state)
        exported_wkv = np.asarray(exported["state_slots"]["wkv_matrix_state"])
        imported_wkv = np.asarray(extract_state_slot(imported, "wkv_matrix_state"))
        roundtrip = compare_trace_arrays(exported_wkv, imported_wkv)
        roundtrip_status = str(roundtrip["status"])
        roundtrip_reason = (
            "allclose" if roundtrip_status == "pass" else roundtrip_status
        )
    except Exception:  # pragma: no cover - defensive evidence path
        return
    for layer_index in range(int(exported_wkv.shape[0])):
        collector.record(
            "exported_wkv_matrix_state",
            exported_wkv[layer_index],
            layer=layer_index,
            token=token_index,
            stage="state_after_exported",
            source_stage_name="exported_wkv_matrix_state",
            capture_kind="exported_state",
            export_path=str(exported.get("export_path", REFERENCE_STATE_EXPORT_PATH)),
            import_path=REFERENCE_STATE_IMPORT_PATH,
            import_roundtrip_status=roundtrip_status,
            import_roundtrip_reason=roundtrip_reason,
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


def _row_balance_state_lane(row: Mapping[str, Any], *, side: str | None = None) -> str:
    lane = row.get("balance_state_lane")
    if lane is not None:
        return str(lane)
    config = row.get("live_config") or row.get("config")
    if isinstance(config, Mapping):
        return classify_balance_state_lane(config)
    if side == "qrwkv_experimental":
        return DIRECT_BALANCE_STATE_LANE
    return NATIVE_OR_UNKNOWN_LANE


def _lanes_from_rows(
    rows: Iterable[Mapping[str, Any]], *, side: str | None = None
) -> list[str]:
    lanes = sorted({_row_balance_state_lane(row, side=side) for row in rows})
    if lanes:
        return lanes
    return [_lane_from_rows([], side=side)]


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
    row = {
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
    for key in (
        "export_path",
        "import_path",
        "import_roundtrip_status",
        "import_roundtrip_reason",
    ):
        if source.get(key) is not None:
            row[key] = source.get(key)
    return row


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
        "balance_state_lane": key[5],
        "stage": key[6],
        "dependency_index": DEPENDENCY_ORDER.index(key[6])
        if key[6] in DEPENDENCY_ORDER
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
        sides = sorted(value)
        for index, left_side in enumerate(sides):
            for right_side in sides[index + 1 :]:
                delta = _config_delta(value[left_side], value[right_side])
                if delta["status"] != "pass":
                    mismatches[(key, left_side, right_side)] = delta
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
    counts = {
        side: sum(
            1
            for row in traces.get(side, [])
            if row.get("capture_kind") == "live_captured"
        )
        for side in SIDES
    }
    counts["radlads_terms"] = _live_row_count_for_lane(
        traces,
        side="radlads",
        lane=BALANCE_STATE_TERMS_LANE,
    )
    counts["radlads_direct"] = _live_row_count_for_lane(
        traces,
        side="radlads",
        lane=DIRECT_BALANCE_STATE_LANE,
    )
    counts["qrwkv_off_terms"] = _live_row_count_for_lane(
        traces,
        side="qrwkv_off",
        lane=BALANCE_STATE_TERMS_LANE,
    )
    counts["qrwkv_experimental_direct"] = _live_row_count_for_lane(
        traces,
        side="qrwkv_experimental",
        lane=DIRECT_BALANCE_STATE_LANE,
    )
    return counts


def _live_row_count_for_lane(
    traces: Mapping[str, list[dict[str, Any]]], *, side: str, lane: str
) -> int:
    return sum(
        1
        for row in traces.get(side, [])
        if row.get("capture_kind") == "live_captured"
        and row.get("balance_state_lane") == lane
    )


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
        "radlads_terms": metadata.get("radlads_terms_config")
        or metadata.get("radlads_config_snapshot"),
        "radlads_direct": metadata.get("radlads_direct_config"),
        "qrwkv_off": metadata.get("qrwkv_off_config"),
        "qrwkv_off_terms": metadata.get("qrwkv_off_config"),
        "qrwkv_experimental": metadata.get("qrwkv_experimental_config"),
        "qrwkv_experimental_direct": metadata.get("qrwkv_experimental_config"),
    }
    result: dict[str, dict[str, Any]] = {}
    surfaces = (
        ("radlads", "radlads", BALANCE_STATE_TERMS_LANE),
        ("qrwkv_off", "qrwkv_off", BALANCE_STATE_TERMS_LANE),
        ("qrwkv_experimental", "qrwkv_experimental", DIRECT_BALANCE_STATE_LANE),
        ("radlads_terms", "radlads", BALANCE_STATE_TERMS_LANE),
        ("qrwkv_off_terms", "qrwkv_off", BALANCE_STATE_TERMS_LANE),
        ("radlads_direct", "radlads", DIRECT_BALANCE_STATE_LANE),
        (
            "qrwkv_experimental_direct",
            "qrwkv_experimental",
            DIRECT_BALANCE_STATE_LANE,
        ),
    )
    surface_lanes: dict[str, str] = {}
    for surface, side, fallback_lane in surfaces:
        config = config_by_side.get(surface)
        if not isinstance(config, Mapping):
            lane_rows = [
                row
                for row in traces.get(side, [])
                if row.get("balance_state_lane") == fallback_lane
            ]
            config = _live_config_from_rows(lane_rows) or {}
        lane = classify_balance_state_lane(config)
        if lane == NATIVE_OR_UNKNOWN_LANE and any(
            row.get("balance_state_lane") == fallback_lane
            for row in traces.get(side, [])
        ):
            lane = fallback_lane
        surface_lanes[surface] = lane
    for surface, side, fallback_lane in surfaces:
        config = config_by_side.get(surface)
        if not isinstance(config, Mapping):
            lane_rows = [
                row
                for row in traces.get(side, [])
                if row.get("balance_state_lane") == fallback_lane
            ]
            config = _live_config_from_rows(lane_rows) or {}
        lane = surface_lanes[surface]
        excluded = (
            sorted(LANE_A_ONLY_STAGES) if lane == DIRECT_BALANCE_STATE_LANE else []
        )
        comparable_to = [
            other
            for other, other_lane in surface_lanes.items()
            if other != surface and other_lane == lane
        ]
        result[surface] = {
            "side": side,
            "surface": surface,
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
    surface_lanes = {
        surface: lane_map.get(surface, {}).get("lane")
        for surface in (
            "radlads_terms",
            "qrwkv_off_terms",
            "radlads_direct",
            "qrwkv_experimental_direct",
        )
    }
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
        surface_lanes.get("radlads_terms") == BALANCE_STATE_TERMS_LANE
        and surface_lanes.get("qrwkv_off_terms") == BALANCE_STATE_TERMS_LANE
        and _lane_has_rows(traces, side="radlads", lane=BALANCE_STATE_TERMS_LANE)
        and _lane_has_rows(traces, side="qrwkv_off", lane=BALANCE_STATE_TERMS_LANE)
    )
    terms_first = _first_pair_failure(
        rows,
        "radlads_vs_qrwkv_off",
        lane=BALANCE_STATE_TERMS_LANE,
    )
    direct_pair_available = (
        surface_lanes.get("radlads_direct") == DIRECT_BALANCE_STATE_LANE
        and surface_lanes.get("qrwkv_experimental_direct") == DIRECT_BALANCE_STATE_LANE
        and _lane_has_rows(traces, side="radlads", lane=DIRECT_BALANCE_STATE_LANE)
        and _lane_has_rows(
            traces,
            side="qrwkv_experimental",
            lane=DIRECT_BALANCE_STATE_LANE,
        )
    )
    direct_first = _first_pair_failure(
        rows,
        "radlads_vs_qrwkv_experimental",
        lane=DIRECT_BALANCE_STATE_LANE,
    )
    direct_lane_present = any(
        row.get("balance_state_lane") == DIRECT_BALANCE_STATE_LANE
        for side in SIDES
        for row in traces.get(side, [])
    )
    terms_valid = bool(terms_pair_available and terms_first is None)
    direct_valid = bool(direct_pair_available and direct_first is None)
    return {
        "lane_classification_valid": classification_valid,
        "overall_same_run_valid": None,
        "balance_state_terms_lane_valid": terms_valid,
        "balance_state_terms_lane_pair_available": terms_pair_available,
        "balance_state_terms_first_failure": None
        if terms_first is None
        else _primary_gap(terms_first),
        "balance_state_terms_math_conclusion_valid": terms_valid,
        "direct_balance_state_lane_valid": direct_valid,
        "direct_balance_state_lane_present": direct_lane_present,
        "direct_balance_state_radlads_available": (
            surface_lanes.get("radlads_direct") == DIRECT_BALANCE_STATE_LANE
            and _lane_has_rows(traces, side="radlads", lane=DIRECT_BALANCE_STATE_LANE)
        ),
        "direct_balance_state_lane_pair_available": direct_pair_available,
        "direct_balance_state_first_failure": None
        if direct_first is None
        else _primary_gap(direct_first),
        "direct_balance_state_math_conclusion_valid": direct_valid,
        "lane_mixed_comparison_valid": len(set(lanes.values())) <= 1,
        "not_applicable_rows": [
            _trace_row_summary(row)
            for side in SIDES
            for row in traces.get(side, [])
            if row.get("capture_kind") == "not_applicable"
        ],
    }


def _lane_has_rows(
    traces: Mapping[str, list[dict[str, Any]]], *, side: str, lane: str
) -> bool:
    return any(row.get("balance_state_lane") == lane for row in traces.get(side, []))


def _first_pair_failure(
    rows: Iterable[Mapping[str, Any]], pair: str, *, lane: str
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("balance_state_lane") != lane:
            continue
        if row.get("stage") not in {*MINIMUM_STAGES, *P71_STRETCH_STAGES}:
            continue
        if lane == DIRECT_BALANCE_STATE_LANE and row.get("stage") in LANE_A_ONLY_STAGES:
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
    del lane_map
    for row in rows:
        if row.get("stage") not in {*MINIMUM_STAGES, *P71_STRETCH_STAGES}:
            continue
        lane = row.get("balance_state_lane")
        if lane == DIRECT_BALANCE_STATE_LANE and row.get("stage") in LANE_A_ONLY_STAGES:
            continue
        pair = _lane_primary_pair(row)
        if pair is None or row.get(pair, {}).get("status") == "pass":
            continue
        if row.get(pair, {}).get("status") == "unavailable":
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
    terms_first = lane_validity.get("balance_state_terms_first_failure")
    if terms_first is not None:
        return _lane_specific_recommendation(
            lane=BALANCE_STATE_TERMS_LANE,
            first=terms_first,
            valid=False,
        )
    if first_comparable is not None:
        return _lane_specific_recommendation(
            lane=str(first_comparable.get("lane")),
            first=first_comparable,
            valid=False,
        )
    if lane_validity.get("direct_balance_state_lane_present") and not lane_validity.get(
        "direct_balance_state_radlads_available"
    ):
        return P75_DIRECT_LANE_REPAIR
    direct_first = lane_validity.get("direct_balance_state_first_failure")
    if direct_first is not None:
        return _lane_specific_recommendation(
            lane=DIRECT_BALANCE_STATE_LANE,
            first=direct_first,
            valid=False,
        )
    if lane_validity.get("balance_state_terms_lane_valid") and lane_validity.get(
        "direct_balance_state_lane_valid"
    ):
        return P75_RESIDUAL_GATE
    return P75_DIRECT_LANE_REPAIR


def _lane_specific_recommendation(
    *, lane: str, first: Mapping[str, Any] | None, valid: bool
) -> str:
    if valid and first is None:
        return P75_RESIDUAL_GATE
    if first is None:
        if lane == DIRECT_BALANCE_STATE_LANE:
            return P75_DIRECT_LANE_REPAIR
        return P72_HOOK_COMPLETION
    if first.get("max_abs_error") is None:
        return P72_HOOK_COMPLETION
    stage = first.get("stage")
    if lane == DIRECT_BALANCE_STATE_LANE:
        if stage == "kk":
            return P75_DIRECT_KK_FIX
        if stage in {"k_for_update", "v_for_update"}:
            return P75_DIRECT_BALANCE_PREP_FIX
        if stage == "ab":
            return P75_DIRECT_AB_FIX
        if stage == "vk":
            return P75_DIRECT_VK_FIX
        if stage == "state_after_live":
            return P75_DIRECT_STATE_AFTER_FIX
        return P75_DIRECT_LANE_REPAIR
    if stage == "kk":
        return P75_TERMS_KK_FIX
    if stage in {"k_for_update", "v_for_update"}:
        return P75_TERMS_BALANCE_PREP_FIX
    if stage == "ab":
        return P75_TERMS_AB_FIX
    if stage == "vk":
        return P75_TERMS_VK_FIX
    return P72_HOOK_COMPLETION


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
    pair = _lane_primary_pair(row)
    if pair is not None:
        return str(row[pair]["status"])
    statuses = {
        row["radlads_vs_qrwkv_off"]["status"],
        row["radlads_vs_qrwkv_experimental"]["status"],
        row["qrwkv_off_vs_qrwkv_experimental"]["status"],
    }
    return "pass" if statuses == {"pass"} else "fail"


def _row_has_unavailable(row: Mapping[str, Any]) -> bool:
    pair = _lane_primary_pair(row)
    if pair is not None:
        return row[pair]["status"] == "unavailable"
    return any(
        row[name]["status"] == "unavailable"
        for name in (
            "radlads_vs_qrwkv_off",
            "radlads_vs_qrwkv_experimental",
            "qrwkv_off_vs_qrwkv_experimental",
        )
    )


def _row_capture_kind(row: Mapping[str, Any]) -> str:
    lane = row.get("balance_state_lane")
    if lane == BALANCE_STATE_TERMS_LANE:
        kinds = [row.get("radlads_capture_kind"), row.get("qrwkv_off_capture_kind")]
    elif lane == DIRECT_BALANCE_STATE_LANE:
        kinds = [
            row.get("radlads_capture_kind"),
            row.get("qrwkv_experimental_capture_kind"),
        ]
    else:
        kinds = [
            row.get("radlads_capture_kind"),
            row.get("qrwkv_off_capture_kind"),
            row.get("qrwkv_experimental_capture_kind"),
        ]
    if any(kind in {None, "unavailable"} for kind in kinds):
        return "unavailable"
    return ",".join(str(kind) for kind in kinds)


def _lane_primary_pair(row: Mapping[str, Any]) -> str | None:
    lane = row.get("balance_state_lane")
    if lane == BALANCE_STATE_TERMS_LANE:
        return "radlads_vs_qrwkv_off"
    if lane == DIRECT_BALANCE_STATE_LANE:
        return "radlads_vs_qrwkv_experimental"
    return None


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
        first.get("balance_state_lane", NATIVE_OR_UNKNOWN_LANE),
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
    pair = _lane_primary_pair(row)
    if pair is not None:
        return row[pair]["max_abs_error"]
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
        "balance_state_lane": row.get("balance_state_lane"),
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
        entry.get("balance_state_lane", NATIVE_OR_UNKNOWN_LANE),
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
        str(key[5]),
        DEPENDENCY_ORDER.index(key[6]) if key[6] in DEPENDENCY_ORDER else 999,
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
    terms_first = report.get("balance_state_terms_first_differing_stage")
    direct_first = report.get("direct_balance_state_first_differing_stage")
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
            "- Lane comparisons:",
            "  - balance_state_terms: `"
            f"{report.get('balance_state_terms_lane_valid')}`",
            "  - direct_balance_state: `"
            f"{report.get('direct_balance_state_lane_valid')}`",
            "- First comparable differing stage by lane:",
            f"  - balance_state_terms: `{terms_first}`",
            f"  - direct_balance_state: `{direct_first}`",
            f"- Overall recommended next phase: {report['recommended_next_phase']}",
            f"- Kernel-ready: `{report['kernel_ready']}`",
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
    p75_gate = report.get("p75_residual_impact_gate", {})
    lines = [
        "# P68 Decision",
        "",
        "P75 residual-impact / kernel-readiness outcome:",
        f"- same_run_valid: `{report['same_run_valid']}`",
        f"- minimum_stage_valid: `{report.get('minimum_stage_valid')}`",
        f"- stretch_stages_available: `{report.get('stretch_stages_available')}`",
        f"- math_conclusion_valid: `{report.get('math_conclusion_valid')}`",
        "- balance_state_terms_lane_valid: `"
        f"{report.get('balance_state_terms_lane_valid')}`",
        "- direct_balance_state_lane_valid: `"
        f"{report.get('direct_balance_state_lane_valid')}`",
        "- balance_state_terms_first_differing_stage: `"
        f"{report.get('balance_state_terms_first_differing_stage')}`",
        "- direct_balance_state_first_differing_stage: `"
        f"{report.get('direct_balance_state_first_differing_stage')}`",
        f"- lane_mixed_comparison_valid: `{report.get('lane_mixed_comparison_valid')}`",
        f"- kernel_ready: `{report['kernel_ready']}`",
        f"- kernel_readiness_reason: `{report.get('kernel_readiness_reason')}`",
        f"- blocking_gates: `{p75_gate.get('blocking_gates')}`",
        f"- warning_gates: `{p75_gate.get('warning_gates')}`",
        f"- recommended_next_phase: {report['recommended_next_phase']}",
        "- math_fix_recommended: `False`",
        "- pallas_gate_recommended: `False`",
        "- residual_impact_gate_completed: `True`",
        "",
    ]
    p79 = report.get("p79_broader_fixture_residual_matrix")
    if isinstance(p79, Mapping):
        lines.extend(
            [
                "P79 broader fixture validation outcome:",
                "- all_expected_cases_present: `"
                f"{p79.get('all_expected_cases_present')}`",
                f"- all_expected_cases_pass: `{p79.get('all_expected_cases_pass')}`",
                f"- recommended_action: `{p79.get('recommended_action')}`",
                "",
            ]
        )
    return "\n".join(lines)


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


def _p76_state_export_import_markdown(
    report: Mapping[str, Any], gate: Mapping[str, Any]
) -> str:
    semantics = report.get("state_export_semantics", {})
    lines = [
        "# P76 State Export / Import Report",
        "",
        f"status: `{report.get('status')}`",
        f"schema: `{report.get('schema')}`",
        f"fixture_id: `{report.get('fixture_id')}`",
        f"parameter_id: `{report.get('parameter_id')}`",
        f"same_run_group_id: `{report.get('same_run_group_id')}`",
        "",
        "## Export/Import Path",
        "",
        f"- export function/path: `{semantics.get('export_path')}`",
        f"- import function/path: `{semantics.get('import_path')}`",
        f"- state representation: `{semantics.get('meaning')}`",
        "- state slots: `wkv_matrix_state`, `shift_state`, "
        "`next_position` when present",
        "- shape convention: reference state slot shapes are preserved",
        "- dtype convention: exported slot dtypes are preserved",
        "- lane identity: preserved by trace row `balance_state_lane` keys",
        f"- trace_stage: `{semantics.get('trace_stage')}`",
        f"- capture_kind: `{semantics.get('capture_kind')}`",
        "",
        "## Lane-Aware Exported State Rows",
        "",
    ]
    for surface, item in report.get("surface_status", {}).items():
        lines.append(
            f"- {surface}: status=`{item.get('status')}`, reason=`{item.get('reason')}`"
        )
    lines.extend(["", "## Intra-side Consistency", ""])
    lines.append(
        "state_after_live vs state_after_exported: see required surfaces above"
    )
    lines.extend(["", "## Inter-side Exported-State Parity", ""])
    for lane, item in report.get("lane_pair_status", {}).items():
        lines.append(
            f"- {lane}: pair=`{item.get('pair')}`, "
            f"status=`{item.get('status')}`, reason=`{item.get('reason')}`"
        )
    exported_gate = gate.get("output_gates", {}).get("exported_state", {})
    lines.extend(
        [
            "",
            "## Import Round Trip",
            "",
            "exported -> imported: included in required surface status above",
            "",
            "## P75 Gate Feed",
            "",
            "- exported_state gate: "
            f"status=`{exported_gate.get('status')}`, "
            f"reason=`{exported_gate.get('reason')}`",
            f"- kernel_ready: `{gate.get('kernel_ready')}`",
            f"- blocking_gates: `{gate.get('blocking_gates')}`",
            f"- recommended_next_phase: `{gate.get('recommended_next_phase')}`",
            "",
            "## Decision",
            "",
            f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
            "No recurrence math, balance-state math, dtype policy, tolerances, "
            "fixture values, RADLADS source, Pallas code, training path, or "
            "default experimental balance_state behavior changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _p76_blocker_report_markdown(report: Mapping[str, Any]) -> str:
    blockers = []
    for surface, item in report.get("surface_status", {}).items():
        if item.get("status") != "pass":
            blockers.append(f"{surface}:{item.get('reason')}")
    for lane, item in report.get("lane_pair_status", {}).items():
        if item.get("status") != "pass":
            blockers.append(f"{lane}:{item.get('reason')}")
    return "\n".join(
        [
            "# P76 Blocker Report",
            "",
            f"status: `{report.get('status')}`",
            "",
            "## Blocking Or Missing Evidence",
            "",
            *[f"- `{item}`" for item in blockers],
            "",
            f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p77_full_vs_stepwise_markdown(
    report: Mapping[str, Any], gate: Mapping[str, Any]
) -> str:
    paths = report.get("paths", {})
    state_gate = gate.get("output_gates", {}).get("state_after", {})
    exported_gate = gate.get("output_gates", {}).get("exported_state", {})
    full_gate = gate.get("output_gates", {}).get("full_vs_stepwise", {})
    logits_gate = gate.get("output_gates", {}).get("logits_output", {})
    lines = [
        "# P77 Full-vs-Stepwise Report",
        "",
        f"status: `{report.get('status')}`",
        f"schema: `{report.get('schema')}`",
        f"fixture_id: `{report.get('fixture_id')}`",
        f"parameter_id: `{report.get('parameter_id')}`",
        f"same_run_group_id: `{report.get('same_run_group_id')}`",
        "",
        "## Execution Paths",
        "",
        f"full path: `{paths.get('full')}`",
        f"stepwise path: `{paths.get('stepwise')}`",
        f"state initialization: `{paths.get('initial_state')}`",
        f"state carry: `{paths.get('state_carry')}`",
        f"final state representation: `{paths.get('state_slots')}`",
        "",
        "## Lane Surfaces",
        "",
    ]
    for surface, item in report.get("surface_status", {}).items():
        lines.append(
            f"- {surface}: status=`{item.get('status')}`, reason=`{item.get('reason')}`"
        )
    lines.extend(["", "## Lane Pair Policy", ""])
    for lane, item in report.get("lanes", {}).items():
        lines.append(
            f"- {lane}: left=`{item.get('left')}`, right=`{item.get('right')}`, "
            f"status=`{item.get('status')}`"
        )
    lines.extend(["", "## Final State Comparisons", ""])
    lines.append(
        f"full final state vs stepwise final state: `{report.get('final_state')}`"
    )
    lines.extend(["", "## Exported State Comparisons", ""])
    lines.append(
        "full exported state vs stepwise exported state: "
        f"`{report.get('exported_state')}`"
    )
    lines.extend(["", "## Output Comparisons", ""])
    lines.append(f"hidden/output/logits if available: `{report.get('outputs')}`")
    lines.extend(
        [
            "",
            "## P75 Gate Update",
            "",
            "- state_after gate: "
            f"status=`{state_gate.get('status')}`, "
            f"reason=`{state_gate.get('reason')}`",
            "- exported_state gate: "
            f"status=`{exported_gate.get('status')}`, "
            f"reason=`{exported_gate.get('reason')}`",
            "- full_vs_stepwise gate: "
            f"status=`{full_gate.get('status')}`, "
            f"reason=`{full_gate.get('reason')}`",
            "- logits_output gate: "
            f"status=`{logits_gate.get('status')}`, "
            f"reason=`{logits_gate.get('reason')}`",
            f"- kernel_ready: `{gate.get('kernel_ready')}`",
            f"- blocking_gates: `{gate.get('blocking_gates')}`",
            "",
            "## Decision",
            "",
            f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
            "No recurrence math, balance-state math, dtype policy, tolerances, "
            "fixture values, RADLADS source, Pallas code, training path, or "
            "default experimental balance_state behavior changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _p77_blocker_report_markdown(report: Mapping[str, Any]) -> str:
    blockers = []
    for surface, item in report.get("surface_status", {}).items():
        if item.get("status") != "pass":
            blockers.append(f"{surface}:{item.get('reason')}")
    return "\n".join(
        [
            "# P77 Blocker Report",
            "",
            f"status: `{report.get('status')}`",
            "",
            "## Blocking Or Missing Evidence",
            "",
            *[f"- `{item}`" for item in blockers],
            "",
            f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p77_fix_note_markdown(report: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    before_reason = "missing_evidence:full_vs_stepwise"
    after_gate = gate.get("output_gates", {}).get("full_vs_stepwise", {})
    return "\n".join(
        [
            "# P77 Fix Note",
            "",
            "problem: P75/P76 had no same-run, lane-aware full-vs-stepwise "
            "residual evidence, so the readiness gate stopped at "
            f"`{before_reason}`.",
            "",
            "source evidence: current invocation reruns the same fixture and "
            "same imported RADLADS parameter payload through the documented "
            "full and token-by-token paths, preserving case, side, lane, "
            "fixture_id, parameter_id, and same_run_group_id in every row.",
            "",
            "exact file/function changed: "
            "`src/qrwkv_xla/parity/radlads_live_same_run_trace.py` "
            "(`_capture_full_vs_stepwise_case`, "
            "`build_p77_full_vs_stepwise_residual`, and report writers).",
            "",
            "why this is state-carry/reporting only: the patch calls existing "
            "`RWKV7QwenReferenceStudent.apply_with_state` and "
            "`RWKV7QwenReferenceStudent.step` APIs, explicitly carries the "
            "returned stepwise state, and compares/report-gates the resulting "
            "state slots. It does not change recurrence math, balance math, "
            "dtype policy, tolerances, fixtures, RADLADS source, Pallas code, "
            "or default experimental balance_state behavior.",
            "",
            f"before gate result: `{before_reason}`",
            "after gate result: "
            f"`{after_gate.get('status')}` / `{after_gate.get('reason')}`",
            f"p77_status: `{report.get('status')}`",
            f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p78_logits_output_markdown(
    report: Mapping[str, Any], gate: Mapping[str, Any]
) -> str:
    logits_gate = gate.get("output_gates", {}).get("logits_output", {})
    logits_available = (
        "yes" if report.get("logits_path", {}).get("status") == "pass" else "no"
    )
    lines = [
        "# P78 Logits / Output Report",
        "",
        f"status: `{report.get('status')}`",
        f"schema: `{report.get('schema')}`",
        f"fixture_id: `{report.get('fixture_id')}`",
        f"parameter_id: `{report.get('parameter_id')}`",
        f"same_run_group_id: `{report.get('same_run_group_id')}`",
        "",
        "## Output Semantics",
        "",
        "hidden path: `RWKV7QwenReferenceStudent.apply_with_state` and "
        "`RWKV7QwenReferenceStudent.step` returned hidden states, reported as "
        "`post_block_hidden_output`.",
        "output path: `post_block_hidden_output` compares returned hidden/output "
        "tensors; `final_normalized_hidden` is reported separately when captured.",
        "logits path: `StudentOutput.logits` from the existing student full and "
        "stepwise paths, never synthesized from hidden states.",
        "LM head path: existing student LM-head emission behind `emit_logits`; "
        "missing logits are marked unavailable with an exact reason.",
        "normalization path: `final_normalized_hidden` is not captured by this "
        "P78 hook and remains separate from logits.",
        "tokens compared: final token plus full emitted logits sequence where "
        "available.",
        "lanes compared: RADLADS terms vs QRWKV off terms; RADLADS direct vs "
        "QRWKV experimental direct.",
        f"full-vocab logits available: `{logits_available}`",
        f"selected-token logits available: `{logits_available}`",
        "top-k logits/logprobs available: `no`",
        "",
        "## Lane Surfaces",
        "",
    ]
    for lane, item in report.get("lanes", {}).items():
        lines.append(
            f"- {lane}: left=`{item.get('left')}`, right=`{item.get('right')}`, "
            f"status=`{item.get('status')}`"
        )
    lines.extend(["", "## Inter-side Output Parity", ""])
    for lane, item in report.get("inter_side_parity", {}).items():
        lines.append(
            f"- {lane}: status=`{item.get('status')}`, "
            f"reason=`{item.get('reason')}`, row_count=`{item.get('row_count')}`"
        )
    lines.extend(["", "## Full-vs-Stepwise Output Parity", ""])
    fvs = report.get("full_vs_stepwise_output", {})
    lines.append(f"overall: `{fvs.get('status')}` / `{fvs.get('reason')}`")
    for surface, item in fvs.get("surface_status", {}).items():
        lines.append(
            f"- {surface}: status=`{item.get('status')}`, reason=`{item.get('reason')}`"
        )
    lines.extend(
        [
            "",
            "## P75 Gate Update",
            "",
            f"- logits_output gate: status=`{logits_gate.get('status')}`, "
            f"reason=`{logits_gate.get('reason')}`",
            f"- kernel_ready: `{gate.get('kernel_ready')}`",
            f"- blocking_gates: `{gate.get('blocking_gates')}`",
            "",
            "## Decision",
            "",
            f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
            "No recurrence math, balance-state math, dtype policy, tolerances, "
            "fixture values, RADLADS source, Pallas code, training path, or "
            "default experimental balance_state behavior changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _p78_fix_note_markdown(report: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    logits_gate = gate.get("output_gates", {}).get("logits_output", {})
    return "\n".join(
        [
            "# P78 Fix Note",
            "",
            "P78 adds same-run, lane-aware hidden/output/logits evidence to the "
            "existing P75 residual gate.",
            "",
            "It compares RADLADS terms vs QRWKV off terms and RADLADS direct vs "
            "QRWKV experimental direct. Hidden/output evidence remains separate "
            "from true LM-head logits; logits are marked unavailable when "
            "`StudentOutput.logits` is absent.",
            "",
            f"- p78_status: `{report.get('status')}`",
            f"- logits_output gate: `{logits_gate.get('status')}` / "
            f"`{logits_gate.get('reason')}`",
            f"- recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p74_direct_lane_comparison(report: Mapping[str, Any]) -> dict[str, Any]:
    direct = report.get("lane_validity", {}).get("direct_balance_state_first_failure")
    terms = report.get("lane_validity", {}).get("balance_state_terms_first_failure")
    return {
        "schema": "qrwkv_xla.p74_direct_balance_lane_comparison.v1",
        "same_run_valid": bool(report.get("same_run_valid")),
        "fixture_id": report.get("fixture_id"),
        "parameter_id": report.get("parameter_id"),
        "lane_aware_keys": True,
        "terms_lane": {
            "left": "radlads_terms",
            "right": "qrwkv_off_terms",
            "valid": bool(report.get("balance_state_terms_lane_valid")),
            "first_differing_stage": report.get(
                "balance_state_terms_first_differing_stage"
            ),
            "first_differing_capture_kind": report.get(
                "balance_state_terms_first_differing_capture_kind"
            ),
            "math_conclusion_valid": bool(
                report.get("balance_state_terms_math_conclusion_valid")
            ),
            "recommended_next_phase": report.get(
                "balance_state_terms_recommended_next_phase"
            ),
            "first_failure": terms,
        },
        "direct_lane": {
            "left": "radlads_direct",
            "right": "qrwkv_experimental_direct",
            "valid": bool(report.get("direct_balance_state_lane_valid")),
            "first_differing_stage": report.get(
                "direct_balance_state_first_differing_stage"
            ),
            "first_differing_capture_kind": report.get(
                "direct_balance_state_first_differing_capture_kind"
            ),
            "math_conclusion_valid": bool(
                report.get("direct_balance_state_math_conclusion_valid")
            ),
            "not_applicable_stages": sorted(LANE_A_ONLY_STAGES),
            "recommended_next_phase": report.get(
                "direct_balance_state_recommended_next_phase"
            ),
            "first_failure": direct,
        },
        "recommended_next_phase": report.get("recommended_next_phase"),
        "kernel_ready": report.get("kernel_ready"),
    }


def _p74_direct_lane_markdown(report: Mapping[str, Any]) -> str:
    lane_map = report.get("balance_state_lane_map", {})
    counts = report.get("live_rows_captured", {})
    not_applicable = report.get("lane_validity", {}).get("not_applicable_rows", [])
    return "\n".join(
        [
            "# P74 Direct Balance-State Lane Report",
            "",
            "## Lane Inventory",
            "",
            f"RADLADS terms: `{lane_map.get('radlads_terms', {}).get('lane')}`",
            f"QRWKV off terms: `{lane_map.get('qrwkv_off_terms', {}).get('lane')}`",
            f"RADLADS direct: `{lane_map.get('radlads_direct', {}).get('lane')}`",
            "QRWKV experimental direct: `"
            f"{lane_map.get('qrwkv_experimental_direct', {}).get('lane')}`",
            "",
            "## Direct Lane Comparison",
            "",
            f"same_run_valid: `{report.get('same_run_valid')}`",
            f"fixture_id: `{report.get('fixture_id')}`",
            f"parameter_id: `{report.get('parameter_id')}`",
            f"RADLADS direct live rows: `{counts.get('radlads_direct', 0)}`",
            "QRWKV experimental direct live rows: `"
            f"{counts.get('qrwkv_experimental_direct', 0)}`",
            f"not-applicable stages: `{sorted(LANE_A_ONLY_STAGES)}`",
            "first comparable differing stage: `"
            f"{report.get('direct_balance_state_first_differing_stage')}`",
            "math conclusion valid: `"
            f"{report.get('direct_balance_state_math_conclusion_valid')}`",
            "recommended next phase: `"
            f"{report.get('direct_balance_state_recommended_next_phase')}`",
            "",
            "## Terms Lane Comparison",
            "",
            f"RADLADS terms live rows: `{counts.get('radlads_terms', 0)}`",
            f"QRWKV off terms live rows: `{counts.get('qrwkv_off_terms', 0)}`",
            "first comparable differing stage: `"
            f"{report.get('balance_state_terms_first_differing_stage')}`",
            "math conclusion valid: `"
            f"{report.get('balance_state_terms_math_conclusion_valid')}`",
            "recommended next phase: `"
            f"{report.get('balance_state_terms_recommended_next_phase')}`",
            "",
            "## Overall Decision",
            "",
            f"recommended next phase: `{report.get('recommended_next_phase')}`",
            f"kernel_ready: `{report.get('kernel_ready')}`",
            "lane-aware row keys: `True`",
            f"not_applicable_rows: `{len(not_applicable)}`",
            "",
        ]
    )


def _p74_fix_note_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P74 Fix Note",
            "",
            "P74 adds the missing RADLADS direct-balance-state capture by "
            "running the same local reference path with "
            "`radlads_balance_state=True`. It preserves the existing RADLADS "
            "balance-state-terms capture.",
            "",
            "Row identity is lane-aware via `balance_state_lane`, so RADLADS "
            "terms rows and RADLADS direct rows for the same fixture/context "
            "do not collide in comparison maps.",
            "",
            f"- same_run_valid: `{report.get('same_run_valid')}`",
            "- direct_balance_state_lane_valid: `"
            f"{report.get('direct_balance_state_lane_valid')}`",
            "- balance_state_terms_lane_valid: `"
            f"{report.get('balance_state_terms_lane_valid')}`",
            f"- recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
            "No recurrence math, balance-prep math, dtype policy, tolerance, "
            "Pallas/kernel code, RADLADS upstream/vendor code, or default "
            "experimental balance_state behavior is changed.",
            "",
        ]
    )


def _p75_residual_gate_markdown(
    report: Mapping[str, Any], gate: Mapping[str, Any]
) -> str:
    tolerances = gate.get("tolerances", {})
    lines = [
        "# P75 Residual-Impact Gate",
        "",
        "## Inputs",
        "",
        f"same_run_valid: `{gate.get('same_run_valid')}`",
        f"fixture_id: `{gate.get('fixture_id')}`",
        f"parameter_id: `{gate.get('parameter_id')}`",
        f"lane_aware_keys: `{gate.get('lane_aware_keys')}`",
        f"tolerances: `{json.dumps(tolerances, sort_keys=True)}`",
        "",
        "## Lane Comparisons",
        "",
    ]
    for lane in (BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE):
        item = gate.get("lane_comparisons", {}).get(lane, {})
        lines.extend(
            [
                f"{lane}:",
                f"- left: `{item.get('left')}`",
                f"- right: `{item.get('right')}`",
                f"- valid: `{item.get('valid')}`",
                f"- first_differing_stage: `{item.get('first_differing_stage')}`",
                f"- math_conclusion_valid: `{item.get('math_conclusion_valid')}`",
                "",
            ]
        )
    lines.extend(["## Residual Measurements", ""])
    for lane in (BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE):
        lane_report = gate.get("residuals", {}).get(lane, {})
        lines.extend([f"{lane}:", ""])
        measurements = lane_report.get("measurements", {})
        for stage in P75_RESIDUAL_STAGES:
            item = measurements.get(stage, {})
            lines.append(
                "- "
                f"{stage}: status=`{item.get('status')}`, "
                f"severity=`{item.get('severity')}`, "
                f"max_abs=`{_fmt(item.get('max_abs'))}`, "
                f"rms=`{_fmt(item.get('rms'))}`, "
                f"reason=`{item.get('reason')}`"
            )
        lines.append("")
    lines.extend(["## Output/State Gates", ""])
    for gate_name, item in gate.get("output_gates", {}).items():
        lines.append(
            f"- {gate_name}: status=`{item.get('status')}`, "
            f"required=`{item.get('required')}`, reason=`{item.get('reason')}`"
        )
    lines.extend(
        [
            "",
            "## Blocking Conditions",
            "",
            f"blocking_gates: `{gate.get('blocking_gates')}`",
            f"warning_gates: `{gate.get('warning_gates')}`",
            "",
            "## Decision",
            "",
            f"kernel_ready: `{gate.get('kernel_ready')}`",
            f"reason: `{gate.get('kernel_readiness', {}).get('reason')}`",
            f"recommended_next_phase: `{gate.get('recommended_next_phase')}`",
            "",
            "This gate preserves the P74 lane comparisons exactly: RADLADS "
            "terms vs QRWKV off terms, and RADLADS direct vs QRWKV "
            "experimental direct.",
            f"P68 recommended_next_phase: `{report.get('recommended_next_phase')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _p75_kernel_decision_markdown(
    report: Mapping[str, Any], gate: Mapping[str, Any]
) -> str:
    lines = [
        "# P75 Kernel Readiness Decision",
        "",
        f"kernel_ready: `{gate.get('kernel_ready')}`",
        f"reason: `{gate.get('kernel_readiness', {}).get('reason')}`",
        f"blocking_gates: `{gate.get('blocking_gates')}`",
        f"warning_gates: `{gate.get('warning_gates')}`",
        f"recommended_next_phase: `{gate.get('recommended_next_phase')}`",
        "",
    ]
    p79 = report.get("p79_broader_fixture_residual_matrix")
    if isinstance(p79, Mapping):
        lines.extend(
            [
                "P79 broader_fixture_recommendation:",
                f"- all_expected_cases_pass: `{p79.get('all_expected_cases_pass')}`",
                f"- recommended_action: `{p79.get('recommended_action')}`",
                f"- effective_next_phase: `{p79.get('recommended_action')}`",
                "",
            ]
        )
    lines.extend(
        [
            "why_this_is_not_a_math_fix: P75 only evaluates residual impact and "
            "readiness evidence from lane-aligned live traces. It does not "
            "change recurrence math, balance-state math, parameter mapping, "
            "dtype policy, tolerances, fixture values, RADLADS upstream code, "
            "or default experimental balance_state behavior.",
            "",
            "why_this_is_or_is_not_ready_for_Pallas: Pallas work is blocked "
            "unless `kernel_ready` is `yes`. Missing or failing state/export, "
            "full-vs-stepwise, logits/output evidence, or P79 broader fixture "
            "coverage keeps the recommendation on the exact P80 follow-up.",
            "",
        ]
    )
    return "\n".join(lines)


def _p75_blocker_report_markdown(gate: Mapping[str, Any]) -> str:
    lines = [
        "# P75 Blocker Report",
        "",
        f"kernel_ready: `{gate.get('kernel_ready')}`",
        "",
        "## Blocking Gates",
        "",
    ]
    lines.extend(f"- `{item}`" for item in gate.get("blocking_gates", []))
    lines.extend(
        ["", f"recommended_next_phase: `{gate.get('recommended_next_phase')}`", ""]
    )
    return "\n".join(lines)


def _p75_fix_note_markdown(report: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    del report
    return "\n".join(
        [
            "# P75 Fix Note",
            "",
            "P75 adds report-only residual-impact and kernel-readiness gates on "
            "top of the existing P74 lane-aware trace comparison.",
            "",
            f"- kernel_ready: `{gate.get('kernel_ready')}`",
            f"- recommended_next_phase: `{gate.get('recommended_next_phase')}`",
            "- fix type: `reporting/gate only`",
            "",
            "No recurrence math, balance-state math, parameter mapping, dtype "
            "policy, tolerances, Pallas/kernel code, RADLADS upstream/vendor "
            "code, fixture values, hidden-state layout, or default "
            "experimental balance_state behavior changed.",
            "",
        ]
    )


def _p79_matrix_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P79 Broader Fixture Residual Matrix",
        "",
        "| requested_case | canonical_case | resolved_case | resolution | "
        "fixture_id | same_run_valid | state_after | exported_state | "
        "full_vs_stepwise | logits_output | kernel_ready_for_case | "
        "recommended_action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in _p79_case_rows(report):
        lines.append(
            "| "
            f"{row.get('requested_case', row.get('case'))} | "
            f"{row.get('canonical_case')} | "
            f"{row.get('resolved_case')} | "
            f"{row.get('resolution')} | "
            f"{row.get('fixture_id')} | "
            f"{row.get('same_run_valid')} | "
            f"{row.get('state_after')} | "
            f"{row.get('exported_state')} | "
            f"{row.get('full_vs_stepwise')} | "
            f"{row.get('logits_output')} | "
            f"{row.get('kernel_ready_for_case')} | "
            f"{row.get('recommended_action')} |"
        )
    lines.extend(
        [
            "",
            f"overall_recommended_action: `{report.get('recommended_action')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _p79_case_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("case_rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    cases = report.get("cases", {})
    if isinstance(cases, Mapping):
        return [row for row in cases.values() if isinstance(row, Mapping)]
    if isinstance(cases, list):
        return [row for row in cases if isinstance(row, Mapping)]
    return []


def _p79_validation_report_markdown(report: Mapping[str, Any]) -> str:
    cases_missing = report.get("cases_missing", [])
    cases_found = report.get("cases_found", [])
    lines = [
        "# P79 Broader Fixture Residual-Impact Validation",
        "",
        "## Inputs",
        "",
        f"schema: `{report.get('schema')}`",
        f"fixture_manifest: `{report.get('fixture_manifest')}`",
        f"parameters: `{report.get('parameters')}`",
        f"cases_requested: `{report.get('cases_requested')}`",
        f"cases_found: `{cases_found}`",
        f"cases_missing: `{cases_missing}`",
        f"same_run_policy: `{report.get('same_run_policy')}`",
        f"tolerances: `{report.get('tolerances')}`",
        f"active_expected_cases: `{report.get('active_expected_cases')}`",
        f"accepted_aliases: `{report.get('accepted_aliases')}`",
        f"deprecated_cases: `{report.get('deprecated_cases')}`",
        f"optional_cases: `{report.get('optional_cases')}`",
        f"remaining_missing_cases: `{report.get('remaining_missing_cases')}`",
        f"fixture_id: `{report.get('fixture_id')}`",
        f"parameter_id: `{report.get('parameter_id')}`",
        f"same_run_group_id: `{report.get('same_run_group_id')}`",
        f"all_expected_cases_present: `{report.get('all_expected_cases_present')}`",
        f"all_expected_cases_pass: `{report.get('all_expected_cases_pass')}`",
        f"recommended_action: `{report.get('recommended_action')}`",
        "",
        "## Fixture Matrix",
        "",
        _p79_matrix_markdown(report),
        "## Lane Details",
        "",
        *_p79_lane_detail_lines(report),
        "## Blocking Gates",
        "",
        *(
            [f"- {case}: fixture_case_not_found" for case in cases_missing]
            if cases_missing
            else ["None"]
        ),
        "",
        "## Warnings",
        "",
        "None",
        "",
        "## Decision",
        "",
        f"recommended_next_phase: `{report.get('recommended_next_phase')}`",
        f"kernel_ready_scope: `{report.get('kernel_ready_scope')}`",
        "",
        "## Lane Policy",
        "",
        "P79 preserves the fair lanes from P74-P78: RADLADS terms vs QRWKV "
        "off terms, and RADLADS direct vs QRWKV experimental direct.",
        "",
    ]
    return "\n".join(lines)


def _p79_lane_detail_lines(report: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for row in _p79_case_rows(report):
        lines.append(f"### {row.get('requested_case', row.get('case'))}")
        lines.append("")
        lines.extend(
            [
                f"- canonical_case: `{row.get('canonical_case')}`",
                f"- resolved_case: `{row.get('resolved_case')}`",
                f"- resolution: `{row.get('resolution')}`",
                "",
            ]
        )
        lane_details = row.get("lane_details", {})
        if not isinstance(lane_details, Mapping):
            lines.append("lane_details: `unavailable`")
            lines.append("")
            continue
        for lane in (BALANCE_STATE_TERMS_LANE, DIRECT_BALANCE_STATE_LANE):
            detail = lane_details.get(lane, {})
            lines.append(f"{lane}:")
            if isinstance(detail, Mapping):
                lines.extend(
                    [
                        f"- valid: `{detail.get('valid')}`",
                        "- first_differing_stage: `"
                        f"{detail.get('first_differing_stage')}`",
                        f"- state_after: `{detail.get('state_after')}`",
                        f"- exported_state: `{detail.get('exported_state')}`",
                        f"- full_vs_stepwise: `{detail.get('full_vs_stepwise')}`",
                        f"- logits_output: `{detail.get('logits_output')}`",
                    ]
                )
            else:
                lines.append("- unavailable")
        lines.append("")
    return lines


def _p79_blocker_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P79 Blocker Report",
        "",
        f"recommended_action: `{report.get('recommended_action')}`",
        "",
        "## Blocking Cases",
        "",
    ]
    for row in _p79_case_rows(report):
        if row.get("kernel_ready_for_case") == "yes":
            continue
        lane_details = row.get("lane_details", {})
        lane_items = lane_details.items() if isinstance(lane_details, Mapping) else []
        first_lane = next(
            (
                lane
                for lane, detail in lane_items
                if isinstance(detail, Mapping) and not detail.get("valid")
            ),
            None,
        )
        lines.append(
            "- "
            f"fixture_case=`{row.get('case')}`, lane=`{first_lane}`, "
            f"gate=`{row.get('blocking_gates')}`, stage=`{row.get('state_after')}/"
            f"{row.get('exported_state')}/{row.get('full_vs_stepwise')}/"
            f"{row.get('logits_output')}`, first_failure=`"
            f"{row.get('unavailable_reason') or row.get('blocking_gates')}`, "
            "max_abs=`None`, shape_match=`None`, finite=`None`, reason=`"
            f"{row.get('unavailable_reason') or row.get('blocking_gates')}`, "
            f"kernel_ready_for_case=`"
            f"{row.get('kernel_ready_for_case')}`, blocking_gates=`"
            f"{row.get('blocking_gates')}`, recommended_next_phase=`"
            f"{row.get('recommended_action')}`"
        )
    lines.append("")
    return "\n".join(lines)


def _p80_fixture_lineage_resolution(report: Mapping[str, Any]) -> dict[str, Any]:
    alias_rows = [
        row for row in _p79_case_rows(report) if row.get("resolution") == "alias"
    ]
    missing_case = (
        alias_rows[0].get("requested_case")
        if alias_rows
        else (report.get("cases_missing") or [None])[0]
    )
    canonical_case = alias_rows[0].get("canonical_case") if alias_rows else None
    resolution = "alias" if alias_rows else "missing"
    evidence = [
        {
            "artifact": "artifacts/parity/radlads_source_bridge/manifest.json",
            "case": "tiny_prefix_padding_or_left_padding",
            "finding": (
                "historical case uses attention_mask.kind prefix_or_left_padding"
            ),
        },
        {
            "artifact": (
                "artifacts/p54_radlads_loader_export_repair/qrwkv_outputs/manifest.json"
            ),
            "case": "tiny_prefix_or_left_padding",
            "finding": (
                "current QRWKV export case uses attention_mask.kind "
                "prefix_or_left_padding"
            ),
        },
        {
            "artifact": (
                "artifacts/p54_radlads_loader_export_repair/"
                "radlads_outputs/manifest.json"
            ),
            "case": "tiny_prefix_or_left_padding",
            "finding": (
                "current RADLADS export case uses attention_mask.kind "
                "prefix_or_left_padding"
            ),
        },
        {
            "artifact": "artifacts/p65_balance_state_experiment/off/mode_report.json",
            "case": "tiny_prefix_padding_or_left_padding",
            "finding": (
                "historical experiment retained the longer case name for the same "
                "mask kind"
            ),
        },
        {
            "artifact": str(report.get("fixture_manifest")),
            "case": canonical_case,
            "finding": "current live fixture manifest contains the canonical case",
        },
    ]
    remaining_missing = list(report.get("remaining_missing_cases") or [])
    return {
        "schema": P80_FIXTURE_LINEAGE_RESOLUTION_SCHEMA,
        "phase": "P80",
        "missing_expected_case": missing_case,
        "resolution": resolution,
        "canonical_case": canonical_case,
        "evidence": evidence,
        "active_expected_cases": list(report.get("active_expected_cases") or []),
        "accepted_aliases": dict(report.get("accepted_aliases") or {}),
        "deprecated_cases": list(report.get("deprecated_cases") or []),
        "optional_cases": list(report.get("optional_cases") or []),
        "remaining_missing_cases": remaining_missing,
        "kernel_ready": "yes" if report.get("all_expected_cases_pass") else "no",
        "kernel_ready_scope": report.get("kernel_ready_scope"),
        "recommended_next_phase": report.get("recommended_next_phase"),
    }


def _p80_fixture_lineage_report_markdown(
    resolution: Mapping[str, Any], p79_report: Mapping[str, Any]
) -> str:
    evidence = resolution.get("evidence", [])
    evidence_lines = [
        f"- `{item.get('artifact')}`: `{item.get('case')}` - {item.get('finding')}"
        for item in evidence
        if isinstance(item, Mapping)
    ]
    alias_rows = [
        row for row in _p79_case_rows(p79_report) if row.get("resolution") == "alias"
    ]
    alias_lines = [
        "- requested_case: `"
        f"{row.get('requested_case')}`, canonical_case: "
        f"`{row.get('canonical_case')}`, resolved_case: "
        f"`{row.get('resolved_case')}`, resolution: `{row.get('resolution')}`"
        for row in alias_rows
    ]
    return "\n".join(
        [
            "# P80 Fixture Lineage / Harness Repair Report",
            "",
            "## Problem",
            "",
            "P79 used a flat expected-case list and treated the historical "
            "`tiny_prefix_padding_or_left_padding` name as a missing fixture, "
            "even though the current manifest carries the same fixture family "
            "under `tiny_prefix_or_left_padding`.",
            "",
            "## Evidence Inspected",
            "",
            *(evidence_lines or ["None"]),
            "",
            "## Resolution",
            "",
            f"missing_expected_case: `{resolution.get('missing_expected_case')}`",
            f"resolution: `{resolution.get('resolution')}`",
            f"canonical_case: `{resolution.get('canonical_case')}`",
            f"accepted_aliases: `{resolution.get('accepted_aliases')}`",
            "",
            "Alias rows reuse the canonical case evidence and are marked "
            "`resolved_by_alias`; no duplicate fixture run evidence is created.",
            "",
            *(alias_lines or ["No alias rows were resolved."]),
            "",
            "## Regenerated Broader Fixture Matrix",
            "",
            f"active_expected_cases: `{resolution.get('active_expected_cases')}`",
            f"remaining_missing_cases: `{resolution.get('remaining_missing_cases')}`",
            f"kernel_ready: `{resolution.get('kernel_ready')}`",
            f"kernel_ready_scope: `{resolution.get('kernel_ready_scope')}`",
            "",
            "## Decision",
            "",
            f"recommended_next_phase: `{resolution.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p80_fix_note_markdown(resolution: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P80 Fix Note",
            "",
            "Harness/reporting now uses structured fixture expectation metadata "
            "with active, alias, deprecated, optional, and missing case buckets.",
            "",
            f"resolution: `{resolution.get('resolution')}`",
            f"canonical_case: `{resolution.get('canonical_case')}`",
            f"remaining_missing_cases: `{resolution.get('remaining_missing_cases')}`",
            "",
        ]
    )


def _p80_blocker_report_markdown(resolution: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P80 Blocker Report",
            "",
            f"remaining_missing_cases: `{resolution.get('remaining_missing_cases')}`",
            f"recommended_next_phase: `{resolution.get('recommended_next_phase')}`",
            "",
        ]
    )


def _p79_fix_note_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P79 Fix Note",
            "",
            "P79 adds breadth validation and reporting over the existing fixture "
            "family using the same P75-P78 gates.",
            "",
            "- all_expected_cases_present: `"
            f"{report.get('all_expected_cases_present')}`",
            f"- all_expected_cases_pass: `{report.get('all_expected_cases_pass')}`",
            f"- recommended_action: `{report.get('recommended_action')}`",
            "- fix type: `reporting/breadth validation only`",
            "",
            "No recurrence math, parameter remapping, dtype policy, tolerance, "
            "fixture values, RADLADS upstream code, Pallas code, training path, "
            "or default experimental balance_state behavior changed.",
            "",
        ]
    )
