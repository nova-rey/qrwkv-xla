from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from qrwkv_xla.parity.radlads_numerical_fixtures import (
    load_numerical_case_arrays,
    load_numerical_manifest,
)
from qrwkv_xla.parity.radlads_parameter_import import (
    QRWKV_DEFAULTED_SURFACES,
    import_radlads_parameters_for_replay,
    write_parameter_import_report,
)
from qrwkv_xla.parity.radlads_replay_diagnostics import (
    ReplayDiagnosticsCollector,
    find_first_nonfinite,
)
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

REPLAY_REPORT_SCHEMA = "radlads_parameter_replay_comparison.v1"

P49_REPLAY_SURFACES = (
    ("hidden_states", "radlads_hidden_states", "qrwkv_hidden_states"),
    ("logits", "radlads_logits", "qrwkv_logits"),
    ("wkv_matrix_state", "radlads_wkv_matrix_state", "qrwkv_wkv_matrix_state"),
    ("shift_state", "radlads_shift_state", "qrwkv_shift_state"),
    (
        "stepwise_hidden_states",
        "radlads_stepwise_hidden_states",
        "qrwkv_stepwise_hidden_states",
    ),
    ("stepwise_logits", "radlads_stepwise_logits", "qrwkv_stepwise_logits"),
    (
        "stepwise_wkv_matrix_state",
        "radlads_stepwise_wkv_matrix_state",
        "qrwkv_stepwise_wkv_matrix_state",
    ),
    (
        "stepwise_shift_state",
        "radlads_stepwise_shift_state",
        "qrwkv_stepwise_shift_state",
    ),
)


@dataclass(frozen=True)
class ReplayCaseProfile:
    case_name: str
    all_radlads_math: bool
    attention_qkv_bias: bool
    active_defaulted_surfaces: tuple[str, ...]
    reason: str


_SIMPLE_CASES = {
    "tiny_no_mask",
    "tiny_attention_mask",
    "tiny_prefix_or_left_padding",
    "tiny_stepwise_state",
}


_SIMPLE_CASE_DEFAULTS = (
    "layers.self_attn.b_proj.weight",
    "layers.self_attn.time_mix",
    "layers.self_attn.time_bias",
    "lm_head.bias",
)


_ALL_MATH_DEFAULTS = ("lm_head.bias",)


def replay_profile_for_case(case: Mapping[str, Any]) -> ReplayCaseProfile:
    case_name = str(case.get("name"))
    all_math = bool(
        case.get("all_radlads_math")
        if "all_radlads_math" in case
        else case_name == "tiny_all_radlads_math_enabled"
    )
    if all_math:
        return ReplayCaseProfile(
            case_name=case_name,
            all_radlads_math=True,
            attention_qkv_bias=True,
            active_defaulted_surfaces=_ALL_MATH_DEFAULTS,
            reason=(
                "This fixture was generated with explicit RADLADS math flags enabled, "
                "so replay keeps the low-rank path active."
            ),
        )
    if case_name in _SIMPLE_CASES:
        return ReplayCaseProfile(
            case_name=case_name,
            all_radlads_math=False,
            attention_qkv_bias=True,
            active_defaulted_surfaces=_SIMPLE_CASE_DEFAULTS,
            reason=(
                "P49 generated this fixture with all_radlads_math=False, so replay "
                "must keep the low-rank RADLADS path disabled instead of forcing "
                "the all-math profile."
            ),
        )
    return ReplayCaseProfile(
        case_name=case_name,
        all_radlads_math=False,
        attention_qkv_bias=True,
        active_defaulted_surfaces=_SIMPLE_CASE_DEFAULTS,
        reason="Unknown fixture defaults to the safer non-all-math replay profile.",
    )


def student_for_replay_profile(
    base_config: RWKV7QwenReferenceConfig,
    profile: ReplayCaseProfile,
) -> RWKV7QwenReferenceStudent:
    return RWKV7QwenReferenceStudent(
        replace(
            base_config,
            radlads_compatible_math=profile.all_radlads_math,
            radlads_attention_group_norm=profile.all_radlads_math,
            radlads_balance_state=profile.all_radlads_math,
            radlads_replay_mode=profile.all_radlads_math,
            radlads_low_rank_gate=profile.all_radlads_math,
            attention_qkv_bias=profile.attention_qkv_bias,
        )
    )


def replay_radlads_tiny_numerical_fixtures(
    manifest_path: Path,
    *,
    parameter_npz: Path | None = None,
    out_dir: Path | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    allow_defaults: bool = True,
    report_prefix: str = "P50",
) -> dict[str, Any]:
    manifest = load_numerical_manifest(manifest_path)
    fixture_dir = manifest_path.parent
    parameter_path = parameter_npz or fixture_dir / str(
        manifest.get("parameter_payload", "radlads_parameters.npz")
    )
    if not parameter_path.exists():
        report = _not_replayed_report(
            manifest_path,
            parameter_path,
            status="missing_source",
            reason="RADLADS parameter payload is absent.",
        )
        if out_dir is not None:
            write_replay_reports(report, out_dir, report_prefix=report_prefix)
        return report

    try:
        import_result = import_radlads_parameters_for_replay(
            parameter_path,
            manifest_path=manifest_path,
            allow_defaults=allow_defaults,
        )
    except Exception as exc:  # pragma: no cover
        report = _not_replayed_report(
            manifest_path,
            parameter_path,
            status="not_replayed_due_to_import_failure",
            reason=f"parameter import crashed: {type(exc).__name__}: {exc}",
        )
        if out_dir is not None:
            write_replay_reports(report, out_dir, report_prefix=report_prefix)
        return report

    if import_result.overall_status != "pass":
        report = _not_replayed_report(
            manifest_path,
            parameter_path,
            status="not_replayed_due_to_import_failure",
            reason="Parameter import report did not pass.",
            import_report=import_result.report,
        )
        if out_dir is not None:
            write_replay_reports(report, out_dir, report_prefix=report_prefix)
            write_parameter_import_report(import_result.report, out_dir)
        return report

    cases = [
        _replay_case(
            manifest_path,
            case,
            base_config=import_result.qrwkv_config,
            params=import_result.params,
            atol=atol,
            rtol=rtol,
        )
        for case in manifest["cases"]
    ]

    counts = _counts(
        comparison["status"]
        for case in cases
        for comparison in case.get("comparisons", [])
    )
    attempted = sum(
        1
        for case in cases
        for comparison in case.get("comparisons", [])
        if comparison["status"]
        not in {
            "unsupported",
            "missing_source",
            "not_replayed_due_to_import_failure",
        }
    )
    overall = "pass" if counts and set(counts) == {"pass"} else "fail"
    if attempted == 0:
        overall = "unsupported"
    best_passing_surface = _best_passing_surface(cases)
    largest_failure = _largest_failure(cases)
    baseline_non_finite_count = _load_baseline_non_finite_count(manifest_path)
    report = {
        "schema": REPLAY_REPORT_SCHEMA,
        "manifest": str(manifest_path),
        "parameter_payload": str(parameter_path),
        "overall_status": overall,
        "counts": counts,
        "surface_status_counts": counts,
        "attempted_comparisons": attempted,
        "cases_attempted": len(cases),
        "cases_finite": sum(
            1
            for case in cases
            if all(
                row.get("status")
                not in {"non_finite", "not_replayed_due_to_import_failure"}
                for row in case.get("comparisons", [])
            )
        ),
        "baseline_non_finite_count": baseline_non_finite_count,
        "non_finite_count_after": counts.get("non_finite", 0),
        "best_passing_surface": best_passing_surface,
        "largest_failure": largest_failure,
        "import_report": import_result.report,
        "cases": cases,
    }
    if out_dir is not None:
        write_replay_reports(report, out_dir, report_prefix=report_prefix)
        write_parameter_import_report(import_result.report, out_dir)
    return report


def diagnose_replay_case(
    manifest_path: Path,
    case: Mapping[str, Any],
    *,
    base_config: RWKV7QwenReferenceConfig,
    params: dict[str, object],
) -> dict[str, Any]:
    arrays = load_numerical_case_arrays(manifest_path, case)
    profile = replay_profile_for_case(case)
    student = student_for_replay_profile(base_config, profile)
    diagnostics = ReplayDiagnosticsCollector()
    qrwkv = _run_qrwkv_replay(
        student,
        params,
        arrays,
        diagnostics=diagnostics,
    )
    final_outputs_finite = all(np.isfinite(value).all() for value in qrwkv.values())
    first_nonfinite = find_first_nonfinite(diagnostics.summaries)
    return {
        "case": case["name"],
        "first_nonfinite": first_nonfinite,
        "final_outputs_finite": final_outputs_finite,
        "instrumented_stages": diagnostics.instrumented_stages,
        "tensor_summaries": diagnostics.summaries,
        "replay_profile": {
            "all_radlads_math": profile.all_radlads_math,
            "attention_qkv_bias": profile.attention_qkv_bias,
            "reason": profile.reason,
            "active_defaulted_surfaces": list(profile.active_defaulted_surfaces),
            "qrwkv_only_default_used_in_active_path": bool(
                profile.active_defaulted_surfaces
            ),
        },
        "suspected_root_cause": (
            "active_path_profile_mismatch"
            if not profile.all_radlads_math
            else "source_parameter_nonfinite_or_overflow"
        ),
    }


def write_replay_reports(
    report: Mapping[str, Any],
    out_dir: Path,
    *,
    report_prefix: str = "P50",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    (out_dir / "replay_comparison_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{report_prefix}_RESULTS.md").write_text(
        _results_markdown(payload, report_prefix=report_prefix),
        encoding="utf-8",
    )
    (out_dir / f"{report_prefix}_SURFACE_COMPARISON.md").write_text(
        _surface_markdown(payload, report_prefix=report_prefix),
        encoding="utf-8",
    )


def _replay_case(
    manifest_path: Path,
    case: Mapping[str, Any],
    *,
    base_config: RWKV7QwenReferenceConfig,
    params: dict[str, object],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if case.get("generation_status") not in {None, "generated"}:
        return {
            "name": case["name"],
            "status": "missing_source",
            "reason": case.get("generation_failed_reason", "case was not generated"),
            "comparisons": [],
        }
    arrays = load_numerical_case_arrays(manifest_path, case)
    if not any(key.startswith("radlads_") for key in arrays):
        return {
            "name": case["name"],
            "status": "missing_source",
            "reason": "case has no RADLADS arrays",
            "comparisons": [],
        }
    profile = replay_profile_for_case(case)
    student = student_for_replay_profile(base_config, profile)
    try:
        qrwkv = _run_qrwkv_replay(student, params, arrays)
    except Exception as exc:
        return {
            "name": case["name"],
            "status": "fail",
            "reason": f"QRWKV replay execution failed: {exc}",
            "replay_profile": _profile_payload(profile),
            "comparisons": [
                {
                    "name": name,
                    "status": "not_replayed_due_to_import_failure",
                    "reason": str(exc),
                }
                for name, _left, _right in P49_REPLAY_SURFACES
            ],
        }

    comparisons = [
        _compare_surface(
            name,
            arrays.get(left),
            qrwkv.get(right),
            atol=atol,
            rtol=rtol,
        )
        for name, left, right in P49_REPLAY_SURFACES
    ]
    status = "pass" if all(row["status"] == "pass" for row in comparisons) else "fail"
    return {
        "name": case["name"],
        "status": status,
        "replay_profile": _profile_payload(profile),
        "comparisons": comparisons,
    }


def _run_qrwkv_replay(
    student: RWKV7QwenReferenceStudent,
    params: dict[str, object],
    arrays: Mapping[str, np.ndarray],
    *,
    diagnostics: ReplayDiagnosticsCollector | None = None,
) -> dict[str, np.ndarray]:
    input_ids = np.asarray(arrays["input_ids"], dtype=np.int32)
    attention_mask = None
    if "attention_mask" in arrays:
        attention_mask = np.asarray(arrays["attention_mask"], dtype=np.int32)

    output, state = student.apply_with_state(
        params,
        input_ids,
        attention_mask,
        diagnostics=diagnostics,
    )
    result = {
        "qrwkv_hidden_states": _np(output.hidden_states)[:, -1],
        "qrwkv_logits": _np(output.logits),
        "qrwkv_wkv_matrix_state": _np(state.wkv_matrix_state),
        "qrwkv_shift_state": _np(state.shift_state)[:, :, None, :],
    }

    if "radlads_stepwise_hidden_states" in arrays:
        step_state = student.init_state(input_ids.shape[0])
        step_hidden = []
        step_logits = []
        for index in range(input_ids.shape[1]):
            step_mask = None
            if attention_mask is not None:
                step_mask = attention_mask[:, index : index + 1]
            step_output, step_state = student.step(
                params,
                input_ids[:, index : index + 1],
                step_state,
                attention_mask=step_mask,
                diagnostics=diagnostics,
            )
            step_hidden.append(_np(step_output.hidden_states)[:, -1])
            step_logits.append(_np(step_output.logits))
        result["qrwkv_stepwise_hidden_states"] = np.concatenate(step_hidden, axis=1)
        result["qrwkv_stepwise_logits"] = np.concatenate(step_logits, axis=1)
        result["qrwkv_stepwise_wkv_matrix_state"] = _np(step_state.wkv_matrix_state)
        result["qrwkv_stepwise_shift_state"] = _np(step_state.shift_state)[
            :, :, None, :
        ]
    return result


def _compare_surface(
    name: str,
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    base = {
        "name": name,
        "shape_match": False,
        "dtype_match": False,
        "finite_radlads": False,
        "finite_qrwkv": False,
        "max_abs_error": None,
        "mean_abs_error": None,
        "max_relative_error": None,
        "allclose": False,
        "atol": atol,
        "rtol": rtol,
        "reason": "",
    }
    if left is None:
        return {**base, "status": "missing_source", "reason": "RADLADS array missing"}
    if right is None:
        return {**base, "status": "unsupported", "reason": "QRWKV replay array missing"}

    left_value = np.asarray(left)
    right_value = np.asarray(right)
    base["finite_radlads"] = bool(np.isfinite(left_value).all())
    base["finite_qrwkv"] = bool(np.isfinite(right_value).all())

    if left_value.shape != right_value.shape:
        return {
            **base,
            "status": "shape_mismatch",
            "reason": "surface shapes differ",
            "left_shape": list(left_value.shape),
            "right_shape": list(right_value.shape),
        }
    base["shape_match"] = True

    if str(left_value.dtype) != str(right_value.dtype):
        return {
            **base,
            "status": "dtype_mismatch",
            "reason": "surface dtypes differ",
            "left_dtype": str(left_value.dtype),
            "right_dtype": str(right_value.dtype),
        }
    base["dtype_match"] = True

    if not base["finite_radlads"] or not base["finite_qrwkv"]:
        return {
            **base,
            "status": "non_finite",
            "reason": "one or both surfaces contain non-finite values",
            "shape": list(left_value.shape),
            "dtype": str(left_value.dtype),
        }

    left_float = left_value.astype(np.float32)
    right_float = right_value.astype(np.float32)
    diff = np.abs(left_float - right_float)
    denom = np.maximum(np.abs(right_float), 1e-12)
    allclose = bool(np.allclose(left_float, right_float, atol=atol, rtol=rtol))
    return {
        **base,
        "status": "pass" if allclose else "fail",
        "shape": list(left_value.shape),
        "dtype": str(left_value.dtype),
        "finite_radlads": True,
        "finite_qrwkv": True,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_relative_error": float(np.max(diff / denom)) if diff.size else 0.0,
        "allclose": allclose,
        "reason": "allclose within tolerance" if allclose else "numerical mismatch",
    }


def _not_replayed_report(
    manifest_path: Path,
    parameter_path: Path,
    *,
    status: str,
    reason: str,
    import_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": REPLAY_REPORT_SCHEMA,
        "manifest": str(manifest_path),
        "parameter_payload": str(parameter_path),
        "overall_status": status,
        "reason": reason,
        "counts": {status: 1},
        "attempted_comparisons": 0,
        "best_passing_surface": None,
        "largest_failure": None,
        "import_report": import_report,
        "cases": [],
    }


def _results_markdown(report: Mapping[str, Any], *, report_prefix: str) -> str:
    lines = [
        f"# {report_prefix} Results",
        "",
        f"- Overall status: `{report.get('overall_status')}`",
        f"- Attempted comparisons: `{report.get('attempted_comparisons', 0)}`",
        f"- Cases attempted: `{report.get('cases_attempted', 0)}`",
        f"- Cases finite: `{report.get('cases_finite', 0)}`",
        f"- Baseline P50 non_finite count: `{report.get('baseline_non_finite_count')}`",
        f"- P51 non_finite count: `{report.get('non_finite_count_after')}`",
        f"- Best passing surface: `{report.get('best_passing_surface')}`",
        f"- Largest failure: `{report.get('largest_failure')}`",
        "",
        "## Surface Status Counts",
        "",
    ]
    for status, count in sorted(dict(report.get("counts", {})).items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Replay profiles", ""])
    for case in report.get("cases", []):
        profile = case.get("replay_profile") or {}
        if not profile:
            continue
        lines.append(
            f"- `{case.get('name')}` "
            f"all_radlads_math=`{profile.get('all_radlads_math')}` "
            "active_defaults="
            f"`{profile.get('active_defaulted_surfaces')}`"
        )
    lines.extend(
        [
            "",
            f"{report_prefix} proves replay diagnostics only. It does not prove full "
            "RADLADS checkpoint compatibility, Pallas kernels, production training "
            "throughput, or model quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _surface_markdown(report: Mapping[str, Any], *, report_prefix: str) -> str:
    lines = [
        f"# {report_prefix} Surface Comparison",
        "",
        "| Case | Surface | Status | Max abs | Mean abs | Max rel | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.get("cases", []):
        for row in case.get("comparisons", []):
            lines.append(
                (
                    "| {case} | {surface} | `{status}` | {max_abs} | "
                    "{mean_abs} | {max_rel} | {reason} |"
                ).format(
                    case=case.get("name"),
                    surface=row.get("name"),
                    status=row.get("status"),
                    max_abs=row.get("max_abs_error", ""),
                    mean_abs=row.get("mean_abs_error", ""),
                    max_rel=row.get("max_relative_error", ""),
                    reason=row.get("reason", ""),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _best_passing_surface(cases: list[Mapping[str, Any]]) -> str | None:
    for case in cases:
        for row in case.get("comparisons", []):
            if row.get("status") == "pass":
                return f"{case.get('name')}:{row.get('name')}"
    return None


def _largest_failure(cases: list[Mapping[str, Any]]) -> str | None:
    biggest = None
    biggest_key = None
    for case in cases:
        for row in case.get("comparisons", []):
            if row.get("status") != "fail":
                continue
            value = row.get("max_abs_error")
            if value is None:
                continue
            if biggest is None or float(value) > biggest:
                biggest = float(value)
                biggest_key = f"{case.get('name')}:{row.get('name')}:{value}"
    return biggest_key


def _counts(statuses) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _np(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("expected emitted array")
    return np.asarray(jax.device_get(value), dtype=np.float32)


def _profile_payload(profile: ReplayCaseProfile) -> dict[str, Any]:
    active_defaults = list(profile.active_defaulted_surfaces)
    return {
        "all_radlads_math": profile.all_radlads_math,
        "attention_qkv_bias": profile.attention_qkv_bias,
        "active_defaulted_surfaces": active_defaults,
        "qrwkv_only_default_used_in_active_path": bool(active_defaults),
        "defaulted_surface_metadata": {
            name: QRWKV_DEFAULTED_SURFACES.get(name, {}) for name in active_defaults
        },
        "reason": profile.reason,
    }


def _load_baseline_non_finite_count(manifest_path: Path) -> int | None:
    candidate = (
        manifest_path.parents[2]
        / "p50_radlads_replay_compatibility"
        / "replay_comparison_report.json"
    )
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    return int(payload.get("counts", {}).get("non_finite", 0))
