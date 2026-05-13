from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional CI dependency
    torch = None

from qrwkv_xla.parity.radlads_clean_loader import (
    load_case_output_arrays,
    load_clean_output_manifest,
    load_radlads_clean_payload,
)
from qrwkv_xla.parity.radlads_fixture_validation import (
    audit_parameter_payload,
    to_audit_report,
    validate_parameter_payload,
)
from qrwkv_xla.parity.radlads_numerical_fixtures import (
    DEFAULT_RADLADS_SOURCE,
    PARAMETER_EXTREME_THRESHOLD,
    _build_radlads_model,
    _load_radlads_runtime,
    _qrwkv_case_arrays,
    _tiny_cases,
    generate_radlads_tiny_numerical_fixtures,
    load_parameter_arrays,
)

HEAD_TO_HEAD_SCHEMA = "radlads_qrwkv_head_to_head.v1"
HEAD_TO_HEAD_REPORT_SCHEMA = "radlads_qrwkv_head_to_head_report.v1"
HEAD_TO_HEAD_CASES = tuple(case["name"] for case in _tiny_cases())
VALID_INIT_POLICIES = {"deterministic_finite", "radlads_source"}
DEFAULT_OUT = Path("artifacts/p53_radlads_qrwkv_head_to_head")
DEFAULT_SEED = 5353
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-5

PAIRWISE_SURFACES = (
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


def load_head_to_head_manifest(path: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(path).read_text(encoding="utf-8"))


def validate_head_to_head_manifest(path: Path) -> dict[str, Any]:
    manifest = load_head_to_head_manifest(path)
    manifest_path = _manifest_path(path)
    _require(manifest.get("schema") == HEAD_TO_HEAD_SCHEMA, "bad schema")
    _require(manifest.get("phase") == "P53", "bad phase")
    _require(manifest.get("init_policy") in VALID_INIT_POLICIES, "bad init_policy")
    _require(isinstance(manifest.get("cases"), list), "cases must be a list")
    case_names = {case.get("name") for case in manifest["cases"]}
    _require(set(HEAD_TO_HEAD_CASES).issubset(case_names), "missing required cases")
    parameter_payload = manifest.get("parameter_payload")
    if parameter_payload is not None:
        _require(
            (manifest_path.parent / parameter_payload).is_file(),
            f"missing parameter payload {parameter_payload}",
        )
    for case in manifest["cases"]:
        _validate_case_manifest(manifest_path.parent, case)
    return manifest


def generate_radlads_qrwkv_head_to_head_fixtures(
    out: Path,
    *,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    radlads_source_path: Path = DEFAULT_RADLADS_SOURCE,
    init_policy: str = "deterministic_finite",
) -> dict[str, Any]:
    _validate_init_policy(init_policy)
    _prepare_out_dir(out, overwrite=overwrite)

    clean_fixture_dir = out / "fixtures_clean"
    generate_radlads_tiny_numerical_fixtures(
        clean_fixture_dir,
        seed=seed,
        overwrite=True,
        radlads_source_path=radlads_source_path,
        init_policy="deterministic_finite",
    )
    parameter_arrays = load_parameter_arrays(clean_fixture_dir / "manifest.json")
    parameter_path = out / "radlads_parameters.npz"
    shutil.copy2(clean_fixture_dir / "radlads_parameters.npz", parameter_path)

    audit_results = audit_parameter_payload(
        parameter_arrays,
        stage="saved_npz",
        extreme_threshold=PARAMETER_EXTREME_THRESHOLD,
    )
    audit_report = to_audit_report(audit_results)
    is_valid, blocking = validate_parameter_payload(audit_results)

    from qrwkv_xla.parity.radlads_parameter_import import (
        import_radlads_parameters_for_replay,
    )

    qrwkv_import = import_radlads_parameters_for_replay(
        parameter_path,
        allow_defaults=True,
        seed=seed,
    )

    manifest_cases: list[dict[str, Any]] = []
    radlads_runtime: dict[str, Any] | None = None
    radlads_load_report: dict[str, Any] | None = None
    radlads_blocker: str | None = None
    try:
        radlads_runtime = _load_radlads_runtime(radlads_source_path)
    except Exception as exc:  # pragma: no cover - environment-specific
        radlads_blocker = f"{type(exc).__name__}: {exc}"

    radlads_load_report = _radlads_load_report(
        parameter_path,
        radlads_runtime=radlads_runtime,
        blocker=radlads_blocker,
        seed=seed,
        radlads_source_path=radlads_source_path,
    )
    case_radlads_runtime = (
        radlads_runtime if radlads_load_report.get("status") == "pass" else None
    )
    case_radlads_blocker = radlads_load_report.get("reason") or radlads_blocker

    for case in _tiny_cases():
        case_record = _run_case_pair(
            out,
            case,
            seed=seed + int(case["seed_offset"]),
            parameter_arrays=parameter_arrays,
            qrwkv_import=qrwkv_import,
            radlads_runtime=case_radlads_runtime,
            radlads_blocker=case_radlads_blocker,
        )
        manifest_cases.append(_case_manifest_record(case_record))

    manifest = {
        "schema_version": 1,
        "schema": HEAD_TO_HEAD_SCHEMA,
        "phase": "P53",
        "source": "radlads_qrwkv_head_to_head",
        "created_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "seed": seed,
        "init_policy": init_policy,
        "radlads_commit": _git_head(radlads_source_path),
        "radlads_source_path": str(radlads_source_path),
        "parameter_payload": "radlads_parameters.npz",
        "source_fixture_manifest": "fixtures_clean/manifest.json",
        "parameter_payload_init_policy": init_policy,
        "parameter_payload_source": "P52 deterministic_finite path",
        "parameter_payload_sha256": _hash_numerical_arrays(parameter_arrays),
        "parameter_validation": {
            "status": "pass" if is_valid else "blocked",
            "all_finite": is_valid,
            "non_finite_count": audit_report["summary"]["non_finite"],
            "extreme_count": audit_report["summary"]["extreme_value"],
            "max_abs": _max_abs_from_payload(parameter_arrays),
            "blocking_count": len(blocking),
            "summary": audit_report["summary"],
        },
        "radlads_load": radlads_load_report,
        "qrwkv_load": qrwkv_import.report,
        "radlads": {
            "available": radlads_load_report.get("status") == "pass",
            "blocker": radlads_load_report.get("reason"),
            "parameter_import": radlads_load_report,
        },
        "qrwkv": {
            "available": qrwkv_import.overall_status == "pass",
            "parameter_import": qrwkv_import.report,
        },
        "required_cases": list(HEAD_TO_HEAD_CASES),
        "cases": manifest_cases,
        "notes": [
            "P53 pairs clean deterministic-finite RADLADS-shaped parameters "
            "with QRWKV replay.",
            "RADLADS live execution is optional but attempted when the local "
            "source can load.",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_head_to_head_manifest(out)


def compare_radlads_qrwkv_head_to_head(
    manifest_path: Path,
    *,
    parameter_npz: Path | None = None,
    out_dir: Path | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    report_prefix: str = "P53",
    radlads_outputs: Path | None = None,
    qrwkv_outputs: Path | None = None,
) -> dict[str, Any]:
    manifest = validate_head_to_head_manifest(manifest_path)
    fixture_dir = manifest_path.parent
    parameter_path = parameter_npz or fixture_dir / str(
        manifest.get("parameter_payload", "radlads_parameters.npz")
    )
    if not parameter_path.exists():
        report = _missing_source_report(
            manifest_path,
            parameter_path,
            reason="clean parameter payload is missing",
        )
        if out_dir is not None:
            write_head_to_head_reports(report, out_dir, report_prefix=report_prefix)
        return report

    radlads_output_arrays = None
    qrwkv_output_arrays = None
    radlads_output_report = None
    qrwkv_output_report = None
    if radlads_outputs is not None:
        radlads_output_report = load_clean_output_manifest(radlads_outputs)
        radlads_output_arrays = load_case_output_arrays(radlads_outputs)
    if qrwkv_outputs is not None:
        qrwkv_output_report = load_clean_output_manifest(qrwkv_outputs)
        qrwkv_output_arrays = load_case_output_arrays(qrwkv_outputs)

    qrwkv_import = None
    if qrwkv_output_arrays is None:
        try:
            from qrwkv_xla.parity.radlads_parameter_import import (
                import_radlads_parameters_for_replay,
            )

            qrwkv_import = import_radlads_parameters_for_replay(
                parameter_path,
                allow_defaults=True,
                seed=int(manifest.get("seed", DEFAULT_SEED)),
            )
        except Exception as exc:  # pragma: no cover - environment specific
            report = _missing_source_report(
                manifest_path,
                parameter_path,
                reason=f"QRWKV import failed: {type(exc).__name__}: {exc}",
            )
            if out_dir is not None:
                write_head_to_head_reports(report, out_dir, report_prefix=report_prefix)
            return report

    cases = []
    for case in manifest["cases"]:
        if radlads_output_arrays is not None:
            if case["name"] not in radlads_output_arrays:
                cases.append(
                    {
                        "name": case["name"],
                        "status": "missing_source",
                        "reason": "RADLADS output manifest did not include this case",
                        "comparisons": [],
                    }
                )
                continue
            radlads_arrays = radlads_output_arrays[case["name"]]
        elif case.get("radlads_status") == "unsupported":
            reason = case.get("radlads_reason", "RADLADS output unavailable")
            cases.append(
                {
                    "name": case["name"],
                    "status": "unsupported",
                    "reason": reason,
                    "comparisons": [
                        {
                            "name": name,
                            "status": "unsupported",
                            "reason": reason,
                        }
                        for name, _left, _right in PAIRWISE_SURFACES
                    ],
                }
            )
            continue
        else:
            case_path = fixture_dir / str(case["payload"])
            if not case_path.exists():
                cases.append(
                    {
                        "name": case["name"],
                        "status": "missing_source",
                        "reason": f"missing paired case payload: {case_path.name}",
                        "comparisons": [],
                    }
                )
                continue
            radlads_arrays = _load_npz(case_path)

        if qrwkv_output_arrays is not None:
            if case["name"] not in qrwkv_output_arrays:
                cases.append(
                    {
                        "name": case["name"],
                        "status": "missing_source",
                        "reason": "QRWKV output manifest did not include this case",
                        "comparisons": [],
                    }
                )
                continue
            qrwkv_arrays = _normalize_qrwkv_arrays(qrwkv_output_arrays[case["name"]])
        else:
            if qrwkv_import is None:
                cases.append(
                    {
                        "name": case["name"],
                        "status": "missing_source",
                        "reason": "QRWKV output unavailable",
                        "comparisons": [],
                    }
                )
                continue
            from qrwkv_xla.students import RWKV7QwenReferenceStudent

            qrwkv_student = RWKV7QwenReferenceStudent(qrwkv_import.qrwkv_config)
            qrwkv_arrays = _normalize_qrwkv_arrays(
                _qrwkv_case_arrays(
                    qrwkv_student,
                    qrwkv_import.params,
                    case,
                )
            )

        comparisons = []
        for name, left, right in PAIRWISE_SURFACES:
            if name.startswith("stepwise_") and case["name"] != "tiny_stepwise_state":
                comparisons.append(
                    {
                        "name": name,
                        "status": "not_applicable",
                        "reason": "stepwise surface not requested for this case",
                        "shape_match": False,
                        "finite_radlads": False,
                        "finite_qrwkv": False,
                        "allclose": False,
                        "atol": atol,
                        "rtol": rtol,
                        "dtype_match": False,
                        "shape": None,
                        "left_shape": None,
                        "right_shape": None,
                        "max_abs_error": None,
                        "mean_abs_error": None,
                        "max_relative_error": None,
                    }
                )
                continue
            comparisons.append(
                compare_surface_arrays(
                    name,
                    {**radlads_arrays, **qrwkv_arrays}.get(left),
                    {**radlads_arrays, **qrwkv_arrays}.get(right),
                    atol=atol,
                    rtol=rtol,
                )
            )
        case_status = _case_status(comparisons)
        cases.append(
            {
                "name": case["name"],
                "status": case_status,
                "comparisons": comparisons,
            }
        )

    counts = _counts(row["status"] for case in cases for row in case["comparisons"])
    for case in cases:
        if (
            case["status"] in {"unsupported", "missing_source"}
            and not case["comparisons"]
        ):
            counts[case["status"]] = counts.get(case["status"], 0) + 1
    attempted = sum(
        1
        for case in cases
        for row in case["comparisons"]
        if row["status"] not in {"unsupported", "missing_source"}
    )
    radlads_load = (
        radlads_output_report.get("load_report", {})
        if radlads_output_report is not None
        else manifest.get("radlads_load", {})
    )
    qrwkv_load = (
        qrwkv_output_report.get("load_report", {})
        if qrwkv_output_report is not None
        else (qrwkv_import.report if qrwkv_import is not None else {})
    )
    overall_status = "pass" if counts and set(counts) == {"pass"} else "fail"
    if attempted == 0:
        overall_status = "unsupported"
    report = {
        "schema": HEAD_TO_HEAD_REPORT_SCHEMA,
        "manifest": str(manifest_path),
        "parameter_payload": str(parameter_path),
        "overall_status": overall_status,
        "cases_attempted": len(cases),
        "cases_ran_both_sides": attempted,
        "cases_finite": sum(1 for case in cases if case["status"] == "pass"),
        "attempted_comparisons": attempted,
        "surface_comparisons_count": attempted,
        "non_finite_count": counts.get("non_finite_radlads", 0)
        + counts.get("non_finite_qrwkv", 0)
        + counts.get("non_finite_both", 0),
        "counts": counts,
        "surface_status_counts": counts,
        "best_passing_surface": _best_passing_surface(cases),
        "largest_failure": _largest_failure(cases),
        "hidden_states_convention": {
            "radlads": "final_hidden",
            "qrwkv": "layer_major_all_hidden",
            "comparison": "final_hidden_selected_from_layer_major",
        },
        "surface_conventions": {
            "hidden_states": {
                "radlads": "final_hidden",
                "qrwkv": "layer_major_all_hidden",
                "comparison": "final_hidden_selected_from_layer_major",
            },
            "wkv_matrix_state": {
                "radlads": "full_sequence_final_state",
                "qrwkv": "full_sequence_final_state",
                "comparison": "as_exported",
            },
            "stepwise": {
                "radlads": "stepwise_only_for_tiny_stepwise_state",
                "qrwkv": "stepwise_only_for_tiny_stepwise_state",
                "comparison": "not_applicable_for_non_stepwise_cases",
            },
        },
        "cases": cases,
        "parameter_validation": manifest.get("parameter_validation", {}),
        "radlads_load": radlads_load,
        "radlads_blocker": radlads_load.get("reason"),
        "qrwkv_load": qrwkv_load,
        "radlads_outputs": None if radlads_outputs is None else str(radlads_outputs),
        "qrwkv_outputs": None if qrwkv_outputs is None else str(qrwkv_outputs),
    }

    if _needs_intermediate_trace(report):
        _write_intermediate_trace(manifest_path.parent, cases, out_dir, report_prefix)

    if out_dir is not None:
        write_head_to_head_reports(report, out_dir, report_prefix=report_prefix)
        if qrwkv_import is not None and qrwkv_output_arrays is None:
            from qrwkv_xla.parity.radlads_parameter_import import (
                write_parameter_import_report,
            )

            write_parameter_import_report(qrwkv_import.report, out_dir)
    return report


def generate_head_to_head_fixtures(
    out_dir: Path = DEFAULT_OUT,
    *,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    radlads_source_path: Path = DEFAULT_RADLADS_SOURCE,
) -> dict[str, Any]:
    return generate_radlads_qrwkv_head_to_head_fixtures(
        out_dir,
        seed=seed,
        overwrite=overwrite,
        radlads_source_path=radlads_source_path,
        init_policy="deterministic_finite",
    )


def compare_head_to_head_manifest(
    manifest_path: Path,
    *,
    parameter_npz: Path | None = None,
    radlads_outputs: Path | None = None,
    qrwkv_outputs: Path | None = None,
    out_dir: Path | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    return compare_radlads_qrwkv_head_to_head(
        manifest_path,
        parameter_npz=parameter_npz,
        radlads_outputs=radlads_outputs,
        qrwkv_outputs=qrwkv_outputs,
        out_dir=out_dir,
        atol=atol,
        rtol=rtol,
    )


def compare_surface_arrays(
    name: str,
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
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
        return {**base, "status": "unsupported", "reason": "QRWKV array missing"}

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

    if not base["finite_radlads"] and not base["finite_qrwkv"]:
        return {
            **base,
            "status": "non_finite_both",
            "reason": "both sides are non-finite",
        }
    if not base["finite_radlads"]:
        return {
            **base,
            "status": "non_finite_radlads",
            "reason": "RADLADS surface is non-finite",
        }
    if not base["finite_qrwkv"]:
        return {
            **base,
            "status": "non_finite_qrwkv",
            "reason": "QRWKV surface is non-finite",
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


def write_head_to_head_reports(
    report: dict[str, Any],
    out_dir: Path,
    *,
    report_prefix: str = "P53",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "head_to_head_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{report_prefix}_RESULTS.md").write_text(
        _results_markdown(report, report_prefix=report_prefix),
        encoding="utf-8",
    )
    (out_dir / f"{report_prefix}_HEAD_TO_HEAD_REPORT.md").write_text(
        _results_markdown(report, report_prefix=report_prefix),
        encoding="utf-8",
    )
    (out_dir / f"{report_prefix}_SURFACE_COMPARISON.md").write_text(
        _surface_markdown(report, report_prefix=report_prefix),
        encoding="utf-8",
    )


def _run_case_pair(
    out: Path,
    case: dict[str, Any],
    *,
    seed: int,
    parameter_arrays: dict[str, np.ndarray],
    qrwkv_import,
    radlads_runtime: dict[str, Any] | None,
    radlads_blocker: str | None,
) -> dict[str, Any]:
    payload_name = f"{case['name']}.npz"
    case_path = out / payload_name
    inputs = {
        "input_ids": np.asarray(case["input_ids"], dtype=np.int32),
        "position_ids": np.broadcast_to(
            np.arange(case["input_ids"].shape[1], dtype=np.int32),
            case["input_ids"].shape,
        ),
    }
    if case["attention_mask"] is not None:
        inputs["attention_mask"] = np.asarray(case["attention_mask"], dtype=np.int32)

    case_record: dict[str, Any] = {
        "name": case["name"],
        "description": case["description"],
        "all_radlads_math": bool(case["all_radlads_math"]),
        "payload": payload_name,
        "input_shape": list(case["input_ids"].shape),
        "attention_mask": {
            "present": case["attention_mask"] is not None,
            "kind": case["mask_kind"],
            "shape": None
            if case["attention_mask"] is None
            else list(case["attention_mask"].shape),
        },
        "radlads_status": "unsupported" if radlads_runtime is None else "pass",
        "qrwkv_status": "pass",
        "comparisons": [],
    }

    if radlads_runtime is None:
        case_record["radlads_reason"] = radlads_blocker or "RADLADS runtime unavailable"
    else:
        try:
            radlads_arrays = _radlads_case_arrays_with_payload(
                radlads_runtime,
                case=case,
                parameter_arrays=parameter_arrays,
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover - environment specific
            case_record["radlads_status"] = "unsupported"
            case_record["radlads_reason"] = f"{type(exc).__name__}: {exc}"
            radlads_arrays = None
    try:
        from qrwkv_xla.students import RWKV7QwenReferenceStudent

        qrwkv_student = RWKV7QwenReferenceStudent(qrwkv_import.qrwkv_config)
        qrwkv_arrays = _qrwkv_case_arrays(
            qrwkv_student,
            qrwkv_import.params,
            case,
        )
        qrwkv_arrays = _normalize_qrwkv_arrays(qrwkv_arrays)
    except Exception as exc:  # pragma: no cover - should be rare
        case_record["qrwkv_status"] = "unsupported"
        case_record["qrwkv_reason"] = f"{type(exc).__name__}: {exc}"
        qrwkv_arrays = None

    if radlads_runtime is not None and qrwkv_arrays is not None:
        case_record["comparisons"] = []
        paired = {**inputs}
        paired.update(radlads_arrays or {})
        paired.update(qrwkv_arrays or {})
        case_record["comparisons"] = [
            compare_surface_arrays(
                name,
                paired.get(left),
                paired.get(right),
            )
            for name, left, right in PAIRWISE_SURFACES
        ]
        case_record["status"] = _case_status(case_record["comparisons"])
        np.savez(case_path, **paired)
    else:
        case_record["status"] = "unsupported"
        np.savez(case_path, **inputs)
    return case_record


def _radlads_case_arrays_with_payload(
    runtime: dict[str, Any],
    *,
    case: dict[str, Any],
    parameter_arrays: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    if torch is None:  # pragma: no cover - optional CI dependency
        raise ModuleNotFoundError("torch")
    torch.manual_seed(seed)
    _, model = _build_radlads_model(
        runtime, seed=seed, all_math=case["all_radlads_math"]
    )
    state = model.state_dict()
    named = dict(model.named_parameters())
    loaded = 0
    missing = []
    mismatches = []
    unsupported = []
    for name, value in sorted(parameter_arrays.items()):
        if name not in named:
            unsupported.append(name)
            continue
        if tuple(named[name].shape) != tuple(value.shape):
            mismatches.append(name)
            continue
        state[name] = torch.tensor(value, dtype=state[name].dtype)
        loaded += 1
    for name in named:
        if name not in parameter_arrays:
            missing.append(name)
    model.load_state_dict(state, strict=False)
    input_ids = torch.tensor(case["input_ids"], dtype=torch.long)
    attention_mask = (
        None
        if case["attention_mask"] is None
        else torch.tensor(case["attention_mask"], dtype=torch.long)
    )
    with torch.no_grad():
        full = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    arrays = {
        "input_ids": np.asarray(case["input_ids"], dtype=np.int32),
        "position_ids": np.broadcast_to(
            np.arange(case["input_ids"].shape[1], dtype=np.int32),
            case["input_ids"].shape,
        ),
        "radlads_hidden_states": full.hidden_states[-1]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "radlads_logits": full.logits.detach().cpu().numpy().astype(np.float32),
        "radlads_wkv_matrix_state": _stack_radlads_state(full.past_key_values, index=0),
        "radlads_shift_state": _stack_radlads_state(full.past_key_values, index=1),
    }
    arrays["radlads_shift_state"] = _squeeze_singleton_time_axis(
        arrays["radlads_shift_state"]
    )
    if attention_mask is not None:
        arrays["attention_mask"] = np.asarray(case["attention_mask"], dtype=np.int32)
    if case["name"] == "tiny_stepwise_state":
        step_cache = None
        step_hidden = []
        step_logits = []
        for position in range(case["input_ids"].shape[1]):
            with torch.no_grad():
                step = model(
                    input_ids=input_ids[:, position : position + 1],
                    attention_mask=(
                        None
                        if attention_mask is None
                        else attention_mask[:, position : position + 1]
                    ),
                    past_key_values=step_cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            step_cache = step.past_key_values
            step_hidden.append(
                step.hidden_states[-1].detach().cpu().numpy().astype(np.float32)
            )
            step_logits.append(step.logits.detach().cpu().numpy().astype(np.float32))
        arrays["radlads_stepwise_hidden_states"] = np.concatenate(step_hidden, axis=1)
        arrays["radlads_stepwise_logits"] = np.concatenate(step_logits, axis=1)
        arrays["radlads_stepwise_wkv_matrix_state"] = _stack_radlads_state(
            step_cache,
            index=0,
        )
        arrays["radlads_stepwise_shift_state"] = _squeeze_singleton_time_axis(
            _stack_radlads_state(step_cache, index=1)
        )
    arrays["radlads_loaded_params"] = np.array([loaded], dtype=np.int32)
    arrays["radlads_missing_params"] = np.array([len(missing)], dtype=np.int32)
    arrays["radlads_defaulted_params"] = np.array([0], dtype=np.int32)
    arrays["radlads_unsupported_params"] = np.array([len(unsupported)], dtype=np.int32)
    arrays["radlads_shape_mismatches"] = np.array([len(mismatches)], dtype=np.int32)
    return arrays


def _radlads_load_report(
    parameter_path: Path,
    *,
    radlads_runtime: dict[str, Any] | None,
    blocker: str | None,
    seed: int,
    radlads_source_path: Path,
) -> dict[str, Any]:
    if radlads_runtime is None:
        return {
            "status": "blocked",
            "reason": blocker or "RADLADS runtime unavailable",
            "radlads_loaded_params": 0,
            "radlads_missing_params": 0,
            "radlads_defaulted_params": 0,
            "radlads_unsupported_params": 0,
            "radlads_shape_mismatches": 0,
        }
    result = load_radlads_clean_payload(
        parameter_path,
        radlads_source_path=radlads_source_path,
        seed=seed,
        run_smoke=False,
    )
    report = dict(result.report)
    report.update(
        {
            "status": "pass" if result.overall_status == "pass" else "blocked",
            "reason": blocker or result.reason,
            "radlads_loaded_params": result.loaded_parameter_count,
            "radlads_missing_params": len(result.missing_required),
            "radlads_defaulted_params": len(result.defaulted),
            "radlads_unsupported_params": len(result.unsupported),
            "radlads_shape_mismatches": len(result.shape_mismatches),
        }
    )
    return report


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _case_status(comparisons: list[dict[str, Any]]) -> str:
    statuses = {
        row["status"] for row in comparisons if row["status"] != "not_applicable"
    }
    if not statuses:
        return "unsupported"
    if statuses == {"pass"}:
        return "pass"
    if statuses <= {"unsupported", "missing_source"}:
        return "unsupported"
    return "fail"


def _counts(statuses) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _best_passing_surface(cases: list[Mapping[str, Any]]) -> str | None:
    for case in cases:
        for row in case.get("comparisons", []):
            if row.get("status") == "pass":
                return f"{case.get('name')}:{row.get('name')}"
    return None


def _largest_failure(cases: list[Mapping[str, Any]]) -> str | None:
    biggest = None
    biggest_val = -1.0
    for case in cases:
        for row in case.get("comparisons", []):
            val = row.get("max_abs_error")
            if (
                isinstance(val, (int, float))
                and float(val) > biggest_val
                and row.get("status") != "pass"
            ):
                biggest_val = float(val)
                biggest = f"{case.get('name')}:{row.get('name')}={val}"
    return biggest


def _needs_intermediate_trace(report: Mapping[str, Any]) -> bool:
    failures = [
        case for case in report.get("cases", []) if case.get("status") == "fail"
    ]
    if not failures:
        return False
    worst = _largest_failure(report.get("cases", []))
    if worst is None:
        return False
    try:
        value = float(str(worst).rsplit("=", 1)[-1])
    except Exception:
        return False
    return value > 0.1


def _write_intermediate_trace(
    out_dir: Path,
    cases: list[Mapping[str, Any]],
    out: Path | None,
    report_prefix: str,
) -> None:
    tiny = next((case for case in cases if case.get("name") == "tiny_no_mask"), None)
    if tiny is None:
        return
    trace = {
        "case": tiny.get("name"),
        "stages": [
            "input embeddings",
            "pre-attention norm",
            "q projection",
            "k projection",
            "v projection",
            "q/k/v after bias",
            "q/k/v after reshape/head split",
            "low-rank w path",
            "low-rank a path",
            "gate path",
            "value residual path",
            "WKV output before o_proj",
            "o_proj output",
            "post-attention residual",
            "MLP output",
            "layer output",
            "final norm",
            "logits",
        ],
        "note": (
            "Intermediate trace capture is not yet fully instrumented for both sides; "
            "this placeholder records the surface order and paired comparison "
            "summary."
        ),
        "comparison_summary": tiny.get("comparisons", []),
    }
    target_dir = out or out_dir
    if target_dir is None:
        return
    trace_out = target_dir / "intermediate_trace_tiny_no_mask.json"
    trace_out.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / f"{report_prefix}_INTERMEDIATE_TRACE.md").write_text(
        _intermediate_markdown(trace, report_prefix=report_prefix),
        encoding="utf-8",
    )


def _intermediate_markdown(trace: Mapping[str, Any], *, report_prefix: str) -> str:
    lines = [
        f"# {report_prefix} Intermediate Trace",
        "",
        f"- Case: `{trace.get('case')}`",
        "",
        trace.get("note", ""),
        "",
        "## Stages",
        "",
    ]
    for stage in trace.get("stages", []):
        lines.append(f"- {stage}")
    return "\n".join(lines)


def _results_markdown(report: Mapping[str, Any], *, report_prefix: str) -> str:
    lines = [
        f"# {report_prefix} RADLADS vs QRWKV Head-to-Head",
        "",
        f"- Overall status: `{report.get('overall_status')}`",
        f"- Attempted comparisons: `{report.get('attempted_comparisons', 0)}`",
        f"- RADLADS blocker: `{report.get('radlads_blocker')}`",
        f"- Best passing surface: `{report.get('best_passing_surface')}`",
        f"- Largest failure: `{report.get('largest_failure')}`",
        "",
        "## Surface Status Counts",
        "",
    ]
    for status, count in sorted(dict(report.get("surface_status_counts", {})).items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "P53 uses the P52 deterministic_finite payload for a tiny head-to-head "
            "comparison. Unsupported live RADLADS execution is reported as a "
            "blocker, not counted as parity.",
            "",
        ]
    )
    return "\n".join(lines)


def _surface_markdown(report: Mapping[str, Any], *, report_prefix: str) -> str:
    lines = [
        f"# {report_prefix} Surface Comparison",
        "",
        "| Case | Surface | Status | Shape | Dtype | Finite | Max abs | "
        "Mean abs | Max rel | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.get("cases", []):
        for row in case.get("comparisons", []):
            lines.append(
                "| {case} | {surface} | `{status}` | `{shape}` | `{dtype}` | "
                "`{finite}` | {max_abs} | {mean_abs} | {max_rel} | {reason} |".format(
                    case=case.get("name"),
                    surface=row.get("name"),
                    status=row.get("status"),
                    shape=row.get("shape") or row.get("left_shape"),
                    dtype=row.get("dtype") or row.get("left_dtype"),
                    finite=(row.get("finite_radlads"), row.get("finite_qrwkv")),
                    max_abs=row.get("max_abs_error"),
                    mean_abs=row.get("mean_abs_error"),
                    max_rel=row.get("max_relative_error"),
                    reason=row.get("reason", ""),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _missing_source_report(
    manifest_path: Path,
    parameter_path: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": HEAD_TO_HEAD_SCHEMA,
        "manifest": str(manifest_path),
        "parameter_payload": str(parameter_path),
        "overall_status": "missing_source",
        "reason": reason,
        "cases_attempted": 0,
        "cases_finite": 0,
        "attempted_comparisons": 0,
        "counts": {"missing_source": 1},
        "surface_status_counts": {"missing_source": 1},
        "best_passing_surface": None,
        "largest_failure": None,
        "cases": [],
    }


def _validate_init_policy(init_policy: str) -> None:
    if init_policy not in VALID_INIT_POLICIES:
        raise ValueError(
            f"unsupported init_policy={init_policy!r}; expected one of "
            f"{sorted(VALID_INIT_POLICIES)}"
        )


def _prepare_out_dir(out: Path, *, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to replace fixtures")
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*"):
        if path.is_file():
            path.unlink()


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _validate_case_manifest(base: Path, case: dict[str, Any]) -> None:
    _require(
        (base / case["payload"]).is_file(), f"missing case payload {case['payload']}"
    )


def _git_head(path: Path) -> str:
    if not path.exists():
        return "unknown"
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _hash_numerical_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _max_abs_from_payload(arrays: dict[str, np.ndarray]) -> float:
    maximum = 0.0
    for array in arrays.values():
        if array.size == 0:
            continue
        maximum = max(
            maximum, float(np.max(np.abs(np.asarray(array, dtype=np.float32))))
        )
    return maximum


def _stack_radlads_state(cache: Any, *, index: int) -> np.ndarray:
    values = [
        cache[layer][index].detach().cpu().numpy().astype(np.float32)
        for layer in range(len(cache))
    ]
    return np.stack(values, axis=0)


def _squeeze_singleton_time_axis(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim >= 4 and value.shape[-2] == 1:
        return np.squeeze(value, axis=-2)
    return value


def _case_manifest_record(case_record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case_record.items()
        if key not in {"radlads_arrays", "qrwkv_arrays"}
    }


def _normalize_qrwkv_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized = dict(arrays)
    hidden = normalized.get("qrwkv_hidden_states")
    if hidden is not None:
        hidden_array = np.asarray(hidden)
        if hidden_array.ndim >= 4:
            normalized["qrwkv_hidden_states"] = np.asarray(hidden_array[-1])
    shift = normalized.get("qrwkv_shift_state")
    if shift is not None:
        normalized["qrwkv_shift_state"] = _squeeze_singleton_time_axis(
            np.asarray(shift)
        )
    step_hidden = normalized.get("qrwkv_stepwise_hidden_states")
    if step_hidden is not None:
        step_hidden_array = np.asarray(step_hidden)
        if step_hidden_array.ndim >= 4:
            normalized["qrwkv_stepwise_hidden_states"] = np.asarray(
                step_hidden_array[-1]
            )
    step_shift = normalized.get("qrwkv_stepwise_shift_state")
    if step_shift is not None:
        normalized["qrwkv_stepwise_shift_state"] = _squeeze_singleton_time_axis(
            np.asarray(step_shift)
        )
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
