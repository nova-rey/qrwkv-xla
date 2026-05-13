"""Validation utilities for RADLADS tiny fixture parameters.

This module provides finite parameter payload validation and audit utilities
to detect non-finite values (NaN, Inf) and extreme values that may block replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParameterAuditResult:
    """Result of auditing a single parameter."""

    name: str
    stage: str
    shape: list[int]
    dtype: str
    min: float | str | None
    max: float | str | None
    mean: float | str | None
    std: float | str | None
    abs_max: float | str | None
    finite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    all_zero: bool
    all_same: bool
    sha256: str
    status: str


def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for audit results."""
    if isinstance(obj, ParameterAuditResult):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def compute_sha256_from_array(arr: np.ndarray) -> str:
    """Compute SHA256 hash of array bytes (deterministic)."""
    flat = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(flat).hexdigest()


def analyze_array(
    name: str,
    array: np.ndarray,
    *,
    stage: str,
    extreme_threshold: float = 1e6,
) -> ParameterAuditResult:
    """Analyze a single parameter array and produce audit result.

    Args:
        name: Parameter name
        array: Numpy array to analyze
        stage: Stage identifier (radlads_live, pre_save, saved_npz, qrwkv_imported)
        extreme_threshold: Values with abs() > this are flagged as extreme

    Returns:
        ParameterAuditResult with all diagnostic fields populated
    """
    values = np.asarray(array)
    flat = values.reshape(-1)
    finite_mask = np.isfinite(flat)
    nan_mask = np.isnan(flat)
    posinf_mask = np.isposinf(flat)
    neginf_mask = np.isneginf(flat)

    finite_values = flat[finite_mask].astype(np.float64, copy=False)

    # Stats
    nan_count = int(nan_mask.sum())
    posinf_count = int(posinf_mask.sum())
    neginf_count = int(neginf_mask.sum())
    finite_count = int(finite_mask.sum())

    min_val: float | str | None = None
    max_val: float | str | None = None
    mean_val: float | str | None = None
    std_val: float | str | None = None
    abs_max_val: float | str | None = None

    if finite_count > 0:
        with np.errstate(over="ignore", invalid="ignore"):
            min_val = float(np.min(finite_values))
            max_val = float(np.max(finite_values))
            mean_val = float(np.mean(finite_values))
            std_val = float(np.std(finite_values))
            abs_max_val = float(np.max(np.abs(finite_values)))

    # Deterministic hash
    sha256 = compute_sha256_from_array(values)

    # Binary flags
    all_zero = bool(np.all(flat == 0))
    all_same = bool(np.all(flat == flat.flat[0])) if flat.size > 0 else False

    # Status determination
    status = _determine_status(
        nan_count=nan_count,
        posinf_count=posinf_count,
        neginf_count=neginf_count,
        abs_max_value=abs_max_val,
        finite_count=finite_count,
        all_zero=all_zero,
        all_same=all_same,
        stage=stage,
        extreme_threshold=extreme_threshold,
    )

    return ParameterAuditResult(
        name=name,
        stage=stage,
        shape=[int(dim) for dim in values.shape],
        dtype=str(values.dtype),
        min=min_val,
        max=max_val,
        mean=mean_val,
        std=std_val,
        abs_max=abs_max_val,
        finite_count=finite_count,
        nan_count=nan_count,
        posinf_count=posinf_count,
        neginf_count=neginf_count,
        all_zero=all_zero,
        all_same=all_same,
        sha256=sha256,
        status=status,
    )


def _determine_status(
    *,
    nan_count: int,
    posinf_count: int,
    neginf_count: int,
    abs_max_value: float | None,
    finite_count: int,
    all_zero: bool,
    all_same: bool,
    stage: str,
    extreme_threshold: float,
) -> str:
    """Determine parameter status based on diagnostic flags."""
    # Non-finite takes priority
    if nan_count > 0:
        return "non_finite"
    if posinf_count > 0 or neginf_count > 0:
        return "non_finite"

    # Extreme value check
    if abs_max_value is not None and abs_max_value > extreme_threshold:
        return "extreme_value"

    # Stage-specific checks
    if stage == "qrwkv_imported":
        if finite_count == 0:
            return "defaulted"
        # If all zeros but not all_zero originally, might be intentional
        if all_zero and not all_same:
            return "defaulted"

    if stage == "radlads_live":
        if all_zero and not all_same:
            # Zeroed default - probably intentional
            return "defaulted"
        if all_same and abs_max_value is not None and abs_max_value < 1e-10:
            return "defaulted"

    if finite_count == 0:
        return "missing"

    return "finite_ok"


def audit_parameter_payload(
    parameters: dict[str, np.ndarray],
    *,
    stage: str,
    extreme_threshold: float = 1e6,
    seed: int = 5050,
) -> list[ParameterAuditResult]:
    """Audit all parameters in a payload.

    Args:
        parameters: Dict of parameter name -> numpy array
        stage: Stage identifier
        extreme_threshold: Threshold for flagging extreme values
        seed: RNG seed for deterministic hashing

    Returns:
        List of ParameterAuditResult sorted by name
    """
    np.random.seed(seed)  # For reproducibility if needed

    results: list[ParameterAuditResult] = []
    for name in sorted(parameters):
        array = parameters[name]
        result = analyze_array(
            name=name,
            array=array,
            stage=stage,
            extreme_threshold=extreme_threshold,
        )
        results.append(result)

    return results


def validate_parameter_payload(
    results: list[ParameterAuditResult],
) -> tuple[bool, list[ParameterAuditResult]]:
    """Validate audit results and return blocking issues.

    Args:
        results: List of audit results

    Returns:
        (is_valid, blocking_results) where blocking_results are non-finite or extreme
    """
    blocking = [
        r for r in results if r.status in ("non_finite", "extreme_value", "missing")
    ]
    return len(blocking) == 0, blocking


def to_audit_report(results: list[ParameterAuditResult]) -> dict[str, Any]:
    """Convert audit results to JSON-serializable report."""
    # Group by status
    by_status: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        key = r.status
        if key not in by_status:
            by_status[key] = []
        by_status[key].append(
            {
                "name": r.name,
                "stage": r.stage,
                "shape": r.shape,
                "dtype": r.dtype,
                "min": r.min,
                "max": r.max,
                "mean": r.mean,
                "std": r.std,
                "abs_max": r.abs_max,
                "finite_count": r.finite_count,
                "nan_count": r.nan_count,
                "posinf_count": r.posinf_count,
                "neginf_count": r.neginf_count,
                "all_zero": r.all_zero,
                "all_same": r.all_same,
                "sha256": r.sha256,
            }
        )

    # Count non-finite and extreme
    nonfinite = [r for r in results if r.status in ("non_finite", "extreme_value")]

    return {
        "schema": "radlads_parameter_audit.v1",
        "parameter_count": len(results),
        "nonfinite_parameter_count": len(nonfinite),
        "extreme_threshold": 1e6,
        "by_status": {k: len(v) for k, v in by_status.items()},
        "parameters": {
            r.status: [p for p in by_status[r.status]]
            for r in results
            if r.status in by_status
        },
        "summary": {
            "finite_ok": len([r for r in results if r.status == "finite_ok"]),
            "non_finite": len([r for r in results if r.status == "non_finite"]),
            "extreme_value": len([r for r in results if r.status == "extreme_value"]),
            "defaulted": len([r for r in results if r.status == "defaulted"]),
            "missing": len([r for r in results if r.status == "missing"]),
        },
    }


def write_audit_report(
    report: dict[str, Any],
    out_dir: Path,
    *,
    filename_json: str = "parameter_provenance_report.json",
    filename_markdown: str = "P52_PARAMETER_PROVENANCE.md",
) -> None:
    """Write audit report to files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (out_dir / filename_json).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    # Markdown
    lines = [
        "# P52 RADLADS Tiny Fixture Parameter Provenance Audit",
        "",
        f"- Parameter count: `{report['parameter_count']}`",
        f"- Non-finite parameter count: `{report['summary']['non_finite']}`",
        f"- Extreme value count: `{report['summary']['extreme_value']}`",
        "",
        "## Status Summary",
        "",
    ]
    for status, count in sorted(report["summary"].items()):
        lines.append(f"- `{status}`: {count}")

    lines.extend(["", "## Non-finite Parameters", ""])
    if report["summary"]["non_finite"] > 0:
        for param in report["parameters"].get("non_finite", []):
            lines.append(
                f"- `{param['name']}` shape={param['shape']} "
                f"dtype={param['dtype']} "
                f"min={param['min']} max={param['max']} "
                f"SHA256={param['sha256'][:16]}..."
            )
    else:
        lines.append("None detected.")

    lines.extend(["", "## Extreme Value Parameters", ""])
    if report["summary"]["extreme_value"] > 0:
        for param in report["parameters"].get("extreme_value", []):
            lines.append(
                f"- `{param['name']}` shape={param['shape']} "
                f"abs_max={param['abs_max']} "
                f"SHA256={param['sha256'][:16]}..."
            )
    else:
        lines.append("None detected.")

    lines.extend(["", "## Defaulted Parameters", ""])
    if report["summary"]["defaulted"] > 0:
        for param in report["parameters"].get("defaulted", []):
            lines.append(
                f"- `{param['name']}` shape={param['shape']} "
                f"SHA256={param['sha256'][:16]}..."
            )
    else:
        lines.append("None detected.")

    lines.extend(["", "## Missing Parameters", ""])
    if report["summary"]["missing"] > 0:
        for param in report["parameters"].get("missing", []):
            lines.append(f"- `{param['name']}`")
    else:
        lines.append("None detected.")

    lines.append("")
    (out_dir / filename_markdown).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
