from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - optional CI dependency
    jax = None

import numpy as np

from qrwkv_xla.parity.radlads_parameter_import import QRWKV_DEFAULTED_SURFACES

DIAGNOSTIC_SCHEMA = "radlads_replay_diagnostics.v1"
PARAMETER_SANITY_SCHEMA = "radlads_parameter_sanity.v1"


@dataclass(frozen=True)
class TensorDiagnostic:
    name: str
    shape: list[int]
    dtype: str
    min: float | str | None
    max: float | str | None
    mean: float | str | None
    std: float | str | None
    abs_max: float | str | None
    finite_count: int
    nonfinite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    first_nonfinite_index: list[int] | None
    first_nonfinite_value: float | str | None
    stage: str | None = None
    layer: int | None = None
    time_index: int | None = None
    mapped_qrwkv_path: str | None = None
    expected_shape: list[int] | None = None
    suspicious_reasons: list[str] | None = None


class ReplayDiagnosticsCollector:
    def __init__(self) -> None:
        self.summaries: list[dict[str, Any]] = []
        self.instrumented_stages: list[str] = []

    def record(
        self,
        name: str,
        value: Any,
        *,
        stage: str,
        layer: int | None = None,
        time_index: int | None = None,
    ) -> dict[str, Any]:
        summary = asdict(
            summarize_array(
                name,
                value,
                stage=stage,
                layer=layer,
                time_index=time_index,
            )
        )
        self.summaries.append(summary)
        if stage not in self.instrumented_stages:
            self.instrumented_stages.append(stage)
        return summary


def summarize_array(
    name: str,
    array: Any,
    *,
    stage: str | None = None,
    layer: int | None = None,
    time_index: int | None = None,
) -> TensorDiagnostic:
    values = _to_numpy(array)
    flat = values.reshape(-1)
    finite_mask = np.isfinite(flat)
    nan_mask = np.isnan(flat)
    posinf_mask = np.isposinf(flat)
    neginf_mask = np.isneginf(flat)
    nonfinite_index = np.flatnonzero(~finite_mask)
    finite_values = flat[finite_mask].astype(np.float64, copy=False)
    first_index = None
    first_value: float | str | None = None
    if nonfinite_index.size:
        unravel = np.unravel_index(int(nonfinite_index[0]), values.shape)
        first_index = [int(idx) for idx in unravel]
        first_value = _scalar_for_json(flat[int(nonfinite_index[0])])
    return TensorDiagnostic(
        name=name,
        shape=[int(dim) for dim in values.shape],
        dtype=str(values.dtype),
        min=_stat_or_none(finite_values, np.min),
        max=_stat_or_none(finite_values, np.max),
        mean=_stat_or_none(finite_values, np.mean),
        std=_stat_or_none(finite_values, np.std),
        abs_max=_stat_or_none(np.abs(finite_values), np.max),
        finite_count=int(finite_mask.sum()),
        nonfinite_count=int((~finite_mask).sum()),
        nan_count=int(nan_mask.sum()),
        posinf_count=int(posinf_mask.sum()),
        neginf_count=int(neginf_mask.sum()),
        first_nonfinite_index=first_index,
        first_nonfinite_value=first_value,
        stage=stage,
        layer=layer,
        time_index=time_index,
    )


def find_first_nonfinite(
    summaries: list[dict[str, Any]] | list[TensorDiagnostic],
) -> dict[str, Any] | None:
    for item in summaries:
        row = asdict(item) if isinstance(item, TensorDiagnostic) else dict(item)
        if int(row.get("nonfinite_count", 0)) > 0:
            return row
    return None


def summarize_parameter_payload(
    normalized_arrays: dict[str, np.ndarray],
    *,
    mapping_entries: list[dict[str, Any]],
    active_defaulted_surfaces: set[str] | None = None,
    suspicious_abs_threshold: float = 1e6,
) -> dict[str, Any]:
    mapping_by_radlads = {
        row.get("radlads"): row for row in mapping_entries if row.get("radlads")
    }
    defaulted = [row for row in mapping_entries if row.get("status") == "defaulted"]
    entries: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    for name in sorted(normalized_arrays):
        summary = asdict(summarize_array(name, normalized_arrays[name]))
        mapping = mapping_by_radlads.get(name, {})
        reasons: list[str] = []
        abs_max = summary.get("abs_max")
        abs_max_value = None if isinstance(abs_max, str) else abs_max
        if int(summary["nonfinite_count"]) > 0:
            reasons.append("non-finite source parameter")
        if (
            abs_max_value is not None
            and float(abs_max_value) >= suspicious_abs_threshold
        ):
            reasons.append("very large abs_max")
        if summary["dtype"] not in {"float32", "float64", "int32", "int64"}:
            reasons.append("unexpected dtype")
        if int(summary["finite_count"]) > 0 and abs_max_value == 0.0:
            reasons.append("all zeros where unlikely")
        summary["mapped_qrwkv_path"] = mapping.get("qrwkv")
        summary["expected_shape"] = mapping.get("qrwkv_shape")
        summary["suspicious_reasons"] = reasons
        entries.append(summary)
        if reasons:
            suspicious.append(summary)

    active_defaults = []
    for row in defaulted:
        qrwkv = row.get("qrwkv")
        active = qrwkv in (active_defaulted_surfaces or set())
        active_defaults.append(
            {
                **row,
                "qrwkv_only_default_used_in_active_path": active,
                "default_metadata": QRWKV_DEFAULTED_SURFACES.get(str(qrwkv), {}),
            }
        )

    largest = sorted(
        entries,
        key=lambda item: _numeric_sort_key(item.get("abs_max")),
        reverse=True,
    )[:10]
    return {
        "schema": PARAMETER_SANITY_SCHEMA,
        "entry_count": len(entries),
        "nonfinite_parameter_count": sum(
            1 for item in entries if int(item["nonfinite_count"]) > 0
        ),
        "largest_abs_parameters": [
            {
                "name": item["name"],
                "abs_max": item["abs_max"],
                "dtype": item["dtype"],
                "shape": item["shape"],
            }
            for item in largest
        ],
        "suspicious_parameters": suspicious,
        "defaulted_parameters": active_defaults,
        "entries": entries,
    }


def write_parameter_sanity_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameter_sanity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# P51 Parameter Sanity",
        "",
        f"- Non-finite parameter count: `{report['nonfinite_parameter_count']}`",
        f"- Entry count: `{report['entry_count']}`",
        "",
        "## Largest abs parameters",
        "",
    ]
    for row in report["largest_abs_parameters"]:
        lines.append(
            f"- `{row['name']}` abs_max=`{row['abs_max']}` "
            f"dtype=`{row['dtype']}` shape=`{row['shape']}`"
        )
    lines.extend(["", "## Defaulted params", ""])
    for row in report["defaulted_parameters"]:
        lines.append(
            (
                "- `{qrwkv}` active_path=`{active}` "
                "parity_risk=`{risk}` reason={reason}"
            ).format(
                qrwkv=row.get("qrwkv"),
                active=row.get("qrwkv_only_default_used_in_active_path"),
                risk=row.get("parity_risk"),
                reason=row.get("reason"),
            )
        )
    lines.extend(["", "## Suspicious params", ""])
    for row in report["suspicious_parameters"]:
        lines.append(
            f"- `{row['name']}` reasons={row.get('suspicious_reasons', [])} "
            f"abs_max=`{row['abs_max']}`"
        )
    lines.append("")
    (out_dir / "P51_PARAMETER_SANITY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_diagnostic_report(
    *,
    case_reports: list[dict[str, Any]],
    parameter_sanity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "cases": case_reports,
        "parameter_sanity": {
            "nonfinite_parameter_count": parameter_sanity["nonfinite_parameter_count"],
            "largest_abs_parameters": parameter_sanity["largest_abs_parameters"][:5],
            "suspicious_parameter_count": len(
                parameter_sanity["suspicious_parameters"]
            ),
        },
    }


def write_diagnostic_reports(
    report: dict[str, Any],
    *,
    tensor_summaries: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "tensor_summaries.jsonl").open("w", encoding="utf-8") as handle:
        for row in tensor_summaries:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    lines = ["# P51 Diagnostic Report", "", "## Cases", ""]
    for case in report["cases"]:
        lines.append(f"- `{case['case']}` first_nonfinite=`{case['first_nonfinite']}`")
        lines.append(
            "  - final_outputs_finite="
            f"`{case['final_outputs_finite']}` "
            "instrumented_stages="
            f"`{case['instrumented_stages']}`"
        )
        lines.append(f"  - suspected_root_cause={case.get('suspected_root_cause')}")
    lines.extend(
        [
            "",
            "## Parameter sanity summary",
            "",
            "- nonfinite_parameter_count: "
            f"`{report['parameter_sanity']['nonfinite_parameter_count']}`",
            "- suspicious_parameter_count: "
            f"`{report['parameter_sanity']['suspicious_parameter_count']}`",
            "",
        ]
    )
    (out_dir / "P51_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _to_numpy(value: Any) -> np.ndarray:
    if jax is None:
        return np.asarray(value)
    return np.asarray(jax.device_get(value))


def _stat_or_none(values: np.ndarray, reducer) -> float | str | None:
    if values.size == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        return _scalar_for_json(reducer(values))


def _scalar_for_json(value: Any) -> float | str:
    scalar = float(value)
    if np.isnan(scalar):
        return "nan"
    if np.isposinf(scalar):
        return "inf"
    if np.isneginf(scalar):
        return "-inf"
    return scalar


def _numeric_sort_key(value: Any) -> float:
    if value is None:
        return float("-inf")
    if isinstance(value, str):
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("inf")
        return float("nan")
    return float(value)
