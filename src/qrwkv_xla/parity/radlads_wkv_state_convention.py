from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WKV_STATE_CONVENTION_SCHEMA = "radlads_qrwkv_wkv_state_convention.v1"
WKV_STATE_SLOT_AUDIT_SCHEMA = "radlads_qrwkv_wkv_state_slot_audit.v1"
WKV_STATE_CONVENTION_REPORT_SCHEMA = "radlads_qrwkv_wkv_state_convention_report.v1"
WKV_STATE_EXPORT_SCHEMA = "qrwkv_xla.wkv_state_export.v1"

REFERENCE_STATE_EXPORT_PATH = (
    "qrwkv_xla.parity.radlads_wkv_state_convention.export_reference_state_object"
)
REFERENCE_STATE_IMPORT_PATH = (
    "qrwkv_xla.parity.radlads_wkv_state_convention.import_reference_state_object"
)

VALID_NORMALIZATIONS = {
    "as_is",
    "swap_state_slots",
    "pre_update_to_post_update",
    "post_update_to_pre_update",
    "full_sequence_final_to_stepwise_final",
    "stepwise_returned_to_internal",
    "transpose_matrix_axes",
    "move_layer_axis",
    "move_batch_axis",
    "move_head_axis",
    "dtype_cast_only",
}


@dataclass(frozen=True)
class WKVStateNormalizationResult:
    schema: str
    source_convention: str
    target_convention: str
    source_slot: str
    target_slot: str
    normalization_applied: str
    source_backed: bool
    changed_values: bool
    source_shape: list[int]
    normalized_shape: list[int]
    source_dtype: str
    normalized_dtype: str
    note: str | None = None


def extract_state_slot(
    source: Any,
    slot_name: str,
    *,
    bundle_name: str | None = None,
    alternative_slot: str | None = None,
    normalization: str = "as_is",
) -> np.ndarray:
    if normalization == "swap_state_slots":
        if alternative_slot is None:
            raise ValueError("swap_state_slots requires alternative_slot")
        slot_name = alternative_slot
    if isinstance(source, Mapping):
        if slot_name in source:
            return np.asarray(source[slot_name])
        if "state_slots" in source:
            slots = source["state_slots"]
            if isinstance(slots, Mapping) and slot_name in slots:
                return np.asarray(slots[slot_name])
        if bundle_name is not None and bundle_name in source:
            bundle = source[bundle_name]
            if isinstance(bundle, Mapping) and slot_name in bundle:
                return np.asarray(bundle[slot_name])
        raise KeyError(f"slot {slot_name!r} not found in source")
    if hasattr(source, slot_name):
        return np.asarray(getattr(source, slot_name))
    return np.asarray(source)


def normalize_radlads_wkv_matrix_state(
    source: Any,
    *,
    slot_name: str = "wkv_matrix_state",
    alternative_slot: str | None = None,
    normalization: str = "as_is",
    source_convention: str = "radlads",
    target_convention: str = "comparison",
    axis_order: tuple[int, ...] | None = None,
    dtype: np.dtype | str | None = None,
) -> dict[str, Any]:
    return _normalize_wkv_matrix_state(
        source,
        slot_name=slot_name,
        alternative_slot=alternative_slot,
        normalization=normalization,
        source_convention=source_convention,
        target_convention=target_convention,
        axis_order=axis_order,
        dtype=dtype,
    )


def normalize_qrwkv_wkv_matrix_state(
    source: Any,
    *,
    slot_name: str = "wkv_matrix_state",
    alternative_slot: str | None = None,
    normalization: str = "as_is",
    source_convention: str = "qrwkv",
    target_convention: str = "comparison",
    axis_order: tuple[int, ...] | None = None,
    dtype: np.dtype | str | None = None,
) -> dict[str, Any]:
    return _normalize_wkv_matrix_state(
        source,
        slot_name=slot_name,
        alternative_slot=alternative_slot,
        normalization=normalization,
        source_convention=source_convention,
        target_convention=target_convention,
        axis_order=axis_order,
        dtype=dtype,
    )


def compare_wkv_matrix_state_conventions(
    radlads: Any,
    qrwkv: Any,
    *,
    radlads_slot: str = "wkv_matrix_state",
    qrwkv_slot: str = "wkv_matrix_state",
    slot_audit: Mapping[str, Any] | None = None,
    normalization: str = "as_is",
    axis_order: tuple[int, ...] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    if normalization not in VALID_NORMALIZATIONS:
        raise ValueError(f"unknown normalization: {normalization}")
    radlads_raw = extract_state_slot(radlads, radlads_slot)
    qrwkv_raw = extract_state_slot(qrwkv, qrwkv_slot)

    raw_stats = _compare_arrays(radlads_raw, qrwkv_raw, atol=atol, rtol=rtol)

    source_backed = (
        bool(slot_audit.get("source_backed", False))
        if slot_audit
        else normalization == "as_is"
    )
    if slot_audit is not None:
        recommended = str(slot_audit.get("recommended_normalization", normalization))
        if normalization == "as_is":
            normalization = recommended

    normalized_qrwkv = _apply_normalization(
        qrwkv_raw,
        normalization=normalization,
        axis_order=axis_order,
    )
    normalized_stats = _compare_arrays(
        radlads_raw, normalized_qrwkv, atol=atol, rtol=rtol
    )
    return {
        "schema": WKV_STATE_CONVENTION_REPORT_SCHEMA,
        "radlads_slot": radlads_slot,
        "qrwkv_slot": qrwkv_slot,
        "normalization_applied": normalization,
        "normalization_source_backed": source_backed,
        "normalization_changed_values": not np.array_equal(
            np.asarray(qrwkv_raw), np.asarray(normalized_qrwkv)
        ),
        "raw_wkv_matrix_state_error": raw_stats,
        "normalized_wkv_matrix_state_error": normalized_stats,
        "candidate_normalizations": _candidate_report(
            radlads_raw,
            qrwkv_raw,
            atol=atol,
            rtol=rtol,
        ),
        "source_convention": "radlads-vs-qrwkv",
        "target_convention": "comparison",
    }


def export_reference_state_object(source: Any) -> dict[str, Any]:
    """Observe-only export of the local reference recurrent state object.

    This is the current QRWKV-XLA state export path for P76 evidence. It exports
    the returned reference state object's slots without changing recurrence math
    or serializing a checkpoint.
    """
    slots = {
        "wkv_matrix_state": np.asarray(extract_state_slot(source, "wkv_matrix_state")),
    }
    if hasattr(source, "shift_state") or (
        isinstance(source, Mapping) and "shift_state" in source
    ):
        slots["shift_state"] = np.asarray(extract_state_slot(source, "shift_state"))
    if hasattr(source, "next_position") or (
        isinstance(source, Mapping) and "next_position" in source
    ):
        slots["next_position"] = np.asarray(extract_state_slot(source, "next_position"))
    return {
        "schema": WKV_STATE_EXPORT_SCHEMA,
        "export_path": REFERENCE_STATE_EXPORT_PATH,
        "representation": "reference_state_slots",
        "state_slots": slots,
        "slot_shapes": {
            name: [int(dim) for dim in value.shape] for name, value in slots.items()
        },
        "slot_dtypes": {name: str(value.dtype) for name, value in slots.items()},
    }


def import_reference_state_object(
    payload: Mapping[str, Any], *, template: Any | None = None
) -> Any:
    if payload.get("schema") != WKV_STATE_EXPORT_SCHEMA:
        raise ValueError(f"unsupported state export schema: {payload.get('schema')}")
    slots = payload.get("state_slots")
    if not isinstance(slots, Mapping) or "wkv_matrix_state" not in slots:
        raise ValueError("state export payload missing wkv_matrix_state slot")
    imported = {str(name): np.asarray(value) for name, value in slots.items()}
    if template is not None and all(
        hasattr(template, name)
        for name in ("wkv_matrix_state", "shift_state", "next_position")
    ):
        return type(template)(
            wkv_matrix_state=imported["wkv_matrix_state"],
            shift_state=imported["shift_state"],
            next_position=imported["next_position"],
        )
    if template is not None and not hasattr(template, "wkv_matrix_state"):
        return imported["wkv_matrix_state"]
    return imported


def write_wkv_state_convention_report(report: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "head_to_head_normalized_state_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_wkv_matrix_state(
    source: Any,
    *,
    slot_name: str,
    alternative_slot: str | None,
    normalization: str,
    source_convention: str,
    target_convention: str,
    axis_order: tuple[int, ...] | None,
    dtype: np.dtype | str | None,
) -> dict[str, Any]:
    if normalization not in VALID_NORMALIZATIONS:
        raise ValueError(f"unknown normalization: {normalization}")
    raw = extract_state_slot(
        source,
        slot_name,
        alternative_slot=alternative_slot,
        normalization=normalization,
    )
    normalized = _apply_normalization(
        raw, normalization=normalization, axis_order=axis_order, dtype=dtype
    )
    return asdict(
        WKVStateNormalizationResult(
            schema=WKV_STATE_CONVENTION_SCHEMA,
            source_convention=source_convention,
            target_convention=target_convention,
            source_slot=slot_name,
            target_slot=slot_name,
            normalization_applied=normalization,
            source_backed=True,
            changed_values=not np.array_equal(np.asarray(raw), np.asarray(normalized)),
            source_shape=[int(dim) for dim in np.asarray(raw).shape],
            normalized_shape=[int(dim) for dim in np.asarray(normalized).shape],
            source_dtype=str(np.asarray(raw).dtype),
            normalized_dtype=str(np.asarray(normalized).dtype),
        )
    ) | {"source_array": raw, "normalized_array": normalized}


def _apply_normalization(
    array: Any,
    *,
    normalization: str,
    axis_order: tuple[int, ...] | None,
    dtype: np.dtype | str | None = None,
) -> np.ndarray:
    value = np.asarray(array)
    if normalization == "as_is":
        normalized = value
    elif normalization == "dtype_cast_only":
        normalized = value.astype(np.float32)
    elif normalization == "transpose_matrix_axes":
        normalized = np.swapaxes(value, -1, -2)
    elif normalization == "move_layer_axis":
        if value.ndim < 2:
            raise ValueError("move_layer_axis requires rank >= 2")
        normalized = np.moveaxis(value, 0, -1)
    elif normalization == "move_batch_axis":
        if value.ndim < 2:
            raise ValueError("move_batch_axis requires rank >= 2")
        normalized = np.moveaxis(value, 1, 0)
    elif normalization == "move_head_axis":
        if value.ndim < 3:
            raise ValueError("move_head_axis requires rank >= 3")
        normalized = np.moveaxis(value, 2, 0)
    elif normalization == "pre_update_to_post_update":
        normalized = value
    elif normalization == "post_update_to_pre_update":
        normalized = value
    elif normalization == "full_sequence_final_to_stepwise_final":
        normalized = value
    elif normalization == "stepwise_returned_to_internal":
        normalized = value
    elif normalization == "swap_state_slots":
        normalized = value
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    if axis_order is not None:
        normalized = np.transpose(normalized, axis_order)
    if dtype is not None:
        normalized = normalized.astype(dtype)
    return np.asarray(normalized)


def _candidate_report(
    radlads: Any,
    qrwkv: Any,
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    radlads_arr = np.asarray(radlads)
    qrwkv_arr = np.asarray(qrwkv)
    rows: list[dict[str, Any]] = []
    for candidate in sorted(VALID_NORMALIZATIONS):
        try:
            normalized = _apply_normalization(
                qrwkv_arr, normalization=candidate, axis_order=None
            )
            stats = _compare_arrays(radlads_arr, normalized, atol=atol, rtol=rtol)
            rows.append(
                {
                    "candidate_name": candidate,
                    "applicable": True,
                    "shape_match": stats["shape_match"],
                    "max_abs_error": stats["max_abs_error"],
                    "mean_abs_error": stats["mean_abs_error"],
                    "max_relative_error": stats["max_relative_error"],
                    "allclose": stats["allclose"],
                    "status": stats["status"],
                    "source_backed": candidate == "as_is",
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            rows.append(
                {
                    "candidate_name": candidate,
                    "applicable": False,
                    "shape_match": False,
                    "max_abs_error": None,
                    "mean_abs_error": None,
                    "max_relative_error": None,
                    "allclose": False,
                    "status": "not_applicable",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "source_backed": candidate == "as_is",
                }
            )
    return rows


def _compare_arrays(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if tuple(left_arr.shape) != tuple(right_arr.shape):
        return {
            "shape_match": False,
            "dtype_match": str(left_arr.dtype) == str(right_arr.dtype),
            "finite_both": bool(
                np.isfinite(left_arr).all() and np.isfinite(right_arr).all()
            ),
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
            "status": "shape_mismatch",
        }
    diff = np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64))
    denom = np.maximum(np.abs(right_arr.astype(np.float64)), 1e-12)
    finite_both = bool(np.isfinite(left_arr).all() and np.isfinite(right_arr).all())
    allclose = bool(np.allclose(left_arr, right_arr, atol=atol, rtol=rtol))
    return {
        "shape_match": True,
        "dtype_match": str(left_arr.dtype) == str(right_arr.dtype),
        "finite_both": finite_both,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_relative_error": float(np.max(diff / denom)) if diff.size else 0.0,
        "allclose": allclose,
        "status": "pass" if allclose else ("non_finite" if not finite_both else "fail"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
