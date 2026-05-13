from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.parity.radlads_clean_loader import load_case_output_arrays

HIDDEN_STATES_CONVENTION = {
    "radlads": "final_hidden",
    "qrwkv": "layer_major_all_hidden",
    "comparison": "batch_major_final_hidden",
}

WKV_STATE_CONVENTION = {
    "radlads": "full_sequence_final_state",
    "qrwkv": "full_sequence_final_state",
    "comparison": "as_exported",
}

STEPWISE_CONVENTION = {
    "radlads": "stepwise_only_for_tiny_stepwise_state",
    "qrwkv": "stepwise_only_for_tiny_stepwise_state",
}

HIDDEN_CANDIDATES = (
    "as_is",
    "drop_layer_axis_if_present",
    "select_final_layer_if_all_layers_present",
    "add_single_layer_axis_if_missing",
    "transpose_layer_batch_seq_hidden",
    "transpose_batch_layer_seq_hidden",
    "compare_final_hidden_only",
    "compare_all_layer_hidden_if_available",
)

WKV_CANDIDATES = (
    "as_is",
    "swap_batch_layer_axes",
    "swap_layer_head_axes",
    "swap_batch_head_axes",
    "transpose_last_two_matrix_dims",
    "swap_batch_layer_and_transpose_last_two",
    "swap_layer_head_and_transpose_last_two",
    "compare_pre_update_state_if_available",
    "compare_post_update_state_if_available",
    "cast_both_float32_before_compare",
)

STEPWISE_CANDIDATES = (
    "full_final_vs_stepwise_final",
    "stepwise_last_state_vs_full_final_state",
    "per_time_step_state_if_available",
)


def load_output_pairs(
    radlads_outputs: Path,
    qrwkv_outputs: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    return load_case_output_arrays(radlads_outputs), load_case_output_arrays(
        qrwkv_outputs
    )


def normalize_qrwkv_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
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


def _squeeze_singleton_time_axis(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim >= 4 and value.shape[-2] == 1:
        return np.squeeze(value, axis=-2)
    return value


def surface_stats(array: Any) -> dict[str, Any]:
    if array is None:
        return {
            "present": False,
            "shape": None,
            "dtype": None,
            "finite": False,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
        }
    value = np.asarray(array)
    finite = bool(np.isfinite(value).all()) if value.size else True
    return {
        "present": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "min": float(np.min(value)) if value.size else None,
        "max": float(np.max(value)) if value.size else None,
        "mean": float(np.mean(value)) if value.size else None,
        "std": float(np.std(value)) if value.size else None,
        "abs_max": float(np.max(np.abs(value))) if value.size else None,
    }


def classify_shape_relation(
    radlads: np.ndarray | None,
    qrwkv: np.ndarray | None,
    *,
    surface: str | None = None,
) -> str:
    if radlads is None or qrwkv is None:
        return "missing_source"
    r_shape = tuple(np.asarray(radlads).shape)
    q_shape = tuple(np.asarray(qrwkv).shape)
    if r_shape == q_shape:
        return "same_shape"
    if (
        surface == "hidden_states"
        and len(r_shape) == 3
        and len(q_shape) == 4
        and q_shape[0] == r_shape[0]
    ):
        return "layer_major_all_hidden"
    if len(r_shape) + 1 == len(q_shape) and q_shape[-len(r_shape) :] == r_shape:
        return "layer_axis_present"
    if len(r_shape) == len(q_shape) + 1 and r_shape[-len(q_shape) :] == q_shape:
        return "layer_axis_missing"
    if len(r_shape) == 5 and len(q_shape) == 5:
        return "same_rank_layout_sensitive"
    return "rank_mismatch"


def suspected_axis_meaning(surface: str, relation: str) -> str:
    if surface == "hidden_states":
        if relation == "layer_major_all_hidden":
            return (
                "qrwkv hidden states include layer axis; radlads exports final hidden"
            )
        if relation == "layer_axis_present":
            return "one side includes a layer axis while the other is final hidden only"
        return "final hidden representation"
    if surface == "wkv_matrix_state":
        return (
            "full recurrent matrix state; compare for axis order or pre/post convention"
        )
    if surface.startswith("stepwise_"):
        return "stepwise state; only meaningful on tiny_stepwise_state"
    return "direct surface compare"


def stats_for_surface_pair(
    case: str,
    surface: str,
    radlads: np.ndarray | None,
    qrwkv: np.ndarray | None,
) -> dict[str, Any]:
    r_stats = surface_stats(radlads)
    q_stats = surface_stats(qrwkv)
    relation = classify_shape_relation(radlads, qrwkv, surface=surface)
    return {
        "case": case,
        "surface": surface,
        "radlads_present": radlads is not None,
        "qrwkv_present": qrwkv is not None,
        "radlads_shape": r_stats["shape"],
        "qrwkv_shape": q_stats["shape"],
        "radlads_dtype": r_stats["dtype"],
        "qrwkv_dtype": q_stats["dtype"],
        "radlads_finite": r_stats["finite"],
        "qrwkv_finite": q_stats["finite"],
        "radlads_min": r_stats["min"],
        "radlads_max": r_stats["max"],
        "radlads_mean": r_stats["mean"],
        "radlads_std": r_stats["std"],
        "qrwkv_min": q_stats["min"],
        "qrwkv_max": q_stats["max"],
        "qrwkv_mean": q_stats["mean"],
        "qrwkv_std": q_stats["std"],
        "radlads_abs_max": r_stats["abs_max"],
        "qrwkv_abs_max": q_stats["abs_max"],
        "shape_relation": relation,
        "suspected_axis_meaning": suspected_axis_meaning(surface, relation),
    }


def _shape_match(left: np.ndarray, right: np.ndarray) -> bool:
    return tuple(np.asarray(left).shape) == tuple(np.asarray(right).shape)


def _finite_both(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.isfinite(left).all() and np.isfinite(right).all())


def compare_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    finite_both = _finite_both(left_arr, right_arr)
    if not _shape_match(left_arr, right_arr):
        return {
            "shape_match": False,
            "finite_both": finite_both,
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_relative_error": None,
            "allclose": False,
        }
    diff = np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64))
    denom = np.maximum(np.abs(right_arr.astype(np.float64)), 1e-12)
    return {
        "shape_match": True,
        "finite_both": finite_both,
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_relative_error": float(np.max(diff / denom)) if diff.size else 0.0,
        "allclose": bool(np.allclose(left_arr, right_arr, atol=atol, rtol=rtol)),
    }


def _transformed_pair(
    candidate: str,
    surface: str,
    radlads: np.ndarray | None,
    qrwkv: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    if radlads is None or qrwkv is None:
        return None, None, False
    r = np.asarray(radlads)
    q = np.asarray(qrwkv)
    if surface == "hidden_states":
        if candidate == "as_is":
            return r, q, True
        if candidate in {
            "drop_layer_axis_if_present",
            "select_final_layer_if_all_layers_present",
        }:
            if q.ndim >= 4:
                return r, q[-1], True
            return r, q, False
        if candidate == "add_single_layer_axis_if_missing":
            return r[None, ...], q, True
        if candidate == "transpose_layer_batch_seq_hidden":
            return r, np.transpose(q, (1, 0, 2, 3)) if q.ndim == 4 else q, q.ndim == 4
        if candidate == "transpose_batch_layer_seq_hidden":
            return r, np.transpose(q, (0, 2, 1, 3)) if q.ndim == 4 else q, q.ndim == 4
        if candidate == "compare_final_hidden_only":
            return r, q[-1] if q.ndim >= 4 else q, True
        if candidate == "compare_all_layer_hidden_if_available":
            if q.ndim >= 4:
                expanded = np.broadcast_to(r[:, None, :, :], q.shape)
                return expanded, q, True
            return r, q, False
    if surface == "wkv_matrix_state":
        if candidate == "as_is" or candidate == "cast_both_float32_before_compare":
            return r.astype(np.float32), q.astype(np.float32), True
        if candidate == "swap_batch_layer_axes":
            return r, np.swapaxes(q, 0, 1), True
        if candidate == "swap_layer_head_axes":
            return r, np.swapaxes(q, 1, 2), True
        if candidate == "swap_batch_head_axes":
            return r, np.swapaxes(q, 0, 2), True
        if candidate == "transpose_last_two_matrix_dims":
            return r, np.transpose(q, (0, 1, 2, 4, 3)), True
        if candidate == "swap_batch_layer_and_transpose_last_two":
            return r, np.swapaxes(q, 0, 1).transpose(0, 1, 2, 4, 3), True
        if candidate == "swap_layer_head_and_transpose_last_two":
            return r, np.swapaxes(q, 1, 2).transpose(0, 1, 2, 4, 3), True
        if candidate in {
            "compare_pre_update_state_if_available",
            "compare_post_update_state_if_available",
        }:
            return None, None, False
    if surface.startswith("stepwise_"):
        if candidate == "full_final_vs_stepwise_final":
            return r, q, True
        if candidate == "stepwise_last_state_vs_full_final_state":
            return r, q, True
        if candidate == "per_time_step_state_if_available":
            return r, q, q.ndim >= 2
    return None, None, False


def evaluate_candidates(
    case: str,
    surface: str,
    radlads: np.ndarray | None,
    qrwkv: np.ndarray | None,
    candidates: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = (
        compare_arrays(radlads, qrwkv)
        if radlads is not None and qrwkv is not None
        else None
    )
    baseline_error = (
        baseline["max_abs_error"] if baseline and baseline["shape_match"] else None
    )
    for candidate in candidates:
        left, right, applicable = _transformed_pair(candidate, surface, radlads, qrwkv)
        if not applicable or left is None or right is None:
            rows.append(
                {
                    "case": case,
                    "surface": surface,
                    "candidate_name": candidate,
                    "applicable": False,
                    "transformed_shapes": None,
                    "shape_match": False,
                    "finite_both": False,
                    "max_abs_error": None,
                    "mean_abs_error": None,
                    "max_relative_error": None,
                    "allclose": False,
                    "rank_improvement_vs_as_is": None,
                    "status": "not_applicable",
                }
            )
            continue
        metrics = compare_arrays(left, right)
        improvement = None
        if baseline_error not in (None, 0) and metrics["max_abs_error"] is not None:
            improvement = (
                baseline_error / metrics["max_abs_error"]
                if metrics["max_abs_error"]
                else None
            )
        elif baseline_error is None and metrics["max_abs_error"] is not None:
            improvement = None
        rows.append(
            {
                "case": case,
                "surface": surface,
                "candidate_name": candidate,
                "applicable": True,
                "transformed_shapes": [
                    list(np.asarray(left).shape),
                    list(np.asarray(right).shape),
                ],
                "shape_match": metrics["shape_match"],
                "finite_both": metrics["finite_both"],
                "max_abs_error": metrics["max_abs_error"],
                "mean_abs_error": metrics["mean_abs_error"],
                "max_relative_error": metrics["max_relative_error"],
                "allclose": metrics["allclose"],
                "rank_improvement_vs_as_is": improvement,
                "status": "pass" if metrics["allclose"] else "fail",
            }
        )
    return rows


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    best_error = None
    for row in rows:
        error = row.get("max_abs_error")
        if row.get("status") == "pass" and error is not None:
            if best_error is None or error < best_error:
                best_error = error
                best = row
        elif best is None and error is not None:
            if best_error is None or error < best_error:
                best_error = error
                best = row
    return {
        "best_candidate": None if best is None else best["candidate_name"],
        "best_candidate_max_abs_error": best_error,
        "best_row": best,
    }
