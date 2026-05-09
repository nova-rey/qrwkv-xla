from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

MAPPING_STATUSES = {
    "mapped_exact",
    "mapped_renamed",
    "shape_mismatch",
    "missing_in_qrwkv",
    "missing_in_radlads",
    "unsupported",
    "source_not_found",
}

DEFAULT_PARAMETER_RENAMES = {
    "embed_tokens.weight": "token_embedding.weight",
    "model.embed_tokens.weight": "token_embedding.weight",
    "norm.weight": "final_layernorm.weight",
    "model.norm.weight": "final_layernorm.weight",
    "lm_head.weight": "lm_head.weight",
}

UNSUPPORTED_PARAMETER_SURFACES = {
    "layers.self_attn.g1",
    "layers.self_attn.g2",
    "layers.self_attn.gate",
    "layers.self_attn.gate_w1",
    "layers.self_attn.gate_w2",
    "layers.*.self_attn.gate",
    "layers.*.self_attn.gate_w1",
    "layers.*.self_attn.gate_w2",
}

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

_TRANSPOSE_2D = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}

_SQUEEZE_FRONT = {
    "self_attn.w0",
    "self_attn.a0",
    "self_attn.v0",
    "self_attn.k_k",
    "self_attn.k_a",
}


def flatten_parameter_shapes(
    tree: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            shapes.update(flatten_parameter_shapes(value, prefix=path))
        elif hasattr(value, "shape"):
            shapes[path] = tuple(int(dim) for dim in np.asarray(value).shape)
    return shapes


def normalize_radlads_parameter_arrays(
    named_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    layer_groups: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for raw_name, value in named_arrays.items():
        name = raw_name.removeprefix("model.")
        array = np.asarray(value)
        if name in {"embed_tokens.weight", "norm.weight", "lm_head.weight"}:
            target = DEFAULT_PARAMETER_RENAMES.get(name, name)
            normalized[target] = _normalize_global_array(target, array)
            continue

        match = _LAYER_RE.match(raw_name)
        if match is None:
            continue
        layer_index = int(match.group(1))
        suffix = match.group(2)
        target = f"layers.{suffix}"
        layer_groups[target][layer_index] = _normalize_layer_array(suffix, array)

    for target, per_layer in sorted(layer_groups.items()):
        stacked = [per_layer[index] for index in sorted(per_layer)]
        normalized[target] = np.stack(stacked, axis=0)
    return normalized


def compare_parameter_surfaces(
    radlads_shapes: Mapping[str, tuple[int, ...]] | None,
    qrwkv_shapes: Mapping[str, tuple[int, ...]],
    *,
    rename_map: Mapping[str, str] | None = None,
    unsupported: set[str] | None = None,
) -> dict[str, Any]:
    if radlads_shapes is None:
        return {
            "schema": "radlads_parameter_mapping.v1",
            "overall_status": "source_not_found",
            "counts": {"source_not_found": 1},
            "mappings": [
                {
                    "radlads": "*",
                    "qrwkv": None,
                    "status": "source_not_found",
                    "reason": "RADLADS parameter source was not available.",
                }
            ],
        }

    renames = dict(DEFAULT_PARAMETER_RENAMES)
    if rename_map is not None:
        renames.update(rename_map)
    unsupported_names = set(UNSUPPORTED_PARAMETER_SURFACES)
    if unsupported is not None:
        unsupported_names.update(unsupported)

    rows: list[dict[str, Any]] = []
    seen_qrwkv: set[str] = set()
    for radlads_name in sorted(radlads_shapes):
        qrwkv_name = renames.get(radlads_name, radlads_name)
        radlads_shape = tuple(radlads_shapes[radlads_name])
        qrwkv_shape = qrwkv_shapes.get(qrwkv_name)
        if radlads_name in unsupported_names:
            status = "unsupported"
            reason = "surface is intentionally outside the tiny numerical fixture map"
        elif qrwkv_shape is None:
            status = "missing_in_qrwkv"
            reason = "no QRWKV-XLA parameter surface with the mapped name"
        elif tuple(qrwkv_shape) != radlads_shape:
            status = "shape_mismatch"
            reason = "mapped parameter shapes differ"
            seen_qrwkv.add(qrwkv_name)
        elif qrwkv_name == radlads_name:
            status = "mapped_exact"
            reason = "same name and shape"
            seen_qrwkv.add(qrwkv_name)
        else:
            status = "mapped_renamed"
            reason = "renamed surface with matching shape"
            seen_qrwkv.add(qrwkv_name)
        rows.append(
            {
                "radlads": radlads_name,
                "qrwkv": qrwkv_name,
                "radlads_shape": list(radlads_shape),
                "qrwkv_shape": None if qrwkv_shape is None else list(qrwkv_shape),
                "status": status,
                "reason": reason,
            }
        )

    for qrwkv_name in sorted(set(qrwkv_shapes) - seen_qrwkv):
        if qrwkv_name in renames.values():
            continue
        rows.append(
            {
                "radlads": None,
                "qrwkv": qrwkv_name,
                "radlads_shape": None,
                "qrwkv_shape": list(qrwkv_shapes[qrwkv_name]),
                "status": "missing_in_radlads",
                "reason": "QRWKV-XLA surface has no RADLADS source mapping",
            }
        )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    overall = (
        "pass" if set(counts).issubset({"mapped_exact", "mapped_renamed"}) else "fail"
    )
    return {
        "schema": "radlads_parameter_mapping.v1",
        "overall_status": overall,
        "counts": counts,
        "mappings": rows,
    }


def write_surface_comparison_reports(
    report: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    payload = dict(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surface_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P49_SURFACE_COMPARISON.md").write_text(
        _surface_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _surface_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P49 RADLADS Surface Comparison",
        "",
        f"- Overall status: `{report.get('overall_status', 'unknown')}`",
        "",
        "| RADLADS | QRWKV-XLA | Status | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("mappings", []):
        lines.append(
            "| "
            f"{row.get('radlads')} | "
            f"{row.get('qrwkv')} | "
            f"`{row.get('status')}` | "
            f"{row.get('reason', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _normalize_global_array(target: str, array: np.ndarray) -> np.ndarray:
    if target == "lm_head.weight":
        return np.asarray(array).T
    return np.asarray(array)


def _normalize_layer_array(suffix: str, array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if suffix in _TRANSPOSE_2D:
        value = value.T
    if suffix in _SQUEEZE_FRONT and value.ndim >= 2 and value.shape[:2] == (1, 1):
        value = value.reshape(value.shape[-1])
    return value
