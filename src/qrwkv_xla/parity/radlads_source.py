from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_VERSION = 1
FIXTURE_SCHEMA = "radlads_source_parity.v1"
REQUIRED_CASE_NAMES = (
    "tiny_no_mask",
    "tiny_attention_mask",
    "tiny_prefix_padding_or_left_padding",
)
SUPPORTED_STATUSES = {"pass", "fail", "unsupported"}


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    base = manifest_path.parent
    manifest = load_manifest(manifest_path)
    _require(manifest.get("fixture_version") == FIXTURE_VERSION, "bad fixture_version")
    _require(manifest.get("schema") == FIXTURE_SCHEMA, "bad schema")
    _require(manifest.get("backend") == "rwkv7_qwen_reference", "bad backend")
    _require(isinstance(manifest.get("cases"), list), "cases must be a list")
    case_names = {case.get("name") for case in manifest["cases"]}
    _require(
        set(REQUIRED_CASE_NAMES).issubset(case_names), "missing required tiny cases"
    )

    for case in manifest["cases"]:
        _validate_case(base, case)
    return manifest


def load_case_arrays(
    manifest_path: Path, case: dict[str, Any]
) -> dict[str, np.ndarray]:
    payload_path = _manifest_path(manifest_path).parent / case["payload"]
    with np.load(payload_path) as payload:
        return {name: payload[name] for name in payload.files}


def import_fixture_directory(
    source: Path, out: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    manifest = validate_manifest(source)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to replace fixtures")
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*"):
        if path.is_file():
            path.unlink()
    source_base = _manifest_path(source).parent
    shutil.copy2(source_base / "manifest.json", out / "manifest.json")
    for case in manifest["cases"]:
        shutil.copy2(source_base / case["payload"], out / case["payload"])
    return validate_manifest(out)


def compare_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path)
    results: list[dict[str, Any]] = []
    counts = {"pass": 0, "fail": 0, "unsupported": 0}

    for case in manifest["cases"]:
        arrays = load_case_arrays(manifest_path, case)
        status = str(case.get("status", "unsupported"))
        comparisons = case.get("comparisons", [])
        if status == "unsupported" or not comparisons:
            counts["unsupported"] += 1
            results.append(
                {
                    "name": case["name"],
                    "status": "unsupported",
                    "reason": case.get("unsupported_reason", "no fair comparison"),
                    "comparisons": [],
                }
            )
            continue

        case_comparisons = []
        case_failed = False
        for spec in comparisons:
            result = _compare_arrays(arrays, spec)
            case_comparisons.append(result)
            case_failed = case_failed or result["status"] == "fail"
        final_status = "fail" if case_failed else "pass"
        counts[final_status] += 1
        results.append(
            {
                "name": case["name"],
                "status": final_status,
                "comparisons": case_comparisons,
            }
        )

    overall = (
        "fail" if counts["fail"] else ("pass" if counts["pass"] else "unsupported")
    )
    return {
        "schema": "radlads_source_parity_report.v1",
        "manifest": str(_manifest_path(manifest_path)),
        "overall_status": overall,
        "counts": counts,
        "cases": results,
    }


def write_comparison_reports(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    report = compare_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P40_PARITY_REPORT.md").write_text(
        _comparison_markdown(report),
        encoding="utf-8",
    )
    return report


def build_parameter_surface_map() -> dict[str, Any]:
    rows = [
        _row("embed_tokens.weight", "token_embedding.weight", "direct_shape_role"),
        _row(
            "layers.*.input_layernorm.weight",
            "layers.input_layernorm.weight",
            "direct_role_stacked_layers",
        ),
        _row(
            "layers.*.post_attention_layernorm.weight",
            "layers.post_attention_layernorm.weight",
            "direct_role_stacked_layers",
        ),
        _row(
            "layers.*.self_attn.q_proj.weight",
            "layers.self_attn.q_proj.weight",
            "direct_role_stacked_layers",
        ),
        _row(
            "layers.*.self_attn.k_proj.weight",
            "layers.self_attn.k_proj.weight",
            "direct_role_grouped_kv_shape",
        ),
        _row(
            "layers.*.self_attn.v_proj.weight",
            "layers.self_attn.v_proj.weight",
            "direct_role_grouped_kv_shape",
        ),
        _row(
            "layers.*.self_attn.o_proj.weight",
            "layers.self_attn.o_proj.weight",
            "direct_role_stacked_layers",
        ),
        _row(
            "layers.*.mlp.gate_proj.weight",
            "layers.mlp.gate_proj.weight",
            "direct_role_stacked_layers",
        ),
        _row(
            "layers.*.mlp.up_proj.weight",
            "layers.mlp.up_proj.weight",
            "direct_role_stacked_layers",
        ),
        _row(
            "layers.*.mlp.down_proj.weight",
            "layers.mlp.down_proj.weight",
            "direct_role_stacked_layers",
        ),
        _row("norm.weight", "final_layernorm.weight", "direct_role"),
        _row("lm_head.weight", "lm_head.weight", "direct_role_if_logits_untied"),
        _row(
            "layers.*.self_attn.w0/w1/w2",
            "layers.self_attn.w_proj.weight + time_bias",
            "unsupported",
            "RADLADS low-rank decay is not represented by QRWKV dense "
            "projection without a fitted conversion.",
        ),
        _row(
            "layers.*.self_attn.a0/a1/a2",
            "layers.self_attn.a_proj.weight",
            "unsupported",
            "RADLADS low-rank ICLR path is not represented by QRWKV dense "
            "projection without a fitted conversion.",
        ),
        _row(
            "layers.*.self_attn.v0/v1/v2",
            None,
            "unsupported",
            "Value residual mix against v_first is absent in QRWKV-XLA "
            "current behavior.",
        ),
        _row(
            "layers.*.self_attn.gate or g1/g2",
            "layers.self_attn.g_proj.weight",
            "unsupported",
            "Gate variants and low-rank gate parameterization are not equivalent.",
        ),
        _row(
            "layers.*.self_attn.k_k/k_a/r_k",
            None,
            "unsupported",
            "Balance-state and residual key terms are not present in current "
            "QRWKV parameters.",
        ),
        _row(
            "layers.*.self_attn.ln_x",
            None,
            "unsupported",
            "RADLADS optional attention group norm is not implemented in the "
            "current slow reference.",
        ),
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema": "radlads_parameter_surface_map.v1",
        "radlads_source": {
            "path": "/home/nyx/.openclaw/workspace/_refs/RADLADS",
            "branch": "radlads",
            "head": "1b362eb",
        },
        "qrwkv_backend": "rwkv7_qwen_reference",
        "counts": counts,
        "mappings": rows,
        "claim": (
            "Mapping report only; unsupported rows must not be used for "
            "numerical parity."
        ),
    }


def write_parameter_surface_map_reports(out_dir: Path) -> dict[str, Any]:
    report = build_parameter_surface_map()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameter_surface_map.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P40_PARAMETER_SURFACE_MAP.md").write_text(
        _parameter_markdown(report),
        encoding="utf-8",
    )
    return report


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_case(base: Path, case: dict[str, Any]) -> None:
    _require(case.get("name") in REQUIRED_CASE_NAMES, "unknown case name")
    status = case.get("status")
    _require(status in SUPPORTED_STATUSES, f"bad case status for {case.get('name')}")
    payload_name = case.get("payload")
    _require(isinstance(payload_name, str), "case payload must be a string")
    payload_path = base / payload_name
    _require(payload_path.is_file(), f"missing payload {payload_name}")
    arrays = load_case_arrays(base / "manifest.json", case)
    _require("input_ids" in arrays, f"{case['name']} missing input_ids")
    _require(arrays["input_ids"].ndim == 2, f"{case['name']} input_ids must be [B,T]")
    mask_meta = case.get("attention_mask", {})
    if mask_meta.get("present"):
        _require("attention_mask" in arrays, f"{case['name']} missing attention_mask")
        _require(
            arrays["attention_mask"].shape == arrays["input_ids"].shape,
            f"{case['name']} attention_mask shape mismatch",
        )
    declared_hash = case.get("payload_sha256")
    if declared_hash is not None:
        _require(
            declared_hash == hash_arrays(arrays), f"{case['name']} bad payload hash"
        )
    for spec in case.get("comparisons", []):
        left_name = spec.get("left")
        right_name = spec.get("right")
        _require(left_name in arrays, f"{case['name']} missing array {left_name}")
        _require(right_name in arrays, f"{case['name']} missing array {right_name}")
        _require(
            arrays[left_name].shape == arrays[right_name].shape,
            f"{case['name']} comparison shape mismatch",
        )


def _compare_arrays(
    arrays: dict[str, np.ndarray], spec: dict[str, Any]
) -> dict[str, Any]:
    left_name = spec["left"]
    right_name = spec["right"]
    if left_name not in arrays or right_name not in arrays:
        return {
            "name": spec.get("name", left_name),
            "status": "fail",
            "reason": "missing array",
            "left": left_name,
            "right": right_name,
        }
    left = np.asarray(arrays[left_name])
    right = np.asarray(arrays[right_name])
    if left.shape != right.shape:
        return {
            "name": spec.get("name", left_name),
            "status": "fail",
            "reason": "shape mismatch",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    atol = float(spec.get("atol", 1e-5))
    rtol = float(spec.get("rtol", 1e-5))
    abs_diff = float(np.max(np.abs(left - right))) if left.size else 0.0
    denom = np.maximum(np.abs(right), 1e-12)
    rel_diff = float(np.max(np.abs(left - right) / denom)) if left.size else 0.0
    passed = bool(np.allclose(left, right, atol=atol, rtol=rtol))
    return {
        "name": spec.get("name", left_name),
        "status": "pass" if passed else "fail",
        "left": left_name,
        "right": right_name,
        "shape": list(left.shape),
        "max_abs": abs_diff,
        "max_rel": rel_diff,
        "atol": atol,
        "rtol": rtol,
    }


def _comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P40 RADLADS Source Parity Report",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Pass: {report['counts']['pass']}",
        f"- Fail: {report['counts']['fail']}",
        f"- Unsupported: {report['counts']['unsupported']}",
        "",
        "| Case | Status | Notes |",
        "| --- | --- | --- |",
    ]
    for case in report["cases"]:
        notes = case.get("reason", "")
        if case.get("comparisons"):
            notes = ", ".join(
                f"{item['name']} max_abs={item.get('max_abs', 'n/a')}"
                for item in case["comparisons"]
            )
        lines.append(f"| {case['name']} | `{case['status']}` | {notes} |")
    lines.append("")
    return "\n".join(lines)


def _parameter_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P40 RADLADS Parameter Surface Map",
        "",
        report["claim"],
        "",
        "| RADLADS surface | QRWKV-XLA surface | Status | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["mappings"]:
        lines.append(
            "| {radlads} | {qrwkv} | `{status}` | {notes} |".format(
                radlads=row["radlads"],
                qrwkv=row.get("qrwkv") or "",
                status=row["status"],
                notes=row.get("notes", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _row(
    radlads: str,
    qrwkv: str | None,
    status: str,
    notes: str = "",
) -> dict[str, str | None]:
    return {"radlads": radlads, "qrwkv": qrwkv, "status": status, "notes": notes}


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
