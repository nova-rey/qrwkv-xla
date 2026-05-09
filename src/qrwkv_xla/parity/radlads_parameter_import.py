from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.parity.radlads_numerical_fixtures import load_numerical_manifest
from qrwkv_xla.parity.radlads_parameter_mapping import (
    flatten_parameter_shapes,
    normalize_radlads_parameter_arrays,
)
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

PARAMETER_IMPORT_SCHEMA = "radlads_parameter_replay_import.v1"

IMPORT_STATUSES = {
    "mapped",
    "defaulted",
    "excluded",
    "unsupported",
    "shape_mismatch",
    "missing_required",
}

QRWKV_DEFAULTED_SURFACES = {
    "layers.self_attn.a_proj.weight": {
        "reason": (
            "QRWKV dense ICLR projection is bypassed by RADLADS "
            "low-rank a0/a1/a2 replay."
        ),
        "parity_risk": "low",
    },
    "layers.self_attn.b_proj.weight": {
        "reason": "QRWKV extra state update term is bypassed during RADLADS replay.",
        "parity_risk": "medium",
    },
    "layers.self_attn.g_proj.weight": {
        "reason": "QRWKV dense gate projection is bypassed by RADLADS g1/g2 replay.",
        "parity_risk": "low",
    },
    "layers.self_attn.w_proj.weight": {
        "reason": (
            "QRWKV dense decay projection is bypassed by RADLADS "
            "low-rank w0/w1/w2 replay."
        ),
        "parity_risk": "low",
    },
    "layers.self_attn.time_bias": {
        "reason": (
            "QRWKV dense decay bias is bypassed by RADLADS low-rank decay replay."
        ),
        "parity_risk": "low",
    },
    "layers.self_attn.time_mix": {
        "reason": (
            "RADLADS source has token-shift parameters commented out "
            "for this fixture path."
        ),
        "parity_risk": "medium",
    },
    "lm_head.bias": {
        "reason": (
            "RADLADS tiny causal LM head has bias=False; "
            "QRWKV apply_lm_head requires an explicit zero bias."
        ),
        "parity_risk": "low",
    },
}

SUPPORTED_REPLAY_SURFACES = {
    "token_embedding.weight",
    "layers.input_layernorm.weight",
    "layers.post_attention_layernorm.weight",
    "layers.self_attn.q_proj.weight",
    "layers.self_attn.q_proj.bias",
    "layers.self_attn.k_proj.weight",
    "layers.self_attn.k_proj.bias",
    "layers.self_attn.v_proj.weight",
    "layers.self_attn.v_proj.bias",
    "layers.self_attn.o_proj.weight",
    "layers.self_attn.w0",
    "layers.self_attn.w1",
    "layers.self_attn.w2",
    "layers.self_attn.a0",
    "layers.self_attn.a1",
    "layers.self_attn.a2",
    "layers.self_attn.v0",
    "layers.self_attn.v1",
    "layers.self_attn.v2",
    "layers.self_attn.g1",
    "layers.self_attn.g2",
    "layers.self_attn.k_k",
    "layers.self_attn.k_a",
    "layers.self_attn.ln_x.weight",
    "layers.self_attn.ln_x.bias",
    "layers.mlp.gate_proj.weight",
    "layers.mlp.up_proj.weight",
    "layers.mlp.down_proj.weight",
    "final_layernorm.weight",
    "lm_head.weight",
}

EXCLUDED_RADLADS_SURFACES = {
    "layers.self_attn.r_k": {
        "reason": (
            "RADLADS source has the r_k residual contribution commented out in "
            "the inspected forward path, so bounded replay excludes it."
        ),
        "replay_impact": "low",
    },
}

REQUIRED_REPLAY_SURFACES = SUPPORTED_REPLAY_SURFACES


@dataclass(frozen=True)
class RadladsParameterImportResult:
    params: dict[str, object]
    qrwkv_config: RWKV7QwenReferenceConfig
    mapping_entries: list[dict[str, Any]]
    missing_required: list[dict[str, Any]]
    defaulted: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    unsupported: list[dict[str, Any]]
    shape_mismatches: list[dict[str, Any]]
    mapped: list[dict[str, Any]]
    overall_status: str
    report: dict[str, Any]


def load_radlads_parameter_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def replay_config_from_normalized_parameters(
    arrays: Mapping[str, np.ndarray],
) -> RWKV7QwenReferenceConfig:
    embedding = np.asarray(arrays["token_embedding.weight"])
    q_weight = np.asarray(arrays["layers.self_attn.q_proj.weight"])
    k_weight = np.asarray(arrays["layers.self_attn.k_proj.weight"])
    w1 = np.asarray(arrays["layers.self_attn.w1"])
    a1 = np.asarray(arrays["layers.self_attn.a1"])
    v1 = np.asarray(arrays["layers.self_attn.v1"])
    g1 = np.asarray(arrays["layers.self_attn.g1"])
    r_k = np.asarray(arrays["layers.self_attn.r_k"])
    mlp_up = np.asarray(arrays["layers.mlp.up_proj.weight"])

    vocab_size, hidden_size = embedding.shape
    num_layers = q_weight.shape[0]
    q_out = q_weight.shape[2]
    q_bias = np.asarray(arrays["layers.self_attn.q_proj.bias"])
    if q_bias.shape != (num_layers, q_out):
        raise ValueError("q_proj bias shape is incompatible with q_proj weight")
    if hidden_size != q_weight.shape[1]:
        raise ValueError("q_proj weight input dim does not match hidden size")

    k_out = k_weight.shape[2]
    num_heads = int(r_k.shape[1])
    head_size = int(r_k.shape[2])
    if num_heads <= 0 or head_size <= 0:
        raise ValueError("r_k shape is incompatible with head layout inference")
    if hidden_size != num_heads * head_size:
        raise ValueError("hidden size is incompatible with inferred head layout")
    if k_out % head_size != 0:
        raise ValueError("k_proj output dim is not divisible by inferred head size")
    num_kv_heads = int(k_out // head_size)

    return RWKV7QwenReferenceConfig(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        intermediate_size=int(mlp_up.shape[2]),
        emit_logits=True,
        emit_mixer_outputs=True,
        tie_embeddings=False,
        use_rope=False,
        radlads_compatible_math=True,
        radlads_replay_mode=True,
        radlads_attention_group_norm=True,
        radlads_balance_state=True,
        attention_qkv_bias=True,
        radlads_low_rank_gate=True,
        lora_rank_decay=int(w1.shape[2]),
        lora_rank_iclr=int(a1.shape[2]),
        lora_rank_value_residual_mix=int(v1.shape[2]),
        lora_rank_gate=int(g1.shape[2]),
    )


def import_radlads_parameters_for_replay(
    parameter_payload_path: Path,
    *,
    qrwkv_config: RWKV7QwenReferenceConfig | None = None,
    manifest_path: Path | None = None,
    allow_defaults: bool = False,
    seed: int = 5050,
) -> RadladsParameterImportResult:
    raw_arrays = load_radlads_parameter_npz(parameter_payload_path)
    normalized = normalize_radlads_parameter_arrays(raw_arrays)
    config = qrwkv_config or replay_config_from_normalized_parameters(normalized)
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(seed))
    _zero_defaulted_surfaces(params)

    qrwkv_shapes = flatten_parameter_shapes(params)
    rows: list[dict[str, Any]] = []

    for name in sorted(normalized):
        value = np.asarray(normalized[name])
        if name in EXCLUDED_RADLADS_SURFACES:
            note = EXCLUDED_RADLADS_SURFACES[name]
            rows.append(
                _row(
                    radlads=name,
                    qrwkv=None,
                    status="excluded",
                    reason=note["reason"],
                    radlads_shape=value.shape,
                    replay_impact=note["replay_impact"],
                )
            )
            continue
        if name not in SUPPORTED_REPLAY_SURFACES:
            rows.append(
                _row(
                    radlads=name,
                    qrwkv=None,
                    status="unsupported",
                    reason="RADLADS parameter is not consumed by bounded replay.",
                    radlads_shape=value.shape,
                )
            )
            continue
        expected_shape = qrwkv_shapes.get(name)
        if expected_shape is None:
            rows.append(
                _row(
                    radlads=name,
                    qrwkv=name,
                    status="missing_required",
                    reason="QRWKV replay params do not expose this required surface.",
                    radlads_shape=value.shape,
                )
            )
            continue
        if tuple(expected_shape) != tuple(value.shape):
            rows.append(
                _row(
                    radlads=name,
                    qrwkv=name,
                    status="shape_mismatch",
                    reason="RADLADS value shape does not match QRWKV replay shape.",
                    radlads_shape=value.shape,
                    qrwkv_shape=expected_shape,
                )
            )
            continue
        _set_param(params, name, jnp.asarray(value, dtype=jnp.float32))
        rows.append(
            _row(
                radlads=name,
                qrwkv=name,
                status="mapped",
                reason="RADLADS array imported for replay.",
                radlads_shape=value.shape,
                qrwkv_shape=expected_shape,
            )
        )

    for name in sorted(REQUIRED_REPLAY_SURFACES - set(normalized)):
        rows.append(
            _row(
                radlads=name,
                qrwkv=name,
                status="missing_required",
                reason=(
                    "Required replay surface is absent from RADLADS parameter payload."
                ),
                qrwkv_shape=qrwkv_shapes.get(name),
            )
        )

    for name, metadata in sorted(QRWKV_DEFAULTED_SURFACES.items()):
        rows.append(
            _row(
                radlads=None,
                qrwkv=name,
                status="defaulted",
                reason=metadata["reason"],
                qrwkv_shape=qrwkv_shapes.get(name),
                source="defaulted",
                parity_risk=metadata["parity_risk"],
            )
        )

    mapped = [row for row in rows if row["status"] == "mapped"]
    defaulted = [row for row in rows if row["status"] == "defaulted"]
    excluded = [row for row in rows if row["status"] == "excluded"]
    unsupported = [row for row in rows if row["status"] == "unsupported"]
    shape_mismatches = [row for row in rows if row["status"] == "shape_mismatch"]
    missing_required = [row for row in rows if row["status"] == "missing_required"]

    blocking_statuses = {"shape_mismatch", "missing_required"}
    if not allow_defaults:
        blocking_statuses.add("defaulted")
    overall = (
        "pass"
        if not any(row["status"] in blocking_statuses for row in rows)
        else "fail"
    )

    manifest_summary = None
    if manifest_path is not None and manifest_path.exists():
        manifest = load_numerical_manifest(manifest_path)
        manifest_summary = {
            "manifest": str(manifest_path),
            "real_radlads_fixture_status": manifest.get("real_radlads_fixture_status"),
            "required_cases": manifest.get("required_cases"),
        }

    report = {
        "schema": PARAMETER_IMPORT_SCHEMA,
        "parameter_payload": str(parameter_payload_path),
        "manifest": None if manifest_summary is None else manifest_summary["manifest"],
        "overall_status": overall,
        "allow_defaults": allow_defaults,
        "counts": _counts(rows),
        "mapped": mapped,
        "defaulted": defaulted,
        "excluded": excluded,
        "unsupported": unsupported,
        "shape_mismatches": shape_mismatches,
        "missing_required": missing_required,
        "mapping_entries": rows,
        "g1_g2_status": {
            "status": "implemented",
            "source_reason": (
                "Inspected RADLADS rwkv7qwen2/modeling_rwkv7qwen2.py: "
                "gate_rank_type == 2 computes sigmoid(xg @ g1) @ g2."
            ),
            "replay_impact": "active in replay mode",
        },
        "qkv_bias_status": {
            "status": "implemented",
            "flag": "attention_qkv_bias",
            "source": (
                "RADLADS q_proj.bias/k_proj.bias/v_proj.bias are imported "
                "when replay mode is enabled."
            ),
        },
        "source_notes": {
            "token_shift": (
                "RADLADS x_r/x_w/x_k/x_v/x_a/x_g token-shift parameter "
                "lines are commented out in the inspected source path, so "
                "replay sets QRWKV time_mix/time_bias to deterministic "
                "defaults."
            ),
        },
        "fixture_manifest": manifest_summary,
    }

    return RadladsParameterImportResult(
        params=params,
        qrwkv_config=config,
        mapping_entries=rows,
        missing_required=missing_required,
        defaulted=defaulted,
        excluded=excluded,
        unsupported=unsupported,
        shape_mismatches=shape_mismatches,
        mapped=mapped,
        overall_status=overall,
        report=report,
    )


def write_parameter_import_report(report: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameter_import_report.json").write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P50_PARAMETER_IMPORT_REPORT.md").write_text(
        _parameter_import_markdown(report),
        encoding="utf-8",
    )


def _parameter_import_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P50 Parameter Import Report",
        "",
        f"- Overall status: `{report.get('overall_status', 'unknown')}`",
        f"- Parameter payload: `{report.get('parameter_payload', '')}`",
        f"- allow_defaults: `{report.get('allow_defaults', False)}`",
        "",
        "## Counts",
        "",
    ]
    for status, count in sorted(dict(report.get("counts", {})).items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Entries",
            "",
            "| RADLADS | QRWKV-XLA | Status | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in report.get("mapping_entries", []):
        lines.append(
            "| {radlads} | {qrwkv} | `{status}` | {reason} |".format(
                radlads=row.get("radlads"),
                qrwkv=row.get("qrwkv"),
                status=row.get("status"),
                reason=row.get("reason", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _zero_defaulted_surfaces(params: dict[str, object]) -> None:
    self_attn = params["layers"]["self_attn"]  # type: ignore[index]
    for proj_name in ("a_proj", "b_proj", "g_proj", "w_proj"):
        self_attn[proj_name]["weight"] = jnp.zeros_like(  # type: ignore[index]
            self_attn[proj_name]["weight"]  # type: ignore[index]
        )
    self_attn["time_bias"] = jnp.zeros_like(self_attn["time_bias"])  # type: ignore[index]
    self_attn["time_mix"] = jnp.zeros_like(self_attn["time_mix"])  # type: ignore[index]
    if "lm_head" in params:
        params["lm_head"]["bias"] = jnp.zeros_like(  # type: ignore[index]
            params["lm_head"]["bias"]  # type: ignore[index]
        )


def _set_param(params: dict[str, object], path: str, value: jax.Array) -> None:
    parts = path.split(".")
    target: Any = params
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _row(
    *,
    radlads: str | None,
    qrwkv: str | None,
    status: str,
    reason: str,
    radlads_shape: tuple[int, ...] | None = None,
    qrwkv_shape: tuple[int, ...] | None = None,
    source: str | None = None,
    parity_risk: str | None = None,
    replay_impact: str | None = None,
) -> dict[str, Any]:
    return {
        "radlads": radlads,
        "qrwkv": qrwkv,
        "status": status,
        "reason": reason,
        "source": source,
        "parity_risk": parity_risk,
        "replay_impact": replay_impact,
        "radlads_shape": None if radlads_shape is None else list(radlads_shape),
        "qrwkv_shape": None if qrwkv_shape is None else list(qrwkv_shape),
    }


def _counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts
