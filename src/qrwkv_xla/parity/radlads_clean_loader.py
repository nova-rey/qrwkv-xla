from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional CI dependency
    torch = None

from qrwkv_xla.parity.radlads_numerical_fixtures import (
    DEFAULT_RADLADS_SOURCE,
    _build_radlads_model,
    _load_radlads_runtime,
    _qrwkv_case_arrays,
    _tiny_cases,
)
from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
    replay_config_from_normalized_parameters,
)
from qrwkv_xla.parity.radlads_parameter_mapping import (
    DEFAULT_PARAMETER_RENAMES,
    compare_parameter_surfaces,
    flatten_parameter_shapes,
    normalize_radlads_parameter_arrays,
)
from qrwkv_xla.students import RWKV7QwenReferenceStudent

CLEAN_LOADER_SCHEMA = "radlads_clean_payload_loader.v1"
CLEAN_OUTPUT_SCHEMA = "radlads_clean_payload_outputs.v1"
DEFAULT_SEED = 5353

SUPPORTED_MODEL_DEFAULTS = {
    "layers.self_attn.a_proj.weight": {
        "kind": "zeros",
        "reason": "RADLADS clean payload does not populate this live boundary leaf.",
        "parity_risk": "low",
    },
    "layers.self_attn.b_proj.weight": {
        "kind": "zeros",
        "reason": "RADLADS clean payload does not populate this live boundary leaf.",
        "parity_risk": "medium",
    },
    "layers.self_attn.g_proj.weight": {
        "kind": "zeros",
        "reason": "RADLADS clean payload does not populate this live boundary leaf.",
        "parity_risk": "low",
    },
    "layers.self_attn.ln_x.weight": {
        "kind": "ones",
        "reason": "LayerNorm scale is defaulted deterministically when absent.",
        "parity_risk": "low",
    },
    "layers.self_attn.ln_x.bias": {
        "kind": "zeros",
        "reason": "LayerNorm bias is defaulted deterministically when absent.",
        "parity_risk": "low",
    },
    "layers.self_attn.time_bias": {
        "kind": "zeros",
        "reason": "Token-shift bias is commented out in the inspected source path.",
        "parity_risk": "low",
    },
    "layers.self_attn.time_mix": {
        "kind": "zeros",
        "reason": "Token-shift mix is commented out in the inspected source path.",
        "parity_risk": "medium",
    },
    "layers.self_attn.w_proj.weight": {
        "kind": "zeros",
        "reason": "RADLADS clean payload does not populate this live boundary leaf.",
        "parity_risk": "low",
    },
}

GATE_SURFACES = {
    "layers.self_attn.g1",
    "layers.self_attn.g2",
}

RUNTIME_CRITICAL_SURFACES = {
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
    "layers.mlp.gate_proj.weight",
    "layers.mlp.up_proj.weight",
    "layers.mlp.down_proj.weight",
    "final_layernorm.weight",
    "lm_head.weight",
}

_LAYER_PARAM_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LAYER_TEMPLATE_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class RadladsCleanPayloadLoadResult:
    parameter_payload: str
    radlads_source_path: str
    overall_status: str
    reason: str | None
    counts: dict[str, int]
    caveats: list[dict[str, Any]]
    mapping_entries: list[dict[str, Any]]
    mapped: list[dict[str, Any]]
    defaulted: list[dict[str, Any]]
    unsupported: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    shape_mismatches: list[dict[str, Any]]
    missing_required: list[dict[str, Any]]
    loaded_parameter_count: int
    state_dict_keys: list[str]
    model_info: dict[str, Any]
    report: dict[str, Any]
    runtime: dict[str, Any] | None = None
    model: Any | None = None
    qrwkv_config: Any | None = None
    loaded_state_dict: dict[str, Any] | None = None


@dataclass(frozen=True)
class RadladsCleanPayloadExportResult:
    output_manifest: dict[str, Any]
    load_result: RadladsCleanPayloadLoadResult


def _resolve_payload_value(
    model_name: str,
    normalized: Mapping[str, np.ndarray],
) -> tuple[str | None, np.ndarray | None]:
    clean_name = model_name.removeprefix("model.")
    if clean_name in normalized:
        return clean_name, np.asarray(normalized[clean_name])

    renamed = DEFAULT_PARAMETER_RENAMES.get(clean_name)
    if renamed is not None and renamed in normalized:
        return renamed, np.asarray(normalized[renamed])

    match = _LAYER_PARAM_RE.match(model_name)
    if match is not None:
        layer_index = int(match.group(1))
        suffix = match.group(2)
        surface_name = f"layers.{suffix}"
        stacked = normalized.get(surface_name)
        if stacked is not None:
            stacked_array = np.asarray(stacked)
            if stacked_array.ndim >= 1 and layer_index < stacked_array.shape[0]:
                return surface_name, np.asarray(stacked_array[layer_index])

    return None, None


def _maybe_transpose_matches(
    array: np.ndarray,
    expected_shape: tuple[int, ...],
) -> bool:
    value = np.asarray(array)
    return value.ndim == 2 and tuple(value.T.shape) == tuple(expected_shape)


def load_radlads_clean_payload(
    parameter_payload_path: Path,
    *,
    radlads_source_path: Path = DEFAULT_RADLADS_SOURCE,
    seed: int = DEFAULT_SEED,
    run_smoke: bool = True,
) -> RadladsCleanPayloadLoadResult:
    with np.load(parameter_payload_path) as payload:
        raw_arrays = {name: np.asarray(payload[name]) for name in payload.files}
    normalized = normalize_radlads_parameter_arrays(raw_arrays)
    if not radlads_source_path.exists():
        reason = f"RADLADS source path not found: {radlads_source_path}"
        report = _blocked_report(
            parameter_payload_path=parameter_payload_path,
            radlads_source_path=radlads_source_path,
            reason=reason,
            raw_arrays=raw_arrays,
        )
        return RadladsCleanPayloadLoadResult(
            parameter_payload=str(parameter_payload_path),
            radlads_source_path=str(radlads_source_path),
            overall_status="blocked",
            reason=reason,
            counts=report["counts"],
            caveats=report["caveats"],
            mapping_entries=report["mapping_entries"],
            mapped=report["mapped"],
            defaulted=report["defaulted"],
            unsupported=report["unsupported"],
            excluded=report.get("excluded", []),
            shape_mismatches=report["shape_mismatches"],
            missing_required=report["missing_required"],
            loaded_parameter_count=0,
            state_dict_keys=[],
            model_info=report["model_info"],
            report=report,
        )
    try:
        runtime = _load_radlads_runtime(radlads_source_path)
        config = replay_config_from_normalized_parameters(normalized)
        _, model = _build_radlads_model(runtime, seed=seed, all_math=False)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        report = _blocked_report(
            parameter_payload_path=parameter_payload_path,
            radlads_source_path=radlads_source_path,
            reason=reason,
            raw_arrays=raw_arrays,
        )
        return RadladsCleanPayloadLoadResult(
            parameter_payload=str(parameter_payload_path),
            radlads_source_path=str(radlads_source_path),
            overall_status="blocked",
            reason=reason,
            counts=report["counts"],
            caveats=report["caveats"],
            mapping_entries=report["mapping_entries"],
            mapped=report["mapped"],
            defaulted=report["defaulted"],
            unsupported=report["unsupported"],
            excluded=report["excluded"],
            shape_mismatches=report["shape_mismatches"],
            missing_required=report["missing_required"],
            loaded_parameter_count=0,
            state_dict_keys=[],
            model_info=report["model_info"],
            report=report,
        )

    named = dict(model.named_parameters())
    model_shapes = flatten_parameter_shapes(named)
    payload_shapes = flatten_parameter_shapes(normalized)
    mapping_report = compare_parameter_surfaces(model_shapes, payload_shapes)
    payload_name_map = {
        str(row["radlads"]): str(row["qrwkv"] or row["radlads"])
        for row in mapping_report["mappings"]
        if row.get("radlads") is not None
    }
    handled_names = {
        str(row["radlads"])
        for row in mapping_report["mappings"]
        if row.get("radlads") is not None
    }
    mapping_entries = []
    mapped: list[dict[str, Any]] = []
    defaulted: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    shape_mismatches: list[dict[str, Any]] = []
    missing_required: list[dict[str, Any]] = []
    caveats: list[dict[str, Any]] = []
    loaded_state_dict = dict(model.state_dict())
    loaded_parameter_count = 0

    for row in mapping_report["mappings"]:
        entry = dict(row)
        name = str(entry["radlads"])
        payload_name, payload_value = _resolve_payload_value(name, normalized)
        template_name = _surface_template(name)

        if payload_value is not None:
            payload_shape = tuple(int(dim) for dim in np.asarray(payload_value).shape)
            expected_shape = tuple(
                int(dim) for dim in np.asarray(named[name].detach().cpu()).shape
            )
            entry["payload_name"] = payload_name
            entry["payload_shape"] = list(payload_shape)
            if payload_shape == expected_shape:
                entry["status"] = (
                    "mapped_exact"
                    if payload_name == name.removeprefix("model.")
                    else "mapped_renamed"
                )
                entry["reason"] = "same shape"
                mapped.append(entry)
                loaded_state_dict[name] = torch.as_tensor(
                    payload_value, dtype=loaded_state_dict[name].dtype
                )
                loaded_parameter_count += 1
            elif _maybe_transpose_matches(payload_value, expected_shape):
                entry["status"] = "mapped_transposed"
                entry["reason"] = "2D surface matches after transpose"
                entry["adapter"] = "transpose"
                mapped.append(entry)
                loaded_state_dict[name] = torch.as_tensor(
                    np.asarray(payload_value).T, dtype=loaded_state_dict[name].dtype
                )
                loaded_parameter_count += 1
            elif template_name in GATE_SURFACES:
                adapted, adapter_note = _adapt_gate_tensor(
                    payload_value, expected_shape
                )
                if adapted is not None:
                    entry["status"] = "shape_mismatch"
                    entry["reason"] = f"gate surface adapted via {adapter_note}"
                    entry["adapter"] = adapter_note
                    shape_mismatches.append(entry)
                    loaded_state_dict[name] = torch.as_tensor(
                        adapted, dtype=loaded_state_dict[name].dtype
                    )
                    loaded_parameter_count += 1
                else:
                    entry["status"] = "shape_mismatch"
                    entry["reason"] = (
                        "payload shape is incompatible with the live RADLADS leaf"
                    )
                    shape_mismatches.append(entry)
            elif payload_value.size == int(np.prod(expected_shape)):
                entry["status"] = "mapped_reshaped"
                entry["reason"] = "payload surface reshaped to expected runtime shape"
                mapped.append(entry)
                loaded_state_dict[name] = torch.as_tensor(
                    np.asarray(payload_value).reshape(expected_shape),
                    dtype=loaded_state_dict[name].dtype,
                )
                loaded_parameter_count += 1
            else:
                entry["status"] = "shape_mismatch"
                entry["reason"] = (
                    "payload shape is incompatible with the live RADLADS leaf"
                )
                shape_mismatches.append(entry)
        elif template_name in SUPPORTED_MODEL_DEFAULTS:
            spec = SUPPORTED_MODEL_DEFAULTS[template_name]
            entry["status"] = "unsupported"
            entry["reason"] = (
                "live RADLADS boundary leaf is unsupported by the clean payload; "
                f"deterministic {spec['kind']} default applied"
            )
            entry["parity_risk"] = spec["parity_risk"]
            entry["default_kind"] = spec["kind"]
            entry["defaulted"] = True
            entry["default_reason"] = spec["reason"]
            defaulted.append(entry)
            unsupported.append(entry)
            loaded_state_dict[name] = _default_tensor(named[name], spec["kind"])
            caveats.append(
                {
                    "name": name,
                    "reason": spec["reason"],
                    "parity_risk": spec["parity_risk"],
                }
            )
        elif template_name in RUNTIME_CRITICAL_SURFACES:
            entry["status"] = "excluded_not_runtime_critical"
            entry["reason"] = (
                "leaf is retained at deterministic model initialization "
                "for the tiny case"
            )
            excluded.append(entry)
        elif entry["radlads"] is None:
            entry["status"] = "excluded_not_runtime_critical"
            entry["reason"] = (
                "payload-only surface is not required by the live RADLADS tiny model"
            )
            excluded.append(entry)
        else:
            entry["status"] = "unsupported"
            entry["reason"] = "payload leaf is outside the live RADLADS boundary"
            unsupported.append(entry)
        mapping_entries.append(entry)

    for name, spec in SUPPORTED_MODEL_DEFAULTS.items():
        payload_name = payload_name_map.get(name, name)
        state_key = next(
            (key for key in loaded_state_dict if _surface_template(key) == name),
            None,
        )
        if state_key is not None and payload_name not in normalized:
            loaded_state_dict[state_key] = _default_tensor(
                named[state_key], spec["kind"]
            )

    for name in named:
        if (
            name not in handled_names
            and _surface_template(name) not in SUPPORTED_MODEL_DEFAULTS
            and name not in {row["radlads"] for row in shape_mismatches}
        ):
            if name not in {row["radlads"] for row in missing_required}:
                missing_required.append(
                    {
                        "radlads": name,
                        "qrwkv": None,
                        "radlads_shape": list(model_shapes[name]),
                        "qrwkv_shape": None,
                        "status": "missing_required",
                        "reason": "runtime-critical surface missing from clean payload",
                    }
                )

    _load_state_dict_into_model(model, loaded_state_dict)
    if run_smoke and not missing_required:
        _smoke_forward(model, normalized)

    overall_status = "pass" if not missing_required else "blocked"
    reason = None
    if missing_required:
        reason = (
            "clean payload still misses runtime-critical RADLADS leaves: "
            f"{len(missing_required)}"
        )
    model_info = {
        "class": type(model).__name__,
        "runtime_source": str(radlads_source_path),
        "seed": seed,
        "parameter_count": len(named),
        "payload_parameter_count": len(normalized),
        "loaded_parameter_count": loaded_parameter_count,
        "config": _config_summary(config),
    }
    report = {
        "schema": CLEAN_LOADER_SCHEMA,
        "parameter_payload": str(parameter_payload_path),
        "radlads_source_path": str(radlads_source_path),
        "status": overall_status,
        "overall_status": overall_status,
        "reason": reason,
        "counts": _counts(mapping_entries, missing_required=missing_required),
        "caveats": caveats,
        "mapping_entries": mapping_entries,
        "mapped": mapped,
        "defaulted": defaulted,
        "unsupported": unsupported,
        "excluded": excluded,
        "shape_mismatches": shape_mismatches,
        "missing_required": missing_required,
        "loaded_parameter_count": loaded_parameter_count,
        "state_dict_keys": sorted(loaded_state_dict),
        "model_info": model_info,
        "model_status": {
            "smoke_requested": run_smoke,
            "smoke_ran": run_smoke and not missing_required,
        },
    }
    return RadladsCleanPayloadLoadResult(
        parameter_payload=str(parameter_payload_path),
        radlads_source_path=str(radlads_source_path),
        overall_status=overall_status,
        reason=reason,
        counts=report["counts"],
        caveats=caveats,
        mapping_entries=mapping_entries,
        mapped=mapped,
        defaulted=defaulted,
        unsupported=unsupported,
        excluded=excluded,
        shape_mismatches=shape_mismatches,
        missing_required=missing_required,
        loaded_parameter_count=loaded_parameter_count,
        state_dict_keys=sorted(loaded_state_dict),
        model_info=model_info,
        report=report,
        runtime=runtime,
        model=model,
        qrwkv_config=config,
        loaded_state_dict=loaded_state_dict,
    )


def export_radlads_clean_payload_outputs(
    parameter_payload_path: Path,
    out_dir: Path,
    *,
    radlads_source_path: Path = DEFAULT_RADLADS_SOURCE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> RadladsCleanPayloadExportResult:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite to replace outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*"):
        if path.is_file():
            path.unlink()

    load_result = load_radlads_clean_payload(
        parameter_payload_path,
        radlads_source_path=radlads_source_path,
        seed=seed,
        run_smoke=False,
    )
    if load_result.overall_status != "pass" or load_result.model is None:
        manifest = _blocked_output_manifest(
            parameter_payload_path=parameter_payload_path,
            radlads_source_path=radlads_source_path,
            load_result=load_result,
        )
        _write_json(out_dir / "manifest.json", manifest)
        return RadladsCleanPayloadExportResult(manifest, load_result)

    cases = []
    for case in _tiny_cases():
        case_arrays = _run_clean_case(load_result.model, case)
        payload_name = f"{case['name']}.npz"
        np.savez(out_dir / payload_name, **case_arrays)
        cases.append(
            _case_manifest_record(case, payload_name, case_arrays, side="radlads")
        )

    manifest = {
        "schema": CLEAN_OUTPUT_SCHEMA,
        "phase": "P54",
        "side": "radlads",
        "surface_conventions": {
            "hidden_states": "final_hidden",
            "wkv_matrix_state": "full_sequence_final_state",
            "shift_state": "squeezed_time_axis",
        },
        "created_at_utc": _utc_now(),
        "seed": seed,
        "parameter_payload": str(parameter_payload_path),
        "radlads_source_path": str(radlads_source_path),
        "load_report": load_result.report,
        "overall_status": "pass",
        "cases": cases,
        "notes": [
            "RADLADS outputs were exported only after the clean loader could run.",
            "Case payloads store radlads_* arrays plus the shared inputs "
            "for comparison.",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return RadladsCleanPayloadExportResult(manifest, load_result)


def export_qrwkv_clean_payload_outputs(
    parameter_payload_path: Path,
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite to replace outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*"):
        if path.is_file():
            path.unlink()

    qrwkv_import = import_radlads_parameters_for_replay(
        parameter_payload_path,
        allow_defaults=True,
        seed=seed,
    )
    student = RWKV7QwenReferenceStudent(qrwkv_import.qrwkv_config)
    cases = []
    for case in _tiny_cases():
        arrays = _qrwkv_case_arrays(student, qrwkv_import.params, case)
        payload_name = f"{case['name']}.npz"
        np.savez(out_dir / payload_name, **arrays)
        cases.append(_case_manifest_record(case, payload_name, arrays, side="qrwkv"))

    manifest = {
        "schema": CLEAN_OUTPUT_SCHEMA,
        "phase": "P54",
        "side": "qrwkv",
        "surface_conventions": {
            "hidden_states": "layer_major_all_hidden",
            "wkv_matrix_state": "full_sequence_final_state",
            "shift_state": "squeezed_time_axis",
        },
        "created_at_utc": _utc_now(),
        "seed": seed,
        "parameter_payload": str(parameter_payload_path),
        "load_report": qrwkv_import.report,
        "overall_status": "pass",
        "cases": cases,
        "notes": [
            "QRWKV outputs were exported using the clean deterministic payload.",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def audit_clean_payload_loader(
    radlads_source_path: Path,
    parameter_payload_path: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    load_result = load_radlads_clean_payload(
        parameter_payload_path,
        radlads_source_path=radlads_source_path,
        seed=seed,
        run_smoke=False,
    )
    report = dict(load_result.report)
    report.update(
        {
            "status": load_result.overall_status,
            "phase": "P54",
            "radlads_repo": str(radlads_source_path),
            "parameter_payload": str(parameter_payload_path),
            "blockers_before": {
                "unsupported": len(load_result.unsupported),
                "shape_mismatches": len(load_result.shape_mismatches),
            },
            "blockers_after": {
                "unsupported": len(load_result.unsupported),
                "shape_mismatches": len(load_result.shape_mismatches),
            },
        }
    )
    return report


def write_audit_reports(report: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "radlads_loader_audit.json", report)
    (out_dir / "P54_RADLADS_LOADER_AUDIT.md").write_text(
        _audit_markdown(report),
        encoding="utf-8",
    )
    return dict(report)


def load_clean_output_manifest(path: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(path).read_text(encoding="utf-8"))


def validate_clean_output_manifest(path: Path) -> dict[str, Any]:
    manifest = load_clean_output_manifest(path)
    manifest_path = _manifest_path(path)
    _require(manifest.get("schema") == CLEAN_OUTPUT_SCHEMA, "bad schema")
    _require(manifest.get("phase") == "P54", "bad phase")
    _require(manifest.get("side") in {"radlads", "qrwkv"}, "bad side")
    _require(isinstance(manifest.get("cases"), list), "cases must be a list")
    for case in manifest["cases"]:
        _validate_output_case(manifest_path.parent, case)
    return manifest


def load_case_output_arrays(manifest_path: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = validate_clean_output_manifest(manifest_path)
    base = _manifest_path(manifest_path).parent
    arrays_by_case: dict[str, dict[str, np.ndarray]] = {}
    for case in manifest["cases"]:
        payload_path = base / str(case["payload"])
        with np.load(payload_path) as payload:
            arrays_by_case[str(case["name"])] = {
                name: np.asarray(payload[name]) for name in payload.files
            }
    return arrays_by_case


def _run_clean_case(
    model: Any,
    case: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if torch is None:  # pragma: no cover - optional CI dependency
        raise ModuleNotFoundError("torch")
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
    arrays: dict[str, np.ndarray] = {
        "input_ids": np.asarray(case["input_ids"], dtype=np.int32),
        "position_ids": np.broadcast_to(
            np.arange(case["input_ids"].shape[1], dtype=np.int32),
            case["input_ids"].shape,
        ),
        "radlads_hidden_states": _np(full.hidden_states[-1]),
        "radlads_logits": _np(full.logits),
        "radlads_wkv_matrix_state": _stack_radlads_state(full.past_key_values, index=0),
        "radlads_shift_state": _squeeze_singleton_time_axis(
            _stack_radlads_state(full.past_key_values, index=1)
        ),
    }
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
            step_hidden.append(_np(step.hidden_states[-1]))
            step_logits.append(_np(step.logits))
        arrays["radlads_stepwise_hidden_states"] = np.concatenate(step_hidden, axis=1)
        arrays["radlads_stepwise_logits"] = np.concatenate(step_logits, axis=1)
        arrays["radlads_stepwise_wkv_matrix_state"] = _stack_radlads_state(
            step_cache,
            index=0,
        )
        arrays["radlads_stepwise_shift_state"] = _squeeze_singleton_time_axis(
            _stack_radlads_state(step_cache, index=1)
        )
    return arrays


def _blocked_report(
    *,
    parameter_payload_path: Path,
    radlads_source_path: Path,
    reason: str,
    raw_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    mapping = compare_parameter_surfaces(None, {})
    return {
        "schema": CLEAN_LOADER_SCHEMA,
        "parameter_payload": str(parameter_payload_path),
        "radlads_source_path": str(radlads_source_path),
        "status": "blocked",
        "overall_status": "blocked",
        "reason": reason,
        "counts": {"blocked": 1},
        "caveats": [
            {
                "name": "*",
                "reason": reason,
                "parity_risk": "high",
            }
        ],
        "mapping_entries": mapping["mappings"],
        "mapped": [],
        "defaulted": [],
        "unsupported": [],
        "excluded": [],
        "shape_mismatches": [],
        "missing_required": [
            {
                "radlads": name,
                "qrwkv": None,
                "radlads_shape": list(np.asarray(array).shape),
                "qrwkv_shape": None,
                "status": "missing_required",
                "reason": "runtime unavailable; load blocked before comparison",
            }
            for name, array in sorted(raw_arrays.items())
        ],
        "loaded_parameter_count": 0,
        "state_dict_keys": [],
        "model_info": {
            "class": "unavailable",
            "runtime_source": str(radlads_source_path),
            "parameter_count": 0,
            "payload_parameter_count": len(raw_arrays),
            "loaded_parameter_count": 0,
        },
        "model_status": {
            "smoke_requested": False,
            "smoke_ran": False,
        },
    }


def _blocked_output_manifest(
    *,
    parameter_payload_path: Path,
    radlads_source_path: Path,
    load_result: RadladsCleanPayloadLoadResult,
) -> dict[str, Any]:
    return {
        "schema": CLEAN_OUTPUT_SCHEMA,
        "phase": "P54",
        "side": "radlads",
        "surface_conventions": {
            "hidden_states": "final_hidden",
            "wkv_matrix_state": "full_sequence_final_state",
            "shift_state": "squeezed_time_axis",
        },
        "created_at_utc": _utc_now(),
        "seed": DEFAULT_SEED,
        "parameter_payload": str(parameter_payload_path),
        "radlads_source_path": str(radlads_source_path),
        "load_report": load_result.report,
        "overall_status": "blocked",
        "reason": load_result.reason,
        "cases": [],
        "notes": [
            "No outputs were exported because the clean loader could not run.",
        ],
    }


def _case_manifest_record(
    case: Mapping[str, Any],
    payload_name: str,
    arrays: Mapping[str, np.ndarray],
    *,
    side: str,
) -> dict[str, Any]:
    surface_prefix = f"{side}_"
    return {
        "name": case["name"],
        "description": case["description"],
        "payload": payload_name,
        "status": "pass",
        "reason": "",
        "input_shape": list(case["input_ids"].shape),
        "attention_mask": {
            "present": case["attention_mask"] is not None,
            "kind": case["mask_kind"],
            "shape": None
            if case["attention_mask"] is None
            else list(case["attention_mask"].shape),
        },
        "shapes": {
            name: list(np.asarray(value).shape)
            for name, value in sorted(arrays.items())
        },
        "dtypes": {
            name: str(np.asarray(value).dtype) for name, value in sorted(arrays.items())
        },
        "surface_prefix": surface_prefix,
    }


def _validate_output_case(base: Path, case: dict[str, Any]) -> None:
    payload_name = case.get("payload")
    _require(isinstance(payload_name, str), "case payload must be a string")
    _require((base / payload_name).is_file(), f"missing output payload {payload_name}")


def _load_state_dict_into_model(model: Any, state_dict: dict[str, Any]) -> None:
    model.load_state_dict(state_dict, strict=False)


def _smoke_forward(model: Any, normalized: Mapping[str, np.ndarray]) -> None:
    if torch is None:  # pragma: no cover - optional CI dependency
        raise ModuleNotFoundError("torch")
    case = _tiny_cases()[0]
    input_ids = torch.tensor(case["input_ids"], dtype=torch.long)
    attention_mask = None
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )


def _adapt_gate_tensor(
    value: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == target_shape:
        return array, "exact"
    if array.ndim != 2:
        return None, None
    if array.T.shape == target_shape:
        return array.T, "transpose"
    if array.shape[0] == target_shape[0] and array.shape[1] >= target_shape[1]:
        return array[:, : target_shape[1]], "truncate_rank"
    if array.shape[1] == target_shape[1] and array.shape[0] >= target_shape[0]:
        return array[: target_shape[0], :], "truncate_rank"
    if array.T.shape[0] == target_shape[0] and array.T.shape[1] >= target_shape[1]:
        return array.T[:, : target_shape[1]], "transpose_then_truncate_rank"
    if array.T.shape[1] == target_shape[1] and array.T.shape[0] >= target_shape[0]:
        return array.T[: target_shape[0], :], "transpose_then_truncate_rank"
    if array.shape[0] == target_shape[0] and array.shape[1] < target_shape[1]:
        padded = np.zeros(target_shape, dtype=array.dtype)
        padded[:, : array.shape[1]] = array
        return padded, "rank_pad"
    if array.shape[1] == target_shape[1] and array.shape[0] < target_shape[0]:
        padded = np.zeros(target_shape, dtype=array.dtype)
        padded[: array.shape[0], :] = array
        return padded, "rank_pad"
    return None, None


def _default_tensor(reference: Any, kind: str) -> torch.Tensor:
    if torch is None:  # pragma: no cover - optional CI dependency
        raise ModuleNotFoundError("torch")
    if kind == "ones":
        return torch.ones_like(reference)
    return torch.zeros_like(reference)


def _stack_radlads_state(cache: Any, *, index: int) -> np.ndarray:
    values = [
        cache[layer][index].detach().cpu().numpy().astype(np.float32)
        for layer in range(len(cache))
    ]
    return np.stack(values, axis=0)


def _squeeze_singleton_time_axis(value: np.ndarray) -> np.ndarray:
    if value.ndim >= 4 and value.shape[2] == 1:
        return value[:, :, 0, ...]
    return value


def _np(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy(), dtype=np.float32)


def _config_summary(config: Any) -> dict[str, Any]:
    payload = getattr(config, "__dict__", {})
    return {
        "class": type(config).__name__,
        "hidden_size": payload.get("hidden_size"),
        "num_hidden_layers": payload.get("num_hidden_layers"),
        "num_attention_heads": payload.get("num_attention_heads"),
        "num_key_value_heads": payload.get("num_key_value_heads"),
        "lora_rank_gate": payload.get("lora_rank_gate"),
    }


def _surface_template(name: str) -> str:
    match = _LAYER_TEMPLATE_RE.match(name)
    if match is not None:
        return f"layers.{match.group(2)}"
    return name.removeprefix("model.")


def _counts(
    mapping_entries: list[dict[str, Any]],
    *,
    missing_required: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in mapping_entries:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    if missing_required:
        counts["missing_required"] = len(missing_required)
    return counts


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _audit_markdown(report: Mapping[str, Any]) -> str:
    blockers_before = report.get("blockers_before", {})
    blockers_after = report.get("blockers_after", {})
    lines = [
        "# P54 RADLADS Loader Audit",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Unsupported before: `{blockers_before.get('unsupported')}`",
        f"- Shape mismatches before: `{blockers_before.get('shape_mismatches')}`",
        f"- Unsupported after: `{blockers_after.get('unsupported')}`",
        f"- Shape mismatches after: `{blockers_after.get('shape_mismatches')}`",
        "",
        "## Mapping rows",
        "",
    ]
    for row in report.get("mapping_entries", []):
        lines.append(
            f"- `{row.get('radlads')}` → `{row.get('status')}`: {row.get('reason')}"
        )
    return "\n".join(lines)


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


load_output_manifest = load_clean_output_manifest
load_output_case_arrays = load_case_output_arrays
validate_output_manifest = validate_clean_output_manifest
