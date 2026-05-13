from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qrwkv_xla.students import (  # noqa: F401
        RWKV7QwenReferenceConfig,
        RWKV7QwenReferenceStudent,
    )

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional CI dependency
    jax = None
    jnp = None

import numpy as np

from qrwkv_xla.parity.radlads_fixture_validation import (
    audit_parameter_payload,
    to_audit_report,
    validate_parameter_payload,
)
from qrwkv_xla.parity.radlads_parameter_mapping import (
    compare_parameter_surfaces,
    flatten_parameter_shapes,
    normalize_radlads_parameter_arrays,
    write_surface_comparison_reports,
)

NUMERICAL_FIXTURE_VERSION = 1
NUMERICAL_FIXTURE_SCHEMA = "radlads_tiny_numerical_parity.v1"
NUMERICAL_REPORT_SCHEMA = "radlads_tiny_numerical_parity_report.v1"
REQUIRED_NUMERICAL_CASE_NAMES = (
    "tiny_no_mask",
    "tiny_attention_mask",
    "tiny_prefix_or_left_padding",
    "tiny_stepwise_state",
    "tiny_all_radlads_math_enabled",
)
CASE_STATUSES = {
    "pass",
    "fail",
    "unsupported",
    "missing_source",
    "fail_known_difference",
}
REAL_RADLADS_FIXTURE_STATUSES = {
    "generated",
    "imported",
    "source_unavailable",
    "execution_failed",
}
DEFAULT_RADLADS_SOURCE = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")
LIVE_ENV_VARS = (
    "QRWKV_XLA_RUN_RADLADS_LIVE_FIXTURES",
    "QRWKV_RUN_RADLADS_LIVE",
)
PARAMETER_EXTREME_THRESHOLD = 1e6
INIT_POLICIES = {"radlads_source", "deterministic_finite"}
_TRANSPOSE_RAW_LAYER_SUFFIXES = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}
_SQUEEZE_RAW_LAYER_SUFFIXES = {
    "self_attn.w0",
    "self_attn.a0",
    "self_attn.v0",
    "self_attn.k_k",
    "self_attn.k_a",
}


def load_numerical_manifest(path: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(path).read_text(encoding="utf-8"))


def validate_numerical_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    manifest = load_numerical_manifest(manifest_path)
    _require(
        manifest.get("schema_version") == NUMERICAL_FIXTURE_VERSION,
        "bad schema_version",
    )
    _require(manifest.get("schema") == NUMERICAL_FIXTURE_SCHEMA, "bad schema")
    _require(manifest.get("phase") == "P49", "bad phase")
    _require(manifest.get("source") == "radlads", "bad source")
    _require(
        manifest.get("real_radlads_fixture_status") in REAL_RADLADS_FIXTURE_STATUSES,
        "bad real_radlads_fixture_status",
    )
    _require(isinstance(manifest.get("cases"), list), "cases must be a list")
    case_names = {case.get("name") for case in manifest["cases"]}
    _require(
        set(REQUIRED_NUMERICAL_CASE_NAMES).issubset(case_names),
        "missing required P49 cases",
    )
    parameter_payload = manifest.get("parameter_payload")
    if parameter_payload is not None:
        _require(
            (_manifest_path(path).parent / parameter_payload).is_file(),
            f"missing parameter payload {parameter_payload}",
        )
    for case in manifest["cases"]:
        _validate_numerical_case(manifest_path.parent, case)
    return manifest


def load_numerical_case_arrays(
    manifest_path: Path, case: dict[str, Any]
) -> dict[str, np.ndarray]:
    payload_path = _manifest_path(manifest_path).parent / case["payload"]
    with np.load(payload_path) as payload:
        return {name: payload[name] for name in payload.files}


def load_parameter_arrays(manifest_path: Path) -> dict[str, np.ndarray]:
    manifest = load_numerical_manifest(manifest_path)
    payload_name = manifest.get("parameter_payload")
    if payload_name is None:
        return {}
    payload_path = _manifest_path(manifest_path).parent / payload_name
    with np.load(payload_path) as payload:
        return {name: payload[name] for name in payload.files}


def import_numerical_fixture_directory(
    source: Path,
    out: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = validate_numerical_manifest(source)
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
    if manifest.get("parameter_payload") is not None:
        shutil.copy2(
            source_base / str(manifest["parameter_payload"]),
            out / str(manifest["parameter_payload"]),
        )
    copied = validate_numerical_manifest(out)
    copied["real_radlads_fixture_status"] = "imported"
    (out / "manifest.json").write_text(
        json.dumps(copied, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_numerical_manifest(out)


def write_current_behavior_numerical_fixtures(
    out: Path,
    *,
    seed: int = 4949,
    overwrite: bool = False,
    radlads_source_path: Path = DEFAULT_RADLADS_SOURCE,
    generation_script: str = "scripts/import_radlads_tiny_numerical_fixtures.py",
    init_policy: str = "radlads_source",
) -> dict[str, Any]:
    _validate_init_policy(init_policy)
    _prepare_out_dir(out, overwrite=overwrite)

    from qrwkv_xla.students import RWKV7QwenReferenceStudent

    cases = _tiny_cases()
    manifest_cases = []
    for case in cases:
        config = _tiny_qrwkv_config(all_radlads_math=case["all_radlads_math"])
        student = RWKV7QwenReferenceStudent(config)
        params = student.init_params(
            jax.random.PRNGKey(seed + int(case["seed_offset"]))
        )
        arrays = _qrwkv_case_arrays(student, params, case)
        payload = f"{case['name']}.npz"
        np.savez(out / payload, **arrays)
        manifest_cases.append(
            _case_manifest(
                case,
                arrays,
                status="missing_source",
                reason=(
                    "No live RADLADS numerical arrays are present. Payload records "
                    "QRWKV-XLA current behavior only and is not a RADLADS parity "
                    "result."
                ),
            )
        )

    manifest = _base_manifest(
        seed=seed,
        radlads_source_path=radlads_source_path,
        generation_script=generation_script,
        real_status="source_unavailable",
    )
    manifest["cases"] = manifest_cases
    manifest["parameter_mapping"] = compare_parameter_surfaces(
        None,
        flatten_parameter_shapes(
            RWKV7QwenReferenceStudent(
                _tiny_qrwkv_config(all_radlads_math=True)
            ).init_params(jax.random.PRNGKey(seed))
        ),
    )
    manifest["claim"] = (
        "P49 tiny numerical fixture schema and reporting. Default payloads are "
        "QRWKV-XLA current behavior only, not RADLADS outputs."
    )
    if init_policy == "deterministic_finite":
        parameter_arrays = _deterministic_finite_radlads_parameter_arrays(seed=seed)
        np.savez(out / "radlads_parameters.npz", **parameter_arrays)
        _add_parameter_payload_metadata(
            manifest,
            parameter_arrays,
            init_policy=init_policy,
            source="deterministic_finite",
        )
        manifest["parameter_mapping"] = _parameter_mapping_for_arrays(
            parameter_arrays,
            seed=seed,
        )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_numerical_manifest(out)


def generate_radlads_tiny_numerical_fixtures(
    out: Path,
    *,
    seed: int = 4949,
    overwrite: bool = False,
    radlads_source_path: Path = DEFAULT_RADLADS_SOURCE,
    live_env_var: str | None = None,
    init_policy: str = "radlads_source",
) -> dict[str, Any]:
    _validate_init_policy(init_policy)
    if init_policy == "deterministic_finite":
        return write_current_behavior_numerical_fixtures(
            out,
            seed=seed,
            overwrite=overwrite,
            radlads_source_path=radlads_source_path,
            generation_script="scripts/generate_radlads_tiny_numerical_fixtures.py",
            init_policy=init_policy,
        )
    if not _live_requested(live_env_var):
        return write_current_behavior_numerical_fixtures(
            out,
            seed=seed,
            overwrite=overwrite,
            radlads_source_path=radlads_source_path,
            generation_script="scripts/generate_radlads_tiny_numerical_fixtures.py",
            init_policy=init_policy,
        )
    if not radlads_source_path.exists():
        return write_current_behavior_numerical_fixtures(
            out,
            seed=seed,
            overwrite=overwrite,
            radlads_source_path=radlads_source_path,
            generation_script="scripts/generate_radlads_tiny_numerical_fixtures.py",
            init_policy=init_policy,
        )
    try:
        return _generate_live_radlads_fixtures(
            out,
            seed=seed,
            overwrite=overwrite,
            radlads_source_path=radlads_source_path,
        )
    except Exception as exc:
        manifest = write_current_behavior_numerical_fixtures(
            out,
            seed=seed,
            overwrite=overwrite,
            radlads_source_path=radlads_source_path,
            generation_script="scripts/generate_radlads_tiny_numerical_fixtures.py",
            init_policy=init_policy,
        )
        manifest["real_radlads_fixture_status"] = "execution_failed"
        manifest["live_generation_error"] = f"{type(exc).__name__}: {exc}"
        for case in manifest["cases"]:
            case["generation_status"] = "generation_failed"
            case["generation_failed_reason"] = manifest["live_generation_error"]
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return validate_numerical_manifest(out)


def compare_numerical_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = validate_numerical_manifest(manifest_path)
    parameter_mapping = _parameter_mapping_for_manifest(manifest_path)
    counts = {"pass": 0, "fail": 0, "unsupported": 0, "missing_source": 0}
    cases: list[dict[str, Any]] = []
    known_difference = False
    for case in manifest["cases"]:
        arrays = load_numerical_case_arrays(manifest_path, case)
        status = str(case.get("status", "missing_source"))
        comparisons = case.get("comparisons", [])
        if status == "missing_source":
            counts["missing_source"] += 1
            cases.append(
                {
                    "name": case["name"],
                    "status": "missing_source",
                    "reason": case.get(
                        "missing_source_reason",
                        case.get("reason", "missing RADLADS arrays"),
                    ),
                    "comparisons": [],
                }
            )
            continue
        if status == "unsupported" or not comparisons:
            counts["unsupported"] += 1
            cases.append(
                {
                    "name": case["name"],
                    "status": "unsupported",
                    "reason": case.get(
                        "unsupported_reason",
                        "no supported QRWKV-XLA value comparison yet",
                    ),
                    "comparisons": [],
                }
            )
            continue

        comparison_results = [_compare_arrays(arrays, spec) for spec in comparisons]
        failed = any(item["status"] == "fail" for item in comparison_results)
        case_known = status == "fail_known_difference" or any(
            item.get("known_difference") for item in comparison_results
        )
        known_difference = known_difference or case_known
        final_status = (
            "fail_known_difference"
            if failed and case_known
            else ("fail" if failed else "pass")
        )
        counts["fail" if final_status != "pass" else "pass"] += 1
        cases.append(
            {
                "name": case["name"],
                "status": final_status,
                "comparisons": comparison_results,
            }
        )

    if counts["fail"]:
        overall = "pass_with_known_differences" if known_difference else "fail"
    elif counts["pass"] and not counts["missing_source"] and not counts["unsupported"]:
        overall = "pass"
    elif manifest["real_radlads_fixture_status"] in {"generated", "imported"}:
        overall = "fail"
    else:
        overall = "source_unavailable"
    return {
        "schema": NUMERICAL_REPORT_SCHEMA,
        "manifest": str(_manifest_path(manifest_path)),
        "phase": "P49",
        "overall_status": overall,
        "real_radlads_fixture_status": manifest["real_radlads_fixture_status"],
        "counts": counts,
        "cases": cases,
        "parameter_mapping": parameter_mapping,
    }


def write_numerical_comparison_reports(
    manifest_path: Path, out_dir: Path
) -> dict[str, Any]:
    report = compare_numerical_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "numerical_parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P49_RESULTS.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    write_surface_comparison_reports(report.get("parameter_mapping", {}), out_dir)
    return report


def hash_numerical_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _qrwkv_case_arrays(
    student: RWKV7QwenReferenceStudent,
    params: dict[str, object],
    case: dict[str, Any],
) -> dict[str, np.ndarray]:
    input_ids = case["input_ids"]
    attention_mask = case["attention_mask"]
    output, state = student.apply_with_state(
        params,
        jnp.asarray(input_ids),
        attention_mask=None if attention_mask is None else jnp.asarray(attention_mask),
    )
    arrays = {
        "input_ids": input_ids,
        "position_ids": np.broadcast_to(
            np.arange(input_ids.shape[1], dtype=np.int32),
            input_ids.shape,
        ),
        "qrwkv_hidden_states": _np(output.hidden_states),
        "qrwkv_logits": _np(output.logits),
        "qrwkv_block_outputs": _np(output.mixer_outputs),
        "qrwkv_wkv_matrix_state": _np(state.wkv_matrix_state),
        "qrwkv_shift_state": _np(state.shift_state),
        "qrwkv_next_position": np.asarray(state.next_position, dtype=np.int32),
    }
    if attention_mask is not None:
        arrays["attention_mask"] = attention_mask
    if case["name"] == "tiny_stepwise_state":
        step_state = student.init_state(input_ids.shape[0])
        step_hidden = []
        step_logits = []
        for index in range(input_ids.shape[1]):
            step_output, step_state = student.step(
                params,
                jnp.asarray(input_ids[:, index : index + 1]),
                step_state,
                attention_mask=(
                    None
                    if attention_mask is None
                    else jnp.asarray(attention_mask[:, index : index + 1])
                ),
            )
            step_hidden.append(_np(step_output.hidden_states))
            step_logits.append(_np(step_output.logits))
        arrays["qrwkv_stepwise_hidden_states"] = np.concatenate(step_hidden, axis=2)
        arrays["qrwkv_stepwise_logits"] = np.concatenate(step_logits, axis=1)
        arrays["qrwkv_stepwise_wkv_matrix_state"] = _np(step_state.wkv_matrix_state)
        arrays["qrwkv_stepwise_shift_state"] = _np(step_state.shift_state)
        arrays["qrwkv_stepwise_next_position"] = np.asarray(
            step_state.next_position, dtype=np.int32
        )
    return arrays


def _generate_live_radlads_fixtures(
    out: Path,
    *,
    seed: int,
    overwrite: bool,
    radlads_source_path: Path,
) -> dict[str, Any]:
    _prepare_out_dir(out, overwrite=overwrite)
    runtime = _load_radlads_runtime(radlads_source_path)
    from qrwkv_xla.students import RWKV7QwenReferenceStudent

    qrwkv_shapes = flatten_parameter_shapes(
        RWKV7QwenReferenceStudent(
            _tiny_qrwkv_config(all_radlads_math=True)
        ).init_params(jax.random.PRNGKey(seed))
    )

    parameter_arrays = _radlads_named_parameter_arrays(runtime, seed=seed)
    np.savez(out / "radlads_parameters.npz", **parameter_arrays)
    parameter_mapping = compare_parameter_surfaces(
        flatten_parameter_shapes(normalize_radlads_parameter_arrays(parameter_arrays)),
        qrwkv_shapes,
    )

    manifest_cases = []
    for case in _tiny_cases():
        arrays = _radlads_case_arrays(
            runtime,
            case=case,
            seed=seed + int(case["seed_offset"]),
        )
        payload = f"{case['name']}.npz"
        np.savez(out / payload, **arrays)
        manifest_cases.append(
            _case_manifest(
                case,
                arrays,
                status="unsupported",
                reason=(
                    "Real RADLADS arrays were generated, but QRWKV-XLA output/state "
                    "value replay remains unsupported until parameter-value import is "
                    "completed for the remaining runtime-critical surfaces."
                ),
                comparisons=[],
            )
        )

    manifest = _base_manifest(
        seed=seed,
        radlads_source_path=radlads_source_path,
        generation_script="scripts/generate_radlads_tiny_numerical_fixtures.py",
        real_status="generated",
    )
    manifest["cases"] = manifest_cases
    _add_parameter_payload_metadata(
        manifest,
        parameter_arrays,
        init_policy="radlads_source",
        source="radlads_source",
    )
    manifest["parameter_mapping"] = parameter_mapping
    manifest["claim"] = (
        "P49 real tiny RADLADS source fixtures generated locally via patched CPU "
        "fallback kernels. Parameter-surface comparison is live; QRWKV-XLA "
        "output/state value replay remains explicitly unsupported for now."
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_numerical_manifest(out)


def _base_manifest(
    *,
    seed: int,
    radlads_source_path: Path,
    generation_script: str,
    real_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": NUMERICAL_FIXTURE_VERSION,
        "schema": NUMERICAL_FIXTURE_SCHEMA,
        "phase": "P49",
        "source": "radlads",
        "radlads_commit": _git_head(radlads_source_path),
        "radlads_source_path": str(radlads_source_path),
        "generation_script": generation_script,
        "created_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dtype_policy": {
            "default": "float32",
            "float32": {"atol": 1e-5, "rtol": 1e-5},
            "bfloat16": {"atol": 5e-2, "rtol": 5e-2},
        },
        "seed": seed,
        "backend": "rwkv7_qwen_reference",
        "required_cases": list(REQUIRED_NUMERICAL_CASE_NAMES),
        "real_radlads_fixture_status": real_status,
        "cases": [],
    }


def _validate_init_policy(init_policy: str) -> None:
    if init_policy not in INIT_POLICIES:
        raise ValueError(
            f"unsupported init_policy={init_policy!r}; "
            f"expected one of {sorted(INIT_POLICIES)}"
        )


def _add_parameter_payload_metadata(
    manifest: dict[str, Any],
    parameter_arrays: dict[str, np.ndarray],
    *,
    init_policy: str,
    source: str,
) -> None:
    audit_results = audit_parameter_payload(
        parameter_arrays,
        stage="saved_npz",
        extreme_threshold=PARAMETER_EXTREME_THRESHOLD,
    )
    is_valid, blocking = validate_parameter_payload(audit_results)
    audit_report = to_audit_report(audit_results)
    manifest["parameter_payload"] = "radlads_parameters.npz"
    manifest["parameter_payload_sha256"] = hash_numerical_arrays(parameter_arrays)
    manifest["parameter_payload_init_policy"] = init_policy
    manifest["parameter_payload_source"] = source
    manifest["parameter_payload_validation"] = {
        "status": "clean" if is_valid else "blocked",
        "deterministic": init_policy == "deterministic_finite",
        "finite": all(
            result.nan_count == 0
            and result.posinf_count == 0
            and result.neginf_count == 0
            for result in audit_results
        ),
        "extreme_threshold": PARAMETER_EXTREME_THRESHOLD,
        "blocking_count": len(blocking),
        "summary": audit_report["summary"],
    }


def _parameter_mapping_for_arrays(
    parameter_arrays: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, Any]:
    return compare_parameter_surfaces(
        flatten_parameter_shapes(normalize_radlads_parameter_arrays(parameter_arrays)),
        _deterministic_replay_parameter_shapes(seed=seed),
    )


def _deterministic_finite_radlads_parameter_arrays(
    *, seed: int
) -> dict[str, np.ndarray]:
    qrwkv_shapes = _deterministic_replay_parameter_shapes(seed=seed)
    qrwkv_shapes.update(
        {
            "layers.self_attn.q_proj.bias": (
                qrwkv_shapes["layers.self_attn.q_proj.weight"][0],
                qrwkv_shapes["layers.self_attn.q_proj.weight"][2],
            ),
            "layers.self_attn.k_proj.bias": (
                qrwkv_shapes["layers.self_attn.k_proj.weight"][0],
                qrwkv_shapes["layers.self_attn.k_proj.weight"][2],
            ),
            "layers.self_attn.v_proj.bias": (
                qrwkv_shapes["layers.self_attn.v_proj.weight"][0],
                qrwkv_shapes["layers.self_attn.v_proj.weight"][2],
            ),
        }
    )
    normalized = {
        name: _deterministic_finite_array(name, shape, seed=seed)
        for name, shape in sorted(qrwkv_shapes.items())
        if name != "lm_head.bias"
    }
    return _radlads_raw_arrays_from_normalized(normalized)


def _deterministic_replay_parameter_shapes(
    *,
    seed: int,
) -> dict[str, tuple[int, ...]]:
    config = replace(
        _tiny_qrwkv_config(all_radlads_math=True),
        radlads_replay_mode=True,
        attention_qkv_bias=True,
        radlads_low_rank_gate=True,
    )
    from qrwkv_xla.students import RWKV7QwenReferenceStudent

    return flatten_parameter_shapes(
        RWKV7QwenReferenceStudent(config).init_params(jax.random.PRNGKey(seed))
    )


def _deterministic_finite_array(
    name: str, shape: tuple[int, ...], *, seed: int
) -> np.ndarray:
    size = int(np.prod(shape))
    if size == 0:
        return np.zeros(shape, dtype=np.float32)
    name_offset = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    values = np.arange(size, dtype=np.float32)
    values = ((values + seed + name_offset) % 997) / 9970.0
    return (values.reshape(shape) - 0.05).astype(np.float32)


def _radlads_raw_arrays_from_normalized(
    normalized: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    for name, array in sorted(normalized.items()):
        value = np.asarray(array, dtype=np.float32)
        if name == "token_embedding.weight":
            raw["model.embed_tokens.weight"] = value
            continue
        if name == "final_layernorm.weight":
            raw["model.norm.weight"] = value
            continue
        if name == "lm_head.weight":
            raw["lm_head.weight"] = value.T
            continue
        if not name.startswith("layers."):
            continue
        suffix = name.removeprefix("layers.")
        for layer_index, layer_value in enumerate(value):
            raw_value = np.asarray(layer_value, dtype=np.float32)
            if suffix in _TRANSPOSE_RAW_LAYER_SUFFIXES:
                raw_value = raw_value.T
            if suffix in _SQUEEZE_RAW_LAYER_SUFFIXES:
                raw_value = raw_value.reshape((1, 1, *raw_value.shape))
            raw[f"model.layers.{layer_index}.{suffix}"] = raw_value
    return raw


def _case_manifest(
    case: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    status: str,
    reason: str,
    comparisons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": case["name"],
        "description": case["description"],
        "surface_names": case["surfaces"],
        "status": status,
        "missing_source_reason": reason if status == "missing_source" else "",
        "unsupported_reason": reason if status == "unsupported" else "",
        "generation_status": (
            "generation_failed" if status == "missing_source" else "generated"
        ),
        "generation_failed_reason": reason if status == "missing_source" else "",
        "payload": f"{case['name']}.npz",
        "payload_sha256": hash_numerical_arrays(arrays),
        "input_shape": list(arrays["input_ids"].shape),
        "attention_mask": {
            "present": "attention_mask" in arrays,
            "kind": case["mask_kind"],
            "shape": list(arrays["attention_mask"].shape)
            if "attention_mask" in arrays
            else None,
        },
        "shapes": {
            name: list(np.asarray(value).shape)
            for name, value in sorted(arrays.items())
        },
        "dtypes": {
            name: str(np.asarray(value).dtype) for name, value in sorted(arrays.items())
        },
        "comparisons": comparisons or [],
    }


def _tiny_qrwkv_config(*, all_radlads_math: bool):
    from qrwkv_xla.students import RWKV7QwenReferenceConfig

    return RWKV7QwenReferenceConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        emit_logits=True,
        emit_mixer_outputs=True,
        radlads_compatible_math=all_radlads_math,
        radlads_attention_group_norm=all_radlads_math,
        radlads_balance_state=all_radlads_math,
        lora_rank_decay=4,
        lora_rank_iclr=4,
        lora_rank_value_residual_mix=4,
    )


def _tiny_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "tiny_no_mask",
            "description": "Tiny deterministic no-mask full-sequence case.",
            "input_ids": np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32),
            "attention_mask": None,
            "mask_kind": "none",
            "all_radlads_math": False,
            "seed_offset": 0,
            "surfaces": ["parameters", "hidden", "state", "logits"],
        },
        {
            "name": "tiny_attention_mask",
            "description": "Tiny deterministic interior attention-mask case.",
            "input_ids": np.array([[5, 6, 7, 8], [8, 7, 6, 5]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=np.int32),
            "mask_kind": "attention_mask",
            "all_radlads_math": False,
            "seed_offset": 10,
            "surfaces": ["parameters", "hidden", "state", "logits"],
        },
        {
            "name": "tiny_prefix_or_left_padding",
            "description": "Tiny deterministic prefix/left-padding case.",
            "input_ids": np.array([[0, 0, 9, 10], [0, 11, 12, 13]], dtype=np.int32),
            "attention_mask": np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int32),
            "mask_kind": "prefix_or_left_padding",
            "all_radlads_math": False,
            "seed_offset": 20,
            "surfaces": ["parameters", "hidden", "state", "logits"],
        },
        {
            "name": "tiny_stepwise_state",
            "description": "Tiny full-vs-stepwise recurrent state behavior case.",
            "input_ids": np.array([[14, 15, 16, 17], [17, 16, 15, 14]], dtype=np.int32),
            "attention_mask": None,
            "mask_kind": "none",
            "all_radlads_math": False,
            "seed_offset": 30,
            "surfaces": ["parameters", "hidden", "state", "stepwise_state", "logits"],
        },
        {
            "name": "tiny_all_radlads_math_enabled",
            "description": (
                "Tiny case with all explicit P48 RADLADS math flags enabled."
            ),
            "input_ids": np.array([[18, 19, 20, 21], [21, 20, 19, 18]], dtype=np.int32),
            "attention_mask": None,
            "mask_kind": "none",
            "all_radlads_math": True,
            "seed_offset": 40,
            "surfaces": ["parameters", "hidden", "state", "logits", "p48_math_flags"],
        },
    ]


def _validate_numerical_case(base: Path, case: dict[str, Any]) -> None:
    _require(case.get("name") in REQUIRED_NUMERICAL_CASE_NAMES, "unknown case name")
    _require(
        case.get("status") in CASE_STATUSES,
        f"bad case status for {case.get('name')}",
    )
    payload_name = case.get("payload")
    _require(isinstance(payload_name, str), "case payload must be a string")
    _require((base / payload_name).is_file(), f"missing payload {payload_name}")
    arrays = load_numerical_case_arrays(base / "manifest.json", case)
    _require("input_ids" in arrays, f"{case['name']} missing input_ids")
    _require(arrays["input_ids"].ndim == 2, f"{case['name']} input_ids must be [B,T]")
    if case.get("attention_mask", {}).get("present"):
        _require("attention_mask" in arrays, f"{case['name']} missing attention_mask")
        _require(
            arrays["attention_mask"].shape == arrays["input_ids"].shape,
            "attention_mask shape mismatch",
        )
    if case.get("payload_sha256") is not None:
        _require(
            case["payload_sha256"] == hash_numerical_arrays(arrays),
            f"{case['name']} bad payload hash",
        )
    for spec in case.get("comparisons", []):
        left = spec.get("left")
        right = spec.get("right")
        _require(left in arrays, f"{case['name']} missing array {left}")
        _require(right in arrays, f"{case['name']} missing array {right}")
        _require(
            arrays[left].shape == arrays[right].shape,
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
    diff = np.abs(left.astype(np.float32) - right.astype(np.float32))
    denom = np.maximum(np.abs(right.astype(np.float32)), 1e-12)
    return {
        "name": spec.get("name", left_name),
        "status": "pass" if np.allclose(left, right, atol=atol, rtol=rtol) else "fail",
        "left": left_name,
        "right": right_name,
        "shape": list(left.shape),
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "max_rel": float(np.max(diff / denom)) if diff.size else 0.0,
        "atol": atol,
        "rtol": rtol,
        "known_difference": bool(spec.get("known_difference", False)),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P49 RADLADS Tiny Numerical Parity Report",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Real RADLADS fixture status: `{report['real_radlads_fixture_status']}`",
        f"- Pass: {report['counts']['pass']}",
        f"- Fail: {report['counts']['fail']}",
        f"- Unsupported: {report['counts']['unsupported']}",
        f"- Missing source: {report['counts']['missing_source']}",
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
    lines.extend(
        [
            "",
            "## Parameter Surface Summary",
            "",
            f"- Overall: `"
            f"{report['parameter_mapping'].get('overall_status', 'unknown')}`",
            f"- Counts: `{report['parameter_mapping'].get('counts', {})}`",
            "",
            "P49 proves tiny numerical fixture generation and surface comparison only. "
            "It does not prove full checkpoint compatibility, Pallas kernels, "
            "production throughput, or model quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _parameter_mapping_for_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_numerical_manifest(manifest_path)
    payload_arrays = load_parameter_arrays(manifest_path)
    if not payload_arrays:
        return manifest.get("parameter_mapping") or compare_parameter_surfaces(None, {})
    from qrwkv_xla.students import RWKV7QwenReferenceStudent

    qrwkv_shapes = flatten_parameter_shapes(
        RWKV7QwenReferenceStudent(
            _tiny_qrwkv_config(all_radlads_math=True)
        ).init_params(jax.random.PRNGKey(int(manifest.get("seed", 4949))))
    )
    return compare_parameter_surfaces(
        flatten_parameter_shapes(normalize_radlads_parameter_arrays(payload_arrays)),
        qrwkv_shapes,
    )


def _radlads_named_parameter_arrays(
    runtime: dict[str, Any], *, seed: int
) -> dict[str, np.ndarray]:
    _, model = _build_radlads_model(runtime, seed=seed, all_math=True)
    return {
        name: parameter.detach().cpu().numpy().astype(np.float32)
        for name, parameter in model.named_parameters()
    }


def _radlads_case_arrays(
    runtime: dict[str, Any],
    *,
    case: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    torch = runtime["torch"]
    _, model = _build_radlads_model(
        runtime,
        seed=seed,
        all_math=case["all_radlads_math"],
    )
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
        "input_ids": case["input_ids"],
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
    if attention_mask is not None:
        arrays["attention_mask"] = case["attention_mask"]
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
        arrays["radlads_stepwise_shift_state"] = _stack_radlads_state(
            step_cache,
            index=1,
        )
    return arrays


def _load_radlads_runtime(radlads_source_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(radlads_source_path))
    import rwkv7qwen2.modeling_rwkv7qwen2 as modeling
    import torch
    from rwkv7qwen2.configuration_rwkv7qwen2 import RWKV7Qwen2Config
    from rwkv7qwen2.modeling_rwkv7qwen2 import RWKV7Qwen2ForCausalLM

    class DummyRotary(torch.nn.Module):
        def __init__(self, config=None, device=None):
            super().__init__()

        def forward(self, x, position_ids):
            return None

    class SimpleState:
        def __init__(self):
            self._seen_tokens = 0
            self.layer_kv_states: list[torch.Tensor | None] = []
            self.layer_shift_states: list[torch.Tensor | None] = []

        def __getitem__(self, layer_idx: int):
            return (
                self.layer_kv_states[layer_idx],
                self.layer_shift_states[layer_idx],
            )

        def __len__(self) -> int:
            return len(self.layer_kv_states)

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self._seen_tokens

        def update(
            self,
            kv_state: torch.Tensor,
            shift_state: torch.Tensor,
            q_len: int,
            layer_idx: int,
        ) -> None:
            while len(self.layer_kv_states) <= layer_idx:
                self.layer_kv_states.append(None)
                self.layer_shift_states.append(None)
            self.layer_kv_states[layer_idx] = kv_state.detach().clone()
            self.layer_shift_states[layer_idx] = shift_state.detach().clone()
            self._seen_tokens += q_len

    def simple_rwkv7(
        r: torch.Tensor,
        log_w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neg_kk: torch.Tensor,
        kka: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ):
        batch, steps, heads, head_dim = r.shape
        state = initial_state
        if state is None:
            state = torch.zeros(
                batch,
                heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=r.device,
            )
        outputs = torch.zeros(
            batch,
            steps,
            heads,
            head_dim,
            dtype=v.dtype,
            device=v.device,
        )
        decay = log_w.exp()
        for index in range(steps):
            r_t = r[:, index]
            w_t = decay[:, index]
            k_t = k[:, index]
            v_t = v[:, index]
            kk_t = neg_kk[:, index]
            a_t = kka[:, index]
            vk = v_t.view(batch, heads, head_dim, 1) @ k_t.view(
                batch,
                heads,
                1,
                head_dim,
            )
            ab = kk_t.view(batch, heads, head_dim, 1) @ a_t.view(
                batch,
                heads,
                1,
                head_dim,
            )
            state = (
                state * w_t.view(batch, heads, 1, head_dim)
                + state @ ab.float()
                + vk.float()
            )
            outputs[:, index] = (
                state.to(dtype=outputs.dtype) @ r_t.view(batch, heads, head_dim, 1)
            ).view(batch, heads, head_dim)
        return outputs, state

    modeling.Qwen2RotaryEmbedding = DummyRotary
    modeling.RWKV7State = SimpleState
    modeling.chunk_rwkv7 = simple_rwkv7
    modeling.fused_recurrent_rwkv7 = simple_rwkv7
    return {
        "torch": torch,
        "config_cls": RWKV7Qwen2Config,
        "model_cls": RWKV7Qwen2ForCausalLM,
    }


def _build_radlads_model(runtime: dict[str, Any], *, seed: int, all_math: bool):
    torch = runtime["torch"]
    config_cls = runtime["config_cls"]
    model_cls = runtime["model_cls"]
    torch.manual_seed(seed)
    config = config_cls(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        lora_rank_decay=4,
        lora_rank_iclr=4,
        lora_rank_value_residual_mix=4,
        lora_rank_gate=4,
        max_position_embeddings=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
        use_rope=False,
        balance_state=all_math,
        groupnorm_att=all_math,
        gate_rank_type=2,
    )
    model = model_cls(config)
    model.eval()
    return config, model


def _stack_radlads_state(cache: Any, *, index: int) -> np.ndarray:
    values = [
        cache[layer][index].detach().cpu().numpy().astype(np.float32)
        for layer in range(len(cache))
    ]
    return np.stack(values, axis=0)


def _prepare_out_dir(out: Path, *, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to replace fixtures")
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*"):
        if path.is_file():
            path.unlink()


def _live_requested(live_env_var: str | None) -> bool:
    if live_env_var is not None:
        return os.environ.get(live_env_var) == "1"
    return any(os.environ.get(name) == "1" for name in LIVE_ENV_VARS)


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


def _np(value: jax.Array | None) -> np.ndarray:
    if value is None:
        raise RuntimeError("expected emitted QRWKV array")
    return np.asarray(jax.device_get(value), dtype=np.float32)


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
