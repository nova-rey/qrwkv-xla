from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCHEMA_VERSION = "0.1"
FIXTURE_SET = "tiny_wkv7_correctness"
FIXTURE_VERSION = 1
FIXTURE_SCHEMA = "qrwkv_xla.wkv7_correctness.v1"
DEFAULT_SEED = 4307
DEFAULT_CASES = (
    "tiny_b1_t4_h1_d4_no_mask",
    "tiny_b2_t5_h2_d4_no_mask",
    "tiny_b1_t6_h2_d4_with_attention_mask",
    "tiny_prefix_padding_or_reset_case",
    "tiny_stateful_step_vs_full_scan",
    "tiny_extreme_but_finite_decay_values",
)


@dataclass(frozen=True)
class WKV7Tolerance:
    dtype: str
    atol: float
    rtol: float


TOLERANCES = {
    "float32": WKV7Tolerance(dtype="float32", atol=1e-5, rtol=1e-5),
    "bfloat16": WKV7Tolerance(dtype="bfloat16", atol=5e-2, rtol=5e-2),
}


def generate_wkv7_fixture_bundle(
    out: Path,
    *,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to replace fixtures")
    if out.exists() and overwrite:
        shutil.rmtree(out)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, Any]] = []
    for case in _build_cases(seed):
        case_dir = cases_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        inputs = case["inputs"]
        expected = _expected_for_case(inputs)
        np.savez(case_dir / "inputs.npz", **inputs)
        np.savez(case_dir / "expected.npz", **expected)
        equivalence = _equivalence_metrics(
            expected["output"],
            expected["stepwise_output"],
            expected["next_state"],
            expected["stepwise_next_state"],
            tolerance=TOLERANCES["float32"],
        )
        manifest_cases.append(
            {
                "case_id": case["case_id"],
                "description": case["description"],
                "shapes": {
                    "batch": int(inputs["r"].shape[0]),
                    "time": int(inputs["r"].shape[1]),
                    "num_heads": int(inputs["r"].shape[2]),
                    "head_size": int(inputs["r"].shape[3]),
                    "initial_state": list(inputs["initial_state"].shape),
                    "output": list(expected["output"].shape),
                    "next_state": list(expected["next_state"].shape),
                },
                "dtype": str(inputs["r"].dtype),
                "paths": {
                    "inputs": f"cases/{case['case_id']}/inputs.npz",
                    "expected": f"cases/{case['case_id']}/expected.npz",
                },
                "inputs_sha256": hash_arrays(inputs),
                "expected_sha256": hash_arrays(expected),
                "attention_mask": {
                    "present": "attention_mask" in inputs,
                    "shape": (
                        list(inputs["attention_mask"].shape)
                        if "attention_mask" in inputs
                        else None
                    ),
                },
                "initial_state_nonzero": bool(np.any(inputs["initial_state"] != 0.0)),
                "tolerance": {
                    "dtype": TOLERANCES["float32"].dtype,
                    "atol": TOLERANCES["float32"].atol,
                    "rtol": TOLERANCES["float32"].rtol,
                },
                "full_scan_vs_stepwise": equivalence,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema": FIXTURE_SCHEMA,
        "fixture_version": FIXTURE_VERSION,
        "phase": "P43",
        "fixture_set": FIXTURE_SET,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": _source_metadata(),
        "seed": seed,
        "surface": {
            "implementation_surface": "wkv7_recurrence_core",
            "implementation": "rwkv7_qwen_reference",
            "tested": [
                "per-head projected r/w/k/v/a/b/gate inputs",
                "initial matrix state [B,H,D,D]",
                "optional [B,T] attention_mask",
                "full-scan output [B,T,H,D]",
                "next matrix state [B,H,D,D]",
                "stepwise recurrence equivalence against full scan",
            ],
            "not_tested": [
                "Qwen block norms, RoPE, embeddings, MLP, logits",
                "RADLADS checkpoint parity",
                "training quality",
                "optimized Pallas performance",
            ],
            "note": (
                "P43 fixtures intentionally target the smallest stable "
                "WKV7 recurrence/state surface instead of the full student "
                "model surface."
            ),
            "mask_semantics": (
                "Masked tokens zero v before the state write and zero "
                "output for that token. The decay and in-context state "
                "transform still run."
            ),
        },
        "tolerance_policy": {
            "float32": {"atol": 1e-5, "rtol": 1e-5},
            "bfloat16": {
                "atol": 5e-2,
                "rtol": 5e-2,
                "note": (
                    "Documented for future surfaced bf16 candidates; "
                    "P43 fixtures are float32."
                ),
            },
        },
        "cases": manifest_cases,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "P43_WKV7_FIXTURE_SUMMARY.md").write_text(
        _fixture_summary_markdown(manifest),
        encoding="utf-8",
    )
    return manifest


def validate_wkv7_manifest(
    path: Path, *, verify_payloads: bool = True
) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "bad schema_version")
    _require(manifest.get("schema") == FIXTURE_SCHEMA, "bad schema")
    _require(manifest.get("phase") == "P43", "bad phase")
    _require(manifest.get("fixture_set") == FIXTURE_SET, "bad fixture_set")
    _require(isinstance(manifest.get("cases"), list), "cases must be a list")
    names = {case.get("case_id") for case in manifest["cases"]}
    _require(set(DEFAULT_CASES) == names, "missing required WKV7 cases")
    source = manifest.get("source", {})
    _require(
        source.get("implementation") == "rwkv7_qwen_reference",
        "bad source implementation",
    )
    _require("commit" in source, "missing source commit")
    _require("git_dirty" in source, "missing source git_dirty")
    if verify_payloads:
        for case in manifest["cases"]:
            inputs, expected = load_wkv7_case(manifest_path, case)
            _require(
                case.get("inputs_sha256") == hash_arrays(inputs), "bad inputs hash"
            )
            _require(
                case.get("expected_sha256") == hash_arrays(expected),
                "bad expected hash",
            )
            _validate_case_arrays(case["case_id"], inputs, expected)
    return manifest


def load_wkv7_case(
    manifest_path: Path, case: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = _manifest_path(manifest_path).parent
    paths = case["paths"]
    return _load_npz(base / paths["inputs"]), _load_npz(base / paths["expected"])


def wkv7_reference_full_scan(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output, next_state = _wkv7_scan_jax(_to_jax(inputs), stepwise=False)
    return {
        "output": _np(output),
        "next_state": _np(next_state),
    }


def wkv7_reference_stepwise(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output, next_state = _wkv7_scan_jax(_to_jax(inputs), stepwise=True)
    return {
        "stepwise_output": _np(output),
        "stepwise_next_state": _np(next_state),
    }


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def compare_arrays(
    actual: np.ndarray, expected: np.ndarray, *, tolerance: WKV7Tolerance
) -> dict[str, Any]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        return {
            "status": "shape_mismatch",
            "actual_shape": list(actual_array.shape),
            "expected_shape": list(expected_array.shape),
        }
    if actual_array.dtype != expected_array.dtype:
        return {
            "status": "dtype_mismatch",
            "actual_dtype": str(actual_array.dtype),
            "expected_dtype": str(expected_array.dtype),
            "shape": list(actual_array.shape),
        }
    actual_finite = bool(np.all(np.isfinite(actual_array)))
    expected_finite = bool(np.all(np.isfinite(expected_array)))
    if not actual_finite or not expected_finite:
        return {
            "status": "non_finite",
            "shape": list(actual_array.shape),
            "dtype": str(actual_array.dtype),
            "actual_finite": actual_finite,
            "expected_finite": expected_finite,
        }
    diff = np.abs(actual_array - expected_array)
    denom = np.maximum(np.abs(expected_array), 1e-12)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    allclose = bool(
        np.allclose(
            actual_array,
            expected_array,
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )
    )
    return {
        "status": "pass" if allclose else "fail",
        "shape": list(actual_array.shape),
        "dtype": str(actual_array.dtype),
        "finite": actual_finite,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_relative_error": max_rel,
        "allclose": allclose,
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
    }


def _build_cases(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [
        {
            "case_id": "tiny_b1_t4_h1_d4_no_mask",
            "description": "single batch, short sequence, one head, no mask",
            "inputs": _random_inputs(rng, batch=1, time=4, heads=1, head_size=4),
        },
        {
            "case_id": "tiny_b2_t5_h2_d4_no_mask",
            "description": "two batches, longer sequence, two heads, no mask",
            "inputs": _random_inputs(rng, batch=2, time=5, heads=2, head_size=4),
        },
        {
            "case_id": "tiny_b1_t6_h2_d4_with_attention_mask",
            "description": "single batch masked case with interior attention drops",
            "inputs": _random_inputs(
                rng,
                batch=1,
                time=6,
                heads=2,
                head_size=4,
                attention_mask=np.array([[1, 1, 0, 1, 0, 1]], dtype=np.int32),
            ),
        },
        {
            "case_id": "tiny_prefix_padding_or_reset_case",
            "description": "prefix-masked tokens emulate left padding/reset handling",
            "inputs": _random_inputs(
                rng,
                batch=1,
                time=5,
                heads=2,
                head_size=4,
                attention_mask=np.array([[0, 0, 1, 1, 1]], dtype=np.int32),
                initial_state=_state_grid(batch=1, heads=2, head_size=4, scale=0.03),
            ),
        },
        {
            "case_id": "tiny_stateful_step_vs_full_scan",
            "description": (
                "non-zero initial state used to prove stepwise/full-scan equivalence"
            ),
            "inputs": _random_inputs(
                rng,
                batch=1,
                time=4,
                heads=2,
                head_size=4,
                initial_state=_state_grid(batch=1, heads=2, head_size=4, scale=0.05),
            ),
        },
        {
            "case_id": "tiny_extreme_but_finite_decay_values",
            "description": (
                "finite extreme decay logits to stress stability without overflow"
            ),
            "inputs": _random_inputs(
                rng,
                batch=1,
                time=4,
                heads=2,
                head_size=4,
                w_override=_extreme_decay_values(batch=1, time=4, heads=2, head_size=4),
            ),
        },
    ]


def _random_inputs(
    rng: np.random.Generator,
    *,
    batch: int,
    time: int,
    heads: int,
    head_size: int,
    attention_mask: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    w_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    shape = (batch, time, heads, head_size)
    state_shape = (batch, heads, head_size, head_size)
    inputs = {
        "r": rng.normal(0.0, 0.2, shape).astype(np.float32),
        "w": rng.normal(-0.2, 0.2, shape).astype(np.float32),
        "k": rng.normal(0.0, 0.18, shape).astype(np.float32),
        "v": rng.normal(0.0, 0.22, shape).astype(np.float32),
        "a": rng.normal(0.0, 0.16, shape).astype(np.float32),
        "b": rng.normal(0.0, 0.16, shape).astype(np.float32),
        "gate": rng.uniform(0.2, 0.9, shape).astype(np.float32),
        "initial_state": np.zeros(state_shape, dtype=np.float32),
    }
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask.astype(np.int32)
    if initial_state is not None:
        inputs["initial_state"] = initial_state.astype(np.float32)
    if w_override is not None:
        inputs["w"] = w_override.astype(np.float32)
    return inputs


def _state_grid(*, batch: int, heads: int, head_size: int, scale: float) -> np.ndarray:
    values = np.linspace(
        -scale, scale, num=batch * heads * head_size * head_size, dtype=np.float32
    )
    return values.reshape(batch, heads, head_size, head_size)


def _extreme_decay_values(
    *, batch: int, time: int, heads: int, head_size: int
) -> np.ndarray:
    shape = (batch, time, heads, head_size)
    values = np.array([-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 6.0], dtype=np.float32)
    tiled = np.resize(values, np.prod(shape)).astype(np.float32)
    return tiled.reshape(shape)


def _expected_for_case(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    full = wkv7_reference_full_scan(inputs)
    stepwise = wkv7_reference_stepwise(inputs)
    return {**full, **stepwise}


def _wkv7_scan_jax(
    inputs: dict[str, jax.Array], *, stepwise: bool
) -> tuple[jax.Array, jax.Array]:
    state = jnp.asarray(inputs["initial_state"], dtype=jnp.float32)
    mask = inputs.get("attention_mask")

    def token_at(name: str, index: int | None = None) -> jax.Array:
        value = jnp.asarray(inputs[name], dtype=jnp.float32)
        return value if index is None else value[:, index]

    def one_step(carry: jax.Array, item: tuple[jax.Array, ...]):
        (
            token_r,
            token_w,
            token_k,
            token_v,
            token_a,
            token_b,
            token_gate,
            token_mask,
        ) = item
        token_mask_state = token_mask.reshape(token_mask.shape[0], 1, 1)
        token_v = token_v * token_mask_state
        kk = token_k / jnp.maximum(
            jnp.linalg.norm(token_k, axis=-1, keepdims=True),
            jnp.asarray(1e-6, dtype=token_k.dtype),
        )
        log_w = -jnp.exp(jnp.asarray(-0.5, dtype=token_w.dtype)) * jax.nn.sigmoid(
            token_w
        )
        decay = jnp.exp(log_w)
        vk = jnp.einsum("bhi,bhj->bhij", token_v, token_k)
        ab = jnp.einsum("bhi,bhj->bhij", -kk, kk * token_a + token_b)
        next_state = carry * decay[:, :, None, :] + carry @ ab + vk
        output = jnp.einsum("bhij,bhj->bhi", next_state, token_r) * token_gate
        output = output * token_mask_state
        return next_state, output

    if stepwise:
        outputs = []
        next_state = state
        sequence_length = int(inputs["r"].shape[1])
        for index in range(sequence_length):
            token_mask = (
                jnp.ones(inputs["r"].shape[:1], dtype=jnp.float32)
                if mask is None
                else jnp.asarray(mask[:, index], dtype=jnp.float32)
            )
            next_state, output = one_step(
                next_state,
                (
                    token_at("r", index),
                    token_at("w", index),
                    token_at("k", index),
                    token_at("v", index),
                    token_at("a", index),
                    token_at("b", index),
                    token_at("gate", index),
                    token_mask,
                ),
            )
            outputs.append(output)
        return jnp.stack(outputs, axis=1), next_state

    batch_size, sequence_length = inputs["r"].shape[:2]
    masks = (
        jnp.ones((batch_size, sequence_length), dtype=jnp.float32)
        if mask is None
        else jnp.asarray(mask, dtype=jnp.float32)
    )
    xs = (
        jnp.swapaxes(token_at("r"), 0, 1),
        jnp.swapaxes(token_at("w"), 0, 1),
        jnp.swapaxes(token_at("k"), 0, 1),
        jnp.swapaxes(token_at("v"), 0, 1),
        jnp.swapaxes(token_at("a"), 0, 1),
        jnp.swapaxes(token_at("b"), 0, 1),
        jnp.swapaxes(token_at("gate"), 0, 1),
        jnp.swapaxes(masks, 0, 1),
    )
    next_state, outputs = jax.lax.scan(one_step, state, xs)
    return jnp.swapaxes(outputs, 0, 1), next_state


def _equivalence_metrics(
    full_output: np.ndarray,
    stepwise_output: np.ndarray,
    full_state: np.ndarray,
    stepwise_state: np.ndarray,
    *,
    tolerance: WKV7Tolerance,
) -> dict[str, Any]:
    output = compare_arrays(full_output, stepwise_output, tolerance=tolerance)
    state = compare_arrays(full_state, stepwise_state, tolerance=tolerance)
    return {
        "status": "pass"
        if output["status"] == "pass" and state["status"] == "pass"
        else "fail",
        "full_vs_step_output_max_abs_error": output.get("max_abs_error", 0.0),
        "full_next_state_vs_step_next_state_max_abs_error": state.get(
            "max_abs_error", 0.0
        ),
        "output": output,
        "next_state": state,
    }


def _fixture_summary_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# P43 WKV7 Correctness Fixture Summary",
        "",
        (
            "P43 proves the correctness harness plumbing for the extracted "
            "WKV7 recurrence/state core."
        ),
        "It does not implement or benchmark a production Pallas WKV7 kernel.",
        "",
        f"- Fixture set: `{manifest['fixture_set']}`",
        f"- Schema version: `{manifest['schema_version']}`",
        f"- Implementation surface: `{manifest['surface']['implementation_surface']}`",
        f"- Cases: {len(manifest['cases'])}",
        f"- Source commit: `{manifest['source']['commit']}`",
        (
            "- Float32 tolerance: "
            f"atol={manifest['tolerance_policy']['float32']['atol']} "
            f"rtol={manifest['tolerance_policy']['float32']['rtol']}"
        ),
        "",
        "| Case | Shapes | Mask | Init state | Stepwise vs full-scan |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in manifest["cases"]:
        shapes = case["shapes"]
        lines.append(
            (
                "| {case_id} | `B={batch} T={time} H={heads} D={head_size}` | "
                "{mask} | {state} | `{status}` |"
            ).format(
                case_id=case["case_id"],
                batch=shapes["batch"],
                time=shapes["time"],
                heads=shapes["num_heads"],
                head_size=shapes["head_size"],
                mask="yes" if case["attention_mask"]["present"] else "no",
                state="non-zero" if case["initial_state_nonzero"] else "zero",
                status=case["full_scan_vs_stepwise"]["status"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _validate_case_arrays(
    name: str, inputs: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    for key in ("r", "w", "k", "v", "a", "b", "gate"):
        _require(key in inputs, f"{name} missing {key}")
        _require(inputs[key].ndim == 4, f"{name} {key} must be [B,T,H,D]")
        _require(inputs[key].dtype == np.float32, f"{name} {key} must be float32")
    base_shape = inputs["r"].shape
    for key in ("w", "k", "v", "a", "b", "gate"):
        _require(inputs[key].shape == base_shape, f"{name} {key} shape mismatch")
    state_shape = (base_shape[0], base_shape[2], base_shape[3], base_shape[3])
    _require(
        inputs["initial_state"].shape == state_shape,
        f"{name} initial_state shape mismatch",
    )
    if "attention_mask" in inputs:
        _require(
            inputs["attention_mask"].shape == base_shape[:2],
            f"{name} attention_mask shape mismatch",
        )
    _require(expected["output"].shape == base_shape, f"{name} output shape mismatch")
    _require(expected["output"].dtype == np.float32, f"{name} output dtype mismatch")
    _require(
        expected["next_state"].shape == state_shape, f"{name} next_state shape mismatch"
    )
    _require(
        expected["next_state"].dtype == np.float32, f"{name} next_state dtype mismatch"
    )


def _source_metadata() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    commit = "unknown"
    git_dirty = True
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True
        )
        git_dirty = bool(dirty.strip())
    except Exception:
        pass
    return {
        "implementation": "rwkv7_qwen_reference",
        "commit": commit,
        "git_dirty": git_dirty,
    }


def _to_jax(inputs: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {name: jnp.asarray(value) for name, value in inputs.items()}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}


def _np(value: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float32)


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
