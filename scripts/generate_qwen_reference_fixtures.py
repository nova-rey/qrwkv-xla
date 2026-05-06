from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

FIXTURE_VERSION = 1
DEFAULT_OUT = Path("tests/fixtures/qwen_reference")
DEFAULT_SEED = 1234


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiny deterministic rwkv7_qwen_reference fixtures."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    write_fixtures(args.out, seed=args.seed, overwrite=args.overwrite)


def write_fixtures(out: Path, *, seed: int, overwrite: bool = False) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to replace fixtures")
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*"):
        if path.is_file():
            path.unlink()

    config = RWKV7QwenReferenceConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
        emit_mixer_outputs=True,
    )
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(seed))
    param_surface = _param_surface(params)

    cases = [
        {
            "name": "no_mask",
            "description": "No attention_mask argument; all tokens active.",
            "input_ids": np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.int32),
            "attention_mask": None,
            "mask_kind": "none",
        },
        {
            "name": "interior_mask",
            "description": (
                "Interior masked tokens lock current behavior: outputs/logits are "
                "emitted after value/output/MLP masking, while position and shift "
                "state still advance."
            ),
            "input_ids": np.array([[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]], dtype=np.int32),
            "attention_mask": np.array(
                [[1, 1, 0, 1, 1], [1, 0, 1, 1, 1]],
                dtype=np.int32,
            ),
            "mask_kind": "interior",
        },
        {
            "name": "prefix_left_padding",
            "description": (
                "Prefix/left-padding shape fixture. Left-padding is represented "
                "only by a [B,T] mask; positions still start at zero and advance "
                "through masked tokens."
            ),
            "input_ids": np.array(
                [[0, 0, 11, 12, 13], [0, 14, 15, 16, 17]],
                dtype=np.int32,
            ),
            "attention_mask": np.array(
                [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]],
                dtype=np.int32,
            ),
            "mask_kind": "prefix_left_padding",
        },
    ]

    manifest_cases = []
    for case in cases:
        arrays, equivalence = _run_case(
            student,
            params,
            input_ids=jnp.asarray(case["input_ids"]),
            attention_mask=(
                None
                if case["attention_mask"] is None
                else jnp.asarray(case["attention_mask"])
            ),
        )
        if case["attention_mask"] is not None:
            arrays["attention_mask"] = case["attention_mask"]
        arrays["input_ids"] = case["input_ids"]
        payload_name = f"{case['name']}.npz"
        np.savez(out / payload_name, **arrays)
        manifest_cases.append(
            {
                "name": case["name"],
                "description": case["description"],
                "payload": payload_name,
                "payload_sha256": _hash_arrays(arrays),
                "input_shape": list(case["input_ids"].shape),
                "attention_mask": {
                    "kind": case["mask_kind"],
                    "present": case["attention_mask"] is not None,
                    "shape": (
                        None
                        if case["attention_mask"] is None
                        else list(case["attention_mask"].shape)
                    ),
                },
                "equivalence": equivalence,
            }
        )

    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "backend": "rwkv7_qwen_reference",
        "claim": (
            "Tiny deterministic fixture harness for the local slow JAX "
            "reference only; not full RADLADS numerical parity."
        ),
        "seed": seed,
        "dtype": str(np.dtype(np.float32)),
        "jax": {
            "version": jax.__version__,
            "default_backend": jax.default_backend(),
        },
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "sequence_length": 5,
            "batch_size": 2,
            "emit_logits": config.emit_logits,
            "emit_mixer_outputs": config.emit_mixer_outputs,
            "tie_embeddings": config.tie_embeddings,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
        },
        "masked_token_behavior": {
            "summary": (
                "attention_mask is [B,T]. Masked tokens zero the value stream, "
                "attention output, and MLP output for that token. Recurrence "
                "decay/update from the previous matrix state still runs, "
                "shift_state is updated to the masked token representation, and "
                "next_position advances by sequence length."
            ),
            "left_padding": (
                "Left padding is a mask shape fixture, not a separate position "
                "semantics claim; masked prefix positions still consume positions."
            ),
        },
        "parameter_surface": {
            "sha256": _hash_param_surface(param_surface),
            "leaf_count": len(param_surface),
            "leaves": param_surface,
        },
        "cases": manifest_cases,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_cases)} qwen reference fixture cases to {out}")
    return manifest


def _run_case(
    student: RWKV7QwenReferenceStudent,
    params: dict[str, object],
    *,
    input_ids: jax.Array,
    attention_mask: jax.Array | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    full_output, full_state = student.apply_with_state(
        params,
        input_ids,
        attention_mask=attention_mask,
    )
    step_state = student.init_state(batch_size=input_ids.shape[0])
    step_logits = []
    step_hidden = []
    step_mixer = []
    for index in range(input_ids.shape[1]):
        step_output, step_state = student.step(
            params,
            input_ids[:, index : index + 1],
            step_state,
            attention_mask=(
                None if attention_mask is None else attention_mask[:, index : index + 1]
            ),
        )
        if step_output.logits is None or step_output.mixer_outputs is None:
            raise RuntimeError("fixture config must emit logits and mixer outputs")
        step_logits.append(step_output.logits)
        step_hidden.append(step_output.hidden_states)
        step_mixer.append(step_output.mixer_outputs)

    if full_output.logits is None or full_output.mixer_outputs is None:
        raise RuntimeError("fixture config must emit logits and mixer outputs")
    step_logits_array = jnp.concatenate(step_logits, axis=1)
    step_hidden_array = jnp.concatenate(step_hidden, axis=2)
    step_mixer_array = jnp.concatenate(step_mixer, axis=2)

    arrays = {
        "full_hidden_states": _np(full_output.hidden_states),
        "full_logits": _np(full_output.logits),
        "full_mixer_outputs": _np(full_output.mixer_outputs),
        "full_wkv_matrix_state": _np(full_state.wkv_matrix_state),
        "full_shift_state": _np(full_state.shift_state),
        "full_next_position": np.asarray(full_state.next_position, dtype=np.int32),
        "step_hidden_states": _np(step_hidden_array),
        "step_logits": _np(step_logits_array),
        "step_mixer_outputs": _np(step_mixer_array),
        "step_wkv_matrix_state": _np(step_state.wkv_matrix_state),
        "step_shift_state": _np(step_state.shift_state),
        "step_next_position": np.asarray(step_state.next_position, dtype=np.int32),
    }
    equivalence = {
        "hidden_states_max_abs": _max_abs(
            arrays["full_hidden_states"],
            arrays["step_hidden_states"],
        ),
        "hidden_states_max_rel": _max_rel(
            arrays["full_hidden_states"],
            arrays["step_hidden_states"],
        ),
        "logits_max_abs": _max_abs(arrays["full_logits"], arrays["step_logits"]),
        "logits_max_rel": _max_rel(arrays["full_logits"], arrays["step_logits"]),
        "mixer_outputs_max_abs": _max_abs(
            arrays["full_mixer_outputs"],
            arrays["step_mixer_outputs"],
        ),
        "mixer_outputs_max_rel": _max_rel(
            arrays["full_mixer_outputs"],
            arrays["step_mixer_outputs"],
        ),
        "wkv_matrix_state_max_abs": _max_abs(
            arrays["full_wkv_matrix_state"],
            arrays["step_wkv_matrix_state"],
        ),
        "wkv_matrix_state_max_rel": _max_rel(
            arrays["full_wkv_matrix_state"],
            arrays["step_wkv_matrix_state"],
        ),
        "shift_state_max_abs": _max_abs(
            arrays["full_shift_state"],
            arrays["step_shift_state"],
        ),
        "shift_state_max_rel": _max_rel(
            arrays["full_shift_state"],
            arrays["step_shift_state"],
        ),
        "next_position_abs": float(
            abs(int(arrays["full_next_position"]) - int(arrays["step_next_position"]))
        ),
    }
    return arrays, equivalence


def _param_surface(params: dict[str, object]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], (*path, str(key)))
            return
        array = np.asarray(value)
        leaves.append(
            {
                "path": ".".join(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _hash_arrays({"value": array}),
            }
        )

    visit(params, ())
    return leaves


def _hash_param_surface(surface: list[dict[str, Any]]) -> str:
    payload = json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _np(value: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float32)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def _max_rel(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.maximum(np.abs(left), 1e-12)
    return float(np.max(np.abs(left - right) / denom))


if __name__ == "__main__":
    main()
