from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.parity.radlads_source import (
    FIXTURE_SCHEMA,
    FIXTURE_VERSION,
    REQUIRED_CASE_NAMES,
    hash_arrays,
    import_fixture_directory,
    validate_manifest,
)
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

DEFAULT_OUT = Path("tests/fixtures/radlads_source_parity")
DEFAULT_SEED = 4040


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import or create canonical RADLADS source parity fixtures."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--source-fixtures",
        type=Path,
        default=None,
        help=(
            "Existing canonical RADLADS fixture directory to validate and copy. "
            "When omitted, writes deterministic QRWKV current-behavior-only "
            "fixtures marked unsupported for source parity."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source_fixtures is not None:
        manifest = import_fixture_directory(
            args.source_fixtures,
            args.out,
            overwrite=args.overwrite,
        )
        print(
            "imported "
            f"{len(manifest['cases'])} canonical RADLADS fixtures to "
            f"{args.out}"
        )
        return

    manifest = write_current_behavior_fixtures(
        args.out,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        "wrote "
        f"{len(manifest['cases'])} unsupported current-behavior fixtures to {args.out}"
    )


def write_current_behavior_fixtures(
    out: Path,
    *,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
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
    cases = _tiny_cases()
    manifest_cases = []
    for case in cases:
        output, state = student.apply_with_state(
            params,
            jnp.asarray(case["input_ids"]),
            attention_mask=(
                None
                if case["attention_mask"] is None
                else jnp.asarray(case["attention_mask"])
            ),
        )
        if output.logits is None or output.mixer_outputs is None:
            raise RuntimeError("P40 fixture config must emit logits and mixer outputs")
        arrays = {
            "input_ids": case["input_ids"],
            "qrwkv_hidden_states": _np(output.hidden_states),
            "qrwkv_logits": _np(output.logits),
            "qrwkv_mixer_outputs": _np(output.mixer_outputs),
            "qrwkv_wkv_matrix_state": _np(state.wkv_matrix_state),
            "qrwkv_shift_state": _np(state.shift_state),
            "qrwkv_next_position": np.asarray(state.next_position, dtype=np.int32),
        }
        if case["attention_mask"] is not None:
            arrays["attention_mask"] = case["attention_mask"]
        payload = f"{case['name']}.npz"
        np.savez(out / payload, **arrays)
        manifest_cases.append(
            {
                "name": case["name"],
                "description": case["description"],
                "status": "unsupported",
                "unsupported_reason": (
                    "No source-generated RADLADS arrays are bundled. Payload records "
                    "QRWKV-XLA current behavior only and must not be counted as "
                    "RADLADS numerical parity."
                ),
                "payload": payload,
                "payload_sha256": hash_arrays(arrays),
                "input_shape": list(case["input_ids"].shape),
                "attention_mask": {
                    "present": case["attention_mask"] is not None,
                    "kind": case["mask_kind"],
                    "shape": (
                        None
                        if case["attention_mask"] is None
                        else list(case["attention_mask"].shape)
                    ),
                },
                "comparisons": [],
            }
        )

    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "schema": FIXTURE_SCHEMA,
        "backend": "rwkv7_qwen_reference",
        "seed": seed,
        "dtype": "float32",
        "source": {
            "name": "RADLADS",
            "checkout": "/home/nyx/.openclaw/workspace/_refs/RADLADS",
            "branch": "radlads",
            "head": "1b362eb",
            "generation_mode": "qrwkv_current_behavior_only",
            "live_generation": "not run; source fixtures must be supplied manually",
        },
        "claim": (
            "Canonical fixture schema bridge. Checked-in default payloads are "
            "unsupported QRWKV current behavior only, not RADLADS outputs."
        ),
        "config": {
            "batch_size": 2,
            "sequence_length": 5,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "emit_logits": config.emit_logits,
            "emit_mixer_outputs": config.emit_mixer_outputs,
        },
        "required_cases": list(REQUIRED_CASE_NAMES),
        "cases": manifest_cases,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_manifest(out)
    return manifest


def _tiny_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "tiny_no_mask",
            "description": "Tiny deterministic case without attention_mask.",
            "input_ids": np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.int32),
            "attention_mask": None,
            "mask_kind": "none",
        },
        {
            "name": "tiny_attention_mask",
            "description": (
                "Tiny deterministic padding-mask case. Current QRWKV behavior "
                "is recorded only; source parity is unsupported without RADLADS arrays."
            ),
            "input_ids": np.array([[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]], dtype=np.int32),
            "attention_mask": np.array(
                [[1, 1, 0, 1, 1], [1, 0, 1, 1, 1]],
                dtype=np.int32,
            ),
            "mask_kind": "attention_mask",
        },
        {
            "name": "tiny_prefix_padding_or_left_padding",
            "description": (
                "Tiny deterministic prefix/left-padding shape case. QRWKV "
                "positions still advance through masked tokens; source parity "
                "is unsupported without RADLADS arrays."
            ),
            "input_ids": np.array(
                [[0, 0, 11, 12, 13], [0, 14, 15, 16, 17]],
                dtype=np.int32,
            ),
            "attention_mask": np.array(
                [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]],
                dtype=np.int32,
            ),
            "mask_kind": "prefix_or_left_padding",
        },
    ]


def _np(value: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float32)


if __name__ == "__main__":
    main()
