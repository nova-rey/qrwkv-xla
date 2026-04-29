from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fake teacher target bundle")
    parser.add_argument("--out", required=True, help="Output bundle directory")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    _add_src_to_path()

    from qrwkv_xla.targets import (
        TargetFlags,
        TeacherTargetManifest,
        write_target_bundle,
    )

    rng = np.random.default_rng(args.seed)
    manifest = TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="qwen",
        teacher_model_id=None,
        teacher_policy_label="Qwen3.latest",
        fallback_policy_label="Qwen3.0",
        tokenizer_id=None,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        targets=TargetFlags(),
        dtype="fp32",
        created_by="create_fake_targets.py",
        notes=[],
    )

    shards = []
    for _ in range(args.num_shards):
        input_ids = rng.integers(
            low=0,
            high=32000,
            size=(args.batch_size, args.sequence_length),
            dtype=np.int32,
        )
        attention_mask = np.ones(
            (args.batch_size, args.sequence_length),
            dtype=np.int32,
        )
        hidden_states = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(
                args.batch_size,
                args.num_layers,
                args.sequence_length,
                args.hidden_size,
            ),
        ).astype(np.float32)
        shards.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "hidden_states": hidden_states,
            }
        )

    out_dir = Path(args.out)
    write_target_bundle(out_dir, manifest, shards)

    print(f"Wrote fake target bundle: {out_dir}")
    print(f"Shards: {args.num_shards}")
    print(f"Examples per shard: {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")


if __name__ == "__main__":
    main()
