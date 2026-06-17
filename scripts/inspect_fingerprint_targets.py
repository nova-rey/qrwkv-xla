#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import load_fingerprint_targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect behavioral_fingerprint target batches."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_fingerprint_targets(
        args.artifact,
        batch_size=args.batch_size,
        max_records=args.max_records,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print("artifact_type=behavioral_fingerprint")
    print(f"artifact_version={dataset.manifest.artifact_version}")
    print(f"num_records={dataset.num_records}")
    print(f"batch_size={args.batch_size}")
    print(f"max_seq_len={dataset.max_seq_len}")
    print(f"vocab_size={dataset.vocab_size}")
    print(f"tracked_stats={','.join(dataset.tracked_stats)}")

    for index, batch in enumerate(dataset.iter_batches()):
        if index >= args.max_batches:
            break
        print(f"batch_{index}_input_ids_shape={batch.input_ids.shape}")
        print(f"batch_{index}_mode_ids={batch.mode_id.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
