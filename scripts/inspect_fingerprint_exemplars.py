#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import load_fingerprint_exemplars


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect behavioral_fingerprint exemplar batches."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()

    dataset = load_fingerprint_exemplars(
        args.artifact,
        batch_size=args.batch_size,
    )

    print("artifact_type=behavioral_fingerprint")
    print("payload_type=dense_probs")
    print(f"num_exemplars={dataset.num_records}")
    print(f"vocab_size={dataset.vocab_size}")
    print(f"max_seq_len={dataset.max_seq_len}")

    for batch_index, batch in enumerate(dataset.iter_batches()):
        if batch_index >= args.max_batches:
            break
        print(f"batch_{batch_index}_input_ids_shape={batch.input_ids.shape}")
        print(f"batch_{batch_index}_teacher_probs_shape={batch.teacher_probs.shape}")
        print(f"batch_{batch_index}_positions={batch.position.tolist()}")
        print(f"batch_{batch_index}_mode_ids={batch.mode_id.tolist()}")
        print(f"batch_{batch_index}_reason_codes={batch.reason_codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
