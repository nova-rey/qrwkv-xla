from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny JAX student smoke train on an existing target bundle"
    )
    parser.add_argument("--targets", required=True, help="Existing target bundle path")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum train steps")
    parser.add_argument("--seed", type=int, default=0, help="JAX parameter seed")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Smoke train learning rate"
    )
    args = parser.parse_args()

    from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
    from qrwkv_xla.students import TinyStudent, TinyStudentConfig
    from qrwkv_xla.targets import read_manifest
    from qrwkv_xla.targets.store import manifest_path
    from qrwkv_xla.trainers import train_on_bundle_once

    targets = Path(args.targets)
    manifest = read_manifest(manifest_path(targets))
    dataset = TargetBundleDataset.from_path(targets)
    config = TinyStudentConfig(
        vocab_size=_infer_vocab_size(manifest.extra.get("vocab_size"), dataset),
        hidden_size=manifest.hidden_size,
        num_layers=manifest.num_layers,
    )
    student = TinyStudent(config)
    result = train_on_bundle_once(
        bundle_dir=targets,
        student=student,
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )

    print(f"targets: {targets}")
    print(f"sequence_length: {manifest.sequence_length}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_layers: {config.num_layers}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"steps: {result.steps}")
    print(f"initial_loss: {result.initial_loss:.8f}")
    print(f"final_loss: {result.final_loss:.8f}")


def _infer_vocab_size(
    manifest_vocab_size: object,
    dataset: object,
) -> int:
    if manifest_vocab_size is not None:
        vocab_size = int(manifest_vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"manifest vocab_size must be > 0, got {vocab_size}")
        return vocab_size

    max_token_id = 0
    for batch in dataset.iter_shards():  # type: ignore[attr-defined]
        if batch.input_ids.size:
            max_token_id = max(max_token_id, int(np.max(batch.input_ids)))
    return max_token_id + 1


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Student smoke training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
