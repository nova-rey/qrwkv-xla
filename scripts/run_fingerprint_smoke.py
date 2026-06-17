#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.training import (
    FingerprintCorridorLossConfig,
    FingerprintTrainingSmokeConfig,
    run_tiny_fingerprint_training_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P136 tiny fingerprint-only training smoke."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-remainder", action="store_true")
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--top1-margin-weight", type=float, default=1.0)
    parser.add_argument("--top8-mass-weight", type=float, default=1.0)
    parser.add_argument("--top32-mass-weight", type=float, default=1.0)
    parser.add_argument("--tail-mass-weight", type=float, default=1.0)
    parser.add_argument("--disable-record-weights", action="store_true")
    args = parser.parse_args()

    result = run_tiny_fingerprint_training_smoke(
        FingerprintTrainingSmokeConfig(
            artifact_dir=args.artifact,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            shuffle=args.shuffle,
            max_records=args.max_records,
            drop_remainder=args.drop_remainder,
            loss_config=FingerprintCorridorLossConfig(
                entropy_weight=args.entropy_weight,
                top1_margin_weight=args.top1_margin_weight,
                top8_mass_weight=args.top8_mass_weight,
                top32_mass_weight=args.top32_mass_weight,
                tail_mass_weight=args.tail_mass_weight,
                use_record_weights=not args.disable_record_weights,
            ),
        )
    )
    print(
        f"status={result.status} steps={result.steps} "
        f"initial_loss={result.initial_loss:.8f} "
        f"final_loss={result.final_loss:.8f} "
        f"metrics={result.metrics_path} checkpoint={result.checkpoint_path}"
    )
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
