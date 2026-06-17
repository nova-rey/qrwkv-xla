#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.training import (
    FingerprintCorridorLossConfig,
    FingerprintMixedSmokeConfig,
    FingerprintTrainingSmokeConfig,
    run_mixed_fingerprint_training_smoke,
    run_tiny_fingerprint_training_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run tiny fingerprint smoke modes."
    )
    parser.add_argument(
        "--mode",
        choices=("corridor", "mixed"),
        default="corridor",
        help="corridor keeps the P136 behavior; mixed runs the P138 smoke.",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--corridor-batch-size", type=int, default=None)
    parser.add_argument("--exemplar-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--corridor-max-records", type=int, default=None)
    parser.add_argument("--exemplar-max-records", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--corridor-shuffle", action="store_true")
    parser.add_argument("--exemplar-shuffle", action="store_true")
    parser.add_argument("--drop-remainder", action="store_true")
    parser.add_argument("--corridor-drop-remainder", action="store_true")
    parser.add_argument("--exemplar-drop-remainder", action="store_true")
    parser.add_argument("--corridor-loss-weight", type=float, default=1.0)
    parser.add_argument("--exemplar-loss-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--top1-margin-weight", type=float, default=1.0)
    parser.add_argument("--top8-mass-weight", type=float, default=1.0)
    parser.add_argument("--top32-mass-weight", type=float, default=1.0)
    parser.add_argument("--tail-mass-weight", type=float, default=1.0)
    parser.add_argument("--disable-record-weights", action="store_true")
    args = parser.parse_args()

    corridor_loss_config = FingerprintCorridorLossConfig(
        entropy_weight=args.entropy_weight,
        top1_margin_weight=args.top1_margin_weight,
        top8_mass_weight=args.top8_mass_weight,
        top32_mass_weight=args.top32_mass_weight,
        tail_mass_weight=args.tail_mass_weight,
        use_record_weights=not args.disable_record_weights,
    )
    if args.mode == "mixed":
        result = run_mixed_fingerprint_training_smoke(
            FingerprintMixedSmokeConfig(
                artifact_dir=args.artifact,
                output_dir=args.output_dir,
                steps=args.steps,
                corridor_batch_size=args.corridor_batch_size or args.batch_size,
                exemplar_batch_size=args.exemplar_batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                corridor_shuffle=args.corridor_shuffle or args.shuffle,
                exemplar_shuffle=args.exemplar_shuffle or args.shuffle,
                corridor_max_records=(
                    args.corridor_max_records
                    if args.corridor_max_records is not None
                    else args.max_records
                ),
                exemplar_max_records=args.exemplar_max_records,
                corridor_drop_remainder=(
                    args.corridor_drop_remainder or args.drop_remainder
                ),
                exemplar_drop_remainder=args.exemplar_drop_remainder,
                corridor_loss_weight=args.corridor_loss_weight,
                exemplar_loss_weight=args.exemplar_loss_weight,
                corridor_loss_config=corridor_loss_config,
            )
        )
        print(
            f"status={result.status} steps={result.requested_steps} "
            f"optimizer_steps={result.optimizer_steps_completed} "
            f"corridor_batches={result.corridor_batches_consumed} "
            f"exemplar_batches={result.exemplar_batches_consumed} "
            f"initial_mixed_loss={result.initial_mixed_loss:.8f} "
            f"final_mixed_loss={result.final_mixed_loss:.8f} "
            f"mixed_loss_non_increasing={result.mixed_loss_non_increasing} "
            f"metrics={result.metrics_path} checkpoint={result.checkpoint_path}"
        )
    else:
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
                loss_config=corridor_loss_config,
            )
        )
        print(
            f"status={result.status} steps={result.steps} "
            f"train_batches={result.train_batches_consumed} "
            f"initial_loss={result.initial_loss:.8f} "
            f"final_loss={result.final_loss:.8f} "
            f"loss_non_increasing={result.loss_non_increasing} "
            f"metrics={result.metrics_path} checkpoint={result.checkpoint_path}"
        )
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
