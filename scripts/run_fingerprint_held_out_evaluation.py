#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    HeldOutFingerprintEvaluationConfig,
    run_held_out_fingerprint_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P152 held-out fingerprint evaluation."
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--fingerprint-checkpoint", type=Path, required=True)
    parser.add_argument("--held-out-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--train-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--p151-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--primary-metric",
        choices=("held_out_corridor_loss_total",),
        default="held_out_corridor_loss_total",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_held_out_fingerprint_evaluation(
        HeldOutFingerprintEvaluationConfig(
            baseline_checkpoint=args.baseline_checkpoint,
            fingerprint_checkpoint=args.fingerprint_checkpoint,
            held_out_fingerprint_artifact=args.held_out_fingerprint_artifact,
            train_fingerprint_artifact=args.train_fingerprint_artifact,
            p151_report=args.p151_report,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            tie_tolerance=args.tie_tolerance,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print("comparison_kind=held_out_fingerprint_evaluation")
    print("primary_metric_name=held_out_corridor_loss_total")
    print(f"winner={result.winner}")
    print("winner_scope=held_out_fingerprint_primary_metric_only")
    print("general_quality_claim_made=false")
    print(f"report_path={result.report_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
