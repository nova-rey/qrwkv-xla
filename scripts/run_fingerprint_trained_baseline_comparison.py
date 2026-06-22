#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    FingerprintTrainedBaselineConfig,
    run_fingerprint_trained_baseline_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P151 trained baseline comparison."
    )
    parser.add_argument("--fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_fingerprint_trained_baseline_comparison(
        FingerprintTrainedBaselineConfig(
            fingerprint_artifact=args.fingerprint_artifact,
            source_texts=args.source_texts,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            seed=args.seed,
            student_backend=args.student_backend,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print("comparison_kind=trained_baseline_vs_fingerprint_corridor")
    print("comparison_fairness=matched_training_budget")
    print("winner_declared=false")
    print(f"report_path={result.report_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
