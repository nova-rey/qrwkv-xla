#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    TwoCycleExperimentConfig,
    run_two_cycle_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P155 sequential corridor-to-exemplar experiment."
    )
    parser.add_argument("--training-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--held-out-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--selected-profile-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-architecture")
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--baseline-steps", type=int, default=3)
    parser.add_argument("--corridor-steps", type=int, default=3)
    parser.add_argument("--exemplar-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw"
    )
    parser.add_argument("--baseline-learning-rate", type=float, default=1e-4)
    parser.add_argument("--exemplar-learning-rate", type=float, default=5e-5)
    parser.add_argument("--exemplar-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--corridor-eval-every", type=int, default=1)
    parser.add_argument("--exemplar-eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument(
        "--exemplar-sampling-policy",
        choices=("sequential", "uniform_without_replacement"),
        default="sequential",
    )
    parser.add_argument("--exemplar-max-records", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_two_cycle_experiment(
        TwoCycleExperimentConfig(
            training_fingerprint_artifact=args.training_fingerprint_artifact,
            held_out_fingerprint_artifact=args.held_out_fingerprint_artifact,
            source_texts=args.source_texts,
            selected_profile_receipt=args.selected_profile_receipt,
            output_dir=args.output_dir,
            student_architecture=args.student_architecture,
            student_backend=args.student_backend,
            baseline_steps=args.baseline_steps,
            corridor_steps=args.corridor_steps,
            exemplar_steps=args.exemplar_steps,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            baseline_learning_rate=args.baseline_learning_rate,
            exemplar_learning_rate=args.exemplar_learning_rate,
            exemplar_max_grad_norm=args.exemplar_max_grad_norm,
            corridor_eval_every=args.corridor_eval_every,
            exemplar_eval_every=args.exemplar_eval_every,
            checkpoint_every=args.checkpoint_every,
            exemplar_sampling_policy=args.exemplar_sampling_policy,
            exemplar_max_records=args.exemplar_max_records,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            tie_tolerance=args.tie_tolerance,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print(f"primary_result={result.primary_result}")
    print(f"report_path={result.report_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
