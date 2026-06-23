#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    UnconfoundedQualityExperimentConfig,
    run_unconfounded_quality_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a P156.1 unconfounded CPU efficiency family."
    )
    parser.add_argument("--training-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--calibration-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--final-test-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--selected-profile-receipt", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-family", choices=("bytes", "steps", "time"), required=True
    )
    parser.add_argument("--byte-budgets", default="")
    parser.add_argument("--step-budgets", default="")
    parser.add_argument("--fixed-total-steps", type=int, default=25)
    parser.add_argument("--fixed-artifact-budget", type=int)
    parser.add_argument("--wall-clock-safety-ceiling", type=float, default=1800.0)
    parser.add_argument("--corridor-byte-fraction", type=float, default=0.5)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--student-architecture")
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw"
    )
    parser.add_argument("--baseline-learning-rate", type=float, default=1e-4)
    parser.add_argument("--exemplar-learning-rate", type=float, default=5e-5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--target-quality-threshold", type=float, default=1.0)
    parser.add_argument(
        "--require-backend", choices=("cpu", "gpu", "tpu"), default="cpu"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = run_unconfounded_quality_experiment(
        UnconfoundedQualityExperimentConfig(
            training_fingerprint_artifact=args.training_fingerprint_artifact,
            calibration_fingerprint_artifact=args.calibration_fingerprint_artifact,
            final_test_fingerprint_artifact=args.final_test_fingerprint_artifact,
            selected_profile_receipt=args.selected_profile_receipt,
            source_texts=args.source_texts,
            output_dir=args.output_dir,
            experiment_family=args.experiment_family,
            byte_budgets=_ints(args.byte_budgets),
            step_budgets=_ints(args.step_budgets),
            fixed_total_steps=args.fixed_total_steps,
            fixed_artifact_budget=args.fixed_artifact_budget,
            wall_clock_safety_ceiling=args.wall_clock_safety_ceiling,
            corridor_byte_fraction=args.corridor_byte_fraction,
            selection_seed=args.selection_seed,
            seeds=_ints(args.seeds),
            student_architecture=args.student_architecture,
            student_backend=args.student_backend,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            baseline_learning_rate=args.baseline_learning_rate,
            exemplar_learning_rate=args.exemplar_learning_rate,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            target_quality_threshold=args.target_quality_threshold,
            require_backend=args.require_backend,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print(f"report_path={result.report_path}")
    print(f"integrity_path={result.integrity_path}")
    return 0 if result.status in {"pass", "deferred"} else 1


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
