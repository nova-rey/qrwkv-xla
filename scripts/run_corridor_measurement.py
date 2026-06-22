#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    CorridorMeasurementConfig,
    run_corridor_measurement,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P153 corridor-pass measurement harness."
    )
    parser.add_argument("--fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--held-out-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--p151-report", type=Path)
    parser.add_argument("--stop-on-stable-entry", action="store_true")
    parser.add_argument("--stable-entry-evals", type=int, default=3)
    parser.add_argument("--corridor-entry-threshold", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_corridor_measurement(
        CorridorMeasurementConfig(
            fingerprint_artifact=args.fingerprint_artifact,
            held_out_fingerprint_artifact=args.held_out_fingerprint_artifact,
            source_texts=args.source_texts,
            output_dir=args.output_dir,
            initial_checkpoint=args.initial_checkpoint,
            seed=args.seed,
            student_backend=args.student_backend,
            steps=args.steps,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            p151_report=args.p151_report,
            stop_on_stable_entry=args.stop_on_stable_entry,
            stable_entry_evals=args.stable_entry_evals,
            corridor_entry_threshold=args.corridor_entry_threshold,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print("measurement_kind=corridor_pass_trajectory")
    print(f"completed_steps={result.completed_steps}")
    print(f"stable_entry_achieved={str(result.stable_entry_achieved).lower()}")
    print("general_quality_claim_made=false")
    print(f"report_path={result.report_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
