#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import ExemplarPassConfig, run_exemplar_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P154 standalone exemplar-only training pass."
    )
    parser.add_argument("--corridor-checkpoint", type=Path, required=True)
    parser.add_argument("--fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-architecture")
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--held-out-fingerprint-artifact", type=Path)
    parser.add_argument("--p153-report", type=Path)
    parser.add_argument("--p153-1-selected-profile", type=Path)
    parser.add_argument("--exemplar-max-records", type=int)
    parser.add_argument(
        "--exemplar-sampling-policy",
        choices=("sequential", "uniform_without_replacement"),
        default="sequential",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=args.corridor_checkpoint,
            fingerprint_artifact=args.fingerprint_artifact,
            output_dir=args.output_dir,
            student_architecture=args.student_architecture,
            student_backend=args.student_backend,
            steps=args.steps,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            held_out_fingerprint_artifact=args.held_out_fingerprint_artifact,
            p153_report=args.p153_report,
            selected_profile=args.p153_1_selected_profile,
            exemplar_max_records=args.exemplar_max_records,
            exemplar_sampling_policy=args.exemplar_sampling_policy,
            resume_checkpoint=args.resume_checkpoint,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print("training_cycle=exemplar")
    print("corridor_loss_enabled=false")
    print("mixed_objective_enabled=false")
    print(f"completed_steps={result.completed_steps}")
    print(f"report_path={result.report_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
