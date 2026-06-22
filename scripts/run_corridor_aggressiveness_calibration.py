#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    PROFILE_NAMES,
    AggressivenessCalibrationConfig,
    run_aggressiveness_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run P153.1 corridor profile calibration."
    )
    parser.add_argument("--fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--held-out-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", default=",".join(PROFILE_NAMES))
    parser.add_argument(
        "--aggressiveness-profile", action="append", choices=PROFILE_NAMES
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw"), default="adamw"
    )
    parser.add_argument("--corridor-entry-threshold", type=float, default=0.95)
    parser.add_argument("--stable-entry-evals", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1531)
    for name in (
        "corridor-loss-weight",
        "learning-rate",
        "max-grad-norm",
        "penalty-power",
        "worst-stat-boost",
        "entropy-weight",
        "top1-margin-weight",
        "top8-mass-weight",
        "top32-mass-weight",
        "tail-mass-weight",
    ):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    profiles = tuple(args.aggressiveness_profile or args.profiles.split(","))
    override_names = (
        "corridor_loss_weight",
        "learning_rate",
        "max_grad_norm",
        "penalty_power",
        "worst_stat_boost",
        "entropy_weight",
        "top1_margin_weight",
        "top8_mass_weight",
        "top32_mass_weight",
        "tail_mass_weight",
    )
    result = run_aggressiveness_calibration(
        AggressivenessCalibrationConfig(
            fingerprint_artifact=args.fingerprint_artifact,
            held_out_fingerprint_artifact=args.held_out_fingerprint_artifact,
            source_texts=args.source_texts,
            output_dir=args.output_dir,
            profiles=profiles,
            seeds=tuple(int(x) for x in args.seeds.split(",")),
            steps=args.steps,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            student_backend=args.student_backend,
            optimizer=args.optimizer,
            corridor_entry_threshold=args.corridor_entry_threshold,
            stable_entry_evals=args.stable_entry_evals,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            overrides={name: getattr(args, name) for name in override_names},
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print(f"selected_profile={result.selected_profile}")
    print("general_quality_claim_made=false")
    print(f"report_path={result.report_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
