#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    DEFAULT_TINY_TEXTS,
    RealTeacherFingerprintTrainingRehearsalConfig,
    run_real_teacher_fingerprint_training_rehearsal,
)
from qrwkv_xla.teachers import HFTeacherUnavailable


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P146 real-teacher fingerprint artifact to real student "
            "training rehearsal."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fingerprint-artifact", type=Path)
    mode.add_argument("--build-real-teacher-artifact", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_TINY_REAL_TEACHER)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--texts", type=Path, default=DEFAULT_TINY_TEXTS)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--max-target-positions", type=int, default=64)
    parser.add_argument("--max-exemplars", type=int, default=16)
    parser.add_argument(
        "--bounds-method", choices=("minmax", "quantile"), default="quantile"
    )
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument(
        "--exemplar-selection-policy",
        choices=("top_interestingness_v0", "stratified_interestingness_v0"),
        default="stratified_interestingness_v0",
    )
    parser.add_argument("--per-mode-min", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        result = run_real_teacher_fingerprint_training_rehearsal(
            RealTeacherFingerprintTrainingRehearsalConfig(
                output_dir=args.output_dir,
                fingerprint_artifact=args.fingerprint_artifact,
                build_real_teacher_artifact=args.build_real_teacher_artifact,
                texts_path=args.texts,
                teacher_model=args.teacher_model,
                tokenizer=args.tokenizer,
                sequence_length=args.sequence_length,
                max_examples=args.max_examples,
                max_target_positions=args.max_target_positions,
                max_exemplars=args.max_exemplars,
                bounds_method=args.bounds_method,
                lower_quantile=args.lower_quantile,
                upper_quantile=args.upper_quantile,
                exemplar_selection_policy=args.exemplar_selection_policy,
                per_mode_min=args.per_mode_min,
                local_files_only=args.local_files_only and not args.allow_downloads,
                allow_downloads=args.allow_downloads,
                training_steps=args.training_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                optimizer=args.optimizer,
                seed=args.seed,
                student_backend=args.student_backend,
                resume_from=args.resume_from,
                overwrite=args.overwrite,
            )
        )
    except HFTeacherUnavailable as exc:
        print("status=unavailable", file=sys.stderr)
        print(f"reason={exc}", file=sys.stderr)
        return 2

    print(f"status={result.status}")
    print(f"artifact_source={result.artifact_source}")
    print(f"artifact_dir={result.artifact_dir}")
    print(f"teacher_model={result.teacher_model_name_or_path}")
    print(f"teacher_real={str(result.teacher_real).lower()}")
    print(
        "teacher_required_during_training="
        f"{str(result.teacher_required_during_training).lower()}"
    )
    print(f"optimizer_steps_completed={result.optimizer_steps_completed}")
    print(f"params_changed={str(result.params_changed).lower()}")
    print(f"param_delta_norm={result.param_delta_norm}")
    print(f"final_loss={result.final_loss}")
    print(f"checkpoint_dir={result.checkpoint_dir}")
    print(f"report_path={result.report_path}")
    print(f"summary_path={result.summary_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
