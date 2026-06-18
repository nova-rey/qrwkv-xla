#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    DEFAULT_TINY_TEXTS,
    FingerprintQualityPerByteExperimentConfig,
    run_fingerprint_quality_per_byte_experiment,
)
from qrwkv_xla.teachers import HFTeacherUnavailable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P148 tiny quality-per-byte fingerprint experiment."
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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument(
        "--eval-split",
        choices=("train_artifact_reuse",),
        default="train_artifact_reuse",
    )
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        result = run_fingerprint_quality_per_byte_experiment(
            FingerprintQualityPerByteExperimentConfig(
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
                local_files_only=args.local_files_only and not args.allow_downloads,
                allow_downloads=args.allow_downloads,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                student_backend=args.student_backend,
                eval_split=args.eval_split,
                overwrite=args.overwrite,
            )
        )
    except HFTeacherUnavailable as exc:
        print("status=unavailable", file=sys.stderr)
        print(f"reason={exc}", file=sys.stderr)
        return 2

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    baseline = next(
        arm for arm in report["arms"] if arm["arm_id"] == "baseline_init_only"
    )
    fingerprint = next(
        arm for arm in report["arms"] if arm["arm_id"] == "fingerprint_corridor"
    )
    qpb = report["quality_per_byte"]["reference_delta_vs_init_only"]
    print(f"status={result.status}")
    print(f"eval_split={report['fairness']['eval_split']}")
    print(
        "trained_baseline_available="
        f"{str(report['fairness']['trained_baseline_available']).lower()}"
    )
    print(f"comparison_fairness={report['fairness']['comparison_fairness']}")
    print(
        "fingerprint_artifact_size_bytes="
        f"{report['artifact_budget']['fingerprint_artifact_size_bytes']}"
    )
    print(f"baseline_eval_corridor_loss={baseline['eval']['corridor_loss_total']}")
    print(
        f"fingerprint_eval_corridor_loss={fingerprint['eval']['corridor_loss_total']}"
    )
    print(f"reference_delta_vs_init_only={qpb['absolute_corridor_loss_delta']}")
    print("winner_declared=false")
    print(f"report_path={result.report_path}")
    print(f"summary_path={result.summary_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
