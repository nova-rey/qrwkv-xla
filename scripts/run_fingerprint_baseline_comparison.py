#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    DEFAULT_TINY_TEXTS,
    FingerprintBaselineComparisonConfig,
    run_fingerprint_baseline_comparison,
)
from qrwkv_xla.teachers import HFTeacherUnavailable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P147 tiny fingerprint baseline comparison harness."
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
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        result = run_fingerprint_baseline_comparison(
            FingerprintBaselineComparisonConfig(
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
                overwrite=args.overwrite,
            )
        )
    except HFTeacherUnavailable as exc:
        print("status=unavailable", file=sys.stderr)
        print(f"reason={exc}", file=sys.stderr)
        return 2

    report = _read_report(result.report_path)
    fingerprint = next(
        arm for arm in report["arms"] if arm["arm_id"] == "fingerprint_corridor"
    )
    baseline = next(
        arm for arm in report["arms"] if arm["arm_id"] == "baseline_init_only"
    )
    print(f"status={result.status}")
    print(f"artifact_dir={result.artifact_dir}")
    print(f"arms_run={','.join(result.arms_run)}")
    print(f"fingerprint_final_loss={fingerprint['final_loss']}")
    print(f"baseline_metric={baseline['baseline_loss']}")
    print(f"report_path={result.report_path}")
    print(f"summary_path={result.summary_path}")
    print("quality_claim_made=false")
    return 0 if result.status == "pass" else 1


def _read_report(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
