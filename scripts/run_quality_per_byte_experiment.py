#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    ControlledQualityPerByteConfig,
    QualityBudgetPoint,
    run_controlled_quality_per_byte_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P156 controlled CPU quality-per-byte matrix."
    )
    parser.add_argument("--training-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--calibration-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--final-test-fingerprint-artifact", type=Path, required=True)
    parser.add_argument("--selected-profile-receipt", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-architecture")
    parser.add_argument("--student-backend", default="current_qrwkv")
    parser.add_argument("--byte-budgets", default="small,medium")
    parser.add_argument("--step-budgets", default="10,25")
    parser.add_argument("--wall-clock-budgets", default="300,600")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--target-quality-threshold", type=float, default=1.0)
    parser.add_argument(
        "--require-backend", choices=("cpu", "gpu", "tpu"), default="cpu"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels = [value.strip() for value in args.byte_budgets.split(",")]
    steps = [int(value) for value in args.step_budgets.split(",")]
    wall = [float(value) for value in args.wall_clock_budgets.split(",")]
    if not (len(labels) == len(steps) == len(wall)):
        parser.error("byte, step, and wall-clock budget lists must have equal lengths")
    physical_bytes = sum(
        path.stat().st_size
        for path in args.training_fingerprint_artifact.rglob("*")
        if path.is_file()
    )
    points = tuple(
        QualityBudgetPoint(
            name=label,
            teacher_artifact_bytes=_byte_budget(label, physical_bytes, index),
            total_steps=steps[index],
            wall_clock_seconds=wall[index],
        )
        for index, label in enumerate(labels)
    )
    result = run_controlled_quality_per_byte_experiment(
        ControlledQualityPerByteConfig(
            training_fingerprint_artifact=args.training_fingerprint_artifact,
            calibration_fingerprint_artifact=args.calibration_fingerprint_artifact,
            final_test_fingerprint_artifact=args.final_test_fingerprint_artifact,
            selected_profile_receipt=args.selected_profile_receipt,
            source_texts=args.source_texts,
            output_dir=args.output_dir,
            student_architecture=args.student_architecture,
            student_backend=args.student_backend,
            budget_points=points,
            seeds=tuple(int(value) for value in args.seeds.split(",")),
            batch_size=args.batch_size,
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
    print(f"matrix_state_path={result.matrix_state_path}")
    return 0 if result.status == "pass" else 1


def _byte_budget(label: str, physical_bytes: int, index: int) -> int:
    try:
        return int(label)
    except ValueError:
        return physical_bytes * (index + 1)


if __name__ == "__main__":
    raise SystemExit(main())
