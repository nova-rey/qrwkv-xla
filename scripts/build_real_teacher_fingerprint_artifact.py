#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.fingerprint import (
    DEFAULT_TINY_REAL_TEACHER,
    TinyRealTeacherFingerprintCaptureConfig,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.teachers import HFTeacherUnavailable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a P145 tiny real-teacher behavioral fingerprint artifact."
    )
    parser.add_argument("--teacher-model", default=DEFAULT_TINY_REAL_TEACHER)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--consumer-vocab-limit", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        result = run_tiny_real_teacher_fingerprint_capture(
            TinyRealTeacherFingerprintCaptureConfig(
                output_dir=args.output_dir,
                texts_path=args.texts,
                teacher_model=args.teacher_model,
                tokenizer=args.tokenizer,
                sequence_length=args.sequence_length,
                max_examples=args.max_examples,
                max_target_positions=args.max_target_positions,
                local_files_only=args.local_files_only and not args.allow_downloads,
                allow_downloads=args.allow_downloads,
                overwrite=args.overwrite,
                max_exemplars=args.max_exemplars,
                bounds_method=args.bounds_method,
                lower_quantile=args.lower_quantile,
                upper_quantile=args.upper_quantile,
                exemplar_selection_policy=args.exemplar_selection_policy,
                per_mode_min=args.per_mode_min,
                consumer_vocab_limit=args.consumer_vocab_limit,
            )
        )
    except HFTeacherUnavailable as exc:
        print("status=unavailable", file=sys.stderr)
        print(f"reason={exc}", file=sys.stderr)
        return 2

    print(f"status={result.status}")
    print(f"artifact_dir={result.output_dir}")
    print(f"teacher_model={result.teacher_model_name_or_path}")
    print(f"examples_processed={result.examples_processed}")
    print(f"target_positions_processed={result.target_positions_processed}")
    print(f"modes_discovered={result.modes_discovered}")
    print(f"exemplars_retained={result.exemplars_retained}")
    print(f"artifact_validated={str(result.artifact_validated).lower()}")
    print(f"consumer_sanity_kind={result.consumer_sanity['kind']}")
    print(f"consumer_sanity_status={result.consumer_sanity['status']}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
