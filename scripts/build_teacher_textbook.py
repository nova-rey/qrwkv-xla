#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import (
    TeacherTextbookBuildConfig,
    build_teacher_textbook,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a P116.2 TeacherTextbook. Fake mode is CPU-only and requires "
            "no TPU/GPU/HF/internet; HF mode lazily uses optional teacher-hf "
            "dependencies."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--teacher-mode", choices=("fake", "hf"), default="fake")
    parser.add_argument("--teacher-model", default="fake-deterministic-teacher")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--logits-dtype", default="float32")
    parser.add_argument(
        "--target-type",
        choices=("dense_logits", "topk_with_tail_v0"),
        default="dense_logits",
    )
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument(
        "--top-log-probs-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download HF model/tokenizer files.",
    )
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.local_files_only and args.allow_downloads:
        parser.error("--local-files-only and --allow-downloads cannot both be set")
    if args.teacher_mode == "fake" and args.allow_downloads:
        parser.error("--allow-downloads is not meaningful for --teacher-mode fake")

    config = TeacherTextbookBuildConfig(
        output_dir=args.output,
        dataset_path=args.dataset,
        teacher_mode=args.teacher_mode,
        teacher_model_id=args.teacher_model,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        logits_dtype=args.logits_dtype,
        local_files_only=(
            args.local_files_only
            or args.teacher_mode == "fake"
            or (args.teacher_mode == "hf" and not args.allow_downloads)
        ),
        allow_downloads=args.allow_downloads,
        seed=args.seed,
        overwrite=args.overwrite,
        vocab_size=args.vocab_size,
        target_type=args.target_type,
        top_k=args.top_k,
        top_log_probs_dtype=args.top_log_probs_dtype,
    )
    report = build_teacher_textbook(config)
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} output={args.output}"
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
