#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.teachers import (
    DEFAULT_HF_SPECIMEN_MODEL_ID,
    run_hf_teacher_specimen_smoke,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    prompts = tuple(args.prompt)
    result = run_hf_teacher_specimen_smoke(
        model_id=args.model_id,
        target_store=args.target_store,
        prompts=prompts,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
        allow_downloads=args.allow_downloads,
    )
    result.write_json(args.output)
    print(f"{result.status}: wrote {args.output}")
    return 1 if result.status == "fail" else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optional P104 HF teacher specimen smoke.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_HF_SPECIMEN_MODEL_ID,
        help="HF causal-LM model id. The default is only a lab specimen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/p104_hf_teacher_specimen/hf_teacher_specimen_report.json"
        ),
        help="JSON report path.",
    )
    parser.add_argument(
        "--target-store",
        type=Path,
        default=Path("artifacts/p104_hf_teacher_specimen/target_store"),
        help="TeacherTargetStore output directory.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Maximum token sequence length for the specimen emission.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=["hello"],
        help="Prompt to encode. Repeat to emit multiple examples.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached HF files. Defaults true.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Opt in to HF downloads by forcing local_files_only=False.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
