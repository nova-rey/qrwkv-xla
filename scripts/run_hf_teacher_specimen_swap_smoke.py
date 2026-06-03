#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.teachers import (
    DEFAULT_HF_SPECIMEN_MODEL_ID,
    HFTeacherSpecimenConfig,
    run_hf_teacher_specimen_swap_smoke,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specimens = tuple(
        HFTeacherSpecimenConfig(
            model_id=model_id,
            prompts=tuple(args.prompt),
            sequence_length=args.sequence_length,
            local_files_only=args.local_files_only,
            allow_downloads=args.allow_downloads,
        )
        for model_id in args.model_id
    )
    report = run_hf_teacher_specimen_swap_smoke(
        specimens,
        target_store_root=args.target_store_root,
    )
    report.write_json(args.output)
    print(f"{report.status}: wrote {args.output}")
    return 1 if report.status == "fail" else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optional P105 HF teacher specimen swap smoke.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="HF causal-LM model id. Repeat for multiple teacher specimens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p105_teacher_specimen_swap/specimen_swap_report.json"),
        help="Aggregate JSON report path.",
    )
    parser.add_argument(
        "--target-store-root",
        type=Path,
        default=Path("artifacts/p105_teacher_specimen_swap/stores"),
        help="Root directory for per-specimen TeacherTargetStore outputs.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Maximum token sequence length for each specimen emission.",
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
    parsed = parser.parse_args(argv)
    if parsed.model_id is None:
        parsed.model_id = [
            DEFAULT_HF_SPECIMEN_MODEL_ID,
            "sshleifer/tiny-gpt2",
        ]
    return parsed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
