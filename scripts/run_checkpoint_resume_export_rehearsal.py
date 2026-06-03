#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrwkv_xla.checkpointing import run_checkpoint_resume_export_rehearsal


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P108 checkpoint/resume/export rehearsal."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p108_checkpoint_resume_export"),
    )
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_checkpoint_resume_export_rehearsal(
        output_dir=args.output_dir,
        overwrite=not args.no_overwrite,
    )
    report = result.to_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {result.status}")
        print(f"checkpoint_path: {result.checkpoint_path}")
        print(f"export_path: {result.export_path}")
        print(f"reason: {result.reason}")
    return 0 if result.status in {"pass", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
