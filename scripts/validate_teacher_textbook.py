#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import (
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a P116 TeacherTextbook.")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write validation_report.json into the TeacherTextbook directory.",
    )
    args = parser.parse_args()

    report = validate_teacher_textbook(args.path)
    if args.write_report:
        write_teacher_textbook_validation_report(
            report,
            args.path / "validation_report.json",
        )
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} path={args.path}"
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
