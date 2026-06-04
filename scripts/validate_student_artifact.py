#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import (
    validate_student_artifact,
    write_student_artifact_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a P116 StudentArtifact.")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--teacher-textbook", type=Path)
    parser.add_argument("--expected-architecture-id", default="current_qrwkv")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write validation_report.json into the StudentArtifact directory.",
    )
    args = parser.parse_args()

    report = validate_student_artifact(
        args.path,
        teacher_textbook_path=args.teacher_textbook,
        expected_architecture_id=args.expected_architecture_id,
    )
    if args.write_report:
        write_student_artifact_validation_report(
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
