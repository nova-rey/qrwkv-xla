#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from qrwkv_xla.readiness import (
    BigBurnReadinessReport,
    ReadinessStatus,
    build_big_burn_readiness_report,
    write_big_burn_readiness_report,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    report_builder: Callable[
        ...,
        BigBurnReadinessReport,
    ] = build_big_burn_readiness_report,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run the P111 big burn readiness report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p111_big_burn_readiness/readiness_report.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero for warnings as well as failures.",
    )
    args = parser.parse_args(argv)

    report = report_builder(work_dir=args.output.parent)
    write_big_burn_readiness_report(report, args.output)
    print(
        f"status={report.status.value} checks={len(report.checks)} "
        f"blockers={len(report.blockers)} warnings={len(report.warnings)} "
        f"report={args.output}"
    )
    if report.status is ReadinessStatus.FAIL:
        return 1
    if args.strict and report.status is ReadinessStatus.WARN:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
