#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import (
    FingerprintArc2ReportConfig,
    run_fingerprint_arc2_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the P149 Arc 2 report / go-no-go artifacts."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("docs/QRWKV_SNAPSHOT.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = run_fingerprint_arc2_report(
        FingerprintArc2ReportConfig(
            output_dir=args.output_dir,
            snapshot_path=args.snapshot,
            overwrite=args.overwrite,
        )
    )
    print(f"status={result.status}")
    print(f"recommendation={result.recommendation}")
    print(f"report_path={result.report_path}")
    print(f"summary_path={result.summary_path}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
