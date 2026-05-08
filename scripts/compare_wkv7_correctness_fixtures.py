from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.kernels import SUPPORTED_CANDIDATES, write_wkv7_comparison_reports

DEFAULT_MANIFEST = Path("artifacts/kernels/p43_wkv7_correctness/manifest.json")
DEFAULT_OUT = Path("artifacts/kernels/p43_wkv7_correctness")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a WKV7 candidate against P43 correctness fixtures."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--candidate",
        choices=SUPPORTED_CANDIDATES,
        default="reference",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = write_wkv7_comparison_reports(
        args.manifest,
        args.out,
        candidate=args.candidate,
        overwrite=args.overwrite,
    )
    print(
        "compared WKV7 candidate "
        f"{args.candidate}: overall_status={report['overall_status']}"
    )


if __name__ == "__main__":
    main()
