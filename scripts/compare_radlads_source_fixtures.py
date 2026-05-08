from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity import write_comparison_reports

DEFAULT_MANIFEST = Path("tests/fixtures/radlads_source_parity/manifest.json")
DEFAULT_OUT = Path("artifacts/parity/radlads_source_bridge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare canonical RADLADS source parity fixtures."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = write_comparison_reports(args.manifest, args.out_dir)
    print(
        f"wrote parity report to {args.out_dir} "
        f"with overall_status={report['overall_status']}"
    )


if __name__ == "__main__":
    main()
