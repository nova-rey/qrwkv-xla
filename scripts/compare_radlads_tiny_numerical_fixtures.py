from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity import write_numerical_comparison_reports

DEFAULT_MANIFEST = Path(
    "artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json"
)
DEFAULT_OUT = Path("artifacts/p49_radlads_numerical_parity")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare P49 tiny RADLADS numerical fixtures and write reports."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out-dir",
        "--out",
        dest="out_dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = parser.parse_args()

    report = write_numerical_comparison_reports(args.manifest, args.out_dir)
    print(
        f"wrote P49 reports to {args.out_dir} "
        f"with overall_status={report['overall_status']} "
        f"real_radlads_fixture_status={report['real_radlads_fixture_status']}"
    )


if __name__ == "__main__":
    main()
