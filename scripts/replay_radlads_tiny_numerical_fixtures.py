from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qrwkv_xla.parity import replay_radlads_tiny_numerical_fixtures

DEFAULT_MANIFEST = Path(
    "artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json"
)
DEFAULT_OUT = Path("artifacts/p50_radlads_replay_compatibility")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay P49 tiny RADLADS parameters through QRWKV-XLA P50."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        "--out",
        dest="out_dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--report-prefix", default="P50")
    args = parser.parse_args()

    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = replay_radlads_tiny_numerical_fixtures(
        args.manifest,
        parameter_npz=args.parameters,
        out_dir=args.out_dir,
        atol=args.atol,
        rtol=args.rtol,
        report_prefix=args.report_prefix,
    )
    print(
        f"wrote {args.report_prefix} replay reports to {args.out_dir} "
        f"with overall_status={report['overall_status']} "
        f"attempted_comparisons={report['attempted_comparisons']}"
    )


if __name__ == "__main__":
    main()
