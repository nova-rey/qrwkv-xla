from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity.radlads_head_to_head import (
    DEFAULT_OUT,
    compare_radlads_qrwkv_head_to_head,
)

DEFAULT_MANIFEST = Path("artifacts/p53_radlads_qrwkv_head_to_head/manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare P53 RADLADS-vs-QRWKV head-to-head fixture outputs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "comparison")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        for path in args.out.glob("*"):
            if path.is_file():
                path.unlink()
    args.out.mkdir(parents=True, exist_ok=True)

    report = compare_radlads_qrwkv_head_to_head(
        args.manifest,
        parameter_npz=args.parameters,
        out_dir=args.out,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(
        f"wrote P53 comparison reports to {args.out} "
        f"with overall_status={report['overall_status']} "
        f"attempted_comparisons={report['attempted_comparisons']}"
    )


if __name__ == "__main__":
    main()
