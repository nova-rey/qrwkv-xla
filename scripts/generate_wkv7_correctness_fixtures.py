from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.kernels import generate_wkv7_fixture_bundle

DEFAULT_OUT = Path("artifacts/kernels/p43_wkv7_correctness")
DEFAULT_SEED = 4307


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate P43 tiny deterministic WKV7 correctness fixtures."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = generate_wkv7_fixture_bundle(
        args.out, seed=args.seed, overwrite=args.overwrite
    )
    print(f"wrote {len(manifest['cases'])} WKV7 correctness cases to {args.out}")


if __name__ == "__main__":
    main()
