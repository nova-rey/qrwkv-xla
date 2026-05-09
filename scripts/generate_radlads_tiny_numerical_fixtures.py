from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity import generate_radlads_tiny_numerical_fixtures

DEFAULT_OUT = Path("artifacts/p49_radlads_numerical_parity/radlads_fixtures")
DEFAULT_RADLADS_SOURCE = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")
DEFAULT_SEED = 4949


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate P49 tiny RADLADS numerical fixtures. The default path "
            "writes offline-safe QRWKV current-behavior payloads marked "
            "missing_source. Set QRWKV_XLA_RUN_RADLADS_LIVE_FIXTURES=1 "
            "(or QRWKV_RUN_RADLADS_LIVE=1) to try the local RADLADS hook."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--radlads-source", type=Path, default=DEFAULT_RADLADS_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = generate_radlads_tiny_numerical_fixtures(
        args.out,
        seed=args.seed,
        overwrite=args.overwrite,
        radlads_source_path=args.radlads_source,
    )
    print(
        "wrote "
        f"{len(manifest['cases'])} P49 tiny numerical fixtures to {args.out} "
        f"with real_radlads_fixture_status="
        f"{manifest['real_radlads_fixture_status']}"
    )


if __name__ == "__main__":
    main()
