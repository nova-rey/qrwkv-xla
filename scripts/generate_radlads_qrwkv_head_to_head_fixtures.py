from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity.radlads_head_to_head import (
    DEFAULT_OUT,
    DEFAULT_SEED,
    generate_radlads_qrwkv_head_to_head_fixtures,
)

DEFAULT_RADLADS_SOURCE = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate P53 RADLADS-vs-QRWKV head-to-head fixtures using the "
            "deterministic-finite P52 payload."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--radlads-repo", type=Path, default=DEFAULT_RADLADS_SOURCE)
    parser.add_argument(
        "--init-policy",
        choices=("deterministic_finite", "radlads_source"),
        default="deterministic_finite",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = generate_radlads_qrwkv_head_to_head_fixtures(
        args.out,
        seed=args.seed,
        overwrite=args.overwrite,
        radlads_source_path=args.radlads_repo,
        init_policy=args.init_policy,
    )
    print(
        f"wrote P53 head-to-head fixtures to {args.out} "
        f"with overall_status={manifest['radlads_load']['status']} "
        f"init_policy={manifest['init_policy']}"
    )


if __name__ == "__main__":
    main()
