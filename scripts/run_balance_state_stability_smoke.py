#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwkv_xla.parity.radlads_balance_state_experiment import (
    DEFAULT_EXPERIMENT_OUT,
    run_balance_state_stability_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P65 run a tiny local stability smoke for balance-state mode."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EXPERIMENT_OUT)
    parser.add_argument("--seed", type=int, default=6500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = run_balance_state_stability_smoke(
        out_dir=args.out_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"wrote P65 balance-state stability smoke to {args.out_dir}")
    print(f"status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
