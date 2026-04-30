from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.config import load_config
from qrwkv_xla.xla import format_jax_runtime_info, get_jax_runtime_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a QRWKV-XLA TPU environment smoke"
    )
    parser.add_argument("--config", default="configs/tiny_tpu_smoke.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)

    print(f"Loaded config backend: {config.runtime.backend}")
    print(f"Configured sequence length: {config.model.sequence_length}")

    runtime = get_jax_runtime_info()
    print(format_jax_runtime_info(runtime))

    if not runtime.has_tpu:
        print(
            "No TPU detected; TPU smoke skipped. "
            "Use --require-tpu scripts for hard failure."
        )
        raise SystemExit(0)

    print("TPU detected; TPU smoke environment looks ready.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"TPU smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
