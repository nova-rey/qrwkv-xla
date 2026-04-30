from __future__ import annotations

import argparse
import sys

from qrwkv_xla.xla import format_jax_runtime_info, get_jax_runtime_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the current JAX/XLA runtime")
    parser.add_argument(
        "--require-tpu",
        action="store_true",
        help="Exit nonzero when no TPU is detected.",
    )
    args = parser.parse_args()

    info = get_jax_runtime_info()
    print(format_jax_runtime_info(info))
    if args.require_tpu and not info.has_tpu:
        print("TPU required but not detected.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
