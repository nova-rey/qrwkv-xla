from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a teacher target bundle")
    parser.add_argument("bundle_dir", help="Bundle directory to inspect")
    args = parser.parse_args()

    from qrwkv_xla.targets import inspect_target_bundle

    summary = inspect_target_bundle(args.bundle_dir)
    print("Target bundle summary")
    print("====================")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Invalid target bundle: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
