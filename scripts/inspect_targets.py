from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a teacher target bundle")
    parser.add_argument("bundle_dir", help="Bundle directory to inspect")
    args = parser.parse_args()

    _add_src_to_path()

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
