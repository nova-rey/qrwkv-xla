#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.fingerprint import write_fingerprint_provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a deterministic P152 fingerprint provenance sidecar."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument(
        "--artifact-role",
        choices=("training", "held_out_evaluation"),
        required=True,
    )
    parser.add_argument("--allow-legacy-positional-source-join", action="store_true")
    args = parser.parse_args()
    path = write_fingerprint_provenance(
        args.artifact,
        source_file=args.source_file,
        artifact_role=args.artifact_role,
        allow_legacy_positional_source_join=(args.allow_legacy_positional_source_join),
    )
    print("status=pass")
    print(f"provenance_path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
