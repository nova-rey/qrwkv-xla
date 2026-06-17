#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import validate_fingerprint_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a behavioral_fingerprint artifact."
    )
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    result = validate_fingerprint_artifact(args.artifact)
    metadata = result.metadata
    print(
        f"status={result.status} "
        f"artifact_type={metadata.get('artifact_type', 'unknown')} "
        f"version={metadata.get('artifact_version', 'unknown')} "
        f"shards={metadata.get('shards', 0)} "
        f"records={metadata.get('records', 0)} "
        f"modes={metadata.get('modes', 0)} "
        f"warnings={len(result.warnings)}"
    )
    for blocker in result.blockers:
        print(f"BLOCKER: {blocker}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
