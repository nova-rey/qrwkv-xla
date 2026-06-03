#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrwkv_xla.xla import run_runtime_environment_preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P109 runtime environment preflight."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/p109_runtime_environment/runtime_environment_report.json"
        ),
    )
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--enable-transparent-hugepages", action="store_true")
    parser.add_argument(
        "--hugepage-path",
        type=Path,
        default=Path("/sys/kernel/mm/transparent_hugepage/enabled"),
    )
    args = parser.parse_args()

    report = run_runtime_environment_preflight(
        hugepage_path=args.hugepage_path,
        require_tpu=args.require_tpu,
        enable_hugepages=args.enable_transparent_hugepages,
    )
    payload = report.to_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"status={report.status} jax_available={report.jax_available} "
        f"tpu_detected={report.tpu_devices_detected} "
        f"transparent_hugepages={report.transparent_hugepages.status} "
        f"report={args.output}"
    )
    return 1 if report.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
