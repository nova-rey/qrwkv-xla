from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qrwkv_xla.tracking.reports import P47_DEFAULT_ARTIFACT_DIR
from qrwkv_xla.tracking.smoke import TrackingSmokeConfig, run_tracking_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P47 experiment tracking smoke."
    )
    parser.add_argument(
        "--tracking",
        choices=("local", "wandb-offline", "wandb-online"),
        default="local",
    )
    parser.add_argument("--out", type=Path, default=P47_DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--project", default="qrwkv-xla")
    parser.add_argument("--entity")
    parser.add_argument("--run-name", default="p47-experiment-tracking-smoke")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = TrackingSmokeConfig(
        tracking=args.tracking,
        out=args.out,
        project=args.project,
        entity=args.entity,
        run_name=args.run_name,
        overwrite=args.overwrite,
        steps=args.steps,
    )
    report = run_tracking_smoke(config, command=sys.argv)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print("P47 experiment tracking smoke complete")
    print(f"status: {report['status']}")
    print(f"tracking_mode: {args.tracking}")
    print(report["paths"]["report_markdown"])
    print(report["paths"]["report_json"])


if __name__ == "__main__":
    main()
