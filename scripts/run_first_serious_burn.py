#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from qrwkv_xla.burn import (
    default_first_serious_burn_config,
    load_first_serious_burn_config,
    run_first_serious_burn,
    write_first_serious_burn_config,
    write_first_serious_burn_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the P112 first serious compute burn harness."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("artifacts/p112_first_serious_burn/burn_config.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p112_first_serious_burn/dry_run"),
    )
    parser.add_argument("--mode", choices=("dry_run", "real"), default="dry_run")
    parser.add_argument("--confirm-serious-burn", action="store_true")
    parser.add_argument("--teacher-textbook", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--allow-textbook-reuse",
        dest="allow_textbook_reuse",
        action="store_true",
    )
    parser.add_argument(
        "--no-allow-textbook-reuse",
        dest="allow_textbook_reuse",
        action="store_false",
    )
    parser.set_defaults(allow_textbook_reuse=None)
    args = parser.parse_args()

    config_missing = not args.config.exists()
    if args.config.exists():
        config = load_first_serious_burn_config(args.config)
        config = replace(config, mode=args.mode, output_dir=str(args.output))
    else:
        config = default_first_serious_burn_config(
            output_dir=args.output,
            mode=args.mode,
        )
    if args.teacher_textbook is not None:
        config = replace(config, teacher_textbook_path=str(args.teacher_textbook))
    if args.max_steps is not None:
        config = replace(config, max_steps=args.max_steps)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.allow_textbook_reuse is not None:
        config = replace(config, allow_textbook_reuse=args.allow_textbook_reuse)
    if config_missing:
        write_first_serious_burn_config(config, args.config)

    result = run_first_serious_burn(
        config,
        confirm_serious_burn=args.confirm_serious_burn,
    )
    report_path = args.output / "burn_report.json"
    write_first_serious_burn_report(result, report_path)
    print(
        f"status={result.status} mode={result.mode} dry_run={result.dry_run} "
        f"readiness={result.readiness_status} blockers={len(result.blockers)} "
        f"warnings={len(result.warnings)} report={report_path}"
    )
    return 0 if result.status in {"dry_run_pass", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
