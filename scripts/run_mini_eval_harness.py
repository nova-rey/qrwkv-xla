#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.eval import (
    create_builtin_mini_eval_store,
    run_mini_eval_harness,
    write_mini_eval_report,
)
from qrwkv_xla.targets import TeacherTargetStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the P110 mini eval harness.")
    parser.add_argument("--target-store", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p110_mini_eval/mini_eval_report.json"),
    )
    parser.add_argument("--architecture-id", default="tiny_debug")
    parser.add_argument("--runtime", default=None)
    args = parser.parse_args()

    if args.target_store is None:
        store = create_builtin_mini_eval_store(
            args.output.parent / "builtin_target_store"
        )
    else:
        store = TeacherTargetStore.open(args.target_store)

    result = run_mini_eval_harness(
        store=store,
        architecture_id=args.architecture_id,
        runtime=args.runtime,
    )
    write_mini_eval_report(result, args.output)
    print(
        f"status={result.status} mean_mse_loss={result.mean_mse_loss} "
        f"examples={result.examples_evaluated} shards={result.shard_count} "
        f"report={args.output}"
    )
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
