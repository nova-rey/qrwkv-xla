from __future__ import annotations

import argparse
import json
import sys

from qrwkv_xla.eval.compare import (
    compare_eval_snapshots,
    eval_comparison_to_dict,
    write_eval_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two QRWKV-XLA evaluation snapshots"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = compare_eval_snapshots(
        baseline_dir=args.baseline,
        candidate_dir=args.candidate,
    )
    if args.out:
        write_eval_comparison(result, args.out)
    payload = eval_comparison_to_dict(result)

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"prompt_count: {result.prompt_count}")
        print(f"same_count: {result.same_count}")
        print(f"changed_count: {result.changed_count}")
        print(f"missing_prompt_ids: {list(result.missing_prompt_ids)}")
        if args.out:
            print(f"comparison_path: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
