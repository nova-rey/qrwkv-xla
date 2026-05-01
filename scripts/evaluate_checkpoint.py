from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from qrwkv_xla.eval import (
    load_eval_config,
    replace_eval_overrides,
    run_checkpoint_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a QRWKV-XLA checkpoint on fixed regression prompts"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/eval_regression_smoke.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = replace_eval_overrides(
        load_eval_config(args.config),
        max_new_tokens=args.max_new_tokens,
        prompt_limit=args.prompt_limit,
        output_dir=args.output_dir,
    )
    result = run_checkpoint_evaluation(
        checkpoint_dir=args.checkpoint,
        config=config,
        strict=args.strict,
    )
    payload = asdict(result)
    payload["eval_id"] = result.eval_id
    payload["checkpoint_dir"] = str(result.checkpoint_dir)
    payload["output_dir"] = str(result.output_dir)
    payload["generation_path"] = str(result.generation_path)
    payload["eval_json_path"] = str(result.eval_json_path)
    payload["sanity_path"] = str(result.sanity_path)

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"output_dir: {result.output_dir}")
        print(f"prompt_count: {result.prompt_count}")
        print(f"sanity_passed: {result.sanity_passed}")
        if not result.sanity_passed:
            print("sanity warnings: see sanity.json")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
