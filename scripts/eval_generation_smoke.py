from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qrwkv_xla.generation.eval_smoke import (
    load_generation_smoke_config,
    load_prompts_from_config,
    run_generation_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny QRWKV-XLA generation smoke evaluation"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/generation_smoke.yaml")
    parser.add_argument("--prompt-corpus")
    parser.add_argument("--prompt-split")
    parser.add_argument("--prompt-tag", action="append", default=[])
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--vocab-size", type=int)
    args = parser.parse_args()

    config = load_generation_smoke_config(args.config)
    prompt_tags = args.prompt_tag if args.prompt_tag else None
    prompts = load_prompts_from_config(
        config,
        prompt_corpus=args.prompt_corpus,
        prompt_split=args.prompt_split,
        prompt_tags=prompt_tags,
        prompt_limit=args.prompt_limit,
    )
    output_dir = Path(args.output_dir or config.get("output_dir", "eval_outputs"))
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(config.get("max_new_tokens", 16))
    )
    result = run_generation_smoke(
        checkpoint_dir=args.checkpoint,
        prompts=prompts,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
        vocab_size=args.vocab_size,
    )
    print(f"output_dir: {result.output_dir}")
    print(f"prompt_count: {result.prompt_count}")
    print(f"passed: {result.passed}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Generation smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
