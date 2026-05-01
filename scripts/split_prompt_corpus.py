from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic prompt splits")
    parser.add_argument("corpus", help="Input prompt corpus JSONL")
    parser.add_argument("--out", required=True, help="Output prompt corpus JSONL")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction assigned to validation",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Fraction assigned to test",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output corpus if it already exists",
    )
    args = parser.parse_args()

    from qrwkv_xla.prompting import (
        assign_splits,
        read_prompt_corpus,
        write_prompt_corpus,
    )

    corpus = read_prompt_corpus(args.corpus)
    split_corpus = assign_splits(
        corpus,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    output_path = write_prompt_corpus(
        split_corpus,
        Path(args.out),
        overwrite=args.overwrite,
    )
    print(f"corpus: {output_path}")
    print(f"prompt_count: {len(split_corpus.records)}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Prompt corpus split failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
