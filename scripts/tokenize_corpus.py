from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize and pack a prompt JSONL corpus for Stage 3 LM training"
    )
    parser.add_argument("prompt_corpus", nargs="?")
    parser.add_argument("--input", dest="prompt_corpus_input")
    parser.add_argument("--out", "--output", dest="output_dir", required=True)
    parser.add_argument("--prompt-split", default="train")
    parser.add_argument("--prompt-tag", action="append", default=[])
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--shard-size-tokens", type=int, default=4096)
    parser.add_argument(
        "--tokenizer-backend", "--tokenizer", dest="tokenizer_backend", default="smoke"
    )
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-vocab-size", type=int)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--pad-token-id", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prompt_corpus = args.prompt_corpus_input or args.prompt_corpus
    if prompt_corpus is None:
        parser.error("prompt corpus path is required")
    if args.shard_size_tokens <= 0:
        parser.error("--shard-size-tokens must be > 0")

    from qrwkv_xla.generation import TokenizerConfig, create_tokenizer
    from qrwkv_xla.lm.tokenized_corpus import write_tokenized_corpus_from_prompt_jsonl

    tokenizer = create_tokenizer(
        TokenizerConfig(
            backend=args.tokenizer_backend,
            tokenizer_id=args.tokenizer_id,
            vocab_size=args.tokenizer_vocab_size,
            eos_token_id=args.eos_token_id,
            pad_token_id=args.pad_token_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_fast=not args.slow_tokenizer,
        )
    )
    manifest = write_tokenized_corpus_from_prompt_jsonl(
        prompt_corpus,
        Path(args.output_dir),
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        prompt_split=args.prompt_split,
        prompt_tags=tuple(args.prompt_tag),
        prompt_limit=args.prompt_limit,
        shard_size_tokens=args.shard_size_tokens,
        overwrite=args.overwrite,
    )
    print("Wrote tokenized corpus:")
    print(f" output: {Path(args.output_dir)}")
    print(f" tokenizer: {manifest.tokenizer.backend}")
    print(f" sequence_length: {manifest.packing.sequence_length}")
    print(f" sequences: {manifest.totals.num_sequences}")
    print(f" tokens: {manifest.totals.num_tokens}")
    print(f" shards: {manifest.totals.num_shards}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Tokenize corpus failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
