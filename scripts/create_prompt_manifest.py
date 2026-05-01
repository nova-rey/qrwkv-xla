from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DESCRIPTION = "Tiny checked-in smoke prompt corpus."


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a prompt corpus manifest")
    parser.add_argument("corpus", help="Path to prompt corpus JSONL")
    parser.add_argument("--out", help="Output manifest path")
    parser.add_argument("--description", default="", help="Manifest description")
    parser.add_argument(
        "--note",
        action="append",
        default=None,
        help="Manifest note; may be repeated",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output manifest if it already exists",
    )
    args = parser.parse_args()

    from qrwkv_xla.prompting import (
        build_prompt_corpus_manifest,
        read_prompt_corpus,
        write_prompt_corpus_manifest,
    )

    corpus = read_prompt_corpus(args.corpus)
    output_path = (
        Path(args.out) if args.out else Path(args.corpus).with_suffix(".manifest.json")
    )
    description = args.description or DEFAULT_DESCRIPTION
    manifest = build_prompt_corpus_manifest(
        corpus,
        description=description,
        notes=list(args.note or ()),
    )
    write_prompt_corpus_manifest(manifest, output_path, overwrite=args.overwrite)
    print(f"manifest: {output_path}")
    print(f"corpus_id: {manifest.corpus_id}")
    print(f"prompt_count: {manifest.prompt_count}")
    print(f"sha256: {manifest.sha256}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Prompt manifest creation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
