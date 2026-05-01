from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a prompt corpus JSONL file")
    parser.add_argument("corpus", help="Path to prompt corpus JSONL")
    parser.add_argument("--split", help="Only include this split")
    parser.add_argument("--tag", action="append", default=None, help="Required tag")
    parser.add_argument("--limit", type=int, help="Maximum prompts to include")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    from qrwkv_xla.prompting import (
        build_prompt_corpus_manifest,
        filter_prompt_corpus,
        read_prompt_corpus,
    )

    corpus = read_prompt_corpus(args.corpus)
    filtered = filter_prompt_corpus(
        corpus,
        split=args.split,
        tags=list(args.tag or ()),
        limit=args.limit,
    )
    if not filtered.records:
        raise ValueError("prompt corpus filters resolved to an empty prompt list")
    manifest = build_prompt_corpus_manifest(filtered)
    preview = [
        {"id": record.id, "text": record.text[:80]}
        for record in filtered.records[: min(5, len(filtered.records))]
    ]
    summary = {
        "corpus_id": filtered.corpus_id,
        "source_path": str(filtered.source_path) if filtered.source_path else None,
        "prompt_count": len(filtered.records),
        "splits": manifest.splits,
        "tags": list(manifest.tags),
        "sha256": manifest.sha256,
        "preview": preview,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"corpus_id: {summary['corpus_id']}")
    print(f"source_path: {summary['source_path']}")
    print(f"prompt_count: {summary['prompt_count']}")
    print(f"splits: {summary['splits']}")
    print(f"tags: {', '.join(summary['tags']) if summary['tags'] else '<none>'}")
    print(f"sha256: {summary['sha256']}")
    for item in summary["preview"]:
        print(f"- {item['id']}: {item['text']}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Prompt corpus inspection failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
