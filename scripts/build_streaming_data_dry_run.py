from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrwkv_xla.data import build_streaming_dataset_from_tokenized_corpus
from qrwkv_xla.data.streaming_reports import write_markdown_report
from qrwkv_xla.generation import SmokeTokenizer
from qrwkv_xla.lm.tokenized_corpus import write_tokenized_corpus_from_prompt_jsonl

DEFAULT_OUT = Path("artifacts/data/p44_streaming_dry_run")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the P44 larger/local streaming data dry-run dataset"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--num-documents", type=int, default=1024)
    parser.add_argument("--total-tokens", type=int, default=131072)
    parser.add_argument("--shard-tokens", type=int, default=32768)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_documents <= 0:
        parser.error("--num-documents must be > 0")
    if args.total_tokens <= 0:
        parser.error("--total-tokens must be > 0")
    if args.shard_tokens <= 0:
        parser.error("--shard-tokens must be > 0")
    if args.seq_len <= 1:
        parser.error("--seq-len must be > 1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "synthetic_prompts.jsonl"
    tokenized_dir = out_dir / "_tokenized_corpus"

    approx_chars_per_doc = max(
        args.seq_len * 2,
        args.total_tokens // args.num_documents,
    )
    _write_synthetic_prompts(
        prompt_path,
        num_documents=args.num_documents,
        approx_chars_per_doc=approx_chars_per_doc,
    )
    tokenized_manifest = write_tokenized_corpus_from_prompt_jsonl(
        prompt_path,
        tokenized_dir,
        tokenizer=SmokeTokenizer(),
        sequence_length=args.seq_len,
        shard_size_tokens=args.shard_tokens,
        overwrite=args.overwrite,
        created_at="2026-05-08T00:00:00+00:00",
    )
    manifest = build_streaming_dataset_from_tokenized_corpus(
        tokenized_dir,
        out_dir,
        num_documents=args.num_documents,
        shard_tokens=args.shard_tokens,
        overwrite=args.overwrite,
        created_at_utc="2026-05-08T00:00:00+00:00",
        notes=(
            "P44 larger/local streaming data pipeline dry-run dataset.",
            "Streaming iterator reuses prepacked tokenized-corpus sequences.",
        ),
    )

    summary = {
        "phase": "P44",
        "status": "pass",
        "artifact_dir": str(out_dir),
        "num_documents": args.num_documents,
        "num_shards": len(manifest.shards),
        "total_tokens": manifest.corpus.num_tokens,
        "tokenized_total_tokens": tokenized_manifest.totals.num_tokens,
        "num_sequences": manifest.corpus.num_sequences,
        "shard_tokens": args.shard_tokens,
        "sequence_length": args.seq_len,
        "boundary_policy": manifest.corpus.boundary_policy,
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        out_dir / "P44_DATASET_SUMMARY.md",
        title="P44 Dataset Summary",
        sections=[
            (
                "Dataset",
                [
                    f"phase: {summary['phase']}",
                    f"status: {summary['status']}",
                    f"num_documents: {summary['num_documents']}",
                    f"num_shards: {summary['num_shards']}",
                    f"total_tokens: {summary['total_tokens']}",
                    f"num_sequences: {summary['num_sequences']}",
                    f"shard_tokens: {summary['shard_tokens']}",
                    f"sequence_length: {summary['sequence_length']}",
                    f"boundary_policy: {summary['boundary_policy']}",
                ],
            ),
            (
                "Artifacts",
                [
                    f"manifest: {out_dir / 'manifest.json'}",
                    f"shards_dir: {out_dir / 'shards'}",
                    f"synthetic_prompts: {prompt_path}",
                ],
            ),
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_synthetic_prompts(
    path: Path,
    *,
    num_documents: int,
    approx_chars_per_doc: int,
) -> None:
    lines: list[str] = []
    base_phrase = "p44 streaming dry run offline smoke tokenizer payload "
    for index in range(num_documents):
        repeat = max(1, approx_chars_per_doc // len(base_phrase))
        text = (base_phrase * repeat) + f"document-{index:04d}"
        lines.append(
            json.dumps(
                {
                    "id": f"p44-{index:04d}",
                    "split": "train",
                    "tags": ["p44", "streaming"],
                    "text": text,
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
