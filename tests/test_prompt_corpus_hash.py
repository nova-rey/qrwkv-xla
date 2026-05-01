from __future__ import annotations

from pathlib import Path

from qrwkv_xla.prompting import (
    PromptCorpus,
    PromptRecord,
    build_prompt_corpus_manifest,
    compute_prompt_corpus_hash,
    read_prompt_corpus,
    write_prompt_corpus,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hash_is_stable_across_read_write(tmp_path: Path) -> None:
    corpus = read_prompt_corpus(ROOT / "corpora" / "smoke_prompts.jsonl")
    original_hash = compute_prompt_corpus_hash(corpus)

    output = tmp_path / "rewrite.jsonl"
    write_prompt_corpus(corpus, output, overwrite=True)
    rewritten = read_prompt_corpus(output, corpus_id=corpus.corpus_id)

    assert compute_prompt_corpus_hash(rewritten) == original_hash


def test_hash_changes_when_text_changes() -> None:
    base = PromptCorpus(corpus_id="base", records=(PromptRecord(id="a", text="alpha"),))
    changed = PromptCorpus(
        corpus_id="base", records=(PromptRecord(id="a", text="beta"),)
    )

    assert compute_prompt_corpus_hash(base) != compute_prompt_corpus_hash(changed)


def test_hash_changes_when_order_changes() -> None:
    first = PromptCorpus(
        corpus_id="ordered",
        records=(PromptRecord(id="a", text="alpha"), PromptRecord(id="b", text="beta")),
    )
    second = PromptCorpus(corpus_id="ordered", records=tuple(reversed(first.records)))

    assert compute_prompt_corpus_hash(first) != compute_prompt_corpus_hash(second)


def test_manifest_counts_splits_and_tags() -> None:
    corpus = read_prompt_corpus(ROOT / "corpora" / "smoke_prompts.jsonl")
    manifest = build_prompt_corpus_manifest(corpus, description="Smoke corpus")

    assert manifest.prompt_count == 8
    assert manifest.splits == {"train": 6, "validation": 2}
    assert manifest.tags == (
        "code",
        "generation",
        "greeting",
        "jax",
        "reasoning",
        "recurrence",
        "smoke",
        "teacher",
        "xla",
    )
