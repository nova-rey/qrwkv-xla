from __future__ import annotations

import json
from pathlib import Path

import pytest

from qrwkv_xla.prompting import (
    PromptCorpus,
    PromptRecord,
    filter_prompt_corpus,
    read_prompt_corpus,
    validate_prompt_corpus,
    write_prompt_corpus,
)

ROOT = Path(__file__).resolve().parents[1]


def test_read_smoke_corpus() -> None:
    corpus = read_prompt_corpus(ROOT / "corpora" / "smoke_prompts.jsonl")

    assert corpus.corpus_id == "smoke_prompts"
    assert len(corpus.records) == 8
    assert corpus.records[0].id == "smoke_000001"
    assert corpus.records[0].text == "Explain recurrence in one sentence."


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"ok","text":"hello"}\n{"id":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_prompt_corpus(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dupe.jsonl"
    path.write_text(
        (
            json.dumps({"id": "x", "text": "one"})
            + "\n"
            + json.dumps({"id": "x", "text": "two"})
            + "\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicated"):
        read_prompt_corpus(path)


def test_missing_id_or_text_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "text": ""}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="text"):
        read_prompt_corpus(path)


def test_filter_by_split_tag_and_limit() -> None:
    corpus = read_prompt_corpus(ROOT / "corpora" / "smoke_prompts.jsonl")
    train = filter_prompt_corpus(corpus, split="train")
    tagged = filter_prompt_corpus(corpus, tags=["xla"])
    limited = filter_prompt_corpus(corpus, split="train", limit=2)

    assert len(train.records) == 6
    assert len(tagged.records) == 1
    assert tagged.records[0].id == "smoke_000006"
    assert [record.id for record in limited.records] == ["smoke_000001", "smoke_000002"]


def test_write_read_round_trip(tmp_path: Path) -> None:
    corpus = PromptCorpus(
        corpus_id="round_trip",
        records=(
            PromptRecord(
                id="a",
                text="alpha",
                split="train",
                tags=("tag",),
                metadata={"source": "test"},
            ),
            PromptRecord(id="b", text="beta", split="validation"),
        ),
    )
    path = tmp_path / "round_trip.jsonl"

    write_prompt_corpus(corpus, path, overwrite=True)
    loaded = read_prompt_corpus(path, corpus_id="round_trip")
    validate_prompt_corpus(loaded)

    assert loaded.records == corpus.records
