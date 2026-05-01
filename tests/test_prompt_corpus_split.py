from __future__ import annotations

from collections import Counter

from qrwkv_xla.prompting import PromptCorpus, PromptRecord, assign_splits


def _corpus(count: int) -> PromptCorpus:
    return PromptCorpus(
        corpus_id="tiny",
        records=tuple(
            PromptRecord(id=f"id_{index}", text=f"prompt {index}")
            for index in range(count)
        ),
    )


def test_split_is_deterministic() -> None:
    corpus = _corpus(10)

    first = assign_splits(corpus, validation_fraction=0.2, seed=7)
    second = assign_splits(corpus, validation_fraction=0.2, seed=7)

    assert first.records == second.records


def test_small_corpus_gets_validation_example() -> None:
    corpus = assign_splits(_corpus(2), validation_fraction=0.1, seed=1)
    counts = Counter(record.split for record in corpus.records)

    assert counts["validation"] == 1
    assert counts["train"] == 1


def test_all_records_get_valid_split_and_original_is_unchanged() -> None:
    original = _corpus(5)
    split = assign_splits(original, validation_fraction=0.2, test_fraction=0.2, seed=3)

    assert all(
        record.split in {"train", "validation", "test"} for record in split.records
    )
    assert all(record.split == "unspecified" for record in original.records)
