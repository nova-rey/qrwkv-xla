from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.generation import SmokeTokenizer
from qrwkv_xla.generation.tokenizer import TokenizerConfig
from qrwkv_xla.lm.tokenized_corpus import (
    TOKENIZED_CORPUS_FORMAT,
    LoadedTokenizedCorpus,
    build_tokenized_sequences,
    load_tokenized_corpus,
    read_tokenized_corpus_manifest,
    write_tokenized_corpus_from_prompt_jsonl,
)
from qrwkv_xla.prompting import PromptCorpus, PromptRecord

FIXED_CREATED_AT = "2026-05-03T22:20:00+00:00"


def test_write_and_load_tokenized_corpus_manifest_and_arrays(tmp_path: Path) -> None:
    corpus_path = _write_prompt_corpus(tmp_path / "prompts.jsonl")
    tokenizer = SmokeTokenizer()

    manifest = write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        tmp_path / "tok",
        tokenizer=tokenizer,
        sequence_length=3,
        shard_size_tokens=6,
        overwrite=True,
        created_at=FIXED_CREATED_AT,
    )
    loaded = load_tokenized_corpus(
        tmp_path / "tok",
        expected_sequence_length=3,
        expected_tokenizer=TokenizerConfig(backend="smoke"),
    )

    assert manifest.format == TOKENIZED_CORPUS_FORMAT
    assert manifest.format_version == 1
    assert manifest.created_at == FIXED_CREATED_AT
    assert manifest.source.kind == "jsonl_prompts"
    assert manifest.source.record_count == 2
    assert manifest.source.selected_count == 2
    assert len(manifest.source.sha256) == 64
    assert manifest.packing.sequence_length == 3
    assert manifest.packing.stride == 3
    assert manifest.totals.num_shards == 2
    assert manifest.totals.num_sequences == 3
    assert manifest.totals.num_tokens == 8
    assert manifest.shards[0].path == "shards/shard-00000.npz"
    assert loaded.input_ids.shape == (3, 3)
    assert loaded.labels.shape == (3, 3)
    assert loaded.attention_mask.shape == (3, 3)
    assert loaded.loss_mask.shape == (3, 3)
    np.testing.assert_array_equal(
        loaded.input_ids,
        np.asarray(
            [[98, 109, 113], [105, 98, 0], [99, 102, 117]],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        loaded.labels,
        np.asarray(
            [[109, 113, 105], [98, 0, 99], [102, 117, 98]],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        loaded.attention_mask,
        np.asarray([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        loaded.loss_mask,
        np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32),
    )
    assert loaded.token_sequences[0] == (98, 109, 113, 105)


def test_tokenized_corpus_determinism_uses_stable_hashes(tmp_path: Path) -> None:
    corpus_path = _write_prompt_corpus(tmp_path / "prompts.jsonl")
    tokenizer = SmokeTokenizer()

    first = write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        tmp_path / "first",
        tokenizer=tokenizer,
        sequence_length=3,
        shard_size_tokens=6,
        overwrite=True,
        created_at=FIXED_CREATED_AT,
    )
    second = write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        tmp_path / "second",
        tokenizer=tokenizer,
        sequence_length=3,
        shard_size_tokens=6,
        overwrite=True,
        created_at=FIXED_CREATED_AT,
    )

    assert first.source == second.source
    assert first.tokenizer == second.tokenizer
    assert first.packing == second.packing
    assert first.shards == second.shards
    assert first.totals == second.totals


def test_build_tokenized_sequences_filters_records() -> None:
    corpus = PromptCorpus(
        corpus_id="unit",
        records=(
            PromptRecord(id="keep", text="a", split="train", tags=("ce",)),
            PromptRecord(id="drop", text="b", split="validation", tags=("ce",)),
        ),
    )

    sequences = build_tokenized_sequences(
        corpus,
        SmokeTokenizer(),
        sequence_length=2,
        prompt_split="train",
        prompt_tags=("ce",),
        drop_remainder=False,
    )

    assert sequences == ((98, 0, 0),)


def test_load_rejects_sequence_length_mismatch(tmp_path: Path) -> None:
    _write_basic_tokenized_corpus(tmp_path)

    with pytest.raises(ValueError, match="sequence_length mismatch"):
        load_tokenized_corpus(tmp_path / "tok", expected_sequence_length=4)


def test_load_rejects_missing_shard(tmp_path: Path) -> None:
    _write_basic_tokenized_corpus(tmp_path)
    (tmp_path / "tok" / "shards" / "shard-00000.npz").unlink()

    with pytest.raises(ValueError, match="shard is missing"):
        load_tokenized_corpus(tmp_path / "tok")


def test_load_rejects_wrong_format_version(tmp_path: Path) -> None:
    _write_basic_tokenized_corpus(tmp_path)
    manifest_path = tmp_path / "tok" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["format_version"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="format_version"):
        read_tokenized_corpus_manifest(manifest_path)


def test_load_rejects_tokenizer_vocab_mismatch(tmp_path: Path) -> None:
    _write_basic_tokenized_corpus(tmp_path)

    with pytest.raises(ValueError, match="vocab_size mismatch"):
        load_tokenized_corpus(
            tmp_path / "tok",
            expected_tokenizer=replace(SmokeTokenizer().metadata, vocab_size=513),
        )


def test_write_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    corpus_path = _write_prompt_corpus(tmp_path / "prompts.jsonl")
    tokenizer = SmokeTokenizer()
    write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        tmp_path / "tok",
        tokenizer=tokenizer,
        sequence_length=3,
        overwrite=True,
        created_at=FIXED_CREATED_AT,
    )

    with pytest.raises(ValueError, match="already exists"):
        write_tokenized_corpus_from_prompt_jsonl(
            corpus_path,
            tmp_path / "tok",
            tokenizer=tokenizer,
            sequence_length=3,
            created_at=FIXED_CREATED_AT,
        )


def test_load_rejects_malformed_shard_shape(tmp_path: Path) -> None:
    _write_basic_tokenized_corpus(tmp_path)
    np.savez_compressed(
        tmp_path / "tok" / "shards" / "shard-00000.npz",
        input_ids=np.asarray([[1, 2]], dtype=np.int32),
        labels=np.asarray([[2, 3]], dtype=np.int32),
        attention_mask=np.asarray([[1, 1]], dtype=np.int32),
        loss_mask=np.asarray([[1, 1]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        load_tokenized_corpus(tmp_path / "tok")


def _write_basic_tokenized_corpus(tmp_path: Path) -> LoadedTokenizedCorpus:
    manifest = write_tokenized_corpus_from_prompt_jsonl(
        _write_prompt_corpus(tmp_path / "prompts.jsonl"),
        tmp_path / "tok",
        tokenizer=SmokeTokenizer(),
        sequence_length=3,
        shard_size_tokens=6,
        overwrite=True,
        created_at=FIXED_CREATED_AT,
    )
    assert manifest.totals.num_sequences == 3
    return load_tokenized_corpus(tmp_path / "tok")


def _write_prompt_corpus(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "text": "alpha", "split": "train"}),
                json.dumps({"id": "two", "text": "beta", "split": "train"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
