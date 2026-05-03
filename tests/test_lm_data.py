from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qrwkv_xla.lm import LMDataConfig, build_lm_batches, load_lm_token_sequences


def test_build_lm_batches_shifts_and_masks_tokens() -> None:
    batches = build_lm_batches(
        [[2, 3, 0], [4, 0]],
        sequence_length=3,
        batch_size=2,
    )

    assert len(batches) == 1
    batch = batches[0]
    np.testing.assert_array_equal(np.asarray(batch.input_ids), [[2, 3, 0], [4, 0, 0]])
    np.testing.assert_array_equal(np.asarray(batch.labels), [[3, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(
        np.asarray(batch.attention_mask),
        [[True, True, False], [True, False, False]],
    )
    np.testing.assert_array_equal(
        np.asarray(batch.label_mask),
        [[True, False, False], [False, False, False]],
    )


def test_build_lm_batches_pads_final_batch_to_static_shape() -> None:
    batches = build_lm_batches([[65, 0]], sequence_length=2, batch_size=2)

    assert np.asarray(batches[0].input_ids).shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(batches[0].label_mask)[1], [False, False])


def test_load_lm_token_sequences_uses_prompt_corpus_filters(tmp_path: Path) -> None:
    corpus_path = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "train-1", "text": "abc", "split": "train", "tags": ["ce"]},
        {"id": "val-1", "text": "def", "split": "validation", "tags": ["ce"]},
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    sequences = load_lm_token_sequences(
        LMDataConfig(
            prompt_corpus=corpus_path,
            prompt_split="train",
            prompt_tags=("ce",),
        )
    )

    assert len(sequences) == 1
    assert sequences[0][-1] == 0
