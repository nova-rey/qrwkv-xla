from __future__ import annotations

import random
from dataclasses import dataclass

import jax.numpy as jnp

from qrwkv_xla.generation.tokenizer import SmokeTokenizer
from qrwkv_xla.lm.config import LMDataConfig
from qrwkv_xla.prompting import filter_prompt_corpus, read_prompt_corpus


@dataclass(frozen=True)
class LMBatch:
    input_ids: jnp.ndarray
    labels: jnp.ndarray
    attention_mask: jnp.ndarray
    label_mask: jnp.ndarray


def load_lm_token_sequences(config: LMDataConfig) -> list[list[int]]:
    if config.tokenizer != "smoke":
        raise ValueError("Only the smoke tokenizer is supported for Stage 3")
    corpus = read_prompt_corpus(config.prompt_corpus)
    filtered = filter_prompt_corpus(
        corpus,
        split=config.prompt_split,
        tags=config.prompt_tags,
        limit=config.prompt_limit,
    )
    if not filtered.records:
        raise ValueError("Prompt corpus selection produced no records")
    tokenizer = SmokeTokenizer()
    sequences: list[list[int]] = []
    for record in filtered.records:
        encoded = tokenizer.encode(record.text)
        if not encoded or encoded[-1] != tokenizer.eos_token_id:
            encoded.append(tokenizer.eos_token_id)
        sequences.append(encoded)
    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(sequences)
    return sequences


def build_lm_batches(
    token_sequences: list[list[int]],
    *,
    sequence_length: int,
    batch_size: int,
    pad_token_id: int = 0,
    eos_token_id: int = 0,
    drop_last: bool = False,
) -> list[LMBatch]:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be > 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not token_sequences:
        raise ValueError("token_sequences must be non-empty")

    example_length = sequence_length + 1
    examples = [
        _normalize_sequence(
            sequence,
            example_length=example_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        for sequence in token_sequences
    ]

    batches: list[LMBatch] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        if len(chunk) < batch_size:
            if drop_last:
                continue
            chunk = [
                *chunk,
                *(
                    [pad_token_id] * example_length
                    for _ in range(batch_size - len(chunk))
                ),
            ]
        tokens = jnp.asarray(chunk, dtype=jnp.int32)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        attention_mask = input_ids != pad_token_id
        label_mask = labels != pad_token_id
        batches.append(
            LMBatch(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                label_mask=label_mask,
            )
        )
    if not batches:
        raise ValueError("No LM batches were built")
    return batches


def _normalize_sequence(
    sequence: list[int],
    *,
    example_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> list[int]:
    tokens = [int(token_id) for token_id in sequence]
    if not tokens or tokens[-1] != eos_token_id:
        tokens.append(eos_token_id)
    tokens = tokens[:example_length]
    if len(tokens) < example_length:
        tokens.extend([pad_token_id] * (example_length - len(tokens)))
    return tokens
