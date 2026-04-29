from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

REQUIRED_SHARD_KEYS = ("input_ids", "attention_mask", "hidden_states")


def write_shard(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(Path(path), **arrays)


def read_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def validate_shard_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
) -> None:
    missing = [key for key in REQUIRED_SHARD_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Missing required shard keys: {missing}")

    input_ids = np.asarray(arrays["input_ids"])
    attention_mask = np.asarray(arrays["attention_mask"])
    hidden_states = np.asarray(arrays["hidden_states"])

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be rank 2")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank 2")
    if hidden_states.ndim != 4:
        raise ValueError("hidden_states must be rank 4")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids.shape must match attention_mask.shape")
    if input_ids.shape[1] != sequence_length:
        actual = input_ids.shape[1]
        raise ValueError(
            f"input_ids sequence_length must be {sequence_length}, got {actual}"
        )

    batch_size = input_ids.shape[0]
    if hidden_states.shape[0] != batch_size:
        raise ValueError("hidden_states batch dimension must match input_ids")
    if hidden_states.shape[1] != num_layers:
        actual = hidden_states.shape[1]
        raise ValueError(f"hidden_states num_layers must be {num_layers}, got {actual}")
    if hidden_states.shape[2] != sequence_length:
        raise ValueError(
            "hidden_states sequence_length must match manifest sequence_length"
        )
    if hidden_states.shape[3] != hidden_size:
        actual = hidden_states.shape[3]
        raise ValueError(
            f"hidden_states hidden_size must be {hidden_size}, got {actual}"
        )

    if "logits" in arrays:
        logits = np.asarray(arrays["logits"])
        if logits.ndim != 3:
            raise ValueError("logits must be rank 3")
        if logits.shape[0] != batch_size:
            raise ValueError("logits batch dimension must match input_ids")
        if logits.shape[1] != sequence_length:
            raise ValueError("logits sequence_length must match manifest")

    if "attention_targets" in arrays:
        attention_targets = np.asarray(arrays["attention_targets"])
        if attention_targets.ndim < 2:
            raise ValueError("attention_targets must have at least 2 dimensions")
        if attention_targets.shape[0] != batch_size:
            raise ValueError("attention_targets batch dimension must match input_ids")
        if attention_targets.shape[1] != sequence_length:
            raise ValueError("attention_targets sequence_length must match manifest")
