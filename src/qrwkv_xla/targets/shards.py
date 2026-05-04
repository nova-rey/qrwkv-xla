from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

REQUIRED_SHARD_KEYS = ("input_ids", "attention_mask", "loss_mask")


def write_shard(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(Path(path), **arrays)


def read_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def hash_shard_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_shard_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    require_hidden_states: bool = True,
    require_logits: bool = False,
    require_attention_targets: bool = False,
) -> None:
    missing = [key for key in REQUIRED_SHARD_KEYS if key not in arrays]
    if require_hidden_states and "hidden_states" not in arrays:
        missing.append("hidden_states")
    if require_logits and "logits" not in arrays:
        missing.append("logits")
    if require_attention_targets and "attention_targets" not in arrays:
        missing.append("attention_targets")
    if missing:
        raise ValueError(f"Missing required shard keys: {missing}")

    input_ids = np.asarray(arrays["input_ids"])
    attention_mask = np.asarray(arrays["attention_mask"])
    loss_mask = np.asarray(arrays["loss_mask"])

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be rank 2")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank 2")
    if loss_mask.ndim != 2:
        raise ValueError("loss_mask must be rank 2")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids.shape must match attention_mask.shape")
    if input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids.shape must match loss_mask.shape")
    if input_ids.shape[1] != sequence_length:
        actual = input_ids.shape[1]
        raise ValueError(
            f"input_ids sequence_length must be {sequence_length}, got {actual}"
        )

    batch_size = input_ids.shape[0]
    if "hidden_states" in arrays:
        hidden_states = np.asarray(arrays["hidden_states"])
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must be rank 4")
        if hidden_states.shape[0] != batch_size:
            raise ValueError("hidden_states batch dimension must match input_ids")
        if hidden_states.shape[1] != num_layers:
            actual = hidden_states.shape[1]
            raise ValueError(
                f"hidden_states num_layers must be {num_layers}, got {actual}"
            )
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
        if attention_targets.ndim != 4:
            raise ValueError("attention_targets must be rank 4")
        if attention_targets.shape[0] != batch_size:
            raise ValueError("attention_targets batch dimension must match input_ids")
        if attention_targets.shape[1] != num_layers:
            actual = attention_targets.shape[1]
            raise ValueError(
                f"attention_targets num_layers must be {num_layers}, got {actual}"
            )
        if attention_targets.shape[2] != sequence_length:
            raise ValueError("attention_targets sequence_length must match manifest")
        if attention_targets.shape[3] != hidden_size:
            actual = attention_targets.shape[3]
            raise ValueError(
                f"attention_targets hidden_size must be {hidden_size}, got {actual}"
            )
        if not np.issubdtype(attention_targets.dtype, np.floating):
            raise ValueError("attention_targets dtype must be floating")
