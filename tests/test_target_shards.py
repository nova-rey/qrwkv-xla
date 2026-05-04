from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.targets import read_shard, validate_shard_arrays, write_shard


def _valid_arrays() -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((2, 64), dtype=np.int32),
        "attention_mask": np.ones((2, 64), dtype=np.int32),
        "loss_mask": np.ones((2, 64), dtype=np.int32),
        "hidden_states": np.ones((2, 2, 64, 128), dtype=np.float32),
    }


def test_valid_shard_write_read_round_trip(tmp_path: Path) -> None:
    shard_file = tmp_path / "shard_000000.npz"
    arrays = _valid_arrays()
    write_shard(shard_file, arrays)
    loaded = read_shard(shard_file)
    validate_shard_arrays(loaded, sequence_length=64, hidden_size=128, num_layers=2)
    assert loaded["input_ids"].shape == (2, 64)


def test_required_keys_enforced() -> None:
    arrays = _valid_arrays()
    arrays.pop("loss_mask")
    with pytest.raises(ValueError, match="Missing required shard keys"):
        validate_shard_arrays(arrays, sequence_length=64, hidden_size=128, num_layers=2)


def test_hidden_state_shape_mismatch_raises() -> None:
    arrays = _valid_arrays()
    arrays["hidden_states"] = np.ones((2, 3, 64, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="num_layers"):
        validate_shard_arrays(arrays, sequence_length=64, hidden_size=128, num_layers=2)


def test_logits_shape_validation() -> None:
    arrays = _valid_arrays()
    arrays["logits"] = np.ones((2, 63, 500), dtype=np.float32)
    with pytest.raises(ValueError, match="logits sequence_length"):
        validate_shard_arrays(arrays, sequence_length=64, hidden_size=128, num_layers=2)
