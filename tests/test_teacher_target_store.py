from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
    target_store_metadata_from_dict,
)


def test_teacher_target_store_round_trips_tiny_synthetic_logits(
    tmp_path: Path,
) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    arrays = _arrays()

    shard_path = store.write_shard(0, arrays)
    reopened = TeacherTargetStore.open(tmp_path / "targets")
    loaded = reopened.read_shard(0)

    assert shard_path.name == "shard-00000.npz"
    assert reopened.metadata.target_type == "synthetic"
    np.testing.assert_array_equal(loaded["input_ids"], arrays["input_ids"])
    np.testing.assert_array_equal(loaded["attention_mask"], arrays["attention_mask"])
    np.testing.assert_array_equal(loaded["logits"], arrays["logits"])
    reopened.validate()


def test_teacher_target_store_metadata_round_trip() -> None:
    payload = {
        "schema_version": TEACHER_TARGET_STORE_SCHEMA_VERSION,
        "target_store_version": TEACHER_TARGET_STORE_VERSION,
        "model_id": "synthetic-teacher",
        "model_family": "synthetic",
        "tokenizer_id": "smoke-tokenizer",
        "tokenizer_hash": None,
        "vocab_size": 5,
        "target_type": "synthetic",
        "dtype": "float32",
        "sequence_length": 3,
        "num_examples": 2,
        "shard_count": 1,
        "created_by": "test",
        "created_at": "2026-05-29T00:00:00Z",
        "source": {"kind": "unit"},
        "provenance": {"phase": "P93"},
    }

    metadata = target_store_metadata_from_dict(payload)

    assert metadata.model_id == "synthetic-teacher"
    assert metadata.source == {"kind": "unit"}
    assert metadata.provenance == {"phase": "P93"}


def test_teacher_target_store_validation_fails_for_missing_shard(
    tmp_path: Path,
) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())

    with pytest.raises(ValueError, match="shard_count mismatch"):
        store.validate()


def test_teacher_target_store_rejects_bad_target_type() -> None:
    with pytest.raises(ValueError, match="unsupported target_type"):
        TeacherTargetStore.create(
            Path("/tmp/not-created"),
            replace(_metadata(), target_type="bogus"),
        )


def test_teacher_target_store_rejects_missing_required_array(
    tmp_path: Path,
) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    arrays = _arrays()
    arrays.pop("logits")

    with pytest.raises(ValueError, match="missing required arrays"):
        store.write_shard(0, arrays)


def test_teacher_target_store_rejects_shape_mismatch(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    arrays = _arrays()
    arrays["logits"] = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="vocab_size"):
        store.write_shard(0, arrays)


def test_teacher_target_store_rejects_dtype_mismatch(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(
        tmp_path / "targets",
        replace(_metadata(), dtype="float16"),
    )

    with pytest.raises(ValueError, match="dtype"):
        store.write_shard(0, _arrays())


def _metadata() -> TargetStoreMetadata:
    return TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="synthetic-teacher",
        model_family="synthetic",
        tokenizer_id="smoke-tokenizer",
        tokenizer_hash=None,
        vocab_size=5,
        target_type="synthetic",
        dtype="float32",
        sequence_length=3,
        num_examples=2,
        shard_count=1,
        created_by="test",
        created_at="2026-05-29T00:00:00Z",
        source={"kind": "unit"},
        provenance={"phase": "P93"},
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "input_ids": np.arange(6, dtype=np.int32).reshape(2, 3),
        "attention_mask": np.ones((2, 3), dtype=np.int32),
        "logits": np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 10.0,
    }
