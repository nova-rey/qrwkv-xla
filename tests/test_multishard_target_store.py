from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
    iter_offline_target_batches,
    iter_target_store_shard_ids,
    load_offline_target_batch,
    mse_logits_loss,
    run_multishard_target_store_smoke,
)


def test_create_tiny_target_store_with_two_shards(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    assert store.metadata.shard_count == 2
    store.validate()


def test_list_shards_is_deterministic_order(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    names = [path.name for path in store.list_shards()]

    assert names == ["shard-00000.npz", "shard-00001.npz"]


def test_read_shards_by_id_returns_distinct_arrays(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    first = store.read_shard(0)
    second = store.read_shard(1)

    assert first["input_ids"][0, 0] == 0
    assert second["input_ids"][0, 0] == 100
    assert not np.array_equal(first["logits"], second["logits"])


def test_iter_shard_helper_visits_both_shards(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    assert iter_target_store_shard_ids(store) == (0, 1)
    batches = iter_offline_target_batches(store)

    assert len(batches) == 2
    assert batches[0].input_ids.shape == (2, 3)
    assert batches[1].teacher_logits.shape == (2, 3, 5)


def test_load_offline_target_batch_loads_each_shard(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    first = load_offline_target_batch(store, shard_id=0)
    second = load_offline_target_batch(store, shard_id=1)

    assert first.input_ids[0, 0] == 0
    assert second.input_ids[0, 0] == 100


def test_finite_per_shard_mse_loss_can_be_computed(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    losses = [
        mse_logits_loss(batch.teacher_logits, batch.teacher_logits)
        for batch in iter_offline_target_batches(store)
    ]

    assert len(losses) == 2
    assert all(bool(jnp.isfinite(loss)) for loss in losses)


def test_multishard_smoke_aggregates_examples_and_losses(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    result = run_multishard_target_store_smoke(store)

    assert result.status == "pass"
    assert result.shard_count == 2
    assert result.examples_seen == 4
    assert result.losses == (0.0, 0.0)
    assert result.aggregate_loss == 0.0
    assert result.loss_finite is True


def test_missing_shard_fails_validation_clearly(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    store.write_shard(0, _arrays(offset=0))

    with pytest.raises(ValueError, match="shard_count mismatch"):
        store.validate()


def test_extra_shard_count_mismatch_fails_clearly(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(
        tmp_path / "targets",
        replace(_metadata(), shard_count=1, num_examples=2),
    )
    store.write_shard(0, _arrays(offset=0))
    store.write_shard(1, _arrays(offset=100))

    with pytest.raises(ValueError, match="shard_count mismatch"):
        store.validate()


def test_missing_expected_shard_id_fails_clearly(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    store.write_shard(0, _arrays(offset=0))
    store.write_shard(2, _arrays(offset=200))

    with pytest.raises(ValueError, match="missing target shard"):
        store.validate()


def test_shape_mismatch_in_one_shard_fails_clearly(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    store.write_shard(0, _arrays(offset=0))
    bad = _arrays(offset=100)
    bad["logits"] = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="vocab_size"):
        store.write_shard(1, bad)


def test_canonical_layout_is_preserved(tmp_path: Path) -> None:
    store = _create_multishard_store(tmp_path)

    assert (store.root / "metadata.json").is_file()
    assert (store.root / "shards" / "shard-00000.npz").is_file()
    assert (store.root / "shards" / "shard-00001.npz").is_file()
    assert run_multishard_target_store_smoke(store).canonical_layout == (
        "metadata.json + shards/shard-XXXXX.npz"
    )


def test_multishard_smoke_requires_no_hf_qwen_internet_or_accelerator(
    tmp_path: Path,
) -> None:
    result = run_multishard_target_store_smoke(_create_multishard_store(tmp_path))

    assert result.status == "pass"
    assert "qwen_specific_support" in result.claims_not_made
    assert "training_ready" in result.claims_not_made
    assert "dataset_pipeline_ready" in result.claims_not_made


def _create_multishard_store(tmp_path: Path) -> TeacherTargetStore:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    store.write_shard(0, _arrays(offset=0))
    store.write_shard(1, _arrays(offset=100))
    store.validate()
    return TeacherTargetStore.open(store.root)


def _metadata() -> TargetStoreMetadata:
    return TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="synthetic-multishard-teacher",
        model_family="synthetic",
        tokenizer_id="smoke-tokenizer",
        tokenizer_hash=None,
        vocab_size=5,
        target_type="synthetic",
        dtype="float32",
        sequence_length=3,
        num_examples=4,
        shard_count=2,
        created_by="test",
        created_at="2026-06-03T00:00:00Z",
        source={"kind": "unit"},
        provenance={"phase": "P106"},
    )


def _arrays(*, offset: int) -> dict[str, np.ndarray]:
    input_ids = np.arange(offset, offset + 6, dtype=np.int32).reshape(2, 3)
    logits = np.arange(offset, offset + 30, dtype=np.float32).reshape(2, 3, 5)
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones((2, 3), dtype=np.int32),
        "logits": logits / 10.0,
    }
