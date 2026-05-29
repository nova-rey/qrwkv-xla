from __future__ import annotations

from pathlib import Path

import numpy as np

from qrwkv_xla.targets import TeacherTargetStore, validate_target_store_metadata
from qrwkv_xla.teachers import SyntheticTeacherBackend, emit_teacher_target_store


def test_synthetic_teacher_backend_metadata_is_valid() -> None:
    backend = SyntheticTeacherBackend()
    metadata = backend.build_metadata(num_examples=2, sequence_length=3)

    validate_target_store_metadata(metadata)

    assert metadata.model_id == "synthetic-teacher-v0"
    assert metadata.model_family == "synthetic"
    assert metadata.target_type == "synthetic"
    assert metadata.vocab_size == 8


def test_synthetic_teacher_backend_emits_deterministic_arrays() -> None:
    backend = SyntheticTeacherBackend(vocab_size=8)

    first = backend.emit_targets(num_examples=2, sequence_length=3)
    second = backend.emit_targets(num_examples=2, sequence_length=3)

    assert first["input_ids"].shape == (2, 3)
    assert first["input_ids"].dtype == np.int32
    assert first["attention_mask"].shape == (2, 3)
    assert first["attention_mask"].dtype == np.int32
    assert first["logits"].shape == (2, 3, 8)
    assert first["logits"].dtype == np.float32
    np.testing.assert_array_equal(first["input_ids"], second["input_ids"])
    np.testing.assert_array_equal(first["attention_mask"], second["attention_mask"])
    np.testing.assert_array_equal(first["logits"], second["logits"])
    assert float(first["logits"][1, 2, 7]) == 2.125


def test_emit_teacher_target_store_writes_valid_artifact(tmp_path: Path) -> None:
    backend = SyntheticTeacherBackend()

    store = emit_teacher_target_store(
        backend,
        tmp_path / "synthetic_targets",
        num_examples=2,
        sequence_length=3,
    )

    assert isinstance(store, TeacherTargetStore)
    assert (tmp_path / "synthetic_targets" / "metadata.json").is_file()
    assert (tmp_path / "synthetic_targets" / "shards" / "shard-00000.npz").is_file()
    store.validate()


def test_emitted_teacher_target_store_round_trips_arrays(tmp_path: Path) -> None:
    backend = SyntheticTeacherBackend()
    expected = backend.emit_targets(num_examples=2, sequence_length=3)
    emit_teacher_target_store(
        backend,
        tmp_path / "synthetic_targets",
        num_examples=2,
        sequence_length=3,
    )

    reopened = TeacherTargetStore.open(tmp_path / "synthetic_targets")
    loaded = reopened.read_shard(0)

    np.testing.assert_array_equal(loaded["input_ids"], expected["input_ids"])
    np.testing.assert_array_equal(loaded["attention_mask"], expected["attention_mask"])
    np.testing.assert_array_equal(loaded["logits"], expected["logits"])


def test_emission_is_identical_across_stores(tmp_path: Path) -> None:
    backend = SyntheticTeacherBackend()
    first = emit_teacher_target_store(
        backend,
        tmp_path / "first",
        num_examples=2,
        sequence_length=3,
    )
    second = emit_teacher_target_store(
        backend,
        tmp_path / "second",
        num_examples=2,
        sequence_length=3,
    )

    first_arrays = first.read_shard(0)
    second_arrays = second.read_shard(0)

    np.testing.assert_array_equal(first_arrays["input_ids"], second_arrays["input_ids"])
    np.testing.assert_array_equal(
        first_arrays["attention_mask"],
        second_arrays["attention_mask"],
    )
    np.testing.assert_array_equal(first_arrays["logits"], second_arrays["logits"])


def test_synthetic_teacher_backend_requires_no_live_teacher_dependencies() -> None:
    backend = SyntheticTeacherBackend()

    assert backend.name == "synthetic"
    assert backend.build_metadata(num_examples=1, sequence_length=1).source == {
        "kind": "synthetic"
    }
