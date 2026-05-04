from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.targets import (
    TargetFlags,
    TeacherTargetManifest,
    load_teacher_target_bundle,
    read_manifest,
    validate_target_bundle,
    write_target_bundle,
)
from qrwkv_xla.targets.store import manifest_path, shard_path


def test_teacher_target_bundle_records_deterministic_shard_hashes(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    write_target_bundle(bundle_dir, _manifest(), [_shard()])

    manifest = read_manifest(manifest_path(bundle_dir))
    loaded = load_teacher_target_bundle(bundle_dir, expected_sequence_length=4)

    assert len(manifest.shards) == 1
    assert manifest.shards[0].path == "shards/shard_000000.npz"
    assert len(manifest.shards[0].sha256) == 64
    assert manifest.shards[0].num_examples == 2
    assert manifest.shards[0].arrays == (
        "attention_mask",
        "hidden_states",
        "input_ids",
        "loss_mask",
    )
    assert loaded.manifest.shards == manifest.shards
    np.testing.assert_array_equal(loaded.shards[0]["loss_mask"], _shard()["loss_mask"])


def test_teacher_target_bundle_requires_loss_mask(tmp_path: Path) -> None:
    shard = _shard()
    shard.pop("loss_mask")

    with pytest.raises(ValueError, match="loss_mask"):
        write_target_bundle(tmp_path / "bundle", _manifest(), [shard])


def test_teacher_target_bundle_detects_hash_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    write_target_bundle(bundle_dir, _manifest(), [_shard()])
    np.savez(
        shard_path(bundle_dir, 0),
        input_ids=np.ones((2, 4), dtype=np.int32),
        attention_mask=np.ones((2, 4), dtype=np.int32),
        loss_mask=np.zeros((2, 4), dtype=np.int32),
        hidden_states=np.ones((2, 2, 4, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_target_bundle(bundle_dir)


def test_teacher_target_bundle_allows_hidden_states_disabled(tmp_path: Path) -> None:
    manifest = _manifest(
        targets=TargetFlags(hidden_states=False, logits=False),
    )
    shard = _shard()
    shard.pop("hidden_states")
    write_target_bundle(tmp_path / "bundle", manifest, [shard])

    loaded = load_teacher_target_bundle(tmp_path / "bundle")

    assert loaded.manifest.targets.hidden_states is False
    assert "hidden_states" not in loaded.shards[0]


def test_teacher_target_bundle_sequence_length_validation(tmp_path: Path) -> None:
    write_target_bundle(tmp_path / "bundle", _manifest(), [_shard()])

    with pytest.raises(ValueError, match="sequence_length mismatch"):
        load_teacher_target_bundle(tmp_path / "bundle", expected_sequence_length=5)


def _manifest(
    *,
    targets: TargetFlags | None = None,
) -> TeacherTargetManifest:
    return TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="unit",
        teacher_model_id="unit-teacher",
        teacher_policy_label="unit",
        fallback_policy_label=None,
        tokenizer_id="smoke",
        sequence_length=4,
        hidden_size=3,
        num_layers=2,
        targets=targets or TargetFlags(),
        dtype="fp32",
        created_by="test",
    )


def _shard() -> dict[str, np.ndarray]:
    return {
        "input_ids": np.arange(8, dtype=np.int32).reshape(2, 4),
        "attention_mask": np.ones((2, 4), dtype=np.int32),
        "loss_mask": np.asarray([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.int32),
        "hidden_states": np.ones((2, 2, 4, 3), dtype=np.float32),
    }
