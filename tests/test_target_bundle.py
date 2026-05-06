from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.targets import (
    TargetFlags,
    TeacherTargetManifest,
    inspect_target_bundle,
    validate_target_bundle,
    write_target_bundle,
)
from qrwkv_xla.targets.shards import write_shard
from qrwkv_xla.targets.store import shard_path


def _manifest() -> TeacherTargetManifest:
    return TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="qwen",
        teacher_model_id=None,
        teacher_policy_label="Qwen3.latest",
        fallback_policy_label="Qwen3.0",
        tokenizer_id=None,
        sequence_length=64,
        hidden_size=128,
        num_layers=2,
        targets=TargetFlags(),
        dtype="fp32",
        created_by="test",
        notes=[],
    )


def _shard(batch_size: int = 2) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((batch_size, 64), dtype=np.int32),
        "attention_mask": np.ones((batch_size, 64), dtype=np.int32),
        "loss_mask": np.ones((batch_size, 64), dtype=np.int32),
        "hidden_states": np.ones((batch_size, 2, 64, 128), dtype=np.float32),
    }


def test_bundle_round_trip(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "fake_bundle"
    write_target_bundle(bundle_dir, _manifest(), [_shard(), _shard()])
    validate_target_bundle(bundle_dir)
    summary = inspect_target_bundle(bundle_dir)
    assert summary["shard_count"] == 2
    assert summary["total_examples"] == 4
    assert summary["target_keys"] == [
        "attention_mask",
        "hidden_states",
        "input_ids",
        "loss_mask",
    ]


def test_tiny_hf_target_key_contract_allows_logits_without_requiring_logits_loss(
    tmp_path: Path,
) -> None:
    manifest = TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="hf-causal-lm",
        teacher_model_id="sshleifer/tiny-gpt2",
        teacher_policy_label="tiny-hf-smoke-p29",
        fallback_policy_label=None,
        tokenizer_id="sshleifer/tiny-gpt2",
        sequence_length=8,
        hidden_size=2,
        num_layers=2,
        targets=TargetFlags(logits=True),
        dtype="fp32",
        created_by="test",
        notes=[],
        extra={"vocab_size": 50257},
    )
    shard = {
        "input_ids": np.ones((2, 8), dtype=np.int32),
        "attention_mask": np.ones((2, 8), dtype=np.int32),
        "loss_mask": np.ones((2, 8), dtype=np.int32),
        "hidden_states": np.ones((2, 2, 8, 2), dtype=np.float32),
        "logits": np.ones((2, 8, 50257), dtype=np.float32),
    }
    bundle_dir = tmp_path / "tiny_hf_bundle"

    write_target_bundle(bundle_dir, manifest, [shard])

    summary = inspect_target_bundle(bundle_dir)
    assert summary["teacher_family"] == "hf-causal-lm"
    assert summary["teacher_model_id"] == "sshleifer/tiny-gpt2"
    assert summary["hidden_size"] == 2
    assert summary["num_layers"] == 2
    assert summary["target_keys"] == [
        "attention_mask",
        "hidden_states",
        "input_ids",
        "logits",
        "loss_mask",
    ]


def test_missing_manifest_raises(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "missing_manifest"
    (bundle_dir / "shards").mkdir(parents=True)
    with pytest.raises(ValueError, match="Missing manifest.json"):
        validate_target_bundle(bundle_dir)


def test_missing_shards_directory_raises(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "missing_shards"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing shards directory"):
        validate_target_bundle(bundle_dir)


def test_invalid_shard_is_rejected_at_write_time(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "invalid_shard_write"
    bad_shard = _shard()
    bad_shard["hidden_states"] = np.ones((2, 2, 63, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="hidden_states sequence_length"):
        write_target_bundle(bundle_dir, _manifest(), [bad_shard])


def test_invalid_shard_inside_existing_bundle_raises(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "invalid_shard_existing"
    write_target_bundle(bundle_dir, _manifest(), [_shard()])
    write_shard(
        shard_path(bundle_dir, 0),
        {
            "input_ids": np.ones((2, 64), dtype=np.int32),
            "attention_mask": np.ones((2, 64), dtype=np.int32),
            "loss_mask": np.ones((2, 64), dtype=np.int32),
            "hidden_states": np.ones((2, 2, 63, 128), dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="hidden_states sequence_length"):
        validate_target_bundle(bundle_dir)
