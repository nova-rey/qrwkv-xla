from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.targets import TargetFlags, TeacherTargetManifest, write_target_bundle


@pytest.fixture
def manifest() -> TeacherTargetManifest:
    return TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="fake",
        teacher_model_id="fake-model",
        teacher_policy_label="fake",
        fallback_policy_label=None,
        tokenizer_id="fake-tokenizer",
        sequence_length=3,
        hidden_size=4,
        num_layers=2,
        targets=TargetFlags(attention_targets=True),
        dtype="fp32",
    )


def test_target_bundle_dataset_loads_attention_targets(
    tmp_path: Path,
    manifest: TeacherTargetManifest,
) -> None:
    bundle = tmp_path / "bundle"
    write_target_bundle(
        bundle,
        manifest,
        [
            {
                "input_ids": np.ones((1, 3), dtype=np.int32),
                "attention_mask": np.ones((1, 3), dtype=np.int32),
                "loss_mask": np.ones((1, 3), dtype=np.int32),
                "hidden_states": np.ones((1, 2, 3, 4), dtype=np.float32),
                "attention_targets": np.ones((1, 2, 3, 4), dtype=np.float32),
            }
        ],
    )
    batch = TargetBundleDataset(bundle).first_batch()
    assert batch.attention_targets is not None
    assert batch.attention_targets.shape == (1, 2, 3, 4)


def test_attention_targets_rank_mismatch_fails(
    tmp_path: Path,
    manifest: TeacherTargetManifest,
) -> None:
    bundle = tmp_path / "bad-bundle"
    with pytest.raises(ValueError, match="rank 4"):
        write_target_bundle(
            bundle,
            manifest,
            [
                {
                    "input_ids": np.ones((1, 3), dtype=np.int32),
                    "attention_mask": np.ones((1, 3), dtype=np.int32),
                    "loss_mask": np.ones((1, 3), dtype=np.int32),
                    "hidden_states": np.ones((1, 2, 3, 4), dtype=np.float32),
                    "attention_targets": np.ones((1, 3, 4), dtype=np.float32),
                }
            ],
        )
