from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.datasets import TargetBundleDataset
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)


def test_target_bundle_dataset_reads_fake_export_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=5,
            hidden_size=7,
            num_layers=3,
            include_logits=True,
            vocab_size=17,
        ),
        runtime=replace(
            config.runtime,
            output_dir=bundle_dir,
            batch_size=2,
            num_shards=3,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    dataset = TargetBundleDataset.from_path(bundle_dir)
    first_batch = dataset.first_batch()
    shards = list(dataset.iter_shards())

    assert dataset.bundle_dir == bundle_dir
    with pytest.raises(FrozenInstanceError):
        dataset.bundle_dir = tmp_path  # type: ignore[misc]
    assert len(shards) == 3
    np.testing.assert_array_equal(first_batch.input_ids, shards[0].input_ids)
    assert first_batch.input_ids.shape == (2, 5)
    assert first_batch.attention_mask.shape == (2, 5)
    assert first_batch.loss_mask.shape == (2, 5)
    assert first_batch.hidden_states.shape == (2, 3, 5, 7)
    assert first_batch.logits is not None
    assert first_batch.logits.shape == (2, 5, 17)
