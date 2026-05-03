from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.targets import read_shard
from qrwkv_xla.targets.store import shard_path
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)


def test_fake_attention_export_is_deterministic(tmp_path: Path) -> None:
    base = TeacherExportConfig()
    config = replace(
        base,
        targets=replace(base.targets, include_attention_targets=True),
        runtime=replace(base.runtime, output_dir=tmp_path / "run1", seed=1234),
    )
    other = replace(
        config, runtime=replace(config.runtime, output_dir=tmp_path / "run2")
    )

    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    FakeTeacherExporter().export(
        ExportRequest(config=other, output_dir=other.runtime.output_dir)
    )

    shard_one = read_shard(shard_path(config.runtime.output_dir, 0))
    shard_two = read_shard(shard_path(other.runtime.output_dir, 0))
    assert shard_one["attention_targets"].shape == (2, 2, 64, 128)
    assert np.allclose(shard_one["attention_targets"], shard_two["attention_targets"])
