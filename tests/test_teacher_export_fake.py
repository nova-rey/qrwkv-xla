from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.targets import inspect_target_bundle, read_shard, validate_target_bundle
from qrwkv_xla.targets.store import shard_path
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
    load_teacher_export_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_teacher_export_config_loads() -> None:
    config = load_teacher_export_config(ROOT / "configs" / "teacher_export_stub.yaml")
    assert config.teacher.policy_label == "Qwen3.latest"
    assert config.targets.sequence_length == 64
    assert config.runtime.exporter_backend == "fake"


def test_fake_exporter_writes_valid_bundle(tmp_path: Path) -> None:
    config = TeacherExportConfig()
    config = replace(
        config,
        runtime=replace(config.runtime, output_dir=tmp_path / "bundle"),
    )

    result = FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    validate_target_bundle(result.output_dir)
    summary = inspect_target_bundle(result.output_dir)
    assert result.shard_count == config.runtime.num_shards
    assert (
        result.total_examples == config.runtime.batch_size * config.runtime.num_shards
    )
    assert result.manifest.teacher_policy_label == "Qwen3.latest"
    assert result.manifest.targets.logits is False
    assert summary["target_keys"] == ["attention_mask", "hidden_states", "input_ids"]


def test_fake_exporter_includes_logits_when_requested(tmp_path: Path) -> None:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(config.targets, include_logits=True),
        runtime=replace(config.runtime, output_dir=tmp_path / "bundle_logits"),
    )

    result = FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    shard = read_shard(shard_path(result.output_dir, 0))
    assert "logits" in shard
    assert result.manifest.targets.logits is True


def test_fake_exporter_is_deterministic_for_input_ids(tmp_path: Path) -> None:
    config = TeacherExportConfig()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    config_one = replace(config, runtime=replace(config.runtime, output_dir=first_dir))
    config_two = replace(config, runtime=replace(config.runtime, output_dir=second_dir))

    FakeTeacherExporter().export(
        ExportRequest(config=config_one, output_dir=config_one.runtime.output_dir)
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config_two, output_dir=config_two.runtime.output_dir)
    )

    first_shard = read_shard(shard_path(first_dir, 0))
    second_shard = read_shard(shard_path(second_dir, 0))
    np.testing.assert_array_equal(first_shard["input_ids"], second_shard["input_ids"])


def test_fake_exporter_includes_attention_targets(tmp_path: Path) -> None:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(config.targets, include_attention_targets=True),
        runtime=replace(config.runtime, output_dir=tmp_path / "attention_targets"),
    )

    result = FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    shard = read_shard(shard_path(result.output_dir, 0))
    assert result.manifest.targets.attention_targets is True
    assert shard["attention_targets"].shape == (2, 2, 64, 128)
