from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.generation import SmokeTokenizer
from qrwkv_xla.lm.tokenized_corpus import TokenizedCorpusSource, write_tokenized_corpus
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
    assert summary["target_keys"] == [
        "attention_mask",
        "hidden_states",
        "input_ids",
        "loss_mask",
    ]


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


def test_fake_exporter_uses_tokenized_corpus_inputs_and_loss_mask(
    tmp_path: Path,
) -> None:
    tokenized_dir = tmp_path / "tok"
    write_tokenized_corpus(
        [(10, 11, 12, 13), (20, 0, 21, 0)],
        tokenized_dir,
        sequence_length=3,
        tokenizer=SmokeTokenizer().metadata,
        source=TokenizedCorpusSource(
            kind="jsonl_prompts",
            path=None,
            sha256="a" * 64,
            record_count=2,
            selected_count=2,
        ),
        overwrite=True,
        created_at="2026-05-03T22:20:00+00:00",
    )
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=3,
            tokenized_corpus=tokenized_dir,
        ),
        runtime=replace(config.runtime, output_dir=tmp_path / "bundle"),
    )

    result = FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    shard = read_shard(shard_path(result.output_dir, 0))
    np.testing.assert_array_equal(
        shard["input_ids"],
        np.asarray([[10, 11, 12], [20, 0, 21]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        shard["loss_mask"],
        np.asarray([[1, 1, 1], [0, 1, 0]], dtype=np.int32),
    )
    assert result.manifest.prompt_source["type"] == "tokenized_corpus"
