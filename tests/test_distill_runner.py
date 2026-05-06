from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.distill import (
    DistillLossConfig,
    DistillStageConfig,
    DistillStudentConfig,
    LossWeightConfig,
    run_distill_stage,
)
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)


def test_run_stage_with_tiny_student(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="tiny_student",
                vocab_size=32,
            ),
            training=replace(DistillStageConfig().training, max_steps=2),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        )
    )
    assert result.steps == 2
    assert result.target_bundle == bundle_dir
    assert result.final_hidden_mse is not None
    assert math.isfinite(result.final_loss)
    assert result.final_loss == pytest.approx(result.final_hidden_mse)


def test_run_stage_with_rwkv7_reference(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="rwkv7_reference",
                vocab_size=32,
            ),
            training=replace(DistillStageConfig().training, max_steps=2),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        )
    )
    assert result.steps == 2
    assert result.final_loss == pytest.approx(result.final_hidden_mse)


def test_run_stage_with_rwkv7_radlads_reference(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="rwkv7_radlads_reference",
                vocab_size=32,
                num_heads=2,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        )
    )
    assert result.steps == 1
    assert result.final_hidden_mse is not None
    assert math.isfinite(result.final_loss)


def test_run_stage_with_rwkv7_radlads_reference_logits_loss(
    tmp_path: Path,
) -> None:
    bundle_dir = _fake_bundle(tmp_path, include_logits=True)
    result = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="rwkv7_radlads_reference",
                vocab_size=32,
                num_heads=2,
                emit_logits=True,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
            losses=DistillLossConfig(
                hidden_mse=LossWeightConfig(enabled=True, weight=0.5),
                logits_kl=LossWeightConfig(enabled=True, weight=0.5),
            ),
        )
    )
    assert result.steps == 1
    assert result.final_logits_kl is not None
    assert math.isfinite(result.final_logits_kl)
    assert math.isfinite(result.final_loss)


def test_logits_loss_enabled_without_teacher_logits_raises(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path, include_logits=False)
    with pytest.raises(ValueError, match="teacher targets do not include logits"):
        run_distill_stage(
            DistillStageConfig(
                targets_dir=bundle_dir,
                student=DistillStudentConfig(
                    architecture="rwkv7_radlads_reference",
                    vocab_size=32,
                    num_heads=2,
                    emit_logits=True,
                ),
                losses=DistillLossConfig(
                    hidden_mse=LossWeightConfig(enabled=False, weight=0.0),
                    logits_kl=LossWeightConfig(enabled=True, weight=1.0),
                ),
            )
        )


def test_hidden_size_mismatch_raises(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    with pytest.raises(ValueError, match="hidden_size"):
        run_distill_stage(
            DistillStageConfig(
                targets_dir=bundle_dir,
                student=DistillStudentConfig(
                    architecture="tiny_student",
                    vocab_size=32,
                    hidden_size=99,
                ),
            )
        )


def test_num_layers_mismatch_raises(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    with pytest.raises(ValueError, match="num_layers"):
        run_distill_stage(
            DistillStageConfig(
                targets_dir=bundle_dir,
                student=DistillStudentConfig(
                    architecture="tiny_student",
                    vocab_size=32,
                    num_layers=99,
                ),
            )
        )


def _fake_bundle(tmp_path: Path, *, include_logits: bool = False) -> Path:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=4,
            num_layers=2,
            vocab_size=32,
            include_logits=include_logits,
        ),
        runtime=replace(
            config.runtime,
            output_dir=tmp_path / ("bundle_logits" if include_logits else "bundle"),
            batch_size=2,
            num_shards=1,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    return config.runtime.output_dir
