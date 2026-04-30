from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillStageConfig,
    DistillStudentConfig,
    run_distill_stage,
)
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)


def test_distill_checkpoint_save_then_resume_additional_steps(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    first_dir = tmp_path / "checkpoints" / "first"
    second_dir = tmp_path / "checkpoints" / "second"

    base_config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        training=replace(DistillStageConfig().training, max_steps=2),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
    )
    first = run_distill_stage(
        replace(
            base_config,
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=first_dir,
                overwrite=True,
            ),
        )
    )
    second = run_distill_stage(
        replace(
            base_config,
            training=replace(base_config.training, max_steps=3),
            checkpoint=DistillCheckpointConfig(
                resume_from=first_dir,
                checkpoint_out=second_dir,
                overwrite=True,
            ),
        )
    )

    assert first.start_step == 0
    assert first.end_step == 2
    assert second.start_step == 2
    assert second.end_step == 5
    assert (second_dir / "checkpoint.json").is_file()
    assert (second_dir / "params.npz").is_file()


def test_resume_architecture_mismatch_raises(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "arch"
    run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
            training=replace(DistillStageConfig().training, max_steps=1),
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=checkpoint_dir,
                overwrite=True,
            ),
        )
    )

    with pytest.raises(ValueError, match="architecture mismatch"):
        run_distill_stage(
            DistillStageConfig(
                targets_dir=bundle_dir,
                student=DistillStudentConfig(
                    architecture="rwkv7_reference",
                    vocab_size=32,
                ),
                training=replace(DistillStageConfig().training, max_steps=1),
                checkpoint=DistillCheckpointConfig(resume_from=checkpoint_dir),
            )
        )


def test_same_checkpoint_resume_and_output_without_overwrite_raises(
    tmp_path: Path,
) -> None:
    same_dir = tmp_path / "checkpoints" / "same"
    with pytest.raises(ValueError, match="same path"):
        run_distill_stage(
            DistillStageConfig(
                targets_dir=tmp_path / "bundle",
                checkpoint=DistillCheckpointConfig(
                    resume_from=same_dir,
                    checkpoint_out=same_dir,
                ),
            )
        )


def _fake_bundle(tmp_path: Path) -> Path:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=4,
            num_layers=2,
            vocab_size=32,
        ),
        runtime=replace(
            config.runtime,
            output_dir=tmp_path / "bundle",
            batch_size=2,
            num_shards=1,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    return config.runtime.output_dir
