from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillLRScheduleConfig,
    DistillStageConfig,
    DistillStudentConfig,
    DistillTrackingConfig,
    run_distill_stage,
)
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)


def test_lr_schedule_resume_uses_global_step(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    first_dir = tmp_path / "checkpoints" / "schedule_first"
    second_dir = tmp_path / "checkpoints" / "schedule_second"
    run_root = tmp_path / "runs"
    schedule = DistillLRScheduleConfig(
        type="warmup_cosine",
        warmup_steps=2,
        total_steps=6,
        min_learning_rate=0.01,
    )
    base_config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        training=replace(DistillStageConfig().training, max_steps=2),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.1),
        lr_schedule=schedule,
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
            checkpoint=DistillCheckpointConfig(
                resume_from=first_dir,
                checkpoint_out=second_dir,
                overwrite=True,
            ),
            tracking=DistillTrackingConfig(
                enabled=True,
                run_root=run_root,
                run_name="resume schedule",
            ),
        )
    )

    assert first.end_step == 2
    assert second.start_step == 2
    assert second.end_step == 4
    assert second.initial_learning_rate == 0.1
    assert second.final_learning_rate == pytest.approx(0.08681980515339464)

    loaded = load_checkpoint(second_dir)
    assert loaded.manifest.step == 4
    assert loaded.manifest.lr_schedule["step"] == 4

    assert second.metrics_path is not None
    records = [
        json.loads(line)
        for line in second.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["values"]["global_step"] for record in records] == [2.0, 3.0]
    assert [record["values"]["learning_rate"] for record in records] == [
        pytest.approx(0.1),
        pytest.approx(0.08681980515339464),
    ]


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
