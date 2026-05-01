from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.distill import (
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


def test_distill_stage_tracks_scheduled_learning_rates(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    config = DistillStageConfig(
        targets_dir=_fake_bundle(tmp_path),
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        training=replace(DistillStageConfig().training, max_steps=3),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.1),
        lr_schedule=DistillLRScheduleConfig(
            type="warmup_cosine",
            warmup_steps=2,
            total_steps=6,
            min_learning_rate=0.01,
        ),
        tracking=DistillTrackingConfig(
            enabled=True,
            run_root=run_root,
            run_name="schedule",
        ),
    )

    result = run_distill_stage(config)

    assert result.lr_schedule_type == "warmup_cosine"
    assert result.initial_learning_rate == 0.05
    assert result.final_learning_rate == 0.1

    assert result.metrics_path is not None
    records = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["values"]["learning_rate"] for record in records] == [
        pytest.approx(0.05),
        pytest.approx(0.1),
        pytest.approx(0.1),
    ]
    assert records[-1]["extra"]["lr_schedule_type"] == "warmup_cosine"
    assert records[-1]["values"]["base_learning_rate"] == 0.1

    assert result.summary_path is not None
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["initial_learning_rate"] == 0.05
    assert summary["summary"]["final_learning_rate"] == 0.1
    assert summary["summary"]["lr_schedule"]["type"] == "warmup_cosine"


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
