from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillOptimizerConfig,
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


def test_distill_adamw_checkpoint_and_tracking_include_optimizer(
    tmp_path: Path,
) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "adamw"
    run_root = tmp_path / "runs"
    config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        optimizer=DistillOptimizerConfig(
            type="adamw",
            learning_rate=0.01,
            weight_decay=0.01,
        ),
        training=replace(DistillStageConfig().training, max_steps=2),
        checkpoint=DistillCheckpointConfig(
            checkpoint_out=checkpoint_dir,
            overwrite=True,
        ),
        tracking=DistillTrackingConfig(enabled=True, run_root=run_root),
    )

    result = run_distill_stage(config)
    loaded = load_checkpoint(checkpoint_dir)

    assert result.end_step == 2
    assert loaded.optimizer_state is not None
    assert loaded.optimizer_state.type == "adamw"
    assert int(loaded.optimizer_state.step) == 2

    assert result.metrics_path is not None
    first_metric = json.loads(result.metrics_path.read_text().splitlines()[0])
    assert first_metric["values"]["learning_rate"] == pytest.approx(0.01)
    assert first_metric["values"]["optimizer_step"] == 1.0
    assert first_metric["extra"]["optimizer_type"] == "adamw"


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
