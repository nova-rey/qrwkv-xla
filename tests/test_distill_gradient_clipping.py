from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillGradientConfig,
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
from qrwkv_xla.training.gradients import global_gradient_norm


def test_distill_clipping_metrics_checkpoint_and_tracking(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "clipped"
    run_root = tmp_path / "runs"
    config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        optimizer=DistillOptimizerConfig(type="adamw", learning_rate=0.01),
        gradients=DistillGradientConfig(max_grad_norm=0.01, clip_epsilon=1e-6),
        training=replace(DistillStageConfig().training, max_steps=2),
        checkpoint=DistillCheckpointConfig(
            checkpoint_out=checkpoint_dir,
            overwrite=True,
        ),
        tracking=DistillTrackingConfig(enabled=True, run_root=run_root),
    )

    result = run_distill_stage(config)
    loaded = load_checkpoint(checkpoint_dir)

    assert result.final_grad_global_norm is not None
    assert result.final_grad_clipped_global_norm is not None
    assert result.final_grad_clip_scale is not None
    assert result.final_grad_clipped_global_norm <= 0.010001
    assert loaded.manifest.gradients == {
        "clip_epsilon": 1e-06,
        "max_grad_norm": 0.01,
    }
    assert loaded.optimizer_state is not None
    assert loaded.optimizer_state.type == "adamw"
    first_moment_norm = float(global_gradient_norm(loaded.optimizer_state.slots["m"]))
    max_moment_norm = config.gradients.max_grad_norm * (
        1.0 - config.optimizer.beta1**result.end_step
    )
    assert first_moment_norm <= max_moment_norm + 1e-7

    assert result.metrics_path is not None
    first_metric = json.loads(result.metrics_path.read_text().splitlines()[0])
    values = first_metric["values"]
    assert values["grad_global_norm"] > values["grad_clipped_global_norm"]
    assert values["grad_clip_scale"] < 1.0
    assert values["grad_was_clipped"] == 1.0
    assert values["max_grad_norm"] == pytest.approx(0.01)

    assert result.run_dir is not None
    run_metadata = json.loads((result.run_dir / "run.json").read_text())
    assert run_metadata["distillation"]["gradients"]["max_grad_norm"] == 0.01
    assert result.summary_path is not None
    summary = json.loads(result.summary_path.read_text())
    assert summary["summary"]["final_grad_clip_scale"] == pytest.approx(
        result.final_grad_clip_scale
    )


def test_distill_unclipped_metrics_preserve_norm(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        training=replace(DistillStageConfig().training, max_steps=1),
    )

    result = run_distill_stage(config)

    assert result.final_grad_global_norm == pytest.approx(
        result.final_grad_clipped_global_norm
    )
    assert result.final_grad_clip_scale == pytest.approx(1.0)


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
