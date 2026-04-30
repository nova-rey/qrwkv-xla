from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from qrwkv_xla.distill import (
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


def test_tracking_disabled_preserves_result_shape(tmp_path: Path) -> None:
    result = run_distill_stage(_distill_config(tmp_path, tracking=None))

    assert result.run_dir is None
    assert result.metrics_path is None
    assert result.summary_path is None
    assert result.checkpoint_out is None


def test_tracking_writes_run_metrics_summary_and_default_checkpoint(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    config = _distill_config(
        tmp_path,
        tracking=DistillTrackingConfig(
            enabled=True,
            run_root=run_root,
            run_name="Unit Tracking",
            tags=["unit"],
            notes=["smoke"],
        ),
    )

    result = run_distill_stage(config)

    assert result.run_dir is not None
    assert result.metrics_path == result.run_dir / "metrics.jsonl"
    assert result.summary_path == result.run_dir / "summary.json"
    assert result.checkpoint_out == result.run_dir / "checkpoints" / "final"
    assert (result.checkpoint_out / "checkpoint.json").is_file()

    run_payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["run_name"] == "Unit Tracking"
    assert run_payload["tags"] == ["unit"]
    assert run_payload["notes"] == ["smoke"]
    assert (
        run_payload["checkpoint"]["checkpoint_out"] == "runs/<run_id>/checkpoints/final"
    )

    metric_lines = result.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(metric_lines) == 2
    assert json.loads(metric_lines[-1])["step"] == result.end_step

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["summary"]["final_loss"] == result.final_loss
    assert summary["summary"]["checkpoint_out"] == str(result.checkpoint_out)


def _distill_config(
    tmp_path: Path,
    *,
    tracking: DistillTrackingConfig | None,
) -> DistillStageConfig:
    bundle_dir = _fake_bundle(tmp_path)
    config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
        training=replace(DistillStageConfig().training, max_steps=2),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
    )
    if tracking is not None:
        config = replace(config, tracking=tracking)
    return config


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
