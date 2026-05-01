from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
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


def test_distill_logits_fresh_and_hidden_to_logits_resume(tmp_path: Path) -> None:
    hidden_bundle = _fake_bundle(tmp_path, name="hidden", include_logits=False)
    logits_bundle = _fake_bundle(tmp_path, name="logits", include_logits=True)
    hidden_checkpoint = tmp_path / "checkpoints" / "hidden"
    logits_checkpoint = tmp_path / "checkpoints" / "logits"

    fresh = run_distill_stage(
        DistillStageConfig(
            targets_dir=logits_bundle,
            student=DistillStudentConfig(
                architecture="tiny_student",
                vocab_size=32,
                emit_logits=True,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            losses=_logits_losses(),
        )
    )
    assert fresh.final_logits_kl is not None
    assert np.isfinite(fresh.final_logits_kl)

    hidden = run_distill_stage(
        DistillStageConfig(
            targets_dir=hidden_bundle,
            student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
            training=replace(DistillStageConfig().training, max_steps=1),
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=hidden_checkpoint,
                overwrite=True,
            ),
        )
    )
    resumed = run_distill_stage(
        DistillStageConfig(
            targets_dir=logits_bundle,
            student=DistillStudentConfig(
                architecture="tiny_student",
                vocab_size=32,
                emit_logits=True,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            losses=_logits_losses(),
            checkpoint=DistillCheckpointConfig(
                resume_from=hidden_checkpoint,
                checkpoint_out=logits_checkpoint,
                overwrite=True,
            ),
        )
    )

    assert hidden.end_step == 1
    assert resumed.start_step == 1
    assert resumed.end_step == 2
    assert resumed.final_logits_kl is not None
    assert "initialized missing LM head" in " ".join(resumed.notes)
    loaded = load_checkpoint(logits_checkpoint)
    assert "lm_head" in loaded.params
    assert loaded.manifest.step == 2


def _logits_losses() -> DistillLossConfig:
    return DistillLossConfig(
        hidden_mse=LossWeightConfig(enabled=True, weight=0.5),
        logits_kl=LossWeightConfig(enabled=True, weight=0.5),
    )


def _fake_bundle(tmp_path: Path, *, name: str, include_logits: bool) -> Path:
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
            output_dir=tmp_path / name,
            batch_size=2,
            num_shards=1,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    return config.runtime.output_dir
