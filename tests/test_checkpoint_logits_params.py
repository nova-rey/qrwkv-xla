from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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


def test_logits_checkpoint_saves_and_loads_lm_head(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path, include_logits=True)
    checkpoint_dir = tmp_path / "checkpoints" / "logits"

    run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="tiny_student",
                vocab_size=32,
                emit_logits=True,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            losses=_logits_losses(),
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=checkpoint_dir,
                overwrite=True,
            ),
        )
    )

    loaded = load_checkpoint(checkpoint_dir)
    assert "lm_head" in loaded.params
    np.testing.assert_allclose(
        np.asarray(loaded.params["lm_head"]["bias"]),
        np.asarray(loaded.params["lm_head"]["bias"]),
    )


def test_resume_rejects_incompatible_tie_embeddings(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path, include_logits=True)
    checkpoint_dir = tmp_path / "checkpoints" / "untied"
    run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="tiny_student",
                vocab_size=32,
                emit_logits=True,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            losses=_logits_losses(),
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=checkpoint_dir,
                overwrite=True,
            ),
        )
    )

    with pytest.raises(ValueError, match="tie_embeddings mismatch"):
        run_distill_stage(
            DistillStageConfig(
                targets_dir=bundle_dir,
                student=DistillStudentConfig(
                    architecture="tiny_student",
                    vocab_size=32,
                    emit_logits=True,
                    tie_embeddings=True,
                ),
                training=replace(DistillStageConfig().training, max_steps=1),
                losses=_logits_losses(),
                checkpoint=DistillCheckpointConfig(resume_from=checkpoint_dir),
            )
        )


def _logits_losses() -> DistillLossConfig:
    return DistillLossConfig(
        hidden_mse=LossWeightConfig(enabled=True, weight=0.5),
        logits_kl=LossWeightConfig(enabled=True, weight=0.5),
    )


def _fake_bundle(tmp_path: Path, *, include_logits: bool) -> Path:
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
