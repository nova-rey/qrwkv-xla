from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillStageConfig,
    DistillStudentConfig,
    run_distill_stage,
)
from qrwkv_xla.targets import TargetFlags, TeacherTargetManifest, write_target_bundle
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


def test_radlads_reference_checkpoint_save_then_resume(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    first_dir = tmp_path / "checkpoints" / "radlads_first"
    second_dir = tmp_path / "checkpoints" / "radlads_second"
    base_config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(
            architecture="rwkv7_radlads_reference",
            vocab_size=32,
            num_heads=2,
        ),
        training=replace(DistillStageConfig().training, max_steps=1),
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
            checkpoint=DistillCheckpointConfig(
                resume_from=first_dir,
                checkpoint_out=second_dir,
                overwrite=True,
            ),
        )
    )

    loaded = load_checkpoint(second_dir)
    assert first.start_step == 0
    assert first.end_step == 1
    assert second.start_step == 1
    assert second.end_step == 2
    assert loaded.manifest.student_architecture == "rwkv7_radlads_reference"
    assert loaded.manifest.student_config["num_heads"] == 2


def test_radlads_reference_tiny_hf_shaped_logits_bundle_resume_hidden_only(
    tmp_path: Path,
) -> None:
    bundle_dir = _tiny_hf_shaped_bundle(tmp_path)
    first_dir = tmp_path / "checkpoints" / "tiny_hf_first"
    second_dir = tmp_path / "checkpoints" / "tiny_hf_second"
    base_config = DistillStageConfig(
        targets_dir=bundle_dir,
        student=DistillStudentConfig(
            architecture="rwkv7_radlads_reference",
            vocab_size=50257,
            num_heads=1,
            emit_logits=False,
        ),
        training=replace(DistillStageConfig().training, max_steps=1),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.001),
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
        )
    )

    loaded = load_checkpoint(second_dir)
    assert first.start_step == 0
    assert first.end_step == 1
    assert second.start_step == 1
    assert second.end_step == 2
    assert second.final_logits_kl is None
    assert loaded.manifest.target_manifest is not None
    assert loaded.manifest.target_manifest["targets"]["logits"] is True
    assert loaded.manifest.student_config["hidden_size"] == 2
    assert loaded.manifest.student_config["num_layers"] == 2
    assert loaded.manifest.student_config["vocab_size"] == 50257
    assert loaded.manifest.student_config["emit_logits"] is False


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


def _tiny_hf_shaped_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "tiny_hf_bundle"
    manifest = TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="hf-causal-lm",
        teacher_model_id="sshleifer/tiny-gpt2",
        teacher_policy_label="tiny-hf-smoke-p29",
        fallback_policy_label=None,
        tokenizer_id="sshleifer/tiny-gpt2",
        sequence_length=8,
        hidden_size=2,
        num_layers=2,
        targets=TargetFlags(logits=True),
        dtype="fp32",
        created_by="test",
        notes=["offline tiny HF-shaped fixture"],
        extra={"vocab_size": 50257},
    )
    shard = {
        "input_ids": np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
            dtype=np.int32,
        ),
        "attention_mask": np.ones((2, 8), dtype=np.int32),
        "loss_mask": np.ones((2, 8), dtype=np.int32),
        "hidden_states": np.linspace(
            -0.25,
            0.25,
            num=2 * 2 * 8 * 2,
            dtype=np.float32,
        ).reshape(2, 2, 8, 2),
        "logits": np.zeros((2, 8, 50257), dtype=np.float32),
    }
    write_target_bundle(bundle_dir, manifest, [shard])
    return bundle_dir
