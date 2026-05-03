from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.lm import (
    LMDataConfig,
    LMStageConfig,
    LMStudentConfig,
    LMTrainingConfig,
    run_lm_stage,
)


def test_lm_stage_checkpoint_resume_advances_from_saved_step(tmp_path: Path) -> None:
    config = _lm_config(tmp_path)
    first_checkpoint = tmp_path / "checkpoints" / "stage3"
    resumed_checkpoint = tmp_path / "checkpoints" / "stage3_resume"

    first = run_lm_stage(
        replace(
            config,
            checkpoint=replace(
                config.checkpoint,
                checkpoint_out=first_checkpoint,
                overwrite=True,
            ),
        )
    )
    resumed = run_lm_stage(
        replace(
            config,
            training=replace(config.training, max_steps=1),
            checkpoint=replace(
                config.checkpoint,
                resume_from=first_checkpoint,
                checkpoint_out=resumed_checkpoint,
                overwrite=True,
            ),
        )
    )

    assert first.end_step == 2
    assert resumed.start_step == 2
    assert resumed.end_step == 3
    loaded = load_checkpoint(resumed_checkpoint)
    assert loaded.manifest.step == 3
    assert loaded.manifest.loss_config["next_token_ce"]["enabled"] is True
    assert loaded.manifest.target_manifest["type"] == "prompt_corpus"


def _lm_config(tmp_path: Path) -> LMStageConfig:
    corpus_path = tmp_path / "prompts.jsonl"
    corpus_path.write_text(
        '{"id":"one","text":"hello","split":"train","tags":[]}\n'
        '{"id":"two","text":"world","split":"train","tags":[]}\n',
        encoding="utf-8",
    )
    return LMStageConfig(
        data=LMDataConfig(
            prompt_corpus=corpus_path,
            sequence_length=8,
            batch_size=2,
        ),
        student=LMStudentConfig(
            architecture="tiny_student",
            vocab_size=512,
            hidden_size=8,
            num_layers=2,
            emit_logits=True,
        ),
        training=LMTrainingConfig(max_steps=2),
    )
