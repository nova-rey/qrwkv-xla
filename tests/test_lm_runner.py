from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.lm import (
    LMDataConfig,
    LMStageConfig,
    LMStudentConfig,
    LMTrainingConfig,
    run_lm_stage,
)


def test_run_lm_stage_with_tiny_student(tmp_path: Path) -> None:
    config = _lm_config(tmp_path)

    result = run_lm_stage(config)

    assert result.stage == 3
    assert result.steps == 2
    assert result.end_step == 2
    assert result.final_loss == pytest.approx(result.final_ce_loss)
    assert result.prompt_corpus == config.data.prompt_corpus


def test_lm_stage_requires_emit_logits(tmp_path: Path) -> None:
    config = replace(
        _lm_config(tmp_path),
        student=replace(_lm_config(tmp_path).student, emit_logits=False),
    )

    with pytest.raises(ValueError, match="emit_logits"):
        run_lm_stage(config)


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
