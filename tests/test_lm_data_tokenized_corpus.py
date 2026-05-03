from __future__ import annotations

import json
from pathlib import Path

from qrwkv_xla.generation import SmokeTokenizer
from qrwkv_xla.lm import (
    LMDataConfig,
    LMStageConfig,
    LMStudentConfig,
    LMTrainingConfig,
    load_lm_stage_config,
    load_lm_token_sequences,
    run_lm_stage,
)
from qrwkv_xla.lm.tokenized_corpus import write_tokenized_corpus_from_prompt_jsonl


def test_load_lm_token_sequences_from_tokenized_corpus(tmp_path: Path) -> None:
    corpus_dir = _write_tokenized(tmp_path)

    sequences = load_lm_token_sequences(
        LMDataConfig(
            tokenized_corpus=corpus_dir,
            sequence_length=3,
            batch_size=2,
            tokenizer="smoke",
        )
    )

    assert sequences[0] == [98, 109, 113, 105]
    assert len(sequences) == 3


def test_load_lm_stage_config_with_tokenized_corpus(tmp_path: Path) -> None:
    corpus_dir = _write_tokenized(tmp_path)
    config_path = tmp_path / "lm_tokenized.yaml"
    config_path.write_text(
        f"""
lm:
  data:
    tokenized_corpus: {corpus_dir}
    sequence_length: 3
    batch_size: 2
    tokenizer: smoke
  student:
    architecture: tiny_student
    vocab_size: 512
    hidden_size: 8
    num_layers: 2
    emit_logits: true
  training:
    stage: 3
    max_steps: 1
""",
        encoding="utf-8",
    )

    config = load_lm_stage_config(config_path)

    assert config.data.prompt_corpus is None
    assert config.data.tokenized_corpus == corpus_dir


def test_run_lm_stage_with_tokenized_corpus(tmp_path: Path) -> None:
    corpus_dir = _write_tokenized(tmp_path)
    config = LMStageConfig(
        data=LMDataConfig(
            tokenized_corpus=corpus_dir,
            sequence_length=3,
            batch_size=2,
            tokenizer="smoke",
        ),
        student=LMStudentConfig(
            architecture="tiny_student",
            vocab_size=512,
            hidden_size=8,
            num_layers=2,
            emit_logits=True,
        ),
        training=LMTrainingConfig(max_steps=1),
    )

    result = run_lm_stage(config)

    assert result.prompt_corpus is None
    assert result.tokenized_corpus == corpus_dir
    assert result.final_loss == result.final_ce_loss


def _write_tokenized(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "prompts.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "text": "alpha", "split": "train"}),
                json.dumps({"id": "two", "text": "beta", "split": "train"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tok"
    write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        output_dir,
        tokenizer=SmokeTokenizer(),
        sequence_length=3,
        shard_size_tokens=6,
        overwrite=True,
        created_at="2026-05-03T22:20:00+00:00",
    )
    return output_dir
