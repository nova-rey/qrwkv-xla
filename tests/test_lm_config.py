from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.lm import load_lm_stage_config


def test_load_lm_stage_config_from_smoke_config() -> None:
    config = load_lm_stage_config("configs/lm_stage3_smoke.yaml")

    assert config.training.stage == 3
    assert config.student.emit_logits is True
    assert config.data.tokenizer == "smoke"
    assert config.data.prompt_corpus == Path("corpora/smoke_prompts.jsonl")


def test_lm_stage_config_rejects_hidden_only_student(tmp_path: Path) -> None:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    sequence_length: 8
    batch_size: 1
  student:
    architecture: tiny_student
    vocab_size: 512
    hidden_size: 8
    num_layers: 2
    emit_logits: false
  training:
    stage: 3
    max_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="emit_logits"):
        load_lm_stage_config(config_path)
