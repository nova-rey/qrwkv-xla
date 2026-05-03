from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.lm import load_lm_stage_config


def test_load_lm_stage_config_from_smoke_config() -> None:
    config = load_lm_stage_config("configs/lm_stage3_smoke.yaml")

    assert config.training.stage == 3
    assert config.student.emit_logits is True
    assert config.data.tokenizer.backend == "smoke"
    assert config.data.prompt_corpus == Path("corpora/smoke_prompts.jsonl")


def test_lm_stage_config_loads_tokenizer_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    sequence_length: 8
    batch_size: 1
    tokenizer:
      backend: smoke
      vocab_size: 512
      eos_token_id: 0
      pad_token_id: 0
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

    assert config.data.tokenizer.backend == "smoke"
    assert config.data.tokenizer.vocab_size == 512


def test_lm_stage_config_rejects_tokenizer_vocab_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    sequence_length: 8
    batch_size: 1
    tokenizer:
      backend: smoke
      vocab_size: 513
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

    with pytest.raises(ValueError, match="tokenizer.vocab_size"):
        load_lm_stage_config(config_path)


def test_lm_stage_config_rejects_unknown_tokenizer_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    sequence_length: 8
    batch_size: 1
    tokenizer: nope
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

    with pytest.raises(ValueError, match="data.tokenizer backend"):
        load_lm_stage_config(config_path)


def test_lm_stage_config_rejects_hf_tokenizer_without_id(tmp_path: Path) -> None:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    sequence_length: 8
    batch_size: 1
    tokenizer:
      backend: hf
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

    with pytest.raises(ValueError, match="tokenizer_id"):
        load_lm_stage_config(config_path)


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


def test_lm_stage_config_loads_distributed_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    sequence_length: 8
    batch_size: 2
  distributed:
    enabled: true
    mode: pmap_data_parallel
    axis_name: data
    min_device_count: 2
  training:
    stage: 3
    max_steps: 1
""",
        encoding="utf-8",
    )
    config = load_lm_stage_config(config_path)
    assert config.distributed.enabled is True
    assert config.distributed.mode == "pmap_data_parallel"
    assert config.distributed.min_device_count == 2
