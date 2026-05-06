from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.teacher_export import load_teacher_export_config
from qrwkv_xla.teacher_export.prompts import DEFAULT_TINY_PROMPTS, load_prompt_texts

ROOT = Path(__file__).resolve().parents[1]


def test_hf_tiny_config_loads() -> None:
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_hf_tiny.yaml"
    )

    assert config.runtime.exporter_backend == "hf"
    assert config.teacher.family == "hf-causal-lm"
    assert config.teacher.device == "cpu"
    assert config.teacher.dtype == "auto"
    assert config.targets.hidden_size is None
    assert config.targets.num_layers is None
    assert config.targets.include_logits is True
    assert config.targets.prompt_texts == (
        "The quick brown fox",
        "QRWKV-XLA teacher export smoke",
    )


def test_hf_tiny_corpus_config_loads() -> None:
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_hf_tiny_corpus.yaml"
    )

    assert config.targets.prompt_corpus == ROOT / "corpora" / "smoke_prompts.jsonl"
    assert config.targets.prompt_split == "train"
    assert config.targets.prompt_tags == ("smoke",)
    assert config.targets.prompt_limit == 4


def test_fake_config_still_loads() -> None:
    config = load_teacher_export_config(ROOT / "configs" / "teacher_export_stub.yaml")

    assert config.runtime.exporter_backend == "fake"
    assert config.targets.hidden_size == 128
    assert config.targets.num_layers == 2


def test_load_prompt_texts_uses_defaults_when_no_prompts_present() -> None:
    prompts = load_prompt_texts()
    assert prompts == list(DEFAULT_TINY_PROMPTS)


def test_prompt_file_is_resolved_relative_to_config(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("one\n\ntwo\n", encoding="utf-8")
    config_file = tmp_path / "teacher.yaml"
    config_file.write_text(
        """
teacher:
  family: hf-causal-lm
  resolved_model_id: tiny-model
targets:
  hidden_size: null
  num_layers: null
  prompt_file: prompts.txt
runtime:
  exporter_backend: hf
""",
        encoding="utf-8",
    )

    config = load_teacher_export_config(config_file)

    assert config.targets.prompt_file == prompt_file
    assert load_prompt_texts(prompt_file=config.targets.prompt_file) == ["one", "two"]


def test_config_relative_prompt_corpus_and_output_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    corpus_dir = config_dir / "corpora"
    corpus_dir.mkdir(parents=True)
    prompt_corpus = corpus_dir / "prompts.jsonl"
    prompt_corpus.write_text(
        '{"id":"p0","text":"one","split":"train","tags":["smoke"]}\n',
        encoding="utf-8",
    )
    config_file = config_dir / "teacher.yaml"
    config_file.write_text(
        """
targets:
  prompt_corpus: corpora/prompts.jsonl
runtime:
  output_dir: exports/bundle
""",
        encoding="utf-8",
    )

    config = load_teacher_export_config(config_file)

    assert config.targets.prompt_corpus == prompt_corpus.resolve()
    assert config.runtime.output_dir == (config_dir / "exports" / "bundle").resolve()


def test_absolute_config_paths_stay_absolute(tmp_path: Path) -> None:
    prompt_corpus = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "teacher_targets" / "bundle"
    prompt_corpus.write_text(
        '{"id":"p0","text":"one","split":"train","tags":["smoke"]}\n',
        encoding="utf-8",
    )
    config_file = tmp_path / "teacher.yaml"
    config_file.write_text(
        f"""
targets:
  prompt_corpus: {prompt_corpus}
runtime:
  output_dir: {output_dir}
""",
        encoding="utf-8",
    )

    config = load_teacher_export_config(config_file)

    assert config.targets.prompt_corpus == prompt_corpus
    assert config.runtime.output_dir == output_dir


def test_empty_prompt_source_raises() -> None:
    with pytest.raises(ValueError, match="empty prompt list"):
        load_prompt_texts(prompt_texts=[])
