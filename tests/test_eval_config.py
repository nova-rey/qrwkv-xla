from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.eval.config import (
    EvalConfig,
    EvalGenerationConfig,
    EvalPromptConfig,
    EvalSanityConfig,
    load_eval_config,
    validate_eval_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_eval_regression_smoke_config() -> None:
    config = load_eval_config(ROOT / "configs" / "eval_regression_smoke.yaml")

    assert config.eval_id == "regression_smoke"
    assert config.prompt.prompt_split == "validation"
    assert config.prompt.prompt_tags == ("eval",)
    assert config.generation.tokenizer == "smoke"


def test_invalid_tokenizer_fails() -> None:
    config = EvalConfig(
        prompt=EvalPromptConfig(ROOT / "corpora" / "eval_regression_prompts.jsonl"),
        generation=EvalGenerationConfig(tokenizer="hf"),
    )

    with pytest.raises(ValueError, match="tokenizer"):
        validate_eval_config(config)


def test_invalid_fraction_fails() -> None:
    config = EvalConfig(
        prompt=EvalPromptConfig(ROOT / "corpora" / "eval_regression_prompts.jsonl"),
        sanity=EvalSanityConfig(max_repeated_token_fraction=1.5),
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_eval_config(config)


def test_missing_prompt_corpus_fails(tmp_path: Path) -> None:
    config = EvalConfig(prompt=EvalPromptConfig(tmp_path / "missing.jsonl"))

    with pytest.raises(ValueError, match="prompt corpus path does not exist"):
        validate_eval_config(config)
