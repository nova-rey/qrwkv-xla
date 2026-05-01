from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalPromptConfig:
    prompt_corpus: Path
    prompt_split: str | None = None
    prompt_tags: tuple[str, ...] = ()
    prompt_limit: int | None = None


@dataclass(frozen=True)
class EvalGenerationConfig:
    max_new_tokens: int = 16
    tokenizer: str = "smoke"
    eos_token_id: int | None = 0


@dataclass(frozen=True)
class EvalSanityConfig:
    require_non_empty: bool = True
    max_repeated_token_fraction: float | None = 0.95
    max_unknown_token_fraction: float | None = 0.95


@dataclass(frozen=True)
class EvalConfig:
    prompt: EvalPromptConfig
    eval_id: str | None = None
    generation: EvalGenerationConfig = field(default_factory=EvalGenerationConfig)
    sanity: EvalSanityConfig = field(default_factory=EvalSanityConfig)
    output_dir: Path = Path("eval_outputs/regression_smoke")


def load_eval_config(path: str | Path) -> EvalConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("eval config root must be a mapping")

    prompt_payload = payload.get("prompt")
    if not isinstance(prompt_payload, dict):
        raise ValueError("eval config prompt section must be a mapping")

    config = EvalConfig(
        eval_id=_optional_str(payload.get("eval_id")),
        prompt=EvalPromptConfig(
            prompt_corpus=Path(_required_str(prompt_payload, "prompt_corpus")),
            prompt_split=_optional_str(prompt_payload.get("prompt_split")),
            prompt_tags=tuple(
                str(tag).strip()
                for tag in prompt_payload.get("prompt_tags", ())
                if str(tag).strip()
            ),
            prompt_limit=_optional_int(prompt_payload.get("prompt_limit")),
        ),
        generation=_load_generation_config(payload.get("generation", {})),
        sanity=_load_sanity_config(payload.get("sanity", {})),
        output_dir=Path(
            str(payload.get("output_dir", "eval_outputs/regression_smoke"))
        ),
    )
    validate_eval_config(config)
    return config


def validate_eval_config(config: EvalConfig) -> None:
    if not config.prompt.prompt_corpus.exists():
        raise ValueError(
            f"prompt corpus path does not exist: {config.prompt.prompt_corpus}"
        )
    if config.prompt.prompt_limit is not None and config.prompt.prompt_limit <= 0:
        raise ValueError("prompt_limit must be > 0")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if config.generation.tokenizer != "smoke":
        raise ValueError("eval generation tokenizer must be 'smoke'")
    _validate_fraction(
        config.sanity.max_repeated_token_fraction,
        "max_repeated_token_fraction",
    )
    _validate_fraction(
        config.sanity.max_unknown_token_fraction,
        "max_unknown_token_fraction",
    )


def replace_eval_overrides(
    config: EvalConfig,
    *,
    max_new_tokens: int | None = None,
    prompt_limit: int | None = None,
    output_dir: str | Path | None = None,
) -> EvalConfig:
    updated = EvalConfig(
        eval_id=config.eval_id,
        prompt=EvalPromptConfig(
            prompt_corpus=config.prompt.prompt_corpus,
            prompt_split=config.prompt.prompt_split,
            prompt_tags=config.prompt.prompt_tags,
            prompt_limit=prompt_limit
            if prompt_limit is not None
            else config.prompt.prompt_limit,
        ),
        generation=EvalGenerationConfig(
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else config.generation.max_new_tokens,
            tokenizer=config.generation.tokenizer,
            eos_token_id=config.generation.eos_token_id,
        ),
        sanity=config.sanity,
        output_dir=Path(output_dir) if output_dir is not None else config.output_dir,
    )
    validate_eval_config(updated)
    return updated


def _load_generation_config(payload: Any) -> EvalGenerationConfig:
    if not isinstance(payload, dict):
        raise ValueError("eval config generation section must be a mapping")
    return EvalGenerationConfig(
        max_new_tokens=int(payload.get("max_new_tokens", 16)),
        tokenizer=str(payload.get("tokenizer", "smoke")),
        eos_token_id=_optional_int(payload.get("eos_token_id", 0)),
    )


def _load_sanity_config(payload: Any) -> EvalSanityConfig:
    if not isinstance(payload, dict):
        raise ValueError("eval config sanity section must be a mapping")
    return EvalSanityConfig(
        require_non_empty=bool(payload.get("require_non_empty", True)),
        max_repeated_token_fraction=_optional_float(
            payload.get("max_repeated_token_fraction", 0.95)
        ),
        max_unknown_token_fraction=_optional_float(
            payload.get("max_unknown_token_fraction", 0.95)
        ),
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"eval config prompt.{key} must be non-empty")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _validate_fraction(value: float | None, name: str) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
