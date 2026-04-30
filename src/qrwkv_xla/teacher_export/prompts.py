from __future__ import annotations

from pathlib import Path

from qrwkv_xla.teacher_export.config import TeacherExportConfig

DEFAULT_TINY_PROMPTS = (
    "The quick brown fox",
    "QRWKV-XLA is a recurrent distillation project",
)


def load_prompt_texts(
    *,
    prompt_texts: list[str] | tuple[str, ...] | None = None,
    prompt_file: str | Path | None = None,
) -> list[str]:
    prompts: list[str]
    if prompt_texts is not None:
        prompts = [prompt.strip() for prompt in prompt_texts if prompt.strip()]
    elif prompt_file is not None:
        prompts = [
            line.strip()
            for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        prompts = [prompt.strip() for prompt in DEFAULT_TINY_PROMPTS if prompt.strip()]

    if not prompts:
        raise ValueError("prompt_texts resolved to an empty prompt list")
    return prompts


def resolve_prompt_texts(config: TeacherExportConfig) -> list[str]:
    return load_prompt_texts(
        prompt_texts=list(config.targets.prompt_texts) or None,
        prompt_file=config.targets.prompt_file,
    )
