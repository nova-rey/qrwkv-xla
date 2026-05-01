from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qrwkv_xla.prompting import (
    PromptCorpus,
    build_prompt_corpus_manifest,
    canonical_split,
    compute_prompt_corpus_hash,
    filter_prompt_corpus,
    read_prompt_corpus,
)
from qrwkv_xla.teacher_export.config import TeacherExportConfig

DEFAULT_TINY_PROMPTS = (
    "The quick brown fox",
    "QRWKV-XLA is a recurrent distillation project",
)


@dataclass(frozen=True)
class LoadedPrompts:
    texts: list[str]
    metadata: dict[str, object]


ResolvedPrompts = LoadedPrompts


def load_prompt_texts(
    *,
    prompt_texts: list[str] | tuple[str, ...] | None = None,
    prompt_file: str | Path | None = None,
    prompt_corpus: str | Path | None = None,
    prompt_split: str | None = None,
    prompt_tags: list[str] | tuple[str, ...] | None = None,
    prompt_limit: int | None = None,
) -> list[str]:
    return load_prompt_bundle(
        prompt_texts=prompt_texts,
        prompt_file=prompt_file,
        prompt_corpus=prompt_corpus,
        prompt_split=prompt_split,
        prompt_tags=prompt_tags,
        prompt_limit=prompt_limit,
    ).texts


def load_prompt_bundle(
    *,
    prompt_texts: list[str] | tuple[str, ...] | None = None,
    prompt_file: str | Path | None = None,
    prompt_corpus: str | Path | None = None,
    prompt_split: str | None = None,
    prompt_tags: list[str] | tuple[str, ...] | None = None,
    prompt_limit: int | None = None,
) -> LoadedPrompts:
    prompt_texts_provided = prompt_texts is not None
    inline_prompts = [
        prompt.strip() for prompt in (prompt_texts or ()) if prompt.strip()
    ]
    has_inline_source = prompt_texts_provided or prompt_file is not None
    if prompt_corpus is not None and has_inline_source:
        raise ValueError(
            "Use either prompt_texts/prompt_file or prompt_corpus, not both."
        )

    if prompt_corpus is not None:
        corpus = read_prompt_corpus(prompt_corpus)
        filtered = filter_prompt_corpus(
            corpus,
            split=prompt_split,
            tags=prompt_tags,
            limit=prompt_limit,
        )
        if not filtered.records:
            raise ValueError("prompt_corpus filters resolved to an empty prompt list")
        manifest = build_prompt_corpus_manifest(corpus)
        selected_hash = compute_prompt_corpus_hash(
            PromptCorpus(
                corpus_id=filtered.corpus_id,
                records=filtered.records,
                source_path=filtered.source_path,
            )
        )
        metadata = {
            "type": "corpus",
            "corpus_id": corpus.corpus_id,
            "corpus_sha256": manifest.sha256,
            "corpus_path": str(corpus.source_path) if corpus.source_path else None,
            "prompt_count": len(filtered.records),
            "prompt_ids": [record.id for record in filtered.records],
            "selection_sha256": selected_hash,
        }
        if prompt_split is not None:
            metadata["prompt_split"] = canonical_split(prompt_split)
        normalized_tags = [str(tag) for tag in (prompt_tags or ()) if str(tag).strip()]
        if normalized_tags:
            metadata["prompt_tags"] = normalized_tags
        if prompt_limit is not None:
            metadata["prompt_limit"] = prompt_limit
        return LoadedPrompts(
            texts=[record.text for record in filtered.records],
            metadata=metadata,
        )

    if prompt_texts_provided:
        if not inline_prompts:
            raise ValueError("prompt_texts resolved to an empty prompt list")
        return LoadedPrompts(
            texts=inline_prompts,
            metadata={"type": "inline", "prompt_count": len(inline_prompts)},
        )

    if prompt_file is not None:
        texts = [
            line.strip()
            for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not texts:
            raise ValueError("prompt_file resolved to an empty prompt list")
        return LoadedPrompts(
            texts=texts,
            metadata={
                "type": "file",
                "prompt_count": len(texts),
                "prompt_file": str(prompt_file),
            },
        )

    texts = [prompt.strip() for prompt in DEFAULT_TINY_PROMPTS if prompt.strip()]
    if not texts:
        raise ValueError("default prompt set resolved to an empty prompt list")
    return LoadedPrompts(
        texts=texts,
        metadata={"type": "default", "prompt_count": len(texts)},
    )


def resolve_prompt_texts(config: TeacherExportConfig) -> list[str]:
    return resolve_prompts(config).texts


def resolve_prompts(config: TeacherExportConfig) -> LoadedPrompts:
    return load_prompt_bundle(
        prompt_texts=list(config.targets.prompt_texts) or None,
        prompt_file=config.targets.prompt_file,
        prompt_corpus=config.targets.prompt_corpus,
        prompt_split=config.targets.prompt_split,
        prompt_tags=list(config.targets.prompt_tags) or None,
        prompt_limit=config.targets.prompt_limit,
    )


def load_prompt_source_metadata(config: TeacherExportConfig) -> dict[str, Any]:
    return dict(resolve_prompts(config).metadata)
