from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CANONICAL_SPLITS = ("train", "validation", "test", "unspecified")
_SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


@dataclass(frozen=True)
class PromptRecord:
    id: str
    text: str
    split: str = "unspecified"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptCorpus:
    corpus_id: str
    records: tuple[PromptRecord, ...]
    source_path: Path | None = None


@dataclass(frozen=True)
class PromptCorpusManifest:
    schema_version: str
    corpus_id: str
    description: str
    prompt_count: int
    splits: dict[str, int]
    tags: tuple[str, ...]
    sha256: str
    source_path: str | None
    notes: tuple[str, ...] = ()
    created_by: str = "qrwkv_xla.prompting.corpus"


def normalize_prompt_split(value: str | None) -> str:
    split = (value or "unspecified").strip().lower()
    split = _SPLIT_ALIASES.get(split, split)
    if split not in CANONICAL_SPLITS:
        allowed = ", ".join(CANONICAL_SPLITS)
        raise ValueError(f"Prompt split must be one of {{{allowed}}}, got {value!r}")
    return split


def read_prompt_corpus(
    path: str | Path,
    *,
    corpus_id: str | None = None,
) -> PromptCorpus:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise ValueError(f"Prompt corpus path does not exist: {corpus_path}")
    records: list[PromptRecord] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in prompt corpus {corpus_path} at line "
                    f"{line_number}: {exc.msg}"
                ) from exc
            records.append(prompt_record_from_dict(payload, line_number=line_number))
    corpus = PromptCorpus(
        corpus_id=corpus_id or corpus_path.stem,
        records=tuple(records),
        source_path=corpus_path,
    )
    validate_prompt_corpus(corpus)
    return corpus


def write_prompt_corpus(
    corpus: PromptCorpus,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_prompt_corpus(corpus)
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise ValueError(f"Prompt corpus path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            prompt_record_to_dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for record in corpus.records
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def validate_prompt_corpus(corpus: PromptCorpus) -> None:
    if not corpus.corpus_id.strip():
        raise ValueError("Prompt corpus corpus_id must be non-empty")
    if not corpus.records:
        raise ValueError("Prompt corpus must contain at least one record")

    seen_ids: set[str] = set()
    for index, record in enumerate(corpus.records):
        location = f"records[{index}]"
        normalized_id = record.id.strip()
        if not normalized_id:
            raise ValueError(f"Prompt corpus {location}.id must be non-empty")
        if normalized_id in seen_ids:
            raise ValueError(f"Prompt corpus record id is duplicated: {record.id!r}")
        seen_ids.add(normalized_id)
        if not record.text.strip():
            raise ValueError(f"Prompt corpus {location}.text must be non-empty")
        normalize_prompt_split(record.split)
        for tag_index, tag in enumerate(record.tags):
            if not str(tag).strip():
                raise ValueError(
                    f"Prompt corpus {location}.tags[{tag_index}] must be non-empty"
                )
        if not isinstance(record.metadata, dict):
            raise ValueError(f"Prompt corpus {location}.metadata must be a mapping")


def filter_prompt_corpus(
    corpus: PromptCorpus,
    *,
    split: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> PromptCorpus:
    validate_prompt_corpus(corpus)
    selected = corpus.records
    normalized_split = normalize_prompt_split(split) if split is not None else None
    if normalized_split is not None:
        selected = tuple(
            record for record in selected if record.split == normalized_split
        )
    normalized_tags = tuple(
        str(tag).strip() for tag in (tags or ()) if str(tag).strip()
    )
    for tag in normalized_tags:
        selected = tuple(record for record in selected if tag in record.tags)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"Prompt corpus limit must be > 0, got {limit}")
        selected = selected[:limit]
    return PromptCorpus(
        corpus_id=corpus.corpus_id,
        records=selected,
        source_path=corpus.source_path,
    )


def build_prompt_corpus_manifest(
    corpus: PromptCorpus,
    *,
    description: str = "",
    notes: list[str] | None = None,
) -> PromptCorpusManifest:
    validate_prompt_corpus(corpus)
    split_counts = {
        split: sum(1 for record in corpus.records if record.split == split)
        for split in CANONICAL_SPLITS
    }
    tag_values = sorted({tag for record in corpus.records for tag in record.tags})
    from qrwkv_xla.prompting.hash import compute_prompt_corpus_hash

    return PromptCorpusManifest(
        schema_version="0.1",
        corpus_id=corpus.corpus_id,
        description=description,
        prompt_count=len(corpus.records),
        splits={key: value for key, value in split_counts.items() if value},
        tags=tuple(tag_values),
        sha256=compute_prompt_corpus_hash(corpus),
        source_path=str(corpus.source_path) if corpus.source_path is not None else None,
        notes=tuple(str(note) for note in (notes or [])),
    )


def write_prompt_corpus_manifest(
    manifest: PromptCorpusManifest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise ValueError(f"Prompt manifest path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def prompt_record_from_dict(
    payload: Any,
    *,
    line_number: int | None = None,
) -> PromptRecord:
    location = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt corpus record{location} must be an object")

    record_id = str(payload.get("id", "")).strip()
    text = str(payload.get("text", "")).strip()
    split = normalize_prompt_split(payload.get("split"))
    tags = _normalize_tags(payload.get("tags"), location=location)
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not record_id:
        raise ValueError(f"Prompt corpus record{location} id must be non-empty")
    if not text:
        raise ValueError(f"Prompt corpus record{location} text must be non-empty")
    if not isinstance(metadata, dict):
        raise ValueError(f"Prompt corpus record{location} metadata must be a mapping")
    return PromptRecord(
        id=record_id,
        text=text,
        split=split,
        tags=tags,
        metadata=dict(metadata),
    )


def prompt_record_to_dict(record: PromptRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "metadata": record.metadata,
        "split": record.split,
        "tags": list(record.tags),
        "text": record.text,
    }


def _normalize_tags(value: Any, *, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Prompt corpus record{location} tags must be a sequence")
    tags = tuple(str(item).strip() for item in value)
    if any(not tag for tag in tags):
        raise ValueError(f"Prompt corpus record{location} tags must be non-empty")
    return tags
