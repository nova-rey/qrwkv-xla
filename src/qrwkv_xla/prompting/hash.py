from __future__ import annotations

import hashlib
import json

from qrwkv_xla.prompting.corpus import (
    PromptCorpus,
    PromptRecord,
    normalize_prompt_split,
    prompt_record_to_dict,
)


def hash_prompt_records(records: tuple[PromptRecord, ...] | list[PromptRecord]) -> str:
    return compute_prompt_corpus_hash(
        PromptCorpus(corpus_id="__records__", records=tuple(records))
    )


def prompt_record_to_canonical_json(record: PromptRecord) -> str:
    canonical = prompt_record_to_dict(
        PromptRecord(
            id=record.id.strip(),
            text=record.text.strip(),
            split=normalize_prompt_split(record.split),
            tags=tuple(tag.strip() for tag in record.tags if tag.strip()),
            metadata=dict(record.metadata),
        )
    )
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_prompt_corpus_hash(corpus: PromptCorpus) -> str:
    hasher = hashlib.sha256()
    for record in corpus.records:
        hasher.update(prompt_record_to_canonical_json(record).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()
