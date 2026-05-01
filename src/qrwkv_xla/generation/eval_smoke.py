from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from qrwkv_xla.generation.artifacts import (
    GenerationRecord,
    write_generation_jsonl,
    write_generation_summary,
)
from qrwkv_xla.generation.greedy import greedy_generate
from qrwkv_xla.generation.load import load_student_from_checkpoint
from qrwkv_xla.generation.tokenizer import SmokeTokenizer
from qrwkv_xla.prompting import filter_prompt_corpus, read_prompt_corpus


@dataclass(frozen=True)
class GenerationSmokeResult:
    output_dir: Path
    prompt_count: int
    passed: bool
    summary: dict[str, object]


def run_generation_smoke(
    *,
    checkpoint_dir: str | Path,
    prompts: list[tuple[str, str]],
    output_dir: str | Path,
    max_new_tokens: int = 16,
    vocab_size: int | None = None,
) -> GenerationSmokeResult:
    if not prompts:
        raise ValueError("generation smoke requires at least one prompt")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    loaded = load_student_from_checkpoint(checkpoint_dir)
    checkpoint_vocab_size = int(loaded.manifest.student_config["vocab_size"])
    tokenizer = SmokeTokenizer(vocab_size=vocab_size or checkpoint_vocab_size)
    output_path = Path(output_dir)

    records: list[GenerationRecord] = []
    for prompt_id, prompt_text in prompts:
        prompt_token_ids = tokenizer.encode(prompt_text)
        result = greedy_generate(
            student=loaded.student,
            params=loaded.params,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded_text = tokenizer.decode(result.full_token_ids)
        records.append(
            GenerationRecord(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                prompt_token_ids=result.prompt_token_ids,
                generated_token_ids=result.generated_token_ids,
                full_token_ids=result.full_token_ids,
                decoded_text=decoded_text,
                metadata={
                    "checkpoint_dir": str(loaded.checkpoint_dir),
                    "max_new_tokens": max_new_tokens,
                    "tokenizer": "smoke",
                },
            )
        )

    passed = all(record.generated_token_ids for record in records)
    summary: dict[str, object] = {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "prompt_count": len(records),
        "passed": passed,
        "max_new_tokens": max_new_tokens,
        "tokenizer": "smoke",
        "vocab_size": tokenizer.vocab_size,
        "artifacts": {
            "generations_jsonl": str(output_path / "generations.jsonl"),
            "summary_json": str(output_path / "summary.json"),
        },
        "limitations": [
            "smoke tokenizer only",
            "greedy decoding only",
            "no quality benchmark",
        ],
    }
    write_generation_jsonl(records, output_path / "generations.jsonl")
    write_generation_summary(summary, output_path / "summary.json")
    return GenerationSmokeResult(
        output_dir=output_path,
        prompt_count=len(records),
        passed=passed,
        summary=summary,
    )


def load_generation_smoke_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("generation smoke config root must be a mapping")
    generation = data.get("generation", data)
    if not isinstance(generation, dict):
        raise ValueError("generation smoke config section must be a mapping")
    return dict(generation)


def load_prompts_from_config(
    config: dict[str, Any],
    *,
    prompt_corpus: str | Path | None = None,
    prompt_split: str | None = None,
    prompt_tags: list[str] | tuple[str, ...] | None = None,
    prompt_limit: int | None = None,
) -> list[tuple[str, str]]:
    corpus_path = prompt_corpus or config.get("prompt_corpus")
    if corpus_path is None:
        return [("default", "Hello from QRWKV-XLA")]
    corpus = read_prompt_corpus(corpus_path)
    selected = filter_prompt_corpus(
        corpus,
        split=prompt_split or config.get("prompt_split"),
        tags=prompt_tags if prompt_tags is not None else config.get("prompt_tags", []),
        limit=prompt_limit
        if prompt_limit is not None
        else int(config.get("prompt_limit", 4)),
    )
    if not selected.records:
        raise ValueError("generation smoke prompt selection produced no prompts")
    return [(record.id, record.text) for record in selected.records]
