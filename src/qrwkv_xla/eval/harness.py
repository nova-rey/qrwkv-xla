from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qrwkv_xla.eval.artifacts import (
    write_eval_json,
    write_generation_records,
    write_sanity_json,
)
from qrwkv_xla.eval.config import EvalConfig
from qrwkv_xla.eval.sanity import run_generation_sanity_checks
from qrwkv_xla.generation.artifacts import GenerationRecord
from qrwkv_xla.generation.greedy import greedy_generate
from qrwkv_xla.generation.load import load_student_from_checkpoint
from qrwkv_xla.generation.tokenizer import SmokeTokenizer
from qrwkv_xla.prompting.corpus import (
    build_prompt_corpus_manifest,
    filter_prompt_corpus,
    read_prompt_corpus,
)


@dataclass(frozen=True)
class EvaluationResult:
    eval_id: str
    checkpoint_dir: Path
    output_dir: Path
    prompt_count: int
    generation_path: Path
    eval_json_path: Path
    sanity_path: Path
    sanity_passed: bool


def run_checkpoint_evaluation(
    *,
    checkpoint_dir: str | Path,
    config: EvalConfig,
    output_dir: str | Path | None = None,
    strict: bool = False,
) -> EvaluationResult:
    output_path = Path(output_dir) if output_dir is not None else config.output_dir
    eval_id = config.eval_id or output_path.name
    corpus = read_prompt_corpus(config.prompt.prompt_corpus)
    selected = filter_prompt_corpus(
        corpus,
        split=config.prompt.prompt_split,
        tags=config.prompt.prompt_tags,
        limit=config.prompt.prompt_limit,
    )
    if not selected.records:
        raise ValueError("eval prompt selection produced no prompts")

    loaded = load_student_from_checkpoint(checkpoint_dir)
    tokenizer = SmokeTokenizer(
        vocab_size=int(loaded.manifest.student_config["vocab_size"])
    )
    records: list[GenerationRecord] = []
    for prompt in selected.records:
        prompt_token_ids = tokenizer.encode(prompt.text)
        generation = greedy_generate(
            student=loaded.student,
            params=loaded.params,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=config.generation.max_new_tokens,
            eos_token_id=config.generation.eos_token_id,
        )
        records.append(
            GenerationRecord(
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                prompt_token_ids=generation.prompt_token_ids,
                generated_token_ids=generation.generated_token_ids,
                full_token_ids=generation.full_token_ids,
                decoded_text=tokenizer.decode(generation.full_token_ids),
                metadata={
                    "checkpoint_dir": str(loaded.checkpoint_dir),
                    "max_new_tokens": config.generation.max_new_tokens,
                    "tokenizer": config.generation.tokenizer,
                    "prompt_split": prompt.split,
                    "prompt_tags": list(prompt.tags),
                },
            )
        )

    sanity = run_generation_sanity_checks(
        records,
        require_non_empty=config.sanity.require_non_empty,
        max_repeated_token_fraction=config.sanity.max_repeated_token_fraction,
        max_unknown_token_fraction=config.sanity.max_unknown_token_fraction,
    )
    generation_path = write_generation_records(
        records,
        output_path / "generations.jsonl",
    )
    sanity_path = write_sanity_json(sanity, output_path / "sanity.json")
    corpus_manifest = build_prompt_corpus_manifest(corpus)
    eval_json_path = write_eval_json(
        {
            "schema_version": "0.1",
            "eval_id": eval_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint_dir": str(loaded.checkpoint_dir),
            "checkpoint": _checkpoint_summary(loaded.manifest),
            "prompt_corpus": {
                "corpus_id": corpus.corpus_id,
                "sha256": corpus_manifest.sha256,
                "path": str(config.prompt.prompt_corpus),
                "prompt_split": config.prompt.prompt_split,
                "prompt_tags": list(config.prompt.prompt_tags),
                "prompt_limit": config.prompt.prompt_limit,
                "prompt_ids": [record.id for record in selected.records],
            },
            "generation": {
                "max_new_tokens": config.generation.max_new_tokens,
                "tokenizer": config.generation.tokenizer,
                "eos_token_id": config.generation.eos_token_id,
                "vocab_size": tokenizer.vocab_size,
            },
            "sanity": {
                "passed": sanity.passed,
                "strict": strict,
                "failed_count": sanity.failed_count,
            },
            "output_paths": {
                "eval_json": str(output_path / "eval.json"),
                "generations_jsonl": str(output_path / "generations.jsonl"),
                "sanity_json": str(output_path / "sanity.json"),
            },
            "limitations": [
                "regression snapshot only",
                "smoke tokenizer only",
                "greedy decoding only",
                "no quality benchmark",
            ],
        },
        output_path / "eval.json",
    )
    if strict and not sanity.passed:
        raise ValueError("evaluation sanity checks failed in strict mode")
    return EvaluationResult(
        eval_id=eval_id,
        checkpoint_dir=loaded.checkpoint_dir,
        output_dir=output_path,
        prompt_count=len(records),
        generation_path=generation_path,
        eval_json_path=eval_json_path,
        sanity_path=sanity_path,
        sanity_passed=sanity.passed,
    )


def _checkpoint_summary(manifest: Any) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "student_architecture": manifest.student_architecture,
        "student_config": manifest.student_config,
        "step": manifest.step,
    }
