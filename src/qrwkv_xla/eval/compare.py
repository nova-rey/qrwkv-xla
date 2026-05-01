from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qrwkv_xla.eval.artifacts import read_generation_jsonl, write_eval_json


@dataclass(frozen=True)
class PromptComparison:
    prompt_id: str
    same_output: bool
    old_text: str
    new_text: str
    old_generated_token_count: int
    new_generated_token_count: int


@dataclass(frozen=True)
class EvalComparisonResult:
    baseline_dir: Path
    candidate_dir: Path
    prompt_count: int
    same_count: int
    changed_count: int
    missing_prompt_ids: tuple[str, ...]
    comparisons: tuple[PromptComparison, ...]


def compare_eval_snapshots(
    *,
    baseline_dir: str | Path,
    candidate_dir: str | Path,
) -> EvalComparisonResult:
    baseline_path = Path(baseline_dir)
    candidate_path = Path(candidate_dir)
    baseline = {
        record.prompt_id: record
        for record in read_generation_jsonl(baseline_path / "generations.jsonl")
    }
    candidate = {
        record.prompt_id: record
        for record in read_generation_jsonl(candidate_path / "generations.jsonl")
    }
    missing = tuple(
        sorted(
            set(baseline).symmetric_difference(candidate),
        )
    )
    common_ids = tuple(prompt_id for prompt_id in baseline if prompt_id in candidate)
    comparisons = tuple(
        PromptComparison(
            prompt_id=prompt_id,
            same_output=baseline[prompt_id].decoded_text
            == candidate[prompt_id].decoded_text,
            old_text=baseline[prompt_id].decoded_text,
            new_text=candidate[prompt_id].decoded_text,
            old_generated_token_count=len(baseline[prompt_id].generated_token_ids),
            new_generated_token_count=len(candidate[prompt_id].generated_token_ids),
        )
        for prompt_id in common_ids
    )
    same_count = sum(1 for comparison in comparisons if comparison.same_output)
    return EvalComparisonResult(
        baseline_dir=baseline_path,
        candidate_dir=candidate_path,
        prompt_count=len(common_ids),
        same_count=same_count,
        changed_count=len(comparisons) - same_count,
        missing_prompt_ids=missing,
        comparisons=comparisons,
    )


def write_eval_comparison(result: EvalComparisonResult, path: str | Path) -> Path:
    return write_eval_json(eval_comparison_to_dict(result), path)


def eval_comparison_to_dict(result: EvalComparisonResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["baseline_dir"] = str(result.baseline_dir)
    payload["candidate_dir"] = str(result.candidate_dir)
    return payload
