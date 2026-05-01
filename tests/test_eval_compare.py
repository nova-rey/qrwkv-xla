from __future__ import annotations

from pathlib import Path

from qrwkv_xla.eval.compare import compare_eval_snapshots
from qrwkv_xla.generation.artifacts import GenerationRecord, write_generation_jsonl


def _write_snapshot(
    root: Path,
    records: list[GenerationRecord],
) -> None:
    write_generation_jsonl(records, root / "generations.jsonl")


def _record(prompt_id: str, decoded_text: str) -> GenerationRecord:
    return GenerationRecord(
        prompt_id=prompt_id,
        prompt_text="Prompt",
        prompt_token_ids=(1,),
        generated_token_ids=(2,),
        full_token_ids=(1, 2),
        decoded_text=decoded_text,
    )


def test_compare_identical_snapshots(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    records = [_record("p0", "same")]
    _write_snapshot(baseline, records)
    _write_snapshot(candidate, records)

    result = compare_eval_snapshots(baseline_dir=baseline, candidate_dir=candidate)

    assert result.same_count == 1
    assert result.changed_count == 0


def test_compare_changed_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_snapshot(baseline, [_record("p0", "old")])
    _write_snapshot(candidate, [_record("p0", "new")])

    result = compare_eval_snapshots(baseline_dir=baseline, candidate_dir=candidate)

    assert result.same_count == 0
    assert result.changed_count == 1


def test_compare_reports_missing_prompts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_snapshot(baseline, [_record("p0", "old"), _record("missing", "x")])
    _write_snapshot(candidate, [_record("p0", "old")])

    result = compare_eval_snapshots(baseline_dir=baseline, candidate_dir=candidate)

    assert result.missing_prompt_ids == ("missing",)
