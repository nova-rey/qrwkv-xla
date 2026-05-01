from __future__ import annotations

from qrwkv_xla.eval.sanity import run_generation_sanity_checks
from qrwkv_xla.generation.artifacts import GenerationRecord


def _record(
    generated: tuple[int, ...],
    decoded: str = "Hello",
    *,
    prompt_id: str = "p0",
) -> GenerationRecord:
    return GenerationRecord(
        prompt_id=prompt_id,
        prompt_text="Hello",
        prompt_token_ids=(73,),
        generated_token_ids=generated,
        full_token_ids=(73, *generated),
        decoded_text=decoded,
    )


def test_normal_generation_passes() -> None:
    summary = run_generation_sanity_checks([_record((1, 2, 3), "abc")])

    assert summary.passed is True
    assert summary.failed_count == 0


def test_empty_output_fails_non_empty_check() -> None:
    summary = run_generation_sanity_checks([_record(())])

    assert summary.passed is False
    assert summary.results[0].checks["non_empty_output"] is False


def test_repeated_token_output_triggers_failure() -> None:
    summary = run_generation_sanity_checks(
        [_record((7, 7, 7, 1))],
        max_repeated_token_fraction=0.5,
    )

    assert summary.passed is False
    assert summary.results[0].checks["repeated_token_fraction"] is False


def test_unknown_token_output_triggers_failure() -> None:
    summary = run_generation_sanity_checks(
        [_record((300, 301), "Hi <tok_300><tok_301>")],
        max_unknown_token_fraction=0.2,
    )

    assert summary.passed is False
    assert summary.results[0].checks["unknown_token_fraction"] is False
