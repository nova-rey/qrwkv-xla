from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SanityCheckResult:
    prompt_id: str
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SanitySummary:
    passed: bool
    prompt_count: int
    passed_count: int
    failed_count: int
    results: tuple[SanityCheckResult, ...]


def run_generation_sanity_checks(
    records: list[Any] | tuple[Any, ...],
    *,
    require_non_empty: bool = True,
    max_repeated_token_fraction: float | None = 0.95,
    max_unknown_token_fraction: float | None = 0.95,
) -> SanitySummary:
    results = tuple(
        _check_record(
            record,
            require_non_empty=require_non_empty,
            max_repeated_token_fraction=max_repeated_token_fraction,
            max_unknown_token_fraction=max_unknown_token_fraction,
        )
        for record in records
    )
    passed_count = sum(1 for result in results if result.passed)
    return SanitySummary(
        passed=passed_count == len(results),
        prompt_count=len(results),
        passed_count=passed_count,
        failed_count=len(results) - passed_count,
        results=results,
    )


def sanity_summary_to_dict(summary: SanitySummary) -> dict[str, Any]:
    return asdict(summary)


def _check_record(
    record: Any,
    *,
    require_non_empty: bool,
    max_repeated_token_fraction: float | None,
    max_unknown_token_fraction: float | None,
) -> SanityCheckResult:
    prompt_id = str(_field(record, "prompt_id", ""))
    generated = tuple(
        int(token_id) for token_id in _field(record, "generated_token_ids", ())
    )
    full = tuple(int(token_id) for token_id in _field(record, "full_token_ids", ()))
    decoded_text = str(_field(record, "decoded_text", ""))

    notes: list[str] = []
    checks: dict[str, bool] = {}
    checks["non_empty_output"] = bool(generated) if require_non_empty else True
    if not checks["non_empty_output"]:
        notes.append("generated_token_ids is empty")

    repeated_fraction = _repeated_token_fraction(generated)
    checks["repeated_token_fraction"] = (
        True
        if max_repeated_token_fraction is None
        else repeated_fraction <= max_repeated_token_fraction
    )
    if not checks["repeated_token_fraction"]:
        notes.append(
            "generated_token_ids exceeds max_repeated_token_fraction "
            f"({repeated_fraction:.3f})"
        )

    unknown_fraction = _unknown_token_fraction(decoded_text, full)
    checks["unknown_token_fraction"] = (
        True
        if max_unknown_token_fraction is None
        else unknown_fraction <= max_unknown_token_fraction
    )
    if not checks["unknown_token_fraction"]:
        notes.append(
            f"decoded_text exceeds max_unknown_token_fraction ({unknown_fraction:.3f})"
        )

    checks["serialization_shape"] = bool(prompt_id) and isinstance(decoded_text, str)
    if not checks["serialization_shape"]:
        notes.append("record is missing prompt_id or decoded_text")

    metrics = {
        "generated_token_count": float(len(generated)),
        "repeated_token_fraction": repeated_fraction,
        "unknown_token_fraction": unknown_fraction,
    }
    return SanityCheckResult(
        prompt_id=prompt_id,
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
        notes=tuple(notes),
    )


def _field(record: Any, name: str, default: Any) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _repeated_token_fraction(token_ids: tuple[int, ...]) -> float:
    if not token_ids:
        return 0.0
    return max(Counter(token_ids).values()) / len(token_ids)


def _unknown_token_fraction(decoded_text: str, token_ids: tuple[int, ...]) -> float:
    if not token_ids:
        return 0.0
    return decoded_text.count("<tok_") / len(token_ids)
