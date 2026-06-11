from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

CONTIGUOUS_BY_PROCESS = "contiguous_by_process"
ROUND_ROBIN_BY_PROCESS = "round_robin_by_process"


@dataclass(frozen=True)
class ExampleShard:
    strategy: str
    process_index: int
    process_count: int
    global_example_count: int
    local_indices: tuple[int, ...]
    local_example_count: int
    example_id_min: int | None
    example_id_max: int | None
    example_id_sample: tuple[int, ...]
    coverage_verified: bool | None = None
    overlap_verified: bool | None = None

    def to_report(self, *, include_all_indices: bool = False) -> dict[str, Any]:
        report = asdict(self)
        report["local_indices"] = (
            self.local_indices if include_all_indices else self.local_indices[:16]
        )
        report["local_indices_truncated"] = len(report["local_indices"]) != len(
            self.local_indices
        )
        return report


def contiguous_example_shard(
    *,
    global_example_count: int,
    process_index: int,
    process_count: int,
) -> ExampleShard:
    _validate_shard_inputs(
        global_example_count=global_example_count,
        process_index=process_index,
        process_count=process_count,
    )
    base_count, remainder = divmod(global_example_count, process_count)
    local_count = base_count + (1 if process_index < remainder else 0)
    start = process_index * base_count + min(process_index, remainder)
    return _build_shard(
        strategy=CONTIGUOUS_BY_PROCESS,
        process_index=process_index,
        process_count=process_count,
        global_example_count=global_example_count,
        local_indices=tuple(range(start, start + local_count)),
    )


def round_robin_example_shard(
    *,
    global_example_count: int,
    process_index: int,
    process_count: int,
) -> ExampleShard:
    _validate_shard_inputs(
        global_example_count=global_example_count,
        process_index=process_index,
        process_count=process_count,
    )
    return _build_shard(
        strategy=ROUND_ROBIN_BY_PROCESS,
        process_index=process_index,
        process_count=process_count,
        global_example_count=global_example_count,
        local_indices=tuple(range(process_index, global_example_count, process_count)),
    )


def verify_global_example_shards(shards: Sequence[ExampleShard]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one shard is required")
    strategy = shards[0].strategy
    process_count = shards[0].process_count
    global_example_count = shards[0].global_example_count
    for shard in shards:
        if shard.strategy != strategy:
            raise ValueError("all shards must use the same strategy")
        if shard.process_count != process_count:
            raise ValueError("all shards must use the same process_count")
        if shard.global_example_count != global_example_count:
            raise ValueError("all shards must use the same global_example_count")

    all_indices = [index for shard in shards for index in shard.local_indices]
    counts = Counter(all_indices)
    expected = set(range(global_example_count))
    covered = set(all_indices)
    missing = tuple(sorted(expected - covered))
    duplicates = tuple(sorted(index for index, count in counts.items() if count > 1))
    out_of_range = tuple(
        sorted(index for index in covered if index < 0 or index >= global_example_count)
    )
    coverage_verified = not missing and not out_of_range
    overlap_verified = not duplicates
    return {
        "strategy": strategy,
        "process_count": process_count,
        "global_example_count": global_example_count,
        "coverage_count": len(covered & expected),
        "expected_count": global_example_count,
        "missing_count": len(missing),
        "duplicate_count": len(duplicates),
        "out_of_range_count": len(out_of_range),
        "coverage_verified": coverage_verified,
        "overlap_verified": overlap_verified,
        "missing_sample": missing[:16],
        "duplicate_sample": duplicates[:16],
        "out_of_range_sample": out_of_range[:16],
        "shards": [
            {
                "process_index": shard.process_index,
                "local_example_count": shard.local_example_count,
                "example_id_min": shard.example_id_min,
                "example_id_max": shard.example_id_max,
                "example_id_sample": shard.example_id_sample,
            }
            for shard in sorted(shards, key=lambda value: value.process_index)
        ],
    }


def build_example_shard(
    *,
    strategy: str,
    global_example_count: int,
    process_index: int,
    process_count: int,
) -> ExampleShard:
    if strategy == CONTIGUOUS_BY_PROCESS:
        return contiguous_example_shard(
            global_example_count=global_example_count,
            process_index=process_index,
            process_count=process_count,
        )
    if strategy == ROUND_ROBIN_BY_PROCESS:
        return round_robin_example_shard(
            global_example_count=global_example_count,
            process_index=process_index,
            process_count=process_count,
        )
    raise ValueError(f"unsupported example sharding strategy: {strategy!r}")


def _validate_shard_inputs(
    *,
    global_example_count: int,
    process_index: int,
    process_count: int,
) -> None:
    if global_example_count < 0:
        raise ValueError("global_example_count must be >= 0")
    if process_count <= 0:
        raise ValueError("process_count must be > 0")
    if process_index < 0 or process_index >= process_count:
        raise ValueError("process_index must be in [0, process_count)")


def _build_shard(
    *,
    strategy: str,
    process_index: int,
    process_count: int,
    global_example_count: int,
    local_indices: tuple[int, ...],
) -> ExampleShard:
    return ExampleShard(
        strategy=strategy,
        process_index=process_index,
        process_count=process_count,
        global_example_count=global_example_count,
        local_indices=local_indices,
        local_example_count=len(local_indices),
        example_id_min=min(local_indices) if local_indices else None,
        example_id_max=max(local_indices) if local_indices else None,
        example_id_sample=local_indices[:8],
    )
