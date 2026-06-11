from __future__ import annotations

import pytest

from qrwkv_xla.burn import (
    CONTIGUOUS_BY_PROCESS,
    ROUND_ROBIN_BY_PROCESS,
    contiguous_example_shard,
    round_robin_example_shard,
    verify_global_example_shards,
)


def test_even_contiguous_split_covers_without_overlap() -> None:
    shards = [
        contiguous_example_shard(
            global_example_count=100,
            process_index=process_index,
            process_count=4,
        )
        for process_index in range(4)
    ]

    assert shards[0].local_indices == tuple(range(0, 25))
    assert shards[1].local_indices == tuple(range(25, 50))
    assert shards[2].local_indices == tuple(range(50, 75))
    assert shards[3].local_indices == tuple(range(75, 100))

    report = verify_global_example_shards(shards)
    assert report["strategy"] == CONTIGUOUS_BY_PROCESS
    assert report["coverage_count"] == 100
    assert report["missing_count"] == 0
    assert report["duplicate_count"] == 0
    assert report["coverage_verified"] is True
    assert report["overlap_verified"] is True


def test_uneven_contiguous_split_is_deterministic() -> None:
    shards = [
        contiguous_example_shard(
            global_example_count=103,
            process_index=process_index,
            process_count=4,
        )
        for process_index in range(4)
    ]

    assert [shard.local_example_count for shard in shards] == [26, 26, 26, 25]
    assert shards[0].local_indices == tuple(range(0, 26))
    assert shards[1].local_indices == tuple(range(26, 52))
    assert shards[2].local_indices == tuple(range(52, 78))
    assert shards[3].local_indices == tuple(range(78, 103))
    assert (
        max(shard.local_example_count for shard in shards)
        - min(shard.local_example_count for shard in shards)
        == 1
    )

    report = verify_global_example_shards(shards)
    assert report["coverage_verified"] is True
    assert report["overlap_verified"] is True


def test_more_processes_than_examples_leaves_empty_shards() -> None:
    shards = [
        contiguous_example_shard(
            global_example_count=3,
            process_index=process_index,
            process_count=8,
        )
        for process_index in range(8)
    ]

    assert [shard.local_example_count for shard in shards] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert shards[0].local_indices == (0,)
    assert shards[1].local_indices == (1,)
    assert shards[2].local_indices == (2,)
    assert shards[3].local_indices == ()

    report = verify_global_example_shards(shards)
    assert report["coverage_count"] == 3
    assert report["missing_count"] == 0
    assert report["duplicate_count"] == 0
    assert report["coverage_verified"] is True
    assert report["overlap_verified"] is True


def test_round_robin_split_covers_without_overlap() -> None:
    shards = [
        round_robin_example_shard(
            global_example_count=10,
            process_index=process_index,
            process_count=4,
        )
        for process_index in range(4)
    ]

    assert shards[0].strategy == ROUND_ROBIN_BY_PROCESS
    assert shards[0].local_indices == (0, 4, 8)
    assert shards[1].local_indices == (1, 5, 9)
    assert shards[2].local_indices == (2, 6)
    assert shards[3].local_indices == (3, 7)
    report = verify_global_example_shards(shards)
    assert report["coverage_verified"] is True
    assert report["overlap_verified"] is True


def test_invalid_process_index_fails() -> None:
    with pytest.raises(ValueError, match="process_index"):
        contiguous_example_shard(
            global_example_count=10,
            process_index=4,
            process_count=4,
        )
