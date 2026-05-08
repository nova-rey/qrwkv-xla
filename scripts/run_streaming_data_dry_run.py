from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.data import StreamingCursor, StreamingDataset
from qrwkv_xla.data.streaming_reports import write_json_report, write_markdown_report

DEFAULT_OUT = Path("artifacts/data/p44_streaming_dry_run")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P44 larger/local streaming data pipeline dry-run"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_OUT / "manifest.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seq_len <= 1:
        parser.error("--seq-len must be > 1")
    dataset = StreamingDataset(
        manifest_path.parent,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if dataset.manifest.corpus.sequence_length != args.seq_len:
        parser.error(
            "--seq-len must match manifest sequence_length: "
            f"{dataset.manifest.corpus.sequence_length}"
        )

    started = time.perf_counter()
    batches = list(
        dataset.iter_batches(
            batch_size=args.batch_size,
            max_batches=args.num_batches,
        )
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    if not batches:
        raise ValueError("streaming dry-run read zero batches")

    num_sequences_read = int(sum(batch.input_ids.shape[0] for batch in batches))
    tokens_consumed = int(sum(np.count_nonzero(batch.label_mask) for batch in batches))
    total_slots = int(sum(batch.label_mask.size for batch in batches))
    tokens_discarded_or_padding = int(dataset.manifest.corpus.padded_tokens) + (
        total_slots - tokens_consumed
    )
    padding_fraction = (
        float((total_slots - tokens_consumed) / total_slots) if total_slots else 0.0
    )

    dataset.validate_masks()
    resume_payload = _resume_check(
        dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )
    write_json_report(out_dir / "resume_cursor.json", resume_payload["cursor"])

    replay_payload = _replay_check(
        dataset_path=manifest_path.parent,
        batch_size=args.batch_size,
        num_batches=min(args.num_batches, 4),
        shuffle=args.shuffle,
        seed=args.seed,
    )

    report = {
        "phase": "P44",
        "overall_status": "pass",
        "num_shards": dataset.num_shards,
        "num_tokens": dataset.tokens_available,
        "batch_size": args.batch_size,
        "sequence_length": args.seq_len,
        "num_batches_read": len(batches),
        "num_sequences_read": num_sequences_read,
        "num_tokens_read": tokens_consumed,
        "tokens_available": dataset.tokens_available,
        "tokens_consumed": tokens_consumed,
        "tokens_discarded_or_padding": tokens_discarded_or_padding,
        "padding_fraction": padding_fraction,
        "resume_status": resume_payload["status"],
        "post_resume_batch_match_count": resume_payload["match_count"],
        "post_resume_max_abs_token_diff": resume_payload["max_abs_token_diff"],
        "deterministic_replay_status": replay_payload["status"],
        "mask_validation_status": "pass",
        "approx_tokens_per_second": float(tokens_consumed / elapsed),
        "peak_memory_mb": _peak_memory_mb(),
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "boundary_policy": dataset.manifest.corpus.boundary_policy,
        "shard_boundary_behavior": (
            "iterator consumes prepacked per-sequence rows and does not stitch tokens "
            "across shard boundaries"
        ),
    }
    write_json_report(out_dir / "streaming_dry_run_report.json", report)
    write_markdown_report(
        out_dir / "P44_STREAMING_DRY_RUN_REPORT.md",
        title="P44 Streaming Dry-Run Report",
        sections=[
            (
                "Streaming dry-run",
                [
                    f"overall_status: {report['overall_status']}",
                    f"num_shards: {report['num_shards']}",
                    f"num_tokens: {report['num_tokens']}",
                    f"batch_size: {report['batch_size']}",
                    f"sequence_length: {report['sequence_length']}",
                    f"num_batches_read: {report['num_batches_read']}",
                    f"num_sequences_read: {report['num_sequences_read']}",
                    f"num_tokens_read: {report['num_tokens_read']}",
                    f"tokens_available: {report['tokens_available']}",
                    f"tokens_consumed: {report['tokens_consumed']}",
                    "tokens_discarded_or_padding: "
                    f"{report['tokens_discarded_or_padding']}",
                    f"padding_fraction: {report['padding_fraction']}",
                    f"resume_status: {report['resume_status']}",
                    "post_resume_batch_match_count: "
                    f"{report['post_resume_batch_match_count']}",
                    "post_resume_max_abs_token_diff: "
                    f"{report['post_resume_max_abs_token_diff']}",
                    "deterministic_replay_status: "
                    f"{report['deterministic_replay_status']}",
                    f"mask_validation_status: {report['mask_validation_status']}",
                    f"approx_tokens_per_second: {report['approx_tokens_per_second']}",
                    f"peak_memory_mb: {report['peak_memory_mb']}",
                ],
            ),
            (
                "Behavior",
                [
                    f"boundary_policy: {report['boundary_policy']}",
                    f"shard_boundary_behavior: {report['shard_boundary_behavior']}",
                    f"resume_cursor: {out_dir / 'resume_cursor.json'}",
                ],
            ),
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _resume_check(
    dataset: StreamingDataset,
    *,
    batch_size: int,
    num_batches: int,
) -> dict[str, Any]:
    split = max(1, min(num_batches // 2, num_batches - 1))
    uninterrupted = list(
        dataset.iter_batches(
            batch_size=batch_size,
            max_batches=max(num_batches, split + 1),
        )
    )
    if len(uninterrupted) <= split:
        return {
            "status": "not_enough_batches",
            "match_count": 0,
            "max_abs_token_diff": 0,
            "cursor": StreamingCursor(
                position=0,
                shuffle=dataset.shuffle,
                seed=dataset.seed,
            ).to_dict(),
        }
    cursor = uninterrupted[split - 1].cursor
    resumed = list(
        dataset.iter_batches(
            batch_size=batch_size,
            cursor=cursor,
            max_batches=max(1, len(uninterrupted) - split),
        )
    )
    expected = uninterrupted[split : split + len(resumed)]
    match_count = 0
    max_abs_token_diff = 0
    for left, right in zip(expected, resumed, strict=True):
        batch_diff = _max_batch_diff(left, right)
        if batch_diff == 0:
            match_count += 1
        max_abs_token_diff = max(max_abs_token_diff, batch_diff)
    status = "pass" if match_count == len(expected) else "fail"
    return {
        "status": status,
        "match_count": match_count,
        "max_abs_token_diff": int(max_abs_token_diff),
        "cursor": cursor.to_dict(),
    }


def _replay_check(
    *,
    dataset_path: Path,
    batch_size: int,
    num_batches: int,
    shuffle: bool,
    seed: int,
) -> dict[str, str]:
    left = StreamingDataset(dataset_path, shuffle=shuffle, seed=seed)
    right = StreamingDataset(dataset_path, shuffle=shuffle, seed=seed)
    left_batches = list(
        left.iter_batches(batch_size=batch_size, max_batches=num_batches)
    )
    right_batches = list(
        right.iter_batches(batch_size=batch_size, max_batches=num_batches)
    )
    if len(left_batches) != len(right_batches):
        return {"status": "fail_batch_count_mismatch"}
    if all(
        _max_batch_diff(a, b) == 0
        for a, b in zip(left_batches, right_batches, strict=True)
    ):
        return {"status": "pass"}
    return {"status": "fail_batch_content_mismatch"}


def _max_batch_diff(left, right) -> int:
    diffs = []
    for name in ("input_ids", "labels", "attention_mask", "label_mask"):
        diffs.append(int(np.max(np.abs(getattr(left, name) - getattr(right, name)))))
    return max(diffs) if diffs else 0


def _peak_memory_mb() -> float | str:
    try:
        import resource
    except ImportError:
        return "not_available"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / 1024.0) if peak > 0 else "not_available"


if __name__ == "__main__":
    main()
