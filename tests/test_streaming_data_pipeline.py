from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.data import (
    StreamingCursor,
    StreamingDataset,
    build_streaming_dataset_from_tokenized_corpus,
    read_streaming_dataset_manifest,
)
from qrwkv_xla.data.streaming_reports import write_markdown_report
from qrwkv_xla.generation import SmokeTokenizer
from qrwkv_xla.lm.tokenized_corpus import write_tokenized_corpus_from_prompt_jsonl


def test_manifest_schema_and_shards(tmp_path: Path) -> None:
    streaming_dir = _write_streaming(tmp_path)

    manifest = read_streaming_dataset_manifest(streaming_dir / "manifest.json")

    assert manifest.phase == "P44"
    assert manifest.schema_version == "0.1"
    assert manifest.corpus.num_documents == 12
    assert len(manifest.shards) >= 2
    assert manifest.shards[0].path.startswith("shards/")


def test_iterator_yields_expected_shapes_and_masks(tmp_path: Path) -> None:
    dataset = StreamingDataset(_write_streaming(tmp_path))

    batch = next(dataset.iter_batches(batch_size=3))

    assert batch.input_ids.shape == (3, 5)
    assert batch.labels.shape == (3, 5)
    assert batch.attention_mask.shape == (3, 5)
    assert batch.label_mask.shape == (3, 5)
    dataset.validate_masks()


def test_drop_last_and_partial_batch_behavior(tmp_path: Path) -> None:
    dataset = StreamingDataset(_write_streaming(tmp_path))

    partial_batches = list(dataset.iter_batches(batch_size=4))
    drop_last_batches = list(dataset.iter_batches(batch_size=4, drop_last=True))

    assert partial_batches[-1].input_ids.shape == (4, 5)
    assert (
        np.count_nonzero(partial_batches[-1].label_mask)
        <= partial_batches[-1].label_mask.size
    )
    assert len(drop_last_batches) <= len(partial_batches)


def test_cursor_roundtrip_and_resume_determinism(tmp_path: Path) -> None:
    dataset = StreamingDataset(_write_streaming(tmp_path))

    uninterrupted = list(dataset.iter_batches(batch_size=2, max_batches=3))
    cursor = uninterrupted[0].cursor
    resumed = next(
        dataset.iter_batches(
            batch_size=2,
            max_batches=1,
            cursor=StreamingCursor.from_dict(cursor.to_dict()),
        )
    )

    assert cursor == StreamingCursor(position=2, shuffle=False, seed=0)
    assert np.array_equal(uninterrupted[1].input_ids, resumed.input_ids)
    assert np.array_equal(uninterrupted[1].labels, resumed.labels)


def test_fixed_seed_shuffle_replays_exactly(tmp_path: Path) -> None:
    left = StreamingDataset(_write_streaming(tmp_path), shuffle=True, seed=9)
    right = StreamingDataset(_write_streaming(tmp_path), shuffle=True, seed=9)

    left_batches = list(left.iter_batches(batch_size=2, max_batches=3))
    right_batches = list(right.iter_batches(batch_size=2, max_batches=3))

    assert len(left_batches) == len(right_batches)
    for left_batch, right_batch in zip(left_batches, right_batches, strict=True):
        assert np.array_equal(left_batch.input_ids, right_batch.input_ids)
        assert np.array_equal(left_batch.labels, right_batch.labels)


def test_missing_shard_fails_without_loading_everything(tmp_path: Path) -> None:
    streaming_dir = _write_streaming(tmp_path)
    manifest = read_streaming_dataset_manifest(streaming_dir / "manifest.json")
    (streaming_dir / manifest.shards[0].path).unlink()

    dataset = StreamingDataset(streaming_dir)
    try:
        next(dataset.iter_batches(batch_size=1))
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing shard failure")


def test_report_writer_smoke(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    written = write_markdown_report(
        report_path,
        title="Smoke",
        sections=[("Section", ["a", "b"])],
    )

    assert written == report_path
    assert report_path.read_text(encoding="utf-8").startswith("# Smoke")


def test_cli_help_and_manual_scripts(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    out_dir = tmp_path / "artifacts"

    for script in (
        "scripts/build_streaming_data_dry_run.py",
        "scripts/run_streaming_data_dry_run.py",
        "scripts/run_streaming_trainer_dry_run.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout

    subprocess.run(
        [
            sys.executable,
            "scripts/build_streaming_data_dry_run.py",
            "--out",
            str(out_dir),
            "--num-documents",
            "12",
            "--total-tokens",
            "2048",
            "--shard-tokens",
            "128",
            "--seq-len",
            "8",
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_streaming_data_dry_run.py",
            "--manifest",
            str(out_dir / "manifest.json"),
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--num-batches",
            "3",
            "--out",
            str(out_dir),
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_streaming_trainer_dry_run.py",
            "--manifest",
            str(out_dir / "manifest.json"),
            "--out",
            str(out_dir),
            "--steps",
            "2",
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "P44_DATASET_SUMMARY.md").is_file()
    assert (out_dir / "P44_STREAMING_DRY_RUN_REPORT.md").is_file()
    assert (out_dir / "P44_TRAINER_DRY_RUN_REPORT.md").is_file()
    assert (out_dir / "resume_cursor.json").is_file()


def _write_streaming(tmp_path: Path) -> Path:
    tokenized_dir = _write_tokenized(tmp_path)
    streaming_dir = tmp_path / "streaming"
    build_streaming_dataset_from_tokenized_corpus(
        tokenized_dir,
        streaming_dir,
        num_documents=12,
        shard_tokens=10,
        overwrite=True,
        created_at_utc="2026-05-08T00:00:00+00:00",
    )
    return streaming_dir


def _write_tokenized(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "prompts.jsonl"
    corpus_path.write_text(
        "\n".join(
            json.dumps({"id": f"r{index}", "text": f"record {index}", "split": "train"})
            for index in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tokenized"
    write_tokenized_corpus_from_prompt_jsonl(
        corpus_path,
        output_dir,
        tokenizer=SmokeTokenizer(),
        sequence_length=5,
        shard_size_tokens=10,
        overwrite=True,
        created_at="2026-05-08T00:00:00+00:00",
    )
    return output_dir
