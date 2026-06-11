from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.burn import (
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    run_first_serious_burn,
)
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
)


def test_real_burn_reports_process_local_example_shard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_jax_process(monkeypatch, process_index=2, process_count=4)
    store = _dense_textbook(tmp_path / "teacher_textbook", examples=100)

    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=25, batch_size=4),
        confirm_serious_burn=True,
    )

    assert result.status == "pass"
    assert result.jax_process_index == 2
    assert result.jax_process_count == 4
    assert result.worker_id == "2"
    assert result.sharding_strategy == "contiguous_by_process"
    assert result.distributed_example_sharding_verified is True
    assert result.distributed_sharding_verified is True
    assert result.distributed_training_ready is False
    assert result.batch_size == 4
    assert result.local_batch_size == 1
    assert result.examples_available_global == 100
    assert result.examples_available_local == 25
    assert result.examples_available == 25
    assert result.examples_consumed_local == 25
    assert result.unique_examples_consumed_local == 25
    assert result.examples_consumed == 25
    assert result.unique_examples_consumed == 25
    assert result.example_id_min == 50
    assert result.example_id_max == 74
    assert result.example_id_sample == tuple(range(50, 58))
    assert result.example_sharding_local_report_path is not None
    local_report = json.loads(
        Path(result.example_sharding_local_report_path).read_text(encoding="utf-8")
    )
    assert local_report["local_example_count"] == 25
    assert local_report["example_id_min"] == 50
    assert local_report["example_id_max"] == 74
    assert local_report["local_indices"] == list(range(50, 75))
    assert result.example_sharding_global_report_path is None


def test_process_zero_writes_global_example_sharding_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_jax_process(monkeypatch, process_index=0, process_count=4)
    store = _dense_textbook(tmp_path / "teacher_textbook", examples=100)

    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=1, batch_size=4),
        confirm_serious_burn=True,
    )

    assert result.status == "pass"
    assert result.example_sharding_global_report_path is not None
    global_report = json.loads(
        Path(result.example_sharding_global_report_path).read_text(encoding="utf-8")
    )
    assert global_report["coverage_count"] == 100
    assert global_report["missing_count"] == 0
    assert global_report["duplicate_count"] == 0
    assert global_report["coverage_verified"] is True
    assert global_report["overlap_verified"] is True
    assert [
        (shard["process_index"], shard["example_id_min"], shard["example_id_max"])
        for shard in global_report["shards"]
    ] == [(0, 0, 24), (1, 25, 49), (2, 50, 74), (3, 75, 99)]


def _patch_jax_process(monkeypatch, *, process_index: int, process_count: int) -> None:
    import qrwkv_xla.burn.first_serious_burn as first_serious_burn

    monkeypatch.setattr(first_serious_burn.jax, "process_index", lambda: process_index)
    monkeypatch.setattr(first_serious_burn.jax, "process_count", lambda: process_count)
    monkeypatch.setattr(first_serious_burn.jax, "local_device_count", lambda: 4)
    monkeypatch.setattr(first_serious_burn.jax, "device_count", lambda: 16)
    monkeypatch.setattr(first_serious_burn.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        first_serious_burn.jax,
        "devices",
        lambda: tuple(f"mock_tpu:{index}" for index in range(16)),
    )


def _real_config(
    tmp_path: Path,
    *,
    store: TeacherTargetStore,
    max_steps: int,
    batch_size: int,
) -> FirstSeriousBurnConfig:
    return replace(
        default_first_serious_burn_config(output_dir=tmp_path / "burn", mode="real"),
        phase="P117.1",
        readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        teacher_textbook_path=str(store.root),
        max_steps=max_steps,
        batch_size=batch_size,
        allow_textbook_reuse=False,
        example_sharding="auto",
    )


def _dense_textbook(path: Path, *, examples: int) -> TeacherTargetStore:
    store = TeacherTargetStore.create(
        path,
        TargetStoreMetadata(
            schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
            target_store_version=TEACHER_TARGET_STORE_VERSION,
            model_id="unit-dense-teacher",
            model_family="synthetic",
            tokenizer_id="unit-tokenizer",
            tokenizer_hash=None,
            vocab_size=5,
            target_type="dense_logits",
            dtype="float32",
            sequence_length=3,
            num_examples=examples,
            shard_count=1,
            created_by="test",
            created_at="2026-06-11T00:00:00Z",
            source={"kind": "unit"},
            provenance={"phase": "P128"},
        ),
        overwrite=True,
    )
    input_ids = (np.arange(examples * 3, dtype=np.int32) % 5).reshape(examples, 3)
    vocab = np.arange(5, dtype=np.float32)
    logits = input_ids[:, :, None].astype(np.float32) * 0.05 + vocab[None, None, :]
    store.write_shard(
        0,
        {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=np.int32),
            "logits": logits.astype(np.float32),
        },
    )
    return TeacherTargetStore.open(store.root)


def _readiness_report(tmp_path: Path, *, status: str) -> Path:
    path = tmp_path / f"readiness_{status}.json"
    path.write_text(
        json.dumps(
            {
                "phase": "P111",
                "status": status,
                "blockers": [],
                "warnings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path
