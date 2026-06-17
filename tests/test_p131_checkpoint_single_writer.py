from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.burn import (
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    run_first_serious_burn,
    write_first_serious_burn_report,
)
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
)


def test_process_zero_writes_canonical_checkpoint_and_reports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_jax_process(monkeypatch, process_index=0, process_count=4)
    store = _dense_textbook(tmp_path / "teacher_textbook", examples=100)

    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=1, batch_size=4),
        confirm_serious_burn=True,
    )
    report_path = write_first_serious_burn_report(
        result,
        Path(result.output_dir) / "burn_report.json",
    )

    assert result.status == "pass"
    assert result.artifact_hygiene_ready is True
    assert result.artifact_write_policy == (
        "process_0_canonical_with_per_process_diagnostics"
    )
    assert result.is_canonical_process is True
    assert result.report_role == "canonical"
    assert result.report_writer_process_index == 0
    assert report_path == Path(result.output_dir) / "burn_report.json"
    assert Path(result.canonical_burn_report_path).is_file()
    assert Path(result.per_process_burn_report_path).is_file()
    assert Path(result.launch_commands_path).is_file()
    assert result.checkpoint_write_strategy == "process_0_only"
    assert result.canonical_checkpoint_writer_process_index == 0
    assert result.canonical_checkpoint_written is True
    assert result.canonical_checkpoint_written_by_this_process is True
    assert result.checkpoint_written is True
    assert Path(result.canonical_checkpoint_path).is_file()
    assert result.canonical_checkpoint_fingerprint is not None
    assert result.checkpoint_fingerprint_match is True
    assert result.checkpoint_fingerprint_comparison_scope == "canonical_process_0_only"
    assert result.single_writer_checkpoint_verified is True
    assert result.per_process_reports_written is True
    assert result.per_process_diagnostics_preserved is True
    assert result.production_training_ready is False


def test_nonzero_process_skips_canonical_checkpoint_and_writes_process_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_jax_process(monkeypatch, process_index=2, process_count=4)
    store = _dense_textbook(tmp_path / "teacher_textbook", examples=100)

    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=1, batch_size=4),
        confirm_serious_burn=True,
    )
    report_path = write_first_serious_burn_report(
        result,
        Path(result.output_dir) / "burn_report.json",
    )

    assert result.status == "pass"
    assert result.is_canonical_process is False
    assert result.report_role == "per_process"
    assert result.report_writer_process_index == 2
    assert report_path == Path(result.output_dir) / "burn_report_process_2.json"
    assert not Path(result.canonical_burn_report_path).exists()
    assert Path(result.per_process_burn_report_path).is_file()
    assert not Path(result.launch_commands_path).exists()
    assert result.checkpoint_write_strategy == "process_0_only"
    assert result.canonical_checkpoint_written is False
    assert result.canonical_checkpoint_written_by_this_process is False
    assert result.checkpoint_written is False
    assert not Path(result.canonical_checkpoint_path).exists()
    assert result.canonical_checkpoint_fingerprint is None
    assert result.checkpoint_fingerprint_match is True
    assert result.single_writer_checkpoint_verified is True
    assert result.artifact_hygiene_ready is True
    assert result.production_training_ready is False
    process_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert process_report["report_role"] == "per_process"
    assert process_report["canonical_checkpoint_written_by_this_process"] is False


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
        phase="P131",
        readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        teacher_textbook_path=str(store.root),
        max_steps=max_steps,
        batch_size=batch_size,
        allow_textbook_reuse=False,
        example_sharding="auto",
        distributed_sync="none",
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
            created_at="2026-06-17T00:00:00Z",
            source={"kind": "unit"},
            provenance={"phase": "P131"},
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
