from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.burn import (
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    evaluate_distributed_training_readiness,
    run_first_serious_burn,
)
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
)


def test_readiness_true_only_when_all_sync_predicates_pass() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=True,
        checkpoint_fingerprint_match=True,
    )

    assert readiness.distributed_training_ready is True
    assert readiness.missing_predicates == ()


def test_readiness_false_when_gradient_sync_missing() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=False,
        gradient_sync_verified=False,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=True,
        checkpoint_fingerprint_match=True,
    )

    assert readiness.distributed_training_ready is False
    assert "gradient_sync_enabled" in readiness.missing_predicates
    assert "gradient_sync_verified" in readiness.missing_predicates


def test_readiness_false_when_parameter_or_optimizer_sync_missing() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=False,
        optimizer_state_sync_verified=False,
        loss_is_global=True,
        checkpoint_fingerprint_match=True,
    )

    assert readiness.distributed_training_ready is False
    assert "parameter_sync_verified" in readiness.missing_predicates
    assert "optimizer_state_sync_verified" in readiness.missing_predicates


def test_readiness_false_when_loss_is_local_or_examples_unverified() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=False,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=False,
        checkpoint_fingerprint_match=True,
    )

    assert readiness.distributed_training_ready is False
    assert "distributed_example_sharding_verified" in readiness.missing_predicates
    assert "loss_is_global" in readiness.missing_predicates


def test_burn_sync_report_is_honest_local_update_audit(
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
    assert result.distributed_example_sharding_verified is True
    assert result.distributed_sync_mode == "none"
    assert result.distributed_sync_requested == "none"
    assert result.gradient_sync_enabled is False
    assert result.gradient_sync_verified is False
    assert result.collective_sync_probe_enabled is False
    assert result.collective_sync_probe_verified is False
    assert result.parameter_sync_verified is False
    assert result.optimizer_state_kind == "stateless_sgd"
    assert result.optimizer_state_absent is True
    assert result.optimizer_state_sync_verified is False
    assert result.checkpoint_fingerprint_match is False
    assert result.loss_reduction == "local_only"
    assert result.loss_is_global is False
    assert result.global_batch_size == 4
    assert result.global_batch_size_semantics == "local_batch_size * jax_process_count"
    assert result.distributed_training_ready is False
    assert "gradient_sync_enabled" in result.distributed_training_missing_predicates
    assert result.parameter_fingerprint_initial is not None
    assert result.parameter_fingerprint_final is not None
    assert result.optimizer_state_fingerprint_initial is not None
    assert result.optimizer_state_fingerprint_final is not None
    assert result.checkpoint_fingerprint is not None
    assert result.sync_report_path is not None
    sync_report = json.loads(Path(result.sync_report_path).read_text(encoding="utf-8"))
    assert sync_report["distributed_training_ready"] is False
    assert sync_report["gradient_sync_enabled"] is False
    assert sync_report["loss_reduction"] == "local_only"
    assert result.sync_global_report_path is not None
    assert result.warnings == ()


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
        phase="P129",
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
            created_at="2026-06-11T00:00:00Z",
            source={"kind": "unit"},
            provenance={"phase": "P129"},
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
