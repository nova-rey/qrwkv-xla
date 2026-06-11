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


def test_real_burn_accepts_dense_logits_and_legacy_full_logits(
    tmp_path: Path,
) -> None:
    dense_store = _dense_textbook(tmp_path / "dense", target_type="dense_logits")
    dense_result = run_first_serious_burn(
        _real_config(tmp_path, store=dense_store),
        confirm_serious_burn=True,
    )

    legacy_store = _dense_textbook(tmp_path / "legacy", target_type="full_logits")
    legacy_result = run_first_serious_burn(
        _real_config(tmp_path, store=legacy_store),
        confirm_serious_burn=True,
    )

    assert dense_result.status == "pass"
    assert dense_result.teacher_target_type == "dense_logits"
    assert dense_result.teacher_target_type_legacy_alias == "full_logits"
    assert dense_result.target_loss_kind == "dense_logits_kl"
    assert dense_result.real_training_executed is True

    assert legacy_result.status == "pass"
    assert legacy_result.teacher_target_type == "full_logits"
    assert legacy_result.teacher_target_type_legacy_alias == "full_logits"
    assert legacy_result.target_loss_kind == "dense_logits_kl"
    assert legacy_result.real_training_executed is True


def test_real_burn_dispatches_cascaded_targets_without_bucket_shape_loss(
    tmp_path: Path,
) -> None:
    store = _cascaded_textbook(tmp_path / "cascaded")
    result = run_first_serious_burn(
        _real_config(tmp_path, store=store),
        confirm_serious_burn=True,
    )

    assert result.status == "pass"
    assert result.teacher_target_type == "cascaded_soft_labels_v1"
    assert result.target_loss_kind == "cascaded_soft_labels"
    assert result.top_k == 2
    assert result.bucket_shape_loss_enabled is False
    assert result.bucket_shape_loss_weight == 0.0
    assert result.student_logits_materialization == "dense_full_vocab"
    assert result.real_training_executed is True
    assert result.cascaded_real_training_executed is True
    assert result.example_id_min == 0
    assert result.example_id_max == 0
    assert result.example_id_sample == (0,)


def test_worker_observability_uses_jax_process_index(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        replace(
            default_first_serious_burn_config(output_dir=tmp_path / "burn"),
            readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        )
    )

    assert result.jax_process_index is not None
    assert result.jax_process_count is not None
    assert result.jax_local_device_count is not None
    assert result.jax_global_device_count is not None
    assert result.jax_backend is not None
    assert result.worker_id == str(result.jax_process_index)
    assert result.jax_process_count >= 1
    assert result.jax_local_device_count >= 1
    assert result.jax_global_device_count >= 1
    assert result.distributed_training_ready is False
    assert result.distributed_sharding_verified is False
    assert result.sharding_strategy == "single_process_or_unsharded"
    assert result.distributed_example_sharding_verified is False


def _real_config(
    tmp_path: Path,
    *,
    store: TeacherTargetStore,
) -> FirstSeriousBurnConfig:
    return replace(
        default_first_serious_burn_config(output_dir=tmp_path / "burn", mode="real"),
        phase="P117.1",
        readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        teacher_textbook_path=str(store.root),
        max_steps=1,
        batch_size=1,
        allow_textbook_reuse=False,
    )


def _dense_textbook(path: Path, *, target_type: str) -> TeacherTargetStore:
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
            target_type=target_type,
            dtype="float32",
            sequence_length=3,
            num_examples=2,
            shard_count=1,
            created_by="test",
            created_at="2026-06-11T00:00:00Z",
            source={"kind": "unit"},
            provenance={"phase": "P126"},
        ),
        overwrite=True,
    )
    input_ids = np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
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


def _cascaded_textbook(path: Path) -> TeacherTargetStore:
    bucket_edges = (1.0, 0.1, 0.01, 0.0)
    store = TeacherTargetStore.create(
        path,
        TargetStoreMetadata(
            schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
            target_store_version=TEACHER_TARGET_STORE_VERSION,
            model_id="unit-cascaded-teacher",
            model_family="synthetic",
            tokenizer_id="unit-tokenizer",
            tokenizer_hash=None,
            vocab_size=7,
            target_type="cascaded_soft_labels_v1",
            dtype="float32",
            sequence_length=3,
            num_examples=2,
            shard_count=1,
            created_by="test",
            created_at="2026-06-11T00:00:00Z",
            source={"kind": "unit"},
            provenance={"phase": "P126"},
            target_params={
                "top_k": "2",
                "top_log_probs_dtype": "float32",
                "mass_tolerance": "0.001",
                "bucket_edge_type": "probability",
                "bucket_edges": ",".join(str(edge) for edge in bucket_edges),
                "bucket_count": "3",
                "bucket_mass_dtype": "float32",
                "bucket_mean_logp_dtype": "float32",
            },
        ),
        overwrite=True,
    )
    input_ids = np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
    top_token_ids = np.asarray(
        [[[0, 1], [1, 2], [2, 3]], [[2, 3], [3, 4], [4, 5]]],
        dtype=np.int32,
    )
    top_log_probs = np.log(
        np.asarray([[[0.6, 0.3]] * 3, [[0.6, 0.3]] * 3], dtype=np.float32)
    )
    bucket_mass = np.zeros((2, 3, 3), dtype=np.float32)
    bucket_mass[:, :, 0] = 0.1
    bucket_count = np.zeros((2, 3, 3), dtype=np.int32)
    bucket_count[:, :, 0] = 5
    bucket_mean_logp = np.zeros((2, 3, 3), dtype=np.float32)
    bucket_mean_logp[:, :, 0] = np.log(np.asarray(0.1 / 5, dtype=np.float32))
    store.write_shard(
        0,
        {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=np.int32),
            "top_token_ids": top_token_ids,
            "top_log_probs": top_log_probs.astype(np.float32),
            "top_mass": np.full(input_ids.shape, 0.9, dtype=np.float32),
            "tail_mass": np.full(input_ids.shape, 0.1, dtype=np.float32),
            "teacher_entropy": np.full(input_ids.shape, 0.8, dtype=np.float32),
            "bucket_mass": bucket_mass,
            "bucket_count": bucket_count,
            "bucket_mean_logp": bucket_mean_logp,
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
