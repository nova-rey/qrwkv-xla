from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts import (
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)
from qrwkv_xla.artifacts._json import write_json
from qrwkv_xla.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)
from qrwkv_xla.targets.store import TeacherTargetStore


def test_fake_teacher_textbook_validates(tmp_path: Path) -> None:
    textbook = create_fake_teacher_textbook(tmp_path)

    report = validate_teacher_textbook(textbook)

    assert report.status == "pass"
    assert report.metadata_ok is True
    assert report.vocab_contract_ok is True
    assert report.manifest_ok is True
    assert report.emission_config_ok is True
    assert report.shards_ok is True


def test_missing_teacher_manifest_fails(tmp_path: Path) -> None:
    textbook = create_fake_teacher_textbook(tmp_path)
    (textbook / "teacher_manifest.json").unlink()

    report = validate_teacher_textbook(textbook)

    assert report.status == "fail"
    assert any("teacher_manifest.json" in blocker for blocker in report.blockers)


def test_shard_count_mismatch_fails(tmp_path: Path) -> None:
    textbook = create_fake_teacher_textbook(tmp_path)
    (textbook / "shards" / "shard-00001.npz").unlink()

    report = validate_teacher_textbook(textbook)

    assert report.status == "fail"
    assert any("shard_count mismatch" in blocker for blocker in report.blockers)


def test_teacher_textbook_report_serializes(tmp_path: Path) -> None:
    textbook = create_fake_teacher_textbook(tmp_path)
    report = validate_teacher_textbook(textbook)
    report_path = textbook / "validation_report.json"

    write_teacher_textbook_validation_report(report, report_path)

    assert report_path.is_file()
    assert validate_teacher_textbook(textbook).validation_report_ok is True


def create_fake_teacher_textbook(tmp_path: Path) -> Path:
    root = tmp_path / "teacher_textbook"
    store = TeacherTargetStore.create(root, _metadata())
    store.write_shard(0, _arrays(offset=0))
    store.write_shard(1, _arrays(offset=100))
    store.validate()

    write_json(
        root / "vocab_contract.json",
        {
            "tokenizer_id": "sshleifer/tiny-gpt2",
            "tokenizer_hash": None,
            "vocab_size": 5,
            "model_id": "sshleifer/tiny-gpt2",
            "model_family": "hf-causal-lm",
        },
    )
    write_json(
        root / "teacher_manifest.json",
        {
            "artifact_type": "teacher_textbook",
            "artifact_version": 0,
            "teacher_model_id": "sshleifer/tiny-gpt2",
            "teacher_backend_type": "hf_causal_lm",
            "teacher_revision_or_hash": None,
            "tokenizer_id": "sshleifer/tiny-gpt2",
            "vocab_size": 5,
            "vocab_contract_path": "vocab_contract.json",
            "target_type": "synthetic",
            "dtype": "float32",
            "sequence_length": 3,
            "num_examples": 4,
            "shard_count": 2,
            "created_at": "2026-06-04T00:00:00Z",
            "local_files_only": True,
            "allow_downloads": False,
            "claims_not_made": [
                "no_qwen_parity_claim",
                "no_model_quality_claim",
            ],
        },
    )
    write_json(
        root / "emission_config.json",
        {
            "dataset_source": "unit_fixture",
            "max_examples": 4,
            "batch_size": 2,
            "sequence_length": 3,
            "logits_dtype": "float32",
            "include_hidden_states": False,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "seed": 0,
        },
    )
    assert asdict(_metadata())["vocab_size"] == 5
    return root


def _metadata() -> TargetStoreMetadata:
    return TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="sshleifer/tiny-gpt2",
        model_family="hf-causal-lm",
        tokenizer_id="sshleifer/tiny-gpt2",
        tokenizer_hash=None,
        vocab_size=5,
        target_type="synthetic",
        dtype="float32",
        sequence_length=3,
        num_examples=4,
        shard_count=2,
        created_by="test",
        created_at="2026-06-04T00:00:00Z",
        source={"kind": "unit"},
        provenance={"phase": "P116"},
    )


def _arrays(*, offset: int) -> dict[str, np.ndarray]:
    input_ids = np.arange(offset, offset + 6, dtype=np.int32).reshape(2, 3) % 5
    logits = np.arange(offset, offset + 30, dtype=np.float32).reshape(2, 3, 5)
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones((2, 3), dtype=np.int32),
        "logits": logits / 10.0,
    }
