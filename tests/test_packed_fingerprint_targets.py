from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts import (
    load_fingerprint_targets,
    validate_fingerprint_artifact,
)
from qrwkv_xla.fingerprint import (
    TARGET_PAYLOAD_LEGACY_JSONL,
    TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
    FingerprintCaptureBudgetConfig,
    FingerprintCaptureConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    build_synthetic_capture_examples,
    capture_fingerprint_artifact,
)


def test_packed_corridor_targets_validate_and_match_legacy_loader(
    tmp_path: Path,
) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=4,
        max_seq_len=8,
        vocab_size=16,
    )
    base_config = FingerprintCaptureConfig(
        output_dir=tmp_path / "base",
        overwrite=True,
        capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=21),
    )
    legacy = capture_fingerprint_artifact(
        replace(
            base_config,
            output_dir=tmp_path / "legacy",
            target_payload_type=TARGET_PAYLOAD_LEGACY_JSONL,
        ),
        examples,
    )
    packed = capture_fingerprint_artifact(
        replace(
            base_config,
            output_dir=tmp_path / "packed",
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )

    packed_manifest = _json(packed.manifest_path)
    assert packed.targets_path == packed.output_dir / "targets"
    assert (
        packed_manifest["target_payload"]["kind"]
        == TARGET_PAYLOAD_PACKED_CORRIDOR_V1
    )
    assert packed_manifest["target_payload"]["num_records"] == 21
    assert packed_manifest["target_payload"]["num_examples"] == 3
    assert packed_manifest["target_payload"]["mode_table_path"] == "modes.json"
    assert packed_manifest["target_payload"]["arrays"]["mode_id"]["dtype"] == "int32"
    assert packed_manifest["target_shards"] == []
    assert validate_fingerprint_artifact(packed.output_dir).ok is True

    for batch_size in (2, 7):
        legacy_dataset = load_fingerprint_targets(
            legacy.output_dir,
            batch_size=batch_size,
        )
        packed_dataset = load_fingerprint_targets(
            packed.output_dir,
            batch_size=batch_size,
        )
        assert _records(legacy_dataset) == _records(packed_dataset)
        assert _batches(legacy_dataset) == _batches(packed_dataset)


def test_packed_validator_fails_closed_on_array_contract_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _packed_artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = _json(manifest_path)
    manifest["target_payload"]["arrays"]["position"]["dtype"] = "float32"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    validation = validate_fingerprint_artifact(artifact)

    assert validation.ok is False
    assert any("position.dtype mismatch" in blocker for blocker in validation.blockers)


def test_packed_validator_fails_closed_on_invalid_mode_id(tmp_path: Path) -> None:
    artifact = _packed_artifact(tmp_path)
    mode_path = artifact / "targets" / "mode_id.npy"
    mode_ids = np.load(mode_path, allow_pickle=False)
    mode_ids[0] = 9999
    np.save(mode_path, mode_ids)

    validation = validate_fingerprint_artifact(artifact)

    assert validation.ok is False
    assert "mode_id contains unknown mode ids" in validation.blockers


def test_real_teacher_config_defaults_to_packed_target_payload(tmp_path: Path) -> None:
    config = TinyRealTeacherFingerprintCaptureConfig(
        output_dir=tmp_path / "artifact",
        texts_path=tmp_path / "texts.jsonl",
    )

    assert config.target_payload_type == TARGET_PAYLOAD_PACKED_CORRIDOR_V1


def _packed_artifact(tmp_path: Path) -> Path:
    examples = build_synthetic_capture_examples(
        num_examples=2,
        max_seq_len=4,
        vocab_size=12,
    )
    result = capture_fingerprint_artifact(
        FingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            overwrite=True,
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )
    return result.output_dir


def _records(dataset) -> tuple[tuple, ...]:
    return tuple(
        (
            record.example_id,
            record.position,
            record.input_ids,
            record.mode_id,
            record.entropy_min,
            record.entropy_max,
            record.top1_margin_min,
            record.top1_margin_max,
            record.top8_mass_min,
            record.top8_mass_max,
            record.top32_mass_min,
            record.top32_mass_max,
            record.tail_mass_min,
            record.tail_mass_max,
            record.weight,
        )
        for record in dataset.iter_records()
    )


def _batches(dataset) -> tuple[tuple, ...]:
    return tuple(
        (
            batch.input_ids.tolist(),
            batch.position.tolist(),
            batch.mode_id.tolist(),
            batch.entropy_min.tolist(),
            batch.entropy_max.tolist(),
            batch.top1_margin_min.tolist(),
            batch.top1_margin_max.tolist(),
            batch.top8_mass_min.tolist(),
            batch.top8_mass_max.tolist(),
            batch.top32_mass_min.tolist(),
            batch.top32_mass_max.tolist(),
            batch.tail_mass_min.tolist(),
            batch.tail_mass_max.tolist(),
            batch.weight.tolist(),
        )
        for batch in dataset.iter_batches()
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
