from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_targets,
    validate_fingerprint_artifact,
)
from qrwkv_xla.fingerprint import (
    TARGET_PAYLOAD_LEGACY_JSONL,
    TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
    FingerprintCaptureBudgetConfig,
    FingerprintCaptureConfig,
    FingerprintCorridorBoundsConfig,
    FingerprintExemplarReservoirCaptureConfig,
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
    assert packed_manifest["target_payload"]["target_capacity"] == 21
    assert (
        packed_manifest["target_payload"]["target_capture_memory_kind"]
        == "preallocated_typed_arrays"
    )
    assert packed_manifest["target_payload"]["mode_table_path"] == "modes.json"
    assert packed_manifest["target_payload"]["arrays"]["mode_id"]["dtype"] == "int32"
    assert packed_manifest["target_shards"] == []
    assert packed.summary["target_capture_memory_kind"] == "preallocated_typed_arrays"
    assert packed.summary["actual_target_count"] == 21
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


def test_packed_capture_trims_arrays_when_actual_records_below_capacity(
    tmp_path: Path,
) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=2,
        max_seq_len=4,
        vocab_size=12,
    )
    result = capture_fingerprint_artifact(
        FingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            overwrite=True,
            capture_budget=FingerprintCaptureBudgetConfig(
                max_target_positions=100,
            ),
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )
    manifest = _json(result.manifest_path)

    assert result.summary["target_capacity"] == 100
    assert result.summary["actual_target_count"] == 8
    assert manifest["target_payload"]["arrays"]["position"]["shape"] == [8]
    assert manifest["target_payload"]["arrays"]["examples_input_ids"]["shape"] == [
        2,
        4,
    ]
    assert np.load(result.output_dir / "targets" / "position.npy").shape == (8,)


def test_packed_capture_fails_closed_when_capacity_is_exceeded(
    tmp_path: Path,
) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=2,
        max_seq_len=4,
        vocab_size=12,
    )

    with pytest.raises(ValueError, match="target capacity exceeded"):
        capture_fingerprint_artifact(
            FingerprintCaptureConfig(
                output_dir=tmp_path / "artifact",
                overwrite=True,
                capture_budget=FingerprintCaptureBudgetConfig(
                    packed_target_capacity=3,
                ),
                target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
            ),
            examples,
        )


def test_packed_capture_does_not_use_legacy_target_list_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qrwkv_xla.fingerprint import capture as capture_module

    def fail_legacy_writer(**kwargs):
        raise AssertionError("packed capture used legacy target writer")

    monkeypatch.setattr(capture_module, "_write_target_payload", fail_legacy_writer)
    examples = build_synthetic_capture_examples(
        num_examples=3,
        max_seq_len=4,
        vocab_size=12,
    )

    result = capture_fingerprint_artifact(
        FingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            overwrite=True,
            capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=12),
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )

    assert result.validation_ok is True
    assert result.summary["target_capture_memory_kind"] == "preallocated_typed_arrays"


def test_packed_examples_store_input_ids_once_per_example(tmp_path: Path) -> None:
    result = _packed_capture(
        tmp_path,
        num_examples=3,
        max_seq_len=5,
        max_target_positions=13,
    )
    manifest = _json(result.manifest_path)
    examples = np.load(result.output_dir / "targets" / "examples_input_ids.npy")
    position_example_index = np.load(
        result.output_dir / "targets" / "position_example_index.npy"
    )

    assert manifest["target_payload"]["num_records"] == 13
    assert manifest["target_payload"]["num_examples"] == 3
    assert examples.shape == (3, 5)
    assert set(position_example_index.tolist()) == {0, 1, 2}


def test_packed_exemplar_output_matches_legacy_for_same_fixture(
    tmp_path: Path,
) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=4,
        max_seq_len=8,
        vocab_size=16,
    )
    config = FingerprintCaptureConfig(
        output_dir=tmp_path / "base",
        overwrite=True,
        capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=21),
        exemplar_reservoir=FingerprintExemplarReservoirCaptureConfig(
            enabled=True,
            max_exemplars=5,
            payload_type="cascaded_soft_labels_v1",
        ),
    )
    legacy = capture_fingerprint_artifact(
        replace(
            config,
            output_dir=tmp_path / "legacy",
            target_payload_type=TARGET_PAYLOAD_LEGACY_JSONL,
        ),
        examples,
    )
    packed = capture_fingerprint_artifact(
        replace(
            config,
            output_dir=tmp_path / "packed",
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )

    assert _jsonl(legacy.exemplars_path) == _jsonl(packed.exemplars_path)


def test_packed_quantile_histogram_is_deterministic_and_close_to_exact(
    tmp_path: Path,
) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=4,
        max_seq_len=8,
        vocab_size=16,
    )
    bounds = FingerprintCorridorBoundsConfig(
        method="quantile",
        lower_quantile=0.25,
        upper_quantile=0.75,
        min_width=1.0e-12,
        quantile_bins=4096,
    )
    base = FingerprintCaptureConfig(
        output_dir=tmp_path / "base",
        overwrite=True,
        capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=32),
        corridor_bounds=bounds,
    )
    legacy = capture_fingerprint_artifact(
        replace(
            base,
            output_dir=tmp_path / "legacy",
            target_payload_type=TARGET_PAYLOAD_LEGACY_JSONL,
        ),
        examples,
    )
    packed_a = capture_fingerprint_artifact(
        replace(
            base,
            output_dir=tmp_path / "packed_a",
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )
    packed_b = capture_fingerprint_artifact(
        replace(
            base,
            output_dir=tmp_path / "packed_b",
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )

    assert _json(packed_a.modes_path) == _json(packed_b.modes_path)
    assert packed_a.summary["quantile_aggregation_kind"] == "bounded_fixed_histogram_v1"
    assert packed_a.summary["bounded_quantile_state_size"] > 0

    exact_modes = {
        mode["mode_id"]: mode["bounds"] for mode in _json(legacy.modes_path)["modes"]
    }
    packed_modes = {
        mode["mode_id"]: mode["bounds"] for mode in _json(packed_a.modes_path)["modes"]
    }
    entropy_tol = math.log(16) / bounds.quantile_bins + 1.0e-6
    probability_tol = 1.0 / bounds.quantile_bins + 1.0e-6
    for mode_id, exact_bounds in exact_modes.items():
        for stat, exact in exact_bounds.items():
            tolerance = entropy_tol if stat == "entropy" else probability_tol
            packed = packed_modes[mode_id][stat]
            assert packed["min"] == pytest.approx(exact["min"], abs=tolerance)
            assert packed["max"] == pytest.approx(exact["max"], abs=tolerance)


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
            capture_budget=FingerprintCaptureBudgetConfig(max_target_positions=8),
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )
    return result.output_dir


def _packed_capture(
    tmp_path: Path,
    *,
    num_examples: int,
    max_seq_len: int,
    max_target_positions: int,
) -> object:
    examples = build_synthetic_capture_examples(
        num_examples=num_examples,
        max_seq_len=max_seq_len,
        vocab_size=16,
    )
    return capture_fingerprint_artifact(
        FingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            overwrite=True,
            capture_budget=FingerprintCaptureBudgetConfig(
                max_target_positions=max_target_positions,
            ),
            target_payload_type=TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        ),
        examples,
    )


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


def _jsonl(path: Path | None) -> list[dict]:
    assert path is not None
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
