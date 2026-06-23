from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.artifacts._json import write_json
from qrwkv_xla.fingerprint import write_fingerprint_provenance
from qrwkv_xla.fingerprint.aggressiveness_profiles import (
    resolve_aggressiveness_profile,
)
from qrwkv_xla.fingerprint.provenance import file_sha256, stable_hash
from qrwkv_xla.fingerprint.quality_per_byte import (
    ControlledQualityPerByteConfig,
    QualityBudgetPoint,
    run_controlled_quality_per_byte_experiment,
)
from qrwkv_xla.fingerprint.two_cycle_experiment import (
    ARM_NAMES,
    TwoCycleExperimentConfig,
    build_configuration_freeze,
    paired_lower_is_better_comparison,
    required_arms_present,
    run_two_cycle_experiment,
    sum_stage_resources,
    validate_three_way_split,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)


def test_ci_crossing_zero_is_inconclusive() -> None:
    result = paired_lower_is_better_comparison(
        {"a": 1.0, "b": 3.0, "c": 2.0},
        {"a": 2.0, "b": 2.0, "c": 2.0},
        left_name="two_cycle",
        right_name="exemplar_only",
        bootstrap_samples=500,
        bootstrap_seed=0,
        tie_tolerance=1e-12,
    )
    assert result["result"] == "inconclusive"


def test_paired_comparison_requires_aligned_records() -> None:
    try:
        paired_lower_is_better_comparison(
            {"a": 1.0},
            {"b": 1.0},
            left_name="left",
            right_name="right",
            bootstrap_samples=10,
            bootstrap_seed=0,
            tie_tolerance=0.0,
        )
    except ValueError as exc:
        assert "aligned record keys" in str(exc)
    else:
        raise AssertionError("unaligned records were accepted")


def test_resource_totals_equal_stage_sums() -> None:
    first = _resource(steps=3, records=4)
    second = _resource(steps=5, records=6)
    total = sum_stage_resources(first, second)
    assert total["optimizer_steps"] == 8
    assert total["records_consumed"] == 10
    assert total["artifact_bytes_logically_consumed"] == 20


def test_missing_arm_blocks_comparison() -> None:
    assert required_arms_present({name: {} for name in ARM_NAMES}) is True
    assert required_arms_present({name: {} for name in ARM_NAMES[:-1]}) is False


def test_three_valid_disjoint_splits_pass(tmp_path: Path) -> None:
    config = _split_config(tmp_path)
    receipt = validate_three_way_split(config)
    assert receipt["status"] == "pass"
    assert receipt["three_way_split_valid"] is True
    assert all(receipt["pairwise_disjointness"].values())


@pytest.mark.parametrize(
    ("pair", "overlap_kind", "error"),
    [
        ("training_calibration", "ids", "training_calibration_split_overlap"),
        ("training_calibration", "tokens", "training_calibration_split_overlap"),
        ("training_calibration", "texts", "training_calibration_split_overlap"),
        ("training_final", "ids", "training_final_test_split_overlap"),
        ("training_final", "tokens", "training_final_test_split_overlap"),
        ("training_final", "texts", "training_final_test_split_overlap"),
        ("calibration_final", "ids", "calibration_final_test_split_overlap"),
        ("calibration_final", "tokens", "calibration_final_test_split_overlap"),
        ("calibration_final", "texts", "calibration_final_test_split_overlap"),
    ],
)
def test_exact_split_overlap_fails_closed(
    tmp_path: Path, pair: str, overlap_kind: str, error: str
) -> None:
    specs = {
        "training": {"prefix": "train", "offset": 0, "text": "train"},
        "calibration": {"prefix": "cal", "offset": 2, "text": "cal"},
        "final": {"prefix": "final", "offset": 1, "text": "final"},
    }
    left, right = pair.split("_", 1)
    if right == "final":
        right = "final"
    if overlap_kind == "ids":
        specs[right]["prefix"] = specs[left]["prefix"]
    elif overlap_kind == "tokens":
        specs[right]["offset"] = specs[left]["offset"]
    else:
        specs[right]["text"] = specs[left]["text"]
    config = _split_config(tmp_path, specs=specs)
    with pytest.raises(ValueError, match=error):
        validate_three_way_split(config)


def test_reusing_calibration_as_final_test_fails(tmp_path: Path) -> None:
    config = _split_config(tmp_path)
    with pytest.raises(ValueError, match="calibration_final_test_split_overlap"):
        validate_three_way_split(
            replace(
                config,
                final_test_fingerprint_artifact=config.calibration_fingerprint_artifact,
            )
        )


def test_configuration_freeze_hash_is_deterministic_and_final_test_independent(
    tmp_path: Path,
) -> None:
    config = _split_config(tmp_path)
    kwargs = {
        "shared_initialization_hash": "sha256:init",
        "selected_profile": "rock_hammer",
        "selected_profile_config_sha256": "sha256:profile",
        "exemplar_sampling_receipt": {
            "record_order_sha256": "sha256:order",
            "records_selected": 4,
        },
        "calibration_artifact_sha256": "sha256:calibration",
        "training_artifact_sha256": "sha256:training",
    }
    first = build_configuration_freeze(config, **kwargs)
    second = build_configuration_freeze(
        replace(config, final_test_fingerprint_artifact=tmp_path / "unseen"),
        **kwargs,
    )
    changed = build_configuration_freeze(
        replace(config, exemplar_steps=config.exemplar_steps + 1), **kwargs
    )
    assert first["configuration_freeze_sha256"] == second["configuration_freeze_sha256"]
    assert (
        first["configuration_freeze_sha256"] != changed["configuration_freeze_sha256"]
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "calibration_training_artifact_sha256",
            "sha256:wrong",
            "calibration_training_artifact_mismatch",
        ),
        (
            "calibration_validation_artifact_sha256",
            "sha256:wrong",
            "calibration_validation_artifact_mismatch",
        ),
        (
            "calibration_student_config_sha256",
            "sha256:wrong",
            "calibration_student_config_mismatch",
        ),
        (
            "selected_profile_config_sha256",
            "sha256:wrong",
            "selected_profile_config_mismatch",
        ),
        (
            "calibration_training_artifact_sha256",
            None,
            "calibration_artifact_hashes_missing",
        ),
    ],
)
def test_calibration_lineage_mismatch_fails_before_arms(
    tmp_path: Path, field: str, value: str | None, error: str
) -> None:
    config = _split_config(tmp_path)
    receipt = _selection_receipt(
        tmp_path,
        training_artifact=config.training_fingerprint_artifact,
        calibration_artifact=config.calibration_fingerprint_artifact,
    )
    payload = _json(receipt)
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    write_json(receipt, payload)
    with pytest.raises(ValueError, match=error):
        run_two_cycle_experiment(
            replace(
                config,
                selected_profile_receipt=receipt,
                output_dir=tmp_path / "failed-run",
            )
        )
    assert not (tmp_path / "failed-run" / "arms").exists()


def test_final_test_role_mismatch_fails(tmp_path: Path) -> None:
    config = _split_config(tmp_path)
    provenance_path = (
        config.final_test_fingerprint_artifact / "fingerprint_provenance.json"
    )
    provenance = _json(provenance_path)
    provenance["artifact_role"] = "calibration_validation"
    write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="invalid_final_test_split_provenance"):
        validate_three_way_split(config)


def test_sequential_two_cycle_integration_smoke(tmp_path: Path) -> None:
    train_artifact, train_source = _artifact_copy(
        tmp_path / "train", prefix="train", token_offset=0, role="training"
    )
    calibration_artifact, _ = _artifact_copy(
        tmp_path / "calibration",
        prefix="calibration",
        token_offset=2,
        role="calibration_validation",
    )
    final_test_artifact, _ = _artifact_copy(
        tmp_path / "final-test",
        prefix="final-test",
        token_offset=1,
        role="final_held_out_test",
    )
    selection = _selection_receipt(
        tmp_path,
        training_artifact=train_artifact,
        calibration_artifact=calibration_artifact,
    )
    result = run_two_cycle_experiment(
        TwoCycleExperimentConfig(
            training_fingerprint_artifact=train_artifact,
            calibration_fingerprint_artifact=calibration_artifact,
            final_test_fingerprint_artifact=final_test_artifact,
            source_texts=train_source,
            selected_profile_receipt=selection,
            output_dir=tmp_path / "p155",
            student_backend="tiny_debug",
            baseline_steps=3,
            corridor_steps=3,
            exemplar_steps=3,
            batch_size=1,
            optimizer="sgd",
            baseline_learning_rate=1e-2,
            exemplar_learning_rate=1e-2,
            corridor_eval_every=1,
            exemplar_eval_every=1,
            checkpoint_every=3,
            bootstrap_samples=100,
            bootstrap_seed=0,
        )
    )
    report = _json(result.report_path)
    assert result.status == "pass"
    assert report["phase"] == "P155.1"
    assert report["arms"] == list(ARM_NAMES)
    assert report["shared_initialization_valid"] is True
    assert report["cycle_boundary_valid"] is True
    assert report["mixed_objective_enabled"] is False
    assert report["fresh_exemplar_optimizer_state"] is True
    assert report["primary_result"] in {
        "two_cycle_better",
        "exemplar_only_better",
        "inconclusive",
    }
    assert report["quality_per_byte_claim_made"] is False
    assert report["final_test_independent"] is True
    assert report["publication_grade_final_test"] is True
    assert report["evaluation_artifact_role"] == "final_held_out_test"
    assert report["resources"]["stage_matched_exemplar_budget"]["match"] is True
    boundary = _json(result.output_dir / "cycle_boundary_receipt.json")
    assert boundary["corridor_optimizer_state_loaded_by_exemplar"] is False
    assert boundary["exemplar_local_step_started_at_zero"] is True
    split = _json(result.output_dir / "three_way_split_validation.json")
    access = _json(result.output_dir / "final_test_access_receipt.json")
    assert split["three_way_split_valid"] is True
    assert access["configuration_frozen_before_final_test_access"] is True
    assert access["final_test_used_for_calibration"] is False
    per_record = _jsonl(result.output_dir / "per_record_arm_metrics.jsonl")
    assert all(row["example_id"].startswith("final-test-") for row in per_record)
    assert not any(row["example_id"].startswith("calibration-") for row in per_record)
    for name in (
        "two_cycle_experiment_report.json",
        "two_cycle_experiment_summary.md",
        "experiment_fairness_contract.json",
        "cycle_boundary_receipt.json",
        "shared_initialization_receipt.json",
        "paired_comparison_metrics.json",
        "per_record_arm_metrics.jsonl",
        "resource_accounting.json",
        "three_way_split_validation.json",
        "calibration_lineage_validation.json",
        "experiment_configuration_freeze.json",
        "final_test_access_receipt.json",
    ):
        assert (result.output_dir / name).is_file()


def test_controlled_quality_per_byte_fast_cpu_smoke(tmp_path: Path) -> None:
    train_artifact, train_source = _artifact_copy(
        tmp_path / "train", prefix="train", token_offset=0, role="training"
    )
    calibration_artifact, _ = _artifact_copy(
        tmp_path / "calibration",
        prefix="calibration",
        token_offset=2,
        role="calibration_validation",
    )
    final_test_artifact, _ = _artifact_copy(
        tmp_path / "final-test",
        prefix="final-test",
        token_offset=1,
        role="final_held_out_test",
    )
    selection = _selection_receipt(
        tmp_path,
        training_artifact=train_artifact,
        calibration_artifact=calibration_artifact,
    )
    physical_bytes = sum(
        path.stat().st_size for path in train_artifact.rglob("*") if path.is_file()
    )
    result = run_controlled_quality_per_byte_experiment(
        ControlledQualityPerByteConfig(
            training_fingerprint_artifact=train_artifact,
            calibration_fingerprint_artifact=calibration_artifact,
            final_test_fingerprint_artifact=final_test_artifact,
            source_texts=train_source,
            selected_profile_receipt=selection,
            output_dir=tmp_path / "p156",
            budget_points=(QualityBudgetPoint("smoke", physical_bytes, 2, 120.0),),
            seeds=(0,),
            student_backend="tiny_debug",
            optimizer="sgd",
            baseline_learning_rate=1e-2,
            exemplar_learning_rate=1e-2,
            bootstrap_samples=20,
        )
    )
    report = _json(result.report_path)
    assert result.status == "pass"
    assert report["required_arms_complete"] is True
    assert report["required_seeds_complete"] is False
    assert report["quality_per_byte_claim_allowed"] is False
    assert len(_jsonl(result.output_dir / "quality_budget_curve.jsonl")) == 12
    publication = _json(result.output_dir / "publication_grade_receipt.json")
    assert publication["publication_grade"] is False


def _artifact_copy(
    path: Path,
    *,
    prefix: str,
    token_offset: int,
    role: str,
    text_prefix: str | None = None,
) -> tuple[Path, Path]:
    shutil.copytree(FIXTURE, path)
    manifest_path = path / "manifest.json"
    manifest = _json(manifest_path)
    manifest["created_by"] = f"p155-test-{prefix}"
    write_json(manifest_path, manifest)
    source_rows = []
    target_path = path / "targets" / "targets-00000.jsonl"
    exemplar_path = path / "exemplars" / "exemplars-00000.jsonl"
    targets = _jsonl(target_path)
    exemplars = _jsonl(exemplar_path)
    for index, row in enumerate(targets):
        row["example_id"] = f"{prefix}-{index:03d}"
        row["input_ids"] = _shift_tokens(row["input_ids"], token_offset)
        source_rows.append(
            {
                "example_id": row["example_id"],
                "text": f"{text_prefix or prefix} text {index}",
            }
        )
    for index, row in enumerate(exemplars):
        row["example_id"] = f"{prefix}-{index:03d}"
        row["input_ids"] = _shift_tokens(row["input_ids"], token_offset)
    _write_jsonl(target_path, targets)
    _write_jsonl(exemplar_path, exemplars)
    source = path.parent / f"{path.name}-source.jsonl"
    _write_jsonl(source, source_rows)
    write_fingerprint_provenance(path, source_file=source, artifact_role=role)
    return path, source


def _split_config(
    tmp_path: Path, *, specs: dict[str, dict] | None = None
) -> TwoCycleExperimentConfig:
    resolved = specs or {
        "training": {"prefix": "train", "offset": 0, "text": "train"},
        "calibration": {"prefix": "cal", "offset": 2, "text": "cal"},
        "final": {"prefix": "final", "offset": 1, "text": "final"},
    }
    train, source = _artifact_copy(
        tmp_path / "train",
        prefix=resolved["training"]["prefix"],
        token_offset=resolved["training"]["offset"],
        text_prefix=resolved["training"]["text"],
        role="training",
    )
    calibration, _ = _artifact_copy(
        tmp_path / "calibration",
        prefix=resolved["calibration"]["prefix"],
        token_offset=resolved["calibration"]["offset"],
        text_prefix=resolved["calibration"]["text"],
        role="calibration_validation",
    )
    final_test, _ = _artifact_copy(
        tmp_path / "final",
        prefix=resolved["final"]["prefix"],
        token_offset=resolved["final"]["offset"],
        text_prefix=resolved["final"]["text"],
        role="final_held_out_test",
    )
    return TwoCycleExperimentConfig(
        training_fingerprint_artifact=train,
        calibration_fingerprint_artifact=calibration,
        final_test_fingerprint_artifact=final_test,
        source_texts=source,
        selected_profile_receipt=tmp_path / "unused-selection.json",
        output_dir=tmp_path / "output",
        student_backend="tiny_debug",
    )


def _selection_receipt(
    tmp_path: Path, *, training_artifact: Path, calibration_artifact: Path
) -> Path:
    profile = resolve_aggressiveness_profile("rock_hammer")
    config = profile.to_dict()
    write_json(tmp_path / "selected_profile_config.json", config)
    calibration_report = tmp_path / "aggressiveness_calibration_report.json"
    publication_receipt = tmp_path / "publication_grade_receipt.json"
    write_json(
        calibration_report,
        {
            "status": "pass",
            "selected_profile": "rock_hammer",
        },
    )
    write_json(publication_receipt, {"publication_grade": True})
    receipt = tmp_path / "profile_selection_receipt.json"
    write_json(
        receipt,
        {
            "status": "pass",
            "selection_allowed": True,
            "winner_declared": True,
            "selected_profile": "rock_hammer",
            "selected_profile_config_sha256": stable_hash(config),
            "calibration_training_artifact_sha256": file_sha256(
                training_artifact / "manifest.json"
            ),
            "calibration_validation_artifact_sha256": file_sha256(
                calibration_artifact / "manifest.json"
            ),
            "calibration_student_config_sha256": stable_hash(
                {
                    "architecture_id": "tiny_debug",
                    "backend_name": "TinyDebugStudentBackend",
                    "architecture": "NoneType",
                    "vocab_size": 16,
                    "hidden_size": 0,
                    "num_layers": 0,
                    "num_heads": None,
                    "num_kv_heads": None,
                    "emit_logits": True,
                    "tie_embeddings": False,
                    "emit_mixer_outputs": False,
                }
            ),
            "calibration_report_sha256": file_sha256(calibration_report),
            "publication_grade_receipt_sha256": file_sha256(publication_receipt),
        },
    )
    return receipt


def _shift_tokens(tokens: list[int], offset: int) -> list[int]:
    return [0 if token == 0 else ((token - 1 + offset) % 15) + 1 for token in tokens]


def _resource(*, steps: int, records: int) -> dict[str, int | float]:
    return {
        "optimizer_steps": steps,
        "records_consumed": records,
        "tokens_consumed": records * 8,
        "artifact_bytes_logically_consumed": records * 2,
        "training_seconds": float(steps),
        "evaluation_seconds": 1.0,
        "checkpoint_seconds": 1.0,
        "total_wall_clock_seconds": float(steps + 2),
    }


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
