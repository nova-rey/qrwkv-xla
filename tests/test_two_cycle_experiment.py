from __future__ import annotations

import json
import shutil
from pathlib import Path

from qrwkv_xla.artifacts._json import write_json
from qrwkv_xla.fingerprint import write_fingerprint_provenance
from qrwkv_xla.fingerprint.aggressiveness_profiles import (
    resolve_aggressiveness_profile,
)
from qrwkv_xla.fingerprint.provenance import stable_hash
from qrwkv_xla.fingerprint.two_cycle_experiment import (
    ARM_NAMES,
    TwoCycleExperimentConfig,
    paired_lower_is_better_comparison,
    required_arms_present,
    run_two_cycle_experiment,
    sum_stage_resources,
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


def test_sequential_two_cycle_integration_smoke(tmp_path: Path) -> None:
    train_artifact, train_source = _artifact_copy(
        tmp_path / "train", prefix="train", token_offset=0, role="training"
    )
    held_artifact, _ = _artifact_copy(
        tmp_path / "held",
        prefix="held",
        token_offset=2,
        role="held_out_evaluation",
    )
    selection = _selection_receipt(tmp_path)
    result = run_two_cycle_experiment(
        TwoCycleExperimentConfig(
            training_fingerprint_artifact=train_artifact,
            held_out_fingerprint_artifact=held_artifact,
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
    assert report["resources"]["stage_matched_exemplar_budget"]["match"] is True
    boundary = _json(result.output_dir / "cycle_boundary_receipt.json")
    assert boundary["corridor_optimizer_state_loaded_by_exemplar"] is False
    assert boundary["exemplar_local_step_started_at_zero"] is True
    for name in (
        "two_cycle_experiment_report.json",
        "two_cycle_experiment_summary.md",
        "experiment_fairness_contract.json",
        "cycle_boundary_receipt.json",
        "shared_initialization_receipt.json",
        "paired_comparison_metrics.json",
        "per_record_arm_metrics.jsonl",
        "resource_accounting.json",
    ):
        assert (result.output_dir / name).is_file()


def _artifact_copy(
    path: Path, *, prefix: str, token_offset: int, role: str
) -> tuple[Path, Path]:
    shutil.copytree(FIXTURE, path)
    source_rows = []
    target_path = path / "targets" / "targets-00000.jsonl"
    exemplar_path = path / "exemplars" / "exemplars-00000.jsonl"
    targets = _jsonl(target_path)
    exemplars = _jsonl(exemplar_path)
    for index, row in enumerate(targets):
        row["example_id"] = f"{prefix}-{index:03d}"
        row["input_ids"] = _shift_tokens(row["input_ids"], token_offset)
        source_rows.append(
            {"example_id": row["example_id"], "text": f"{prefix} text {index}"}
        )
    for index, row in enumerate(exemplars):
        row["example_id"] = f"{prefix}-{index:03d}"
        row["input_ids"] = _shift_tokens(row["input_ids"], token_offset)
    _write_jsonl(target_path, targets)
    _write_jsonl(exemplar_path, exemplars)
    source = path.parent / f"{prefix}.jsonl"
    _write_jsonl(source, source_rows)
    write_fingerprint_provenance(path, source_file=source, artifact_role=role)
    return path, source


def _selection_receipt(tmp_path: Path) -> Path:
    profile = resolve_aggressiveness_profile("rock_hammer")
    config = profile.to_dict()
    write_json(tmp_path / "selected_profile_config.json", config)
    receipt = tmp_path / "profile_selection_receipt.json"
    write_json(
        receipt,
        {
            "status": "pass",
            "selection_allowed": True,
            "winner_declared": True,
            "selected_profile": "rock_hammer",
            "selected_profile_config_sha256": stable_hash(config),
        },
    )
    write_json(
        tmp_path / "aggressiveness_calibration_report.json",
        {
            "status": "pass",
            "selected_profile": "rock_hammer",
        },
    )
    write_json(
        tmp_path / "publication_grade_receipt.json",
        {"publication_grade": False},
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
