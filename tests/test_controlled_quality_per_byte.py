from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.fingerprint.quality_per_byte import (
    ControlledQualityPerByteConfig,
    QualityBudgetPoint,
    _curve_auc,
    controlled_matrix_config_hash,
    efficiency_ratios,
    observed_target_quality_cost,
    paired_seed_record_comparison,
    quality_per_byte_claims,
    trapezoidal_auc,
    validate_backend_requirement,
    validate_budget_allocation,
)


def test_cpu_backend_requirement_passes() -> None:
    receipt = validate_backend_requirement(
        "cpu", observed_backend="cpu", device_count=2, process_count=1
    )
    assert receipt["backend_requirement_met"] is True
    assert receipt["distributed"] is False


def test_non_cpu_backend_requirement_fails() -> None:
    with pytest.raises(ValueError, match="backend requirement not met"):
        validate_backend_requirement(
            "cpu", observed_backend="gpu", device_count=1, process_count=1
        )


def test_two_cycle_budget_allocations_match() -> None:
    receipt = validate_budget_allocation(
        total_bytes=101,
        corridor_bytes=50,
        exemplar_bytes=51,
        total_steps=25,
        corridor_steps=12,
        exemplar_steps=13,
    )
    assert receipt == {
        "budget_match_valid": True,
        "step_budget_match_valid": True,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "total_bytes": 100,
            "corridor_bytes": 60,
            "exemplar_bytes": 50,
            "total_steps": 10,
            "corridor_steps": 5,
            "exemplar_steps": 5,
        },
        {
            "total_bytes": 100,
            "corridor_bytes": 50,
            "exemplar_bytes": 50,
            "total_steps": 10,
            "corridor_steps": 6,
            "exemplar_steps": 5,
        },
    ],
)
def test_budget_overflow_fails(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="budget"):
        validate_budget_allocation(**kwargs)


def test_target_quality_uses_first_observed_point() -> None:
    points = [
        _point("medium", score=0.4, steps=20),
        _point("small", score=0.8, steps=10),
    ]
    result = observed_target_quality_cost(points, threshold=0.5)
    assert result["target_reached"] is True
    assert result["observed_budget_point"] == "medium"
    assert result["steps_to_target"] == 20


def test_no_target_has_null_costs() -> None:
    result = observed_target_quality_cost(
        [_point("small", score=0.8, steps=10)], threshold=0.5
    )
    assert result["target_reached"] is False
    assert result["steps_to_target"] is None
    assert result["bytes_to_target"] is None


def test_efficiency_ratio_rejects_nonpositive_denominator() -> None:
    with pytest.raises(ValueError, match="denominators"):
        efficiency_ratios(
            reference_score=1.0,
            final_score=0.5,
            artifact_bytes=0,
            optimizer_steps=10,
            wall_clock_seconds=1.0,
        )


def test_paired_bootstrap_is_deterministic_and_crossing_zero_inconclusive() -> None:
    kwargs = {
        "left": {"0:a": 1.0, "1:a": 3.0, "2:a": 2.0},
        "right": {"0:a": 2.0, "1:a": 2.0, "2:a": 2.0},
        "bootstrap_samples": 500,
        "bootstrap_seed": 7,
        "tie_tolerance": 1e-12,
    }
    first = paired_seed_record_comparison(**kwargs)
    assert first == paired_seed_record_comparison(**kwargs)
    assert first["result"] == "inconclusive"


def test_auc_is_deterministic_trapezoidal_integration() -> None:
    assert trapezoidal_auc([(2.0, 1.0), (0.0, 3.0), (1.0, 2.0)]) == 4.0


def test_auc_is_unavailable_for_zero_range_budget_view() -> None:
    rows = [
        {
            "budget_view": "artifact_bytes",
            "arm": "conventional_baseline",
            "seed": 0,
            "x": 0,
            "final_test_primary_score": score,
        }
        for score in (1.0, 0.9)
    ]
    result = _curve_auc(rows)
    assert result[0]["available"] is False
    assert result[0]["area_under_quality_curve"] is None


def test_resume_hash_is_stable_and_matrix_change_invalidates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = controlled_matrix_config_hash(config)
    assert first == controlled_matrix_config_hash(config)
    changed = replace(
        config,
        budget_points=(QualityBudgetPoint("small", 100, 3, 30.0),),
    )
    assert controlled_matrix_config_hash(config) != controlled_matrix_config_hash(
        changed
    )


def test_claims_reject_incomplete_matrix() -> None:
    claims = quality_per_byte_claims(
        gates={"matrix_complete": False, "split_valid": True}
    )
    assert claims["quality_per_byte_claim_allowed"] is False
    assert claims["winner_declared"] is False


def _point(name: str, *, score: float, steps: int) -> dict[str, float | int | str]:
    return {
        "budget_point": name,
        "final_test_primary_score": score,
        "optimizer_steps": steps,
        "teacher_artifact_bytes": steps * 10,
        "total_wall_clock_seconds": float(steps),
        "records_consumed": steps * 2,
    }


def _config(tmp_path: Path) -> ControlledQualityPerByteConfig:
    artifacts = []
    for name in ("training", "calibration", "final"):
        path = tmp_path / name
        path.mkdir()
        (path / "manifest.json").write_text(f'{{"name":"{name}"}}')
        artifacts.append(path)
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    receipt = tmp_path / "selection.json"
    receipt.write_text("{}")
    return ControlledQualityPerByteConfig(
        training_fingerprint_artifact=artifacts[0],
        calibration_fingerprint_artifact=artifacts[1],
        final_test_fingerprint_artifact=artifacts[2],
        source_texts=source,
        selected_profile_receipt=receipt,
        output_dir=tmp_path / "output",
        budget_points=(QualityBudgetPoint("small", 100, 2, 30.0),),
        seeds=(0,),
    )
