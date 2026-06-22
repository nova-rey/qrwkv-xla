from __future__ import annotations

from pathlib import Path

from qrwkv_xla.fingerprint.aggressiveness_calibration import (
    AggressivenessCalibrationConfig,
    _aggregate_validation,
    _publication_grade_receipt,
    bootstrap_ci95,
    entry_exit_metrics,
    select_profile,
)
from qrwkv_xla.fingerprint.aggressiveness_profiles import (
    PROFILE_NAMES,
    aggressiveness_profiles,
    resolve_aggressiveness_profile,
)


def test_profiles_are_complete_and_ordered() -> None:
    profiles = aggressiveness_profiles()
    assert tuple(profile.profile_name for profile in profiles) == PROFILE_NAMES
    assert [profile.aggressiveness_rank for profile in profiles] == [0, 1, 2, 3]


def test_profile_override_is_explicit() -> None:
    profile = resolve_aggressiveness_profile(
        "ball_peen", {"learning_rate": 0.01, "entropy_weight": 4.0}
    )
    assert profile.learning_rate == 0.01
    assert profile.per_stat_weights["entropy"] == 4.0
    assert profile.profile_overrides_applied == ("entropy_weight", "learning_rate")


def test_unknown_profile_fails() -> None:
    try:
        resolve_aggressiveness_profile("unknown")
    except ValueError as exc:
        assert "unknown aggressiveness profile" in str(exc)
    else:
        raise AssertionError("unknown profile was accepted")


def test_entry_exit_metrics_count_reentry_and_rebound() -> None:
    trajectory = [
        {"optimizer_step": 0, "inside_all_rate": 0.0},
        {"optimizer_step": 1, "inside_all_rate": 0.95},
        {"optimizer_step": 2, "inside_all_rate": 0.5},
        {"optimizer_step": 3, "inside_all_rate": 1.0},
        {"optimizer_step": 4, "inside_all_rate": 1.0},
    ]
    metrics = entry_exit_metrics(trajectory, 0.95)
    assert metrics["first_entry_step"] == 1
    assert metrics["first_exit_after_entry_step"] == 2
    assert metrics["entry_then_exit_count"] == 1
    assert metrics["max_consecutive_inside_evals"] == 2
    assert not metrics["stable_after_first_entry"]


def test_bootstrap_is_deterministic() -> None:
    left = bootstrap_ci95([1.0, 2.0, 3.0], samples=200, seed=17)
    right = bootstrap_ci95([1.0, 2.0, 3.0], samples=200, seed=17)
    assert left == right


def test_destructive_profile_is_excluded() -> None:
    selected, ranking = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary("gallagher", rank=3, steps=(1, 1, 1), abort_rate=1.0),
        ]
    )
    assert selected == "rock_hammer"
    gallagher = _row(ranking, "gallagher")
    assert gallagher["profile_status"] == "destructive"
    assert gallagher["selection_eligible"] is False
    assert "abort_rate_exceeded" in gallagher["selection_exclusion_reasons"]


def test_non_finite_profile_is_excluded() -> None:
    _, ranking = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary("sledgehammer", rank=2, steps=(1, 1, 1), non_finite_run_rate=1.0),
        ]
    )
    row = _row(ranking, "sledgehammer")
    assert row["profile_status"] == "destructive"
    assert row["selection_eligible"] is False
    assert "non_finite_run_rate_exceeded" in row["selection_exclusion_reasons"]


def test_incomplete_profile_is_excluded() -> None:
    _, ranking = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary(
                "ball_peen",
                rank=1,
                steps=(4, 4),
                expected_run_count=3,
                observed_run_count=2,
                completed_run_count=2,
                failed_run_count=0,
                missing_run_count=1,
                completed_run_rate=2 / 3,
                stable_entry_success_rate=2 / 3,
            ),
        ]
    )
    row = _row(ranking, "ball_peen")
    assert row["profile_status"] == "inconclusive"
    assert row["selection_eligible"] is False
    assert "missing_runs_present" in row["selection_exclusion_reasons"]


def test_too_weak_profile_is_excluded() -> None:
    _, ranking = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary(
                "ball_peen",
                rank=1,
                steps=(5, 5, 5),
                stable_entry_success_rate=2 / 3,
            ),
        ]
    )
    row = _row(ranking, "ball_peen")
    assert row["profile_status"] == "too_weak"
    assert row["selection_eligible"] is False


def test_invalid_fairness_excludes_all_candidates() -> None:
    selected, ranking = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary("ball_peen", rank=1, steps=(5, 5, 5)),
        ],
        comparison_valid=False,
    )
    assert selected is None
    assert all(row["profile_status"] == "invalid" for row in ranking)
    assert all(row["selection_eligible"] is False for row in ranking)


def test_only_acceptable_profiles_enter_candidate_pool() -> None:
    _, ranking = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary("ball_peen", rank=1, steps=(6, 6, 6)),
            _summary("sledgehammer", rank=2, steps=(1, 1, 1), abort_rate=1.0),
            _summary(
                "gallagher", rank=3, steps=(2, 2, 2), stable_entry_success_rate=0.0
            ),
        ]
    )
    eligible = {row["profile_name"] for row in ranking if row["selection_eligible"]}
    assert eligible == {"rock_hammer", "ball_peen"}


def test_fast_destructive_profile_cannot_win_regression() -> None:
    selected, _ = select_profile(
        [
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
            _summary("ball_peen", rank=1, steps=(5, 5, 5)),
            _summary("sledgehammer", rank=2, steps=(1, 1, 1), abort_rate=1.0),
        ]
    )
    assert selected == "rock_hammer"


def test_tie_selects_lower_aggressiveness_rank() -> None:
    selected, ranking = select_profile(
        [
            _summary("ball_peen", rank=1, steps=(5, 5, 5)),
            _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
        ]
    )
    assert selected == "rock_hammer"
    assert _row(ranking, "rock_hammer")["selection_status"] == "selected"


def test_no_candidates_is_inconclusive() -> None:
    selected, ranking = select_profile(
        [
            _summary("sledgehammer", rank=2, steps=(1, 1, 1), abort_rate=1.0),
            _summary(
                "gallagher", rank=3, steps=(2, 2, 2), stable_entry_success_rate=0.0
            ),
        ]
    )
    assert selected is None
    assert all(row["selection_eligible"] is False for row in ranking)


def test_selection_is_deterministic_and_input_order_independent() -> None:
    summaries = [
        _summary("ball_peen", rank=1, steps=(5, 5, 5)),
        _summary("rock_hammer", rank=0, steps=(5, 5, 5)),
        _summary("sledgehammer", rank=2, steps=(7, 7, 7)),
    ]
    first, _ = select_profile(summaries, bootstrap_seed=99)
    second, _ = select_profile(list(reversed(summaries)), bootstrap_seed=99)
    assert first == second == "rock_hammer"


def test_publication_grade_requires_valid_fairness() -> None:
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=True),
        fairness={
            "comparison_valid": False,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=_publication_summaries(),
        aggregate_validation=_aggregate_validation_pass(),
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is False
    assert receipt["publication_grade_requirements"]["fairness_valid"] is False


def test_publication_grade_requires_all_profiles_present() -> None:
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=True),
        fairness={
            "comparison_valid": True,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=_publication_summaries()[:-1],
        aggregate_validation=_aggregate_validation_pass(),
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is False
    assert receipt["publication_grade_requirements"]["all_profiles_present"] is False


def test_publication_grade_requires_all_seed_runs_present() -> None:
    summaries = _publication_summaries()
    summaries[0]["observed_run_count"] = 2
    summaries[0]["missing_run_count"] = 1
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=True),
        fairness={
            "comparison_valid": True,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=summaries,
        aggregate_validation=_aggregate_validation_pass(),
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is False
    assert receipt["publication_grade_requirements"]["all_seed_runs_present"] is False


def test_publication_grade_requires_complete_runs() -> None:
    summaries = _publication_summaries()
    summaries[0]["completed_run_count"] = 2
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=True),
        fairness={
            "comparison_valid": True,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=summaries,
        aggregate_validation=_aggregate_validation_pass(),
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is False
    assert receipt["publication_grade_requirements"]["all_runs_complete"] is False


def test_publication_grade_requires_valid_lineage() -> None:
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=False),
        fairness={
            "comparison_valid": True,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=_publication_summaries(),
        aggregate_validation=_aggregate_validation_pass(),
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is False
    assert receipt["publication_grade_requirements"]["lineage_valid"] is False


def test_publication_grade_requires_finite_aggregates() -> None:
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=True),
        fairness={
            "comparison_valid": True,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=_publication_summaries(),
        aggregate_validation={"status": "fail"},
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is False
    assert (
        receipt["publication_grade_requirements"]["aggregate_metrics_finite"] is False
    )


def test_publication_grade_passes_with_three_complete_aligned_seeds() -> None:
    receipt = _publication_grade_receipt(
        config=_config(),
        rows=_rows(lineage_valid=True),
        fairness={
            "comparison_valid": True,
            "only_declared_aggressiveness_fields_differ": True,
        },
        summaries=_publication_summaries(),
        aggregate_validation=_aggregate_validation_pass(),
        selection_receipt={"selection_status": "selected"},
    )
    assert receipt["publication_grade"] is True


def test_aggregate_validation_fails_on_non_finite_summary_metric() -> None:
    summaries = _publication_summaries()
    summaries[0]["mean_trajectory_variance"] = float("inf")
    validation = _aggregate_validation(
        config=_config(),
        rows=_rows(lineage_valid=True),
        summaries=summaries,
        fairness={"comparison_valid": True},
        seed_checks=[_seed_check()],
    )
    assert validation["status"] == "fail"
    assert validation["all_metrics_finite"] is False


def _summary(
    profile_name: str,
    *,
    rank: int,
    steps: tuple[float, ...],
    expected_run_count: int | None = None,
    observed_run_count: int | None = None,
    completed_run_count: int | None = None,
    failed_run_count: int = 0,
    missing_run_count: int = 0,
    completed_run_rate: float = 1.0,
    stable_entry_success_rate: float = 1.0,
    abort_rate: float = 0.0,
    non_finite_run_rate: float = 0.0,
    entry_then_exit_rate: float = 0.0,
    severe_rebound_rate: float = 0.0,
    mean_trajectory_variance: float = 0.0,
) -> dict[str, object]:
    count = len(steps)
    return {
        "profile_name": profile_name,
        "aggressiveness_rank": rank,
        "expected_run_count": count
        if expected_run_count is None
        else expected_run_count,
        "observed_run_count": count
        if observed_run_count is None
        else observed_run_count,
        "completed_run_count": count
        if completed_run_count is None
        else completed_run_count,
        "failed_run_count": failed_run_count,
        "aborted_run_count": round(abort_rate * count),
        "missing_run_count": missing_run_count,
        "run_count": count,
        "completed_run_rate": completed_run_rate,
        "stable_entry_success_rate": stable_entry_success_rate,
        "median_steps_to_stable_entry": float(sorted(steps)[count // 2])
        if steps
        else None,
        "median_seconds_to_stable_entry": float(sorted(steps)[count // 2])
        if steps
        else None,
        "median_records_to_stable_entry": float(sorted(steps)[count // 2])
        if steps
        else None,
        "median_bytes_to_stable_entry": float(sorted(steps)[count // 2])
        if steps
        else None,
        "mean_steps_to_stable_entry": float(sum(steps) / count) if steps else None,
        "steps_to_stable_entry_ci95": [min(steps), max(steps)] if steps else None,
        "mean_best_distance": 0.0,
        "mean_final_distance": 0.0,
        "mean_trajectory_variance": mean_trajectory_variance,
        "mean_entry_then_exit_count": entry_then_exit_rate,
        "entry_then_exit_rate": entry_then_exit_rate,
        "abort_rate": abort_rate,
        "non_finite_run_rate": non_finite_run_rate,
        "severe_rebound_rate": severe_rebound_rate,
        "median_parameter_delta": 1.0,
        "mean_distance_rebound": 0.0,
        "mean_held_out_loss_rebound": 0.0,
        "lineage_valid_for_all_runs": True,
        "seed_metrics": [
            {
                "seed": index,
                "steps_to_stable_entry": step,
                "seconds_to_stable_entry": step,
                "records_to_stable_entry": step,
                "bytes_to_stable_entry": step,
            }
            for index, step in enumerate(steps)
        ],
        "selection_eligible": False,
        "profile_status": "unclassified",
        "selection_exclusion_reasons": [],
        "failed_run_rate": failed_run_count / max(expected_run_count or count, 1),
    }


def _row(ranking: list[dict[str, object]], profile_name: str) -> dict[str, object]:
    return next(row for row in ranking if row["profile_name"] == profile_name)


def _config() -> AggressivenessCalibrationConfig:
    return AggressivenessCalibrationConfig(
        fingerprint_artifact=Path("/tmp/train.json"),
        held_out_fingerprint_artifact=Path("/tmp/held.json"),
        source_texts=Path("/tmp/source.jsonl"),
        output_dir=Path("/tmp/output"),
    )


def _rows(*, lineage_valid: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_name in PROFILE_NAMES:
        for seed in (0, 1, 2):
            rows.append(
                {
                    "profile_name": profile_name,
                    "seed": seed,
                    "lineage_valid": lineage_valid,
                }
            )
    return rows


def _publication_summaries() -> list[dict[str, object]]:
    return [
        _summary(name, rank=index, steps=(5.0, 5.0, 5.0))
        for index, name in enumerate(PROFILE_NAMES)
    ]


def _aggregate_validation_pass() -> dict[str, object]:
    return {"status": "pass"}


def _seed_check() -> dict[str, object]:
    return {
        "all_metrics_finite": True,
        "all_trajectories_ordered": True,
        "all_step_zero_points_present": True,
        "all_final_points_present": True,
        "all_resource_metrics_non_negative": True,
    }
