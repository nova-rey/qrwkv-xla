from __future__ import annotations

from qrwkv_xla.fingerprint.aggressiveness_calibration import (
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
    assert [profile.aggressiveness_rank for profile in profiles] == list(range(4))
    for profile in profiles:
        assert set(profile.per_stat_weights) == {
            "entropy",
            "top1_margin",
            "top8_mass",
            "top32_mass",
            "tail_mass",
        }
        assert profile.stability_abort_enabled
        assert not profile.adaptive_weighting_enabled


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


def test_selection_prefers_reliability_then_smaller_hammer() -> None:
    base = {
        "completed_run_rate": 1.0,
        "stable_entry_success_rate": 1.0,
        "abort_rate": 0.0,
        "mean_entry_then_exit_count": 0.0,
        "mean_distance_rebound": 0.0,
    }
    selected, ranking = select_profile(
        [
            {
                **base,
                "profile_name": "rock_hammer",
                "aggressiveness_rank": 0,
                "median_steps_to_stable_entry": 5.0,
            },
            {
                **base,
                "profile_name": "ball_peen",
                "aggressiveness_rank": 1,
                "median_steps_to_stable_entry": 5.0,
            },
            {
                **base,
                "profile_name": "sledgehammer",
                "aggressiveness_rank": 2,
                "median_steps_to_stable_entry": 1.0,
                "abort_rate": 1.0,
            },
        ]
    )
    assert selected == "rock_hammer"
    assert ranking[0]["selection_status"] == "selected"
