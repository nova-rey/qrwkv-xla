from __future__ import annotations

import json
from dataclasses import replace

import pytest

from qrwkv_xla.fingerprint.mode_plateau_controller import (
    ModePlateauConfig,
    ModeState,
    MultiModePlateauController,
)


def _config(**overrides: object) -> ModePlateauConfig:
    values = {
        "required_modes": ("fast",),
        "minimum_observations": 3,
        "progress_window_observations": 3,
        "plateau_patience_observations": 2,
        "regression_patience_observations": 2,
        "reactivation_cooldown_observations": 2,
        "plateau_absolute_improvement_threshold": 0.02,
        "plateau_relative_improvement_threshold": 0.02,
    }
    values.update(overrides)
    return ModePlateauConfig(**values)


def _metrics(
    loss: float,
    *,
    inside: float = 0.97,
    distance: float = 0.03,
    violation: float = 0.03,
) -> dict[str, float]:
    return {
        "corridor_loss": loss,
        "inside_corridor_rate": inside,
        "mean_distance_outside_corridor": distance,
        "worst_stat_violation": violation,
    }


def _feed(
    controller: MultiModePlateauController,
    losses: list[float],
    *,
    mode_id: str = "fast",
    start: int = 0,
) -> None:
    for offset, loss in enumerate(losses):
        controller.observe(step=start + offset, mode_id=mode_id, metrics=_metrics(loss))


def test_config_validation_rejects_invalid_control_parameters() -> None:
    base = _config()
    invalid_builders = [
        lambda: replace(base, required_modes=()),
        lambda: replace(base, progress_window_observations=0),
        lambda: replace(base, plateau_patience_observations=-1),
        lambda: replace(base, minimum_corridor_steps=10, maximum_corridor_steps=9),
        lambda: replace(base, smoothing_policy="ema"),
        lambda: replace(base, regression_inside_rate_floor=0.95),
        lambda: replace(base, regression_mean_distance_ceiling=0.05),
    ]
    for build in invalid_builders:
        with pytest.raises(ValueError):
            build()


def test_missing_metric_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing metric directions"):
        _config(metric_directions={"corridor_loss": "lower"})


def test_plateau_before_entry_does_not_freeze_and_leaving_resets_patience() -> None:
    controller = MultiModePlateauController(_config())
    for step in range(4):
        controller.observe(
            step=step,
            mode_id="fast",
            metrics=_metrics(1.0, inside=0.5, distance=0.2, violation=0.2),
        )
    assert controller.modes["fast"].state == ModeState.ACTIVE
    controller.observe(step=4, mode_id="fast", metrics=_metrics(1.0))
    assert controller.modes["fast"].state == ModeState.PLATEAU_CANDIDATE
    controller.observe(
        step=5,
        mode_id="fast",
        metrics=_metrics(1.0, inside=0.5, distance=0.2, violation=0.2),
    )
    state = controller.modes["fast"]
    assert state.state == ModeState.ACTIVE
    assert state.entry_condition_first_step == 4
    assert state.plateau_patience_count == 0


def test_strong_progress_resets_plateau_and_patience_must_be_consecutive() -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, [1.0, 0.5, 0.5, 0.5])
    assert controller.modes["fast"].state == ModeState.PLATEAU_CANDIDATE
    controller.observe(step=4, mode_id="fast", metrics=_metrics(0.2))
    assert controller.modes["fast"].state == ModeState.ACTIVE
    assert controller.modes["fast"].plateau_patience_count == 0
    _feed(controller, [0.2, 0.2, 0.2], start=5)
    assert controller.modes["fast"].state == ModeState.FROZEN


def test_lower_is_better_degradation_never_counts_as_plateau() -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, [1.0, 0.5, 0.6, 0.7, 0.8])
    state = controller.modes["fast"]
    assert state.state == ModeState.ACTIVE
    assert state.freeze_step is None
    assert state.plateau_patience_count == 0
    assert state.last_signed_improvement is not None
    assert state.last_signed_improvement < 0
    assert state.last_relative_improvement is not None
    assert state.last_relative_improvement < 0
    assert not state.plateau_observation
    assert not controller.global_cycle_one_complete


def test_higher_is_better_degradation_never_counts_as_plateau() -> None:
    directions = {**_config().metric_directions, "quality": "higher"}
    controller = MultiModePlateauController(
        _config(primary_progress_metric="quality", metric_directions=directions)
    )
    for step, quality in enumerate([0.2, 0.8, 0.7, 0.6, 0.5]):
        controller.observe(
            step=step,
            mode_id="fast",
            metrics={**_metrics(1.0), "quality": quality},
        )
    state = controller.modes["fast"]
    assert state.state == ModeState.ACTIVE
    assert state.freeze_step is None
    assert state.plateau_patience_count == 0
    assert state.last_signed_improvement is not None
    assert state.last_signed_improvement < 0
    assert not state.plateau_observation
    assert not controller.global_cycle_one_complete


def test_degradation_cancels_plateau_candidate_with_explicit_reason() -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, [0.5, 0.5, 0.5])
    state = controller.modes["fast"]
    assert state.state == ModeState.PLATEAU_CANDIDATE
    controller.observe(step=3, mode_id="fast", metrics=_metrics(0.6))
    assert state.state == ModeState.ACTIVE
    assert state.plateau_patience_count == 0
    assert state.plateau_candidate_first_step is None
    assert state.last_signed_improvement is not None
    assert state.last_signed_improvement < 0
    assert not state.plateau_observation
    assert controller.transitions[-1].reason == "primary_metric_degraded"


@pytest.mark.parametrize(
    "losses",
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.500, 0.499, 0.498, 0.497],
    ],
)
def test_zero_and_small_positive_progress_remain_plateau_evidence(
    losses: list[float],
) -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, losses)
    state = controller.modes["fast"]
    assert state.state == ModeState.FROZEN
    assert state.freeze_step == 3
    assert state.last_signed_improvement is not None
    assert state.last_signed_improvement >= 0
    assert state.plateau_observation


def test_minimum_step_rail_prevents_early_freeze() -> None:
    controller = MultiModePlateauController(_config(minimum_corridor_steps=8))
    _feed(controller, [1.0] * 8)
    assert controller.modes["fast"].state == ModeState.PLATEAU_CANDIDATE
    controller.observe(step=8, mode_id="fast", metrics=_metrics(1.0))
    assert controller.modes["fast"].state == ModeState.FROZEN


def test_modes_freeze_independently_and_optional_mode_does_not_block() -> None:
    config = _config(required_modes=("fast", "slow"), optional_modes=("optional",))
    controller = MultiModePlateauController(config)
    fast_losses = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    slow_losses = [1.0, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4]
    for step, (fast, slow) in enumerate(zip(fast_losses, slow_losses, strict=True)):
        controller.observe_all(
            step=step,
            observations={
                "fast": _metrics(fast),
                "slow": _metrics(slow),
                "optional": _metrics(1.0, inside=0.0, distance=1.0, violation=1.0),
            },
        )
    assert controller.modes["fast"].freeze_step < controller.modes["slow"].freeze_step
    assert controller.global_cycle_one_complete
    assert controller.modes["optional"].state == ModeState.ACTIVE


def test_frozen_mode_is_observable_and_outlier_recovers() -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, [1.0] * 4)
    state = controller.modes["fast"]
    assert state.state == ModeState.FROZEN
    assert not state.mode_should_train
    plateau_count = state.plateau_patience_count
    controller.observe(
        step=4,
        mode_id="fast",
        metrics=_metrics(1.0, inside=0.5, distance=0.2, violation=0.2),
    )
    assert state.state == ModeState.REACTIVATION_CANDIDATE
    controller.observe(step=5, mode_id="fast", metrics=_metrics(1.0))
    assert state.state == ModeState.FROZEN
    assert state.reactivation_count == 0
    assert state.plateau_patience_count == plateau_count


def test_persistent_regression_reactivates_and_cooldown_prevents_refreeze() -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, [1.0] * 4)
    bad = _metrics(1.0, inside=0.5, distance=0.2, violation=0.2)
    controller.observe(step=4, mode_id="fast", metrics=bad)
    controller.observe(step=5, mode_id="fast", metrics=bad)
    state = controller.modes["fast"]
    assert state.state == ModeState.ACTIVE
    assert state.reactivation_count == 1
    assert state.last_reactivation_step == 5
    controller.observe(step=6, mode_id="fast", metrics=_metrics(1.0))
    controller.observe(step=7, mode_id="fast", metrics=_metrics(1.0))
    assert state.state == ModeState.ACTIVE
    assert state.cooldown_observations_remaining == 0
    controller.observe(step=8, mode_id="fast", metrics=_metrics(1.0))
    controller.observe(step=9, mode_id="fast", metrics=_metrics(1.0))
    assert state.state == ModeState.FROZEN
    assert state.freeze_step == 9


def test_reactivation_candidate_and_failed_mode_block_completion() -> None:
    controller = MultiModePlateauController(_config())
    _feed(controller, [1.0] * 4)
    controller.observe(
        step=4,
        mode_id="fast",
        metrics=_metrics(1.0, inside=0.5, distance=0.2, violation=0.2),
    )
    assert not controller.global_cycle_one_complete
    assert controller.reactivation_candidate_mode_ids == ["fast"]

    failed = MultiModePlateauController(_config())
    with pytest.raises(ValueError, match="non-finite"):
        failed.observe(step=0, mode_id="fast", metrics=_metrics(float("nan")))
    assert failed.failed_mode_ids == ["fast"]
    assert not failed.global_cycle_one_complete


def test_maximum_step_cap_is_honest_in_report() -> None:
    controller = MultiModePlateauController(_config(maximum_corridor_steps=3))
    for step in range(4):
        controller.observe(
            step=step,
            mode_id="fast",
            metrics=_metrics(1.0, inside=0.0, distance=1.0, violation=1.0),
        )
    report = controller.report()
    assert report["status"] == "incomplete"
    assert report["global_completion_reason"] == "maximum_step_cap"
    assert report["all_required_modes_frozen"] is False


def test_observation_integrity_rejects_bad_batches() -> None:
    controller = MultiModePlateauController(_config(required_modes=("a", "b")))
    with pytest.raises(ValueError, match="missing required modes"):
        controller.observe_all(step=0, observations={"a": _metrics(1.0)})
    with pytest.raises(ValueError, match="unknown modes"):
        controller.observe_all(
            step=0,
            observations={"a": _metrics(1.0), "b": _metrics(1.0), "c": _metrics(1.0)},
        )
    controller.observe_all(
        step=0, observations={"a": _metrics(1.0), "b": _metrics(1.0)}
    )
    with pytest.raises(ValueError, match="strictly monotonic"):
        controller.observe_all(
            step=0, observations={"a": _metrics(1.0), "b": _metrics(1.0)}
        )
    with pytest.raises(ValueError, match="evaluation cadence"):
        controller.observe_all(
            step=2, observations={"a": _metrics(1.0), "b": _metrics(1.0)}
        )


def test_missing_entry_metric_is_rejected() -> None:
    controller = MultiModePlateauController(_config())
    metrics = _metrics(1.0)
    del metrics["worst_stat_violation"]
    with pytest.raises(ValueError, match="missing required metrics"):
        controller.observe(step=0, mode_id="fast", metrics=metrics)


def test_rolling_smoothing_and_receipts_are_deterministic(tmp_path) -> None:
    config = _config(smoothing_window_observations=2)
    left = MultiModePlateauController(config)
    right = MultiModePlateauController(config)
    losses = [1.0, 0.6, 0.5, 0.5, 0.5, 0.5]
    _feed(left, losses)
    _feed(right, losses)
    assert left.to_dict() == right.to_dict()
    assert left.report() == right.report()
    left.write_receipts(tmp_path)
    assert (
        json.loads((tmp_path / "mode_plateau_controller_report.json").read_text())
        == left.report()
    )
    assert (tmp_path / "mode_plateau_transitions.jsonl").read_text().strip()


def test_resume_round_trip_matches_uninterrupted_execution(tmp_path) -> None:
    config = _config()
    uninterrupted = MultiModePlateauController(config)
    _feed(uninterrupted, [1.0] * 6)

    resumed = MultiModePlateauController(config)
    _feed(resumed, [1.0] * 3)
    resumed.write_receipts(tmp_path)
    resumed = MultiModePlateauController.load(
        tmp_path / "mode_plateau_controller_state.json"
    )
    _feed(resumed, [1.0] * 3, start=3)
    assert resumed.to_dict() == uninterrupted.to_dict()
    assert resumed.report() == uninterrupted.report()


def test_resume_immediately_before_degradation_matches_uninterrupted(tmp_path) -> None:
    config = _config()
    uninterrupted = MultiModePlateauController(config)
    _feed(uninterrupted, [0.5, 0.5, 0.5, 0.6])

    resumed = MultiModePlateauController(config)
    _feed(resumed, [0.5, 0.5, 0.5])
    resumed.write_receipts(tmp_path)
    resumed = MultiModePlateauController.load(
        tmp_path / "mode_plateau_controller_state.json"
    )
    resumed.observe(step=3, mode_id="fast", metrics=_metrics(0.6))

    assert resumed.to_dict() == uninterrupted.to_dict()
    assert resumed.report() == uninterrupted.report()
    assert resumed.transitions[-1].reason == "primary_metric_degraded"
