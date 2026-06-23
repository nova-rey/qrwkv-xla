from __future__ import annotations

import math

import pytest

from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorScheduler,
    AdaptiveCorridorSchedulerConfig,
    adaptive_weighted_loss,
    normalized_active_mode_weights,
)
from qrwkv_xla.fingerprint.mode_plateau_controller import ModePlateauConfig, ModeState

MODES = ("0", "1", "2")


def _config(**controller_overrides: object) -> AdaptiveCorridorSchedulerConfig:
    values = {
        "required_modes": MODES,
        "minimum_observations": 3,
        "progress_window_observations": 3,
        "plateau_patience_observations": 2,
        "regression_patience_observations": 2,
        "reactivation_cooldown_observations": 1,
        "plateau_absolute_improvement_threshold": 0.02,
        "plateau_relative_improvement_threshold": 0.02,
        "maximum_corridor_steps": 100,
    }
    values.update(controller_overrides)
    return AdaptiveCorridorSchedulerConfig(
        controller=ModePlateauConfig(**values),
        mode_weights={mode_id: 1.0 for mode_id in MODES},
        global_freeze_confirmation_observations=2,
    )


def _metric(
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


def _observe(
    scheduler: AdaptiveCorridorScheduler,
    step: int,
    losses: tuple[float, float, float],
    *,
    overrides: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    observations = {
        mode_id: _metric(loss) for mode_id, loss in zip(MODES, losses, strict=True)
    }
    observations.update(overrides or {})
    return scheduler.observe_calibration(step=step, observations=observations)


def test_active_weights_renormalize_and_frozen_mode_has_zero_weight() -> None:
    weights = normalized_active_mode_weights(
        {"a": True, "b": False, "c": True},
        {"a": 1.0, "b": 3.0, "c": 1.0},
    )
    assert weights == {"a": 0.5, "b": 0.0, "c": 0.5}
    assert math.isclose(sum(weights.values()), 1.0)
    assert adaptive_weighted_loss({"a": 2.0, "c": 4.0}, weights) == 3.0


def test_mask_rejects_zero_sum_unknown_and_missing_modes() -> None:
    with pytest.raises(ValueError, match="weight sum"):
        normalized_active_mode_weights({"a": False}, {"a": 1.0})
    with pytest.raises(ValueError, match="mask mismatch"):
        normalized_active_mode_weights({"a": True, "b": True}, {"a": 1.0})
    with pytest.raises(ValueError, match="exactly match"):
        adaptive_weighted_loss({"a": 1.0}, {"a": 0.5, "b": 0.5})


def test_freeze_changes_next_step_mask_and_records_event() -> None:
    scheduler = AdaptiveCorridorScheduler(_config())
    _observe(scheduler, 0, (1.0, 1.0, 1.0))
    _observe(scheduler, 1, (0.5, 0.8, 0.8))
    _observe(scheduler, 2, (0.5, 0.6, 0.6))
    _observe(scheduler, 3, (0.5, 0.4, 0.4))
    events = _observe(scheduler, 4, (0.5, 0.2, 0.2))
    assert scheduler.controller.modes["0"].state == ModeState.FROZEN
    assert scheduler.normalized_weights["0"] == 0.0
    assert math.isclose(scheduler.normalized_weights["1"], 0.5)
    assert events[0]["event"] == "mode_frozen"
    assert events[0]["active_modes_before"] == 3
    assert events[0]["active_modes_after"] == 2


def test_frozen_mode_remains_observed_and_reactivation_restores_weight() -> None:
    scheduler = AdaptiveCorridorScheduler(_config())
    for step, losses in enumerate(
        [(1.0, 1.0, 1.0), (0.5, 0.8, 0.8), (0.5, 0.6, 0.6), (0.5, 0.4, 0.4)]
    ):
        _observe(scheduler, step, losses)
    _observe(scheduler, 4, (0.5, 0.2, 0.2))
    bad = _metric(0.5, inside=0.5, distance=0.2, violation=0.2)
    _observe(scheduler, 5, (0.5, 0.1, 0.1), overrides={"0": bad})
    events = _observe(scheduler, 6, (0.5, 0.05, 0.05), overrides={"0": bad})
    assert scheduler.controller.modes["0"].state == ModeState.ACTIVE
    assert scheduler.normalized_weights["0"] > 0
    assert any(event["event"] == "mode_reactivated" for event in events)
    observed = [
        row for row in scheduler.calibration_trajectory if row["mode_id"] == "0"
    ]
    assert len(observed) == 7


def test_training_metrics_do_not_drive_controller_and_accounting_is_exact() -> None:
    scheduler = AdaptiveCorridorScheduler(_config())
    weights = scheduler.record_optimizer_step(1)
    assert weights == {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3}
    assert all(
        state.state == ModeState.WARMUP for state in scheduler.controller.modes.values()
    )
    assert scheduler.full_mode_step_equivalents == 3
    assert scheduler.actual_active_mode_step_equivalents == 3
    assert scheduler.frozen_mode_step_equivalents_saved == 0


def test_frozen_interval_resource_accounting_across_reactivation() -> None:
    scheduler = AdaptiveCorridorScheduler(_config())
    for step in range(4):
        _observe(scheduler, step, (1.0, 1.0 - step * 0.2, 1.0 - step * 0.2))
    assert scheduler.controller.modes["0"].state == ModeState.FROZEN
    scheduler.record_optimizer_step(4)
    assert scheduler.actual_active_mode_step_equivalents == 2
    bad = _metric(1.0, inside=0.5, distance=0.2, violation=0.2)
    _observe(scheduler, 4, (1.0, 0.2, 0.2), overrides={"0": bad})
    _observe(scheduler, 5, (1.0, 0.1, 0.1), overrides={"0": bad})
    scheduler.record_optimizer_step(6)
    assert scheduler.actual_active_mode_step_equivalents == 5
    assert scheduler.full_mode_step_equivalents == 6
    assert scheduler.frozen_mode_step_equivalents_saved == 1
    assert scheduler.accounting["0"].training_steps_while_frozen == 1
    assert scheduler.accounting["0"].direct_loss_contribution_steps == 1


def test_global_completion_waits_for_confirmation_and_resets_on_regression() -> None:
    scheduler = AdaptiveCorridorScheduler(_config())
    for step in range(4):
        _observe(scheduler, step, (1.0, 1.0, 1.0))
    assert scheduler.controller.all_required_modes_frozen
    assert scheduler.global_freeze_confirmation_count == 1
    assert not scheduler.cycle_one_complete
    bad = _metric(1.0, inside=0.5, distance=0.2, violation=0.2)
    _observe(scheduler, 4, (1.0, 1.0, 1.0), overrides={"0": bad})
    assert scheduler.global_freeze_confirmation_count == 0
    assert not scheduler.cycle_one_complete


def test_stable_all_frozen_window_completes_and_optional_mode_does_not_block() -> None:
    config = AdaptiveCorridorSchedulerConfig(
        controller=ModePlateauConfig(
            required_modes=("0", "1"),
            optional_modes=("2",),
            minimum_observations=3,
            progress_window_observations=3,
            plateau_patience_observations=2,
        ),
        mode_weights={mode_id: 1.0 for mode_id in MODES},
        global_freeze_confirmation_observations=2,
    )
    scheduler = AdaptiveCorridorScheduler(config)
    unfinished = _metric(1.0, inside=0.0, distance=1.0, violation=1.0)
    for step in range(4):
        scheduler.observe_calibration(
            step=step,
            observations={"0": _metric(1.0), "1": _metric(1.0), "2": unfinished},
        )
    scheduler.observe_calibration(
        step=4,
        observations={"0": _metric(1.0), "1": _metric(1.0), "2": unfinished},
    )
    assert scheduler.cycle_one_complete
    assert scheduler.global_completion_step == 4
    assert scheduler.controller.modes["2"].state == ModeState.ACTIVE


def test_scheduler_round_trip_preserves_confirmation_and_trajectories() -> None:
    scheduler = AdaptiveCorridorScheduler(_config())
    for step in range(4):
        _observe(scheduler, step, (1.0, 1.0, 1.0))
    restored = AdaptiveCorridorScheduler.from_dict(scheduler.to_dict())
    assert restored.to_dict() == scheduler.to_dict()
    _observe(restored, 4, (1.0, 1.0, 1.0))
    _observe(scheduler, 4, (1.0, 1.0, 1.0))
    assert restored.to_dict() == scheduler.to_dict()


def test_reactivation_caps_fail_closed() -> None:
    base = _config()
    config = AdaptiveCorridorSchedulerConfig(
        controller=base.controller,
        mode_weights=base.mode_weights,
        global_freeze_confirmation_observations=2,
        maximum_reactivations_per_mode=0,
        maximum_total_reactivations=0,
    )
    scheduler = AdaptiveCorridorScheduler(config)
    for step in range(4):
        _observe(scheduler, step, (1.0, 0.8 - 0.1 * step, 0.8 - 0.1 * step))
    bad = _metric(1.0, inside=0.5, distance=0.2, violation=0.2)
    _observe(scheduler, 4, (1.0, 0.3, 0.3), overrides={"0": bad})
    with pytest.raises(ValueError, match="reactivations"):
        _observe(scheduler, 5, (1.0, 0.2, 0.2), overrides={"0": bad})
