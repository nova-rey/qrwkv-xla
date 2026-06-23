from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from qrwkv_xla.fingerprint.mode_plateau_controller import (
    ModePlateauConfig,
    ModeState,
    MultiModePlateauController,
)


@dataclass(frozen=True)
class AdaptiveCorridorSchedulerConfig:
    controller: ModePlateauConfig
    mode_weights: Mapping[str, float]
    global_freeze_confirmation_observations: int = 2
    maximum_confirmation_only_evaluations: int = 8
    maximum_reactivations_per_mode: int = 3
    maximum_total_reactivations: int = 10

    def __post_init__(self) -> None:
        if self.global_freeze_confirmation_observations <= 0:
            raise ValueError("global_freeze_confirmation_observations must be > 0")
        if self.maximum_confirmation_only_evaluations <= 0:
            raise ValueError("maximum_confirmation_only_evaluations must be > 0")
        if (
            self.maximum_confirmation_only_evaluations
            < self.global_freeze_confirmation_observations
        ):
            raise ValueError(
                "maximum_confirmation_only_evaluations must cover confirmation window"
            )
        if self.maximum_reactivations_per_mode < 0:
            raise ValueError("maximum_reactivations_per_mode must be >= 0")
        if self.maximum_total_reactivations < 0:
            raise ValueError("maximum_total_reactivations must be >= 0")
        expected = set(self.controller.all_modes)
        actual = set(self.mode_weights)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"mode weight mismatch: missing={missing} unknown={unknown}"
            )
        if any(
            not math.isfinite(float(weight)) or float(weight) <= 0
            for weight in self.mode_weights.values()
        ):
            raise ValueError("configured mode weights must be finite and > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller.to_dict(),
            "mode_weights": {
                mode_id: float(weight)
                for mode_id, weight in sorted(self.mode_weights.items())
            },
            "global_freeze_confirmation_observations": (
                self.global_freeze_confirmation_observations
            ),
            "maximum_confirmation_only_evaluations": (
                self.maximum_confirmation_only_evaluations
            ),
            "maximum_reactivations_per_mode": self.maximum_reactivations_per_mode,
            "maximum_total_reactivations": self.maximum_total_reactivations,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptiveCorridorSchedulerConfig:
        data = dict(value)
        data["controller"] = ModePlateauConfig.from_dict(data["controller"])
        return cls(**data)


@dataclass
class AdaptiveModeAccounting:
    training_steps_while_active: int = 0
    training_steps_while_frozen: int = 0
    direct_loss_contribution_steps: int = 0
    reactivation_steps: list[int] = field(default_factory=list)
    refreeze_steps: list[int] = field(default_factory=list)


def normalized_active_mode_weights(
    mode_should_train: Mapping[str, bool],
    configured_weights: Mapping[str, float],
) -> dict[str, float]:
    if set(mode_should_train) != set(configured_weights):
        missing = sorted(set(configured_weights) - set(mode_should_train))
        unknown = sorted(set(mode_should_train) - set(configured_weights))
        raise ValueError(f"mode mask mismatch: missing={missing} unknown={unknown}")
    masked = {
        mode_id: float(configured_weights[mode_id]) if train else 0.0
        for mode_id, train in mode_should_train.items()
    }
    total = sum(masked.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("active mode weight sum must be finite and > 0")
    result = {mode_id: weight / total for mode_id, weight in masked.items()}
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("normalized active mode weights must sum to 1")
    return dict(sorted(result.items()))


def adaptive_weighted_loss(
    per_mode_losses: Mapping[str, Any], normalized_weights: Mapping[str, float]
) -> Any:
    if set(per_mode_losses) != {
        mode_id for mode_id, weight in normalized_weights.items() if weight > 0
    }:
        raise ValueError("per-mode losses must exactly match positive mode weights")
    result: Any = 0.0
    for mode_id, loss in per_mode_losses.items():
        result = result + normalized_weights[mode_id] * loss
    return result


class AdaptiveCorridorScheduler:
    def __init__(self, config: AdaptiveCorridorSchedulerConfig) -> None:
        self.config = config
        self.config_sha256 = _stable_hash(config.to_dict())
        self.controller = MultiModePlateauController(config.controller)
        self.global_freeze_confirmation_count = 0
        self.confirmation_phase_active = False
        self.confirmation_only_evaluations_completed = 0
        self.global_completion_step: int | None = None
        self.optimizer_steps_completed = 0
        self.calibration_evaluations_completed = 0
        self.full_mode_step_equivalents = 0
        self.actual_active_mode_step_equivalents = 0
        self.maximum_active_mode_count = 0
        self.minimum_active_mode_count: int | None = None
        self.accounting = {
            mode_id: AdaptiveModeAccounting()
            for mode_id in self.config.controller.all_modes
        }
        self.calibration_trajectory: list[dict[str, Any]] = []
        self.weight_trajectory: list[dict[str, Any]] = []
        self.transition_events: list[dict[str, Any]] = []

    @property
    def mode_should_train(self) -> dict[str, bool]:
        return {
            mode_id: state.mode_should_train
            for mode_id, state in sorted(self.controller.modes.items())
        }

    @property
    def active_mode_ids(self) -> list[str]:
        return [mode_id for mode_id, train in self.mode_should_train.items() if train]

    @property
    def normalized_weights(self) -> dict[str, float]:
        if not self.active_mode_ids:
            return {mode_id: 0.0 for mode_id in self.config.controller.all_modes}
        return normalized_active_mode_weights(
            self.mode_should_train, self.config.mode_weights
        )

    @property
    def cycle_one_complete(self) -> bool:
        return self.global_completion_step is not None

    @property
    def total_reactivations(self) -> int:
        return sum(state.reactivation_count for state in self.controller.modes.values())

    @property
    def frozen_mode_step_equivalents_saved(self) -> int:
        return (
            self.full_mode_step_equivalents - self.actual_active_mode_step_equivalents
        )

    @property
    def fraction_mode_work_saved(self) -> float:
        if self.full_mode_step_equivalents == 0:
            return 0.0
        return self.frozen_mode_step_equivalents_saved / self.full_mode_step_equivalents

    def record_optimizer_step(self, step: int) -> dict[str, float]:
        if self.cycle_one_complete:
            raise ValueError("cannot train after Cycle 1 completion")
        weights = self.normalized_weights
        active = {mode_id for mode_id, weight in weights.items() if weight > 0}
        if not active:
            raise ValueError("cannot run optimizer step with no active modes")
        required_active = active & set(self.config.controller.required_modes)
        required_count = len(self.config.controller.required_modes)
        self.optimizer_steps_completed += 1
        self.full_mode_step_equivalents += required_count
        self.actual_active_mode_step_equivalents += len(required_active)
        self.maximum_active_mode_count = max(
            self.maximum_active_mode_count, len(active)
        )
        self.minimum_active_mode_count = (
            len(active)
            if self.minimum_active_mode_count is None
            else min(self.minimum_active_mode_count, len(active))
        )
        for mode_id, mode_accounting in self.accounting.items():
            if mode_id in active:
                mode_accounting.training_steps_while_active += 1
                mode_accounting.direct_loss_contribution_steps += 1
            else:
                mode_accounting.training_steps_while_frozen += 1
        self.weight_trajectory.append(
            {
                "step": step,
                "active_mode_ids": sorted(active),
                "normalized_active_weights": weights,
                "shared_parameter_update": True,
                "frozen_modes_have_zero_direct_loss_only": True,
            }
        )
        return weights

    def observe_calibration(
        self,
        *,
        step: int,
        observations: Mapping[str, Mapping[str, float]],
        confirmation_only: bool = False,
    ) -> list[dict[str, Any]]:
        before_states = {
            mode_id: state.state.value
            for mode_id, state in self.controller.modes.items()
        }
        before_weights = self.normalized_weights
        event_mask = {mode_id: weight > 0 for mode_id, weight in before_weights.items()}
        transition_start = len(self.controller.transitions)
        self.controller.observe_all(step=step, observations=observations)
        self.calibration_evaluations_completed += 1
        if confirmation_only:
            self.confirmation_only_evaluations_completed += 1
        after_states = {
            mode_id: state.state.value
            for mode_id, state in self.controller.modes.items()
        }
        after_weights = self.normalized_weights
        for mode_id in sorted(observations):
            state = self.controller.modes[mode_id]
            self.calibration_trajectory.append(
                {
                    "step": step,
                    "mode_id": mode_id,
                    "controller_state_before": before_states[mode_id],
                    "metrics": {
                        name: float(value)
                        for name, value in sorted(observations[mode_id].items())
                    },
                    "signed_improvement": state.last_signed_improvement,
                    "plateau_observation": state.plateau_observation,
                    "controller_state_after": after_states[mode_id],
                    "mode_should_train_after": state.mode_should_train,
                }
            )
        events: list[dict[str, Any]] = []
        for transition in self.controller.transitions[transition_start:]:
            if transition.to_state == ModeState.FROZEN.value:
                event_weights_before = _weights_for_mask(
                    event_mask, self.config.mode_weights
                )
                active_modes_before = sum(event_mask.values())
                event_mask[transition.mode_id] = False
                event_weights_after = _weights_for_mask(
                    event_mask, self.config.mode_weights
                )
                accounting = self.accounting[transition.mode_id]
                if accounting.reactivation_steps:
                    accounting.refreeze_steps.append(step)
                event = {
                    "step": step,
                    "mode_id": transition.mode_id,
                    "event": "mode_frozen",
                    "reason": transition.reason,
                    "active_modes_before": active_modes_before,
                    "active_modes_after": sum(event_mask.values()),
                    "mode_weight_before": event_weights_before[transition.mode_id],
                    "mode_weight_after": event_weights_after[transition.mode_id],
                    "normalized_weights_after": event_weights_after,
                }
                events.append(event)
            elif transition.to_state == ModeState.ACTIVE.value and (
                transition.from_state == ModeState.REACTIVATION_CANDIDATE.value
            ):
                event_weights_before = _weights_for_mask(
                    event_mask, self.config.mode_weights
                )
                active_modes_before = sum(event_mask.values())
                event_mask[transition.mode_id] = True
                event_weights_after = _weights_for_mask(
                    event_mask, self.config.mode_weights
                )
                self.accounting[transition.mode_id].reactivation_steps.append(step)
                event = {
                    "step": step,
                    "mode_id": transition.mode_id,
                    "event": "mode_reactivated",
                    "reason": self.controller.modes[
                        transition.mode_id
                    ].reactivation_reason
                    or transition.reason,
                    "active_modes_before": active_modes_before,
                    "active_modes_after": sum(event_mask.values()),
                    "mode_weight_before": event_weights_before[transition.mode_id],
                    "mode_weight_after": event_weights_after[transition.mode_id],
                    "normalized_weights_after": event_weights_after,
                }
                events.append(event)
        if _weights_for_mask(event_mask, self.config.mode_weights) != after_weights:
            raise ValueError("transition events and final scheduler mask disagree")
        self.transition_events.extend(events)
        self._validate_caps()
        if self.controller.failed_mode_ids:
            raise ValueError("required controller mode failed closed")
        reactivated = any(event["event"] == "mode_reactivated" for event in events)
        if self.controller.all_required_modes_frozen:
            self.confirmation_phase_active = True
            self.global_freeze_confirmation_count = (
                self.global_freeze_confirmation_count + 1 if confirmation_only else 0
            )
            if (
                self.global_freeze_confirmation_count
                >= self.config.global_freeze_confirmation_observations
            ):
                self.global_completion_step = step
                self.confirmation_phase_active = False
        else:
            self.global_freeze_confirmation_count = 0
            self.global_completion_step = None
            if reactivated or not confirmation_only:
                self.confirmation_phase_active = False
        self._validate_mask_state_agreement()
        return events

    def _validate_caps(self) -> None:
        if self.total_reactivations > self.config.maximum_total_reactivations:
            raise ValueError("maximum total reactivations exceeded")
        for mode_id, state in self.controller.modes.items():
            if state.reactivation_count > self.config.maximum_reactivations_per_mode:
                raise ValueError(f"maximum reactivations exceeded for {mode_id}")

    def _validate_mask_state_agreement(self) -> None:
        for mode_id, state in self.controller.modes.items():
            expected = state.state in {
                ModeState.WARMUP,
                ModeState.ACTIVE,
                ModeState.PLATEAU_CANDIDATE,
            }
            if self.mode_should_train[mode_id] != expected:
                raise ValueError(
                    f"controller and scheduler mask disagree for {mode_id}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "P156.3.1",
            "scheduler_config": self.config.to_dict(),
            "scheduler_config_sha256": self.config_sha256,
            "controller_state": self.controller.to_dict(),
            "global_freeze_confirmation_count": self.global_freeze_confirmation_count,
            "confirmation_phase_active": self.confirmation_phase_active,
            "confirmation_only_evaluations_completed": (
                self.confirmation_only_evaluations_completed
            ),
            "global_completion_step": self.global_completion_step,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "calibration_evaluations_completed": self.calibration_evaluations_completed,
            "full_mode_step_equivalents": self.full_mode_step_equivalents,
            "actual_active_mode_step_equivalents": (
                self.actual_active_mode_step_equivalents
            ),
            "maximum_active_mode_count": self.maximum_active_mode_count,
            "minimum_active_mode_count": self.minimum_active_mode_count,
            "accounting": {
                mode_id: asdict(value)
                for mode_id, value in sorted(self.accounting.items())
            },
            "calibration_trajectory": self.calibration_trajectory,
            "weight_trajectory": self.weight_trajectory,
            "transition_events": self.transition_events,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptiveCorridorScheduler:
        config = AdaptiveCorridorSchedulerConfig.from_dict(value["scheduler_config"])
        scheduler = cls(config)
        if scheduler.config_sha256 != value["scheduler_config_sha256"]:
            raise ValueError("scheduler config hash mismatch")
        scheduler.controller = MultiModePlateauController.from_dict(
            value["controller_state"]
        )
        scheduler.global_freeze_confirmation_count = value[
            "global_freeze_confirmation_count"
        ]
        scheduler.confirmation_phase_active = value.get(
            "confirmation_phase_active", False
        )
        scheduler.confirmation_only_evaluations_completed = value.get(
            "confirmation_only_evaluations_completed", 0
        )
        scheduler.global_completion_step = value["global_completion_step"]
        scheduler.optimizer_steps_completed = value["optimizer_steps_completed"]
        scheduler.calibration_evaluations_completed = value[
            "calibration_evaluations_completed"
        ]
        scheduler.full_mode_step_equivalents = value["full_mode_step_equivalents"]
        scheduler.actual_active_mode_step_equivalents = value[
            "actual_active_mode_step_equivalents"
        ]
        scheduler.maximum_active_mode_count = value["maximum_active_mode_count"]
        scheduler.minimum_active_mode_count = value["minimum_active_mode_count"]
        scheduler.accounting = {
            mode_id: AdaptiveModeAccounting(**accounting)
            for mode_id, accounting in value["accounting"].items()
        }
        scheduler.calibration_trajectory = list(value["calibration_trajectory"])
        scheduler.weight_trajectory = list(value["weight_trajectory"])
        scheduler.transition_events = list(value["transition_events"])
        scheduler._validate_mask_state_agreement()
        return scheduler


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _weights_for_mask(
    mode_should_train: Mapping[str, bool], configured_weights: Mapping[str, float]
) -> dict[str, float]:
    if not any(mode_should_train.values()):
        return {mode_id: 0.0 for mode_id in sorted(mode_should_train)}
    return normalized_active_mode_weights(mode_should_train, configured_weights)
