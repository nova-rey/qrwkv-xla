from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ModeState(StrEnum):
    WARMUP = "WARMUP"
    ACTIVE = "ACTIVE"
    PLATEAU_CANDIDATE = "PLATEAU_CANDIDATE"
    FROZEN = "FROZEN"
    REACTIVATION_CANDIDATE = "REACTIVATION_CANDIDATE"
    FAILED = "FAILED"


DEFAULT_DIRECTIONS = {
    "corridor_loss": "lower",
    "inside_corridor_rate": "higher",
    "mean_distance_outside_corridor": "lower",
    "worst_stat_violation": "lower",
}


@dataclass(frozen=True)
class ModePlateauConfig:
    required_modes: tuple[str, ...]
    optional_modes: tuple[str, ...] = ()
    primary_progress_metric: str = "corridor_loss"
    entry_metrics: tuple[str, ...] = (
        "inside_corridor_rate",
        "mean_distance_outside_corridor",
        "worst_stat_violation",
    )
    regression_metrics: tuple[str, ...] = (
        "inside_corridor_rate",
        "mean_distance_outside_corridor",
        "worst_stat_violation",
    )
    metric_directions: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DIRECTIONS)
    )
    entry_inside_rate_threshold: float = 0.95
    entry_mean_distance_threshold: float = 0.05
    entry_worst_violation_threshold: float = 0.05
    evaluation_interval_steps: int = 1
    minimum_observations: int = 3
    progress_window_observations: int = 3
    plateau_relative_improvement_threshold: float = 0.01
    plateau_absolute_improvement_threshold: float = 0.01
    plateau_patience_observations: int = 2
    smoothing_policy: str = "rolling_mean"
    smoothing_window_observations: int = 1
    regression_inside_rate_floor: float = 0.90
    regression_mean_distance_ceiling: float = 0.08
    regression_worst_violation_ceiling: float = 0.08
    regression_patience_observations: int = 2
    reactivation_cooldown_observations: int = 2
    minimum_corridor_steps: int = 0
    maximum_corridor_steps: int = 10_000

    def __post_init__(self) -> None:
        _validate_config(self)

    @property
    def all_modes(self) -> tuple[str, ...]:
        return self.required_modes + self.optional_modes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metric_directions"] = dict(sorted(self.metric_directions.items()))
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModePlateauConfig:
        data = dict(value)
        for name in (
            "required_modes",
            "optional_modes",
            "entry_metrics",
            "regression_metrics",
        ):
            data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class ModeObservation:
    step: int
    mode_id: str
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ModeTransition:
    step: int
    mode_id: str
    from_state: str
    to_state: str
    reason: str


@dataclass
class ModeControllerState:
    mode_id: str
    required: bool
    state: ModeState = ModeState.WARMUP
    observation_count: int = 0
    last_observation_step: int | None = None
    entry_condition_reached: bool = False
    entry_condition_first_step: int | None = None
    entry_condition_currently_satisfied: bool = False
    plateau_candidate_first_step: int | None = None
    plateau_patience_count: int = 0
    plateau_patience_satisfied: bool = False
    freeze_step: int | None = None
    freeze_observation_index: int | None = None
    freeze_metric_snapshot: dict[str, float] | None = None
    regression_candidate_first_step: int | None = None
    regression_patience_count: int = 0
    reactivation_count: int = 0
    last_reactivation_step: int | None = None
    reactivation_reason: str | None = None
    cooldown_observations_remaining: int = 0
    final_metric_snapshot: dict[str, float] = field(default_factory=dict)
    metric_history: dict[str, list[float]] = field(default_factory=dict)
    last_signed_improvement: float | None = None
    last_relative_improvement: float | None = None
    plateau_observation: bool = False
    transition_count: int = 0

    @property
    def mode_should_train(self) -> bool:
        return self.state not in {
            ModeState.FROZEN,
            ModeState.REACTIVATION_CANDIDATE,
            ModeState.FAILED,
        }


class MultiModePlateauController:
    def __init__(self, config: ModePlateauConfig) -> None:
        self.config = config
        self.config_sha256 = _stable_hash(config.to_dict())
        required = set(config.required_modes)
        self.modes = {
            mode_id: ModeControllerState(mode_id, mode_id in required)
            for mode_id in config.all_modes
        }
        self.transitions: list[ModeTransition] = []
        self.current_step: int | None = None
        self.global_completion_step: int | None = None

    def observe(
        self, *, step: int, mode_id: str, metrics: Mapping[str, float]
    ) -> ModeControllerState:
        if mode_id not in self.modes:
            raise ValueError(f"unknown mode: {mode_id}")
        state = self.modes[mode_id]
        self._validate_observation(step, state, metrics)
        self._apply_observation(step, state, metrics)
        self.current_step = (
            step if self.current_step is None else max(step, self.current_step)
        )
        self._update_global_completion(step)
        return state

    def observe_all(
        self, *, step: int, observations: Mapping[str, Mapping[str, float]]
    ) -> dict[str, ModeControllerState]:
        unknown = sorted(set(observations) - set(self.modes))
        missing = sorted(set(self.config.required_modes) - set(observations))
        if unknown:
            raise ValueError(f"unknown modes: {unknown}")
        if missing:
            raise ValueError(f"missing required modes: {missing}")
        if self.current_step is not None:
            if step <= self.current_step:
                raise ValueError("batch steps must be strictly monotonic")
            if step - self.current_step != self.config.evaluation_interval_steps:
                raise ValueError("batch step does not match evaluation cadence")
        for mode_id, metrics in observations.items():
            self._validate_observation(step, self.modes[mode_id], metrics)
        for mode_id, metrics in observations.items():
            self._apply_observation(step, self.modes[mode_id], metrics)
        self.current_step = step
        self._update_global_completion(step)
        return {mode_id: self.modes[mode_id] for mode_id in observations}

    def _validate_observation(
        self,
        step: int,
        state: ModeControllerState,
        metrics: Mapping[str, float],
    ) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        if state.last_observation_step is not None:
            if step <= state.last_observation_step:
                raise ValueError(f"duplicate or non-monotonic step for {state.mode_id}")
            if (
                step - state.last_observation_step
                != self.config.evaluation_interval_steps
            ):
                raise ValueError("observation step does not match evaluation cadence")
        required_metrics = set(self.config.entry_metrics)
        required_metrics.update(self.config.regression_metrics)
        required_metrics.add(self.config.primary_progress_metric)
        missing = sorted(required_metrics - set(metrics))
        if missing:
            raise ValueError(f"missing required metrics for {state.mode_id}: {missing}")
        non_finite = [
            name for name in required_metrics if not math.isfinite(float(metrics[name]))
        ]
        if non_finite:
            self._transition(state, step, ModeState.FAILED, "non_finite_metrics")
            raise ValueError(f"non-finite metrics for {state.mode_id}: {non_finite}")

    def _apply_observation(
        self,
        step: int,
        state: ModeControllerState,
        metrics: Mapping[str, float],
    ) -> None:
        if state.state == ModeState.FAILED:
            raise ValueError(f"failed mode cannot be observed: {state.mode_id}")
        snapshot = {name: float(value) for name, value in sorted(metrics.items())}
        state.observation_count += 1
        state.last_observation_step = step
        state.final_metric_snapshot = snapshot
        for name, value in snapshot.items():
            state.metric_history.setdefault(name, []).append(value)

        if state.state in {ModeState.FROZEN, ModeState.REACTIVATION_CANDIDATE}:
            self._evaluate_regression(step, state, snapshot)
            return

        entry = self._entry_satisfied(snapshot)
        state.entry_condition_currently_satisfied = entry
        if entry and not state.entry_condition_reached:
            state.entry_condition_reached = True
            state.entry_condition_first_step = step

        if state.cooldown_observations_remaining > 0:
            state.cooldown_observations_remaining -= 1
            self._reset_plateau(state)
            self._transition(state, step, ModeState.ACTIVE, "reactivation_cooldown")
            return

        if state.observation_count < self.config.minimum_observations:
            self._transition(state, step, ModeState.WARMUP, "minimum_observations")
            return
        if not entry:
            self._reset_plateau(state)
            self._transition(state, step, ModeState.ACTIVE, "entry_condition_not_met")
            return
        plateau, degraded = self._plateau_status(state)
        state.plateau_observation = plateau
        if not plateau:
            self._reset_plateau(state)
            reason = "primary_metric_degraded" if degraded else "meaningful_progress"
            self._transition(state, step, ModeState.ACTIVE, reason)
            return

        if state.state != ModeState.PLATEAU_CANDIDATE:
            state.plateau_candidate_first_step = step
            state.plateau_patience_count = 0
        state.plateau_patience_count += 1
        if (
            state.plateau_patience_count >= self.config.plateau_patience_observations
            and step >= self.config.minimum_corridor_steps
        ):
            state.plateau_patience_satisfied = True
            state.freeze_step = step
            state.freeze_observation_index = state.observation_count
            state.freeze_metric_snapshot = snapshot
            self._transition(
                state, step, ModeState.FROZEN, "plateau_patience_satisfied"
            )
        else:
            self._transition(
                state, step, ModeState.PLATEAU_CANDIDATE, "plateau_detected"
            )

    def _evaluate_regression(
        self, step: int, state: ModeControllerState, metrics: Mapping[str, float]
    ) -> None:
        if not self._regression_detected(metrics):
            state.regression_candidate_first_step = None
            state.regression_patience_count = 0
            self._transition(state, step, ModeState.FROZEN, "regression_recovered")
            return
        if state.state != ModeState.REACTIVATION_CANDIDATE:
            state.regression_candidate_first_step = step
            state.regression_patience_count = 0
        state.regression_patience_count += 1
        if (
            state.regression_patience_count
            >= self.config.regression_patience_observations
        ):
            state.reactivation_count += 1
            state.last_reactivation_step = step
            state.reactivation_reason = "persistent_regression"
            state.cooldown_observations_remaining = (
                self.config.reactivation_cooldown_observations
            )
            state.regression_candidate_first_step = None
            state.regression_patience_count = 0
            self._reset_plateau(state)
            self._transition(
                state, step, ModeState.ACTIVE, "regression_patience_satisfied"
            )
        else:
            self._transition(
                state,
                step,
                ModeState.REACTIVATION_CANDIDATE,
                "regression_detected",
            )

    def _entry_satisfied(self, metrics: Mapping[str, float]) -> bool:
        checks = {
            "inside_corridor_rate": lambda value: (
                value >= self.config.entry_inside_rate_threshold
            ),
            "mean_distance_outside_corridor": lambda value: (
                value <= self.config.entry_mean_distance_threshold
            ),
            "worst_stat_violation": lambda value: (
                value <= self.config.entry_worst_violation_threshold
            ),
        }
        return all(checks[name](metrics[name]) for name in self.config.entry_metrics)

    def _regression_detected(self, metrics: Mapping[str, float]) -> bool:
        checks = {
            "inside_corridor_rate": lambda value: (
                value < self.config.regression_inside_rate_floor
            ),
            "mean_distance_outside_corridor": lambda value: (
                value > self.config.regression_mean_distance_ceiling
            ),
            "worst_stat_violation": lambda value: (
                value > self.config.regression_worst_violation_ceiling
            ),
        }
        return any(
            checks[name](metrics[name]) for name in self.config.regression_metrics
        )

    def _plateau_status(self, state: ModeControllerState) -> tuple[bool, bool]:
        values = state.metric_history[self.config.primary_progress_metric]
        smoothed = _rolling_means(values, self.config.smoothing_window_observations)
        window = self.config.progress_window_observations
        if len(smoothed) < window:
            state.last_signed_improvement = None
            state.last_relative_improvement = None
            return False, False
        direction = self.config.metric_directions[self.config.primary_progress_metric]
        latest_signed = _signed_improvement(smoothed[-2], smoothed[-1], direction)
        latest_relative = latest_signed / max(abs(smoothed[-2]), 1e-12)
        state.last_signed_improvement = latest_signed
        state.last_relative_improvement = latest_relative
        if latest_signed < 0:
            return False, True
        old, new = smoothed[-window], smoothed[-1]
        signed = _signed_improvement(old, new, direction)
        relative = signed / max(abs(old), 1e-12)
        return (
            signed >= 0
            and signed < self.config.plateau_absolute_improvement_threshold
            and relative < self.config.plateau_relative_improvement_threshold
        ), False

    def _reset_plateau(self, state: ModeControllerState) -> None:
        state.plateau_candidate_first_step = None
        state.plateau_patience_count = 0
        state.plateau_patience_satisfied = False
        state.plateau_observation = False

    def _transition(
        self, state: ModeControllerState, step: int, target: ModeState, reason: str
    ) -> None:
        if state.state == target:
            return
        transition = ModeTransition(
            step=step,
            mode_id=state.mode_id,
            from_state=state.state.value,
            to_state=target.value,
            reason=reason,
        )
        self.transitions.append(transition)
        state.transition_count += 1
        state.state = target

    def _update_global_completion(self, step: int) -> None:
        if self.all_required_modes_frozen:
            if self.global_completion_step is None:
                self.global_completion_step = step
        else:
            self.global_completion_step = None

    @property
    def active_mode_ids(self) -> list[str]:
        return sorted(
            mode_id
            for mode_id, state in self.modes.items()
            if state.state
            in {
                ModeState.WARMUP,
                ModeState.ACTIVE,
                ModeState.PLATEAU_CANDIDATE,
            }
        )

    @property
    def frozen_mode_ids(self) -> list[str]:
        return sorted(
            mode_id
            for mode_id, state in self.modes.items()
            if state.state == ModeState.FROZEN
        )

    @property
    def reactivation_candidate_mode_ids(self) -> list[str]:
        return sorted(
            mode_id
            for mode_id, state in self.modes.items()
            if state.state == ModeState.REACTIVATION_CANDIDATE
        )

    @property
    def failed_mode_ids(self) -> list[str]:
        return sorted(
            mode_id
            for mode_id, state in self.modes.items()
            if state.state == ModeState.FAILED
        )

    @property
    def all_required_modes_frozen(self) -> bool:
        return all(
            self.modes[mode_id].state == ModeState.FROZEN
            for mode_id in self.config.required_modes
        )

    @property
    def global_cycle_one_complete(self) -> bool:
        return self.all_required_modes_frozen and not self.hard_stop_reached

    @property
    def hard_stop_reached(self) -> bool:
        return (
            self.current_step is not None
            and self.current_step >= self.config.maximum_corridor_steps
            and not self.all_required_modes_frozen
        )

    @property
    def global_completion_reason(self) -> str | None:
        if self.global_cycle_one_complete:
            return "all_required_modes_frozen"
        if self.hard_stop_reached:
            return "maximum_step_cap"
        return None

    def report(self) -> dict[str, Any]:
        modes = {
            mode_id: {
                "mode_id": mode_id,
                "required": state.required,
                "final_state": state.state.value,
                "mode_should_train": state.mode_should_train,
                "entry_first_step": state.entry_condition_first_step,
                "freeze_step": state.freeze_step,
                "reactivation_count": state.reactivation_count,
                "last_reactivation_step": state.last_reactivation_step,
                "final_metric_snapshot": state.final_metric_snapshot,
                "last_signed_improvement": state.last_signed_improvement,
                "last_relative_improvement": state.last_relative_improvement,
                "plateau_observation": state.plateau_observation,
                "observation_count": state.observation_count,
                "transition_count": state.transition_count,
            }
            for mode_id, state in sorted(self.modes.items())
        }
        payload = {
            "phase": "P156.2",
            "status": "pass" if self.global_cycle_one_complete else "incomplete",
            "controller_config_sha256": self.config_sha256,
            "required_modes": list(self.config.required_modes),
            "active_mode_ids": self.active_mode_ids,
            "frozen_mode_ids": self.frozen_mode_ids,
            "reactivation_candidate_mode_ids": self.reactivation_candidate_mode_ids,
            "failed_mode_ids": self.failed_mode_ids,
            "all_required_modes_frozen": self.all_required_modes_frozen,
            "global_cycle_one_complete": self.global_cycle_one_complete,
            "global_completion_step": self.global_completion_step,
            "global_completion_reason": self.global_completion_reason,
            "modes": modes,
        }
        payload["report_sha256"] = _stable_hash(payload)
        return payload

    def to_dict(self) -> dict[str, Any]:
        transitions = [asdict(item) for item in self.transitions]
        return {
            "phase": "P156.2",
            "controller_config": self.config.to_dict(),
            "controller_config_sha256": self.config_sha256,
            "current_step": self.current_step,
            "global_completion_step": self.global_completion_step,
            "modes": {
                mode_id: {**asdict(state), "state": state.state.value}
                for mode_id, state in sorted(self.modes.items())
            },
            "transitions": transitions,
            "transition_sequence_sha256": _stable_hash(transitions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MultiModePlateauController:
        config = ModePlateauConfig.from_dict(value["controller_config"])
        controller = cls(config)
        if value["controller_config_sha256"] != controller.config_sha256:
            raise ValueError("controller config hash mismatch")
        controller.current_step = value["current_step"]
        controller.global_completion_step = value["global_completion_step"]
        controller.modes = {}
        for mode_id, raw_state in value["modes"].items():
            data = dict(raw_state)
            data["state"] = ModeState(data["state"])
            controller.modes[mode_id] = ModeControllerState(**data)
        controller.transitions = [
            ModeTransition(**transition) for transition in value["transitions"]
        ]
        if value["transition_sequence_sha256"] != _stable_hash(value["transitions"]):
            raise ValueError("transition sequence hash mismatch")
        return controller

    def write_receipts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "mode_plateau_controller_report.json", self.report())
        _write_json(output_dir / "mode_plateau_controller_state.json", self.to_dict())
        transitions = "".join(
            json.dumps(asdict(item), sort_keys=True, separators=(",", ":")) + "\n"
            for item in self.transitions
        )
        (output_dir / "mode_plateau_transitions.jsonl").write_text(
            transitions, encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> MultiModePlateauController:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _validate_config(config: ModePlateauConfig) -> None:
    if not config.required_modes:
        raise ValueError("required_modes must not be empty")
    if len(set(config.all_modes)) != len(config.all_modes):
        raise ValueError("mode IDs must be unique")
    if config.smoothing_policy != "rolling_mean":
        raise ValueError("unknown smoothing policy")
    positive = {
        "evaluation_interval_steps": config.evaluation_interval_steps,
        "minimum_observations": config.minimum_observations,
        "progress_window_observations": config.progress_window_observations,
        "plateau_patience_observations": config.plateau_patience_observations,
        "smoothing_window_observations": config.smoothing_window_observations,
        "regression_patience_observations": config.regression_patience_observations,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    if config.reactivation_cooldown_observations < 0:
        raise ValueError("reactivation_cooldown_observations must be >= 0")
    if config.minimum_corridor_steps < 0:
        raise ValueError("minimum_corridor_steps must be >= 0")
    if config.maximum_corridor_steps < config.minimum_corridor_steps:
        raise ValueError("maximum_corridor_steps must be >= minimum_corridor_steps")
    if (
        config.plateau_relative_improvement_threshold < 0
        or config.plateau_absolute_improvement_threshold < 0
    ):
        raise ValueError("plateau improvement thresholds must be non-negative")
    required_metrics = set(config.entry_metrics) | set(config.regression_metrics)
    required_metrics.add(config.primary_progress_metric)
    missing_directions = required_metrics - set(config.metric_directions)
    if missing_directions:
        raise ValueError(f"missing metric directions: {sorted(missing_directions)}")
    invalid_directions = {
        name
        for name in required_metrics
        if config.metric_directions[name] not in {"lower", "higher"}
    }
    if invalid_directions:
        raise ValueError(f"invalid metric directions: {sorted(invalid_directions)}")
    supported_entry = {
        "inside_corridor_rate",
        "mean_distance_outside_corridor",
        "worst_stat_violation",
    }
    if not config.entry_metrics or not set(config.entry_metrics) <= supported_entry:
        raise ValueError("entry_metrics contains unsupported metrics")
    if (
        not config.regression_metrics
        or not set(config.regression_metrics) <= supported_entry
    ):
        raise ValueError("regression_metrics contains unsupported metrics")
    if config.regression_inside_rate_floor >= config.entry_inside_rate_threshold:
        raise ValueError("reactivation inside-rate threshold lacks hysteresis")
    if config.regression_mean_distance_ceiling <= config.entry_mean_distance_threshold:
        raise ValueError("reactivation mean-distance threshold lacks hysteresis")
    if (
        config.regression_worst_violation_ceiling
        <= config.entry_worst_violation_threshold
    ):
        raise ValueError("reactivation worst-violation threshold lacks hysteresis")


def _rolling_means(values: list[float], window: int) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _signed_improvement(previous: float, current: float, direction: str) -> float:
    return previous - current if direction == "lower" else current - previous


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
