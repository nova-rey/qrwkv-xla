from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

PROFILE_NAMES = ("rock_hammer", "ball_peen", "sledgehammer", "gallagher")


@dataclass(frozen=True)
class CorridorAggressivenessProfile:
    profile_name: str
    aggressiveness_rank: int
    corridor_loss_weight: float
    learning_rate: float
    max_grad_norm: float
    penalty_kind: str
    penalty_power: float
    per_stat_weights: dict[str, float]
    worst_stat_boost: float
    adaptive_weighting_enabled: bool
    distance_normalization: str
    stability_abort_enabled: bool
    parameter_norm_limit: float = 1e6
    gradient_norm_hard_limit: float = 1e4
    held_out_loss_abort_multiple: float = 25.0
    profile_overrides_applied: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["profile_overrides_applied"] = list(self.profile_overrides_applied)
        return value


_PROFILES = {
    "rock_hammer": CorridorAggressivenessProfile(
        "rock_hammer",
        0,
        1.0,
        1e-4,
        1.0,
        "powered_hinge",
        2.0,
        {
            "entropy": 1.0,
            "top1_margin": 1.0,
            "top8_mass": 1.0,
            "top32_mass": 1.0,
            "tail_mass": 1.0,
        },
        1.0,
        False,
        "none",
        True,
    ),
    "ball_peen": CorridorAggressivenessProfile(
        "ball_peen",
        1,
        6.0,
        2e-4,
        1.0,
        "powered_hinge",
        2.0,
        {
            "entropy": 1.25,
            "top1_margin": 1.0,
            "top8_mass": 1.25,
            "top32_mass": 1.25,
            "tail_mass": 1.25,
        },
        1.5,
        False,
        "corridor_width",
        True,
    ),
    "sledgehammer": CorridorAggressivenessProfile(
        "sledgehammer",
        2,
        25.0,
        1e-3,
        1.0,
        "powered_hinge",
        2.0,
        {
            "entropy": 2.0,
            "top1_margin": 0.5,
            "top8_mass": 2.0,
            "top32_mass": 2.0,
            "tail_mass": 2.0,
        },
        4.0,
        False,
        "corridor_width",
        True,
    ),
    "gallagher": CorridorAggressivenessProfile(
        "gallagher",
        3,
        64.0,
        2e-3,
        0.75,
        "powered_hinge",
        3.0,
        {
            "entropy": 3.0,
            "top1_margin": 1.0,
            "top8_mass": 3.0,
            "top32_mass": 3.0,
            "tail_mass": 3.0,
        },
        8.0,
        False,
        "corridor_width",
        True,
        gradient_norm_hard_limit=5e3,
        held_out_loss_abort_multiple=10.0,
    ),
}


def resolve_aggressiveness_profile(
    name: str, overrides: dict[str, float | str | bool | None] | None = None
) -> CorridorAggressivenessProfile:
    try:
        profile = _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown aggressiveness profile: {name}") from exc
    applied: list[str] = []
    updates: dict[str, Any] = {}
    weights = dict(profile.per_stat_weights)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key.endswith("_weight") and key.removesuffix("_weight") in weights:
            weights[key.removesuffix("_weight")] = float(value)
        elif hasattr(profile, key) and key not in {
            "profile_name",
            "aggressiveness_rank",
            "profile_overrides_applied",
        }:
            updates[key] = value
        else:
            raise ValueError(f"unknown profile override: {key}")
        applied.append(key)
    if weights != profile.per_stat_weights:
        updates["per_stat_weights"] = weights
    return replace(profile, **updates, profile_overrides_applied=tuple(sorted(applied)))


def aggressiveness_profiles() -> tuple[CorridorAggressivenessProfile, ...]:
    return tuple(_PROFILES[name] for name in PROFILE_NAMES)
