from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "sgd"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0


def validate_optimizer_config(config: OptimizerConfig) -> None:
    if config.type not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer.type must be one of {'sgd', 'adam', 'adamw'}")
    if config.learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be > 0")
    if not 0 <= config.beta1 < 1:
        raise ValueError("optimizer.beta1 must satisfy 0 <= beta1 < 1")
    if not 0 <= config.beta2 < 1:
        raise ValueError("optimizer.beta2 must satisfy 0 <= beta2 < 1")
    if config.epsilon <= 0:
        raise ValueError("optimizer.epsilon must be > 0")
    if config.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be >= 0")
    if config.type == "adam" and config.weight_decay != 0:
        raise ValueError("Use adamw for weight_decay")
