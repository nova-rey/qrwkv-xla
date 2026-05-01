from __future__ import annotations

import pytest

from qrwkv_xla.optimizers import OptimizerConfig, validate_optimizer_config


def test_optimizer_config_accepts_supported_types() -> None:
    for optimizer_type in ("sgd", "adam", "adamw"):
        validate_optimizer_config(OptimizerConfig(type=optimizer_type))


def test_adam_weight_decay_requires_adamw() -> None:
    with pytest.raises(ValueError, match="Use adamw"):
        validate_optimizer_config(OptimizerConfig(type="adam", weight_decay=0.01))


def test_optimizer_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="optimizer.type"):
        validate_optimizer_config(OptimizerConfig(type="nope"))
    with pytest.raises(ValueError, match="learning_rate"):
        validate_optimizer_config(OptimizerConfig(learning_rate=0.0))
    with pytest.raises(ValueError, match="beta1"):
        validate_optimizer_config(OptimizerConfig(beta1=1.0))
    with pytest.raises(ValueError, match="beta2"):
        validate_optimizer_config(OptimizerConfig(beta2=-0.1))
    with pytest.raises(ValueError, match="epsilon"):
        validate_optimizer_config(OptimizerConfig(epsilon=0.0))
    with pytest.raises(ValueError, match="weight_decay"):
        validate_optimizer_config(OptimizerConfig(weight_decay=-0.1))
