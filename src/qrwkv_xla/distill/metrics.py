from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np


def scalar_to_float(value: Any) -> float:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, jax.Array):
        return float(np.asarray(value))
    return float(value)


def metrics_to_floats(metrics: Mapping[str, object]) -> dict[str, float]:
    return {name: scalar_to_float(value) for name, value in metrics.items()}
