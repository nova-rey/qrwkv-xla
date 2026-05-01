from __future__ import annotations

from typing import Any, NamedTuple


class TrainState(NamedTuple):
    params: Any
    step: int
    learning_rate: float
    optimizer_state: Any = None
