from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax


@dataclass(frozen=True)
class StudentOutput:
    hidden_states: jax.Array
    logits: jax.Array | None = None


class StudentModel(Protocol):
    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        raise NotImplementedError

    def apply(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> StudentOutput:
        raise NotImplementedError
