from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OptimizerState:
    type: str
    step: Any
    slots: dict[str, Any]

    def tree_flatten(self):
        return (self.step, self.slots), self.type

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        step, slots = children
        return cls(type=aux_data, step=step, slots=slots)
