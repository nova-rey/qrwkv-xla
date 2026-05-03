from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StudentOutput:
    hidden_states: jax.Array
    logits: jax.Array | None = None
    mixer_outputs: jax.Array | None = None

    def tree_flatten(self):
        return (self.hidden_states, self.logits, self.mixer_outputs), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        hidden_states, logits, mixer_outputs = children
        return cls(
            hidden_states=hidden_states,
            logits=logits,
            mixer_outputs=mixer_outputs,
        )


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
