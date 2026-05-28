from __future__ import annotations

from typing import Any, Protocol

import jax

from qrwkv_xla.students.base import StudentOutput


class StudentBackend(Protocol):
    """Stable wrapper interface for QRWKV/Radjax student behavior.

    P91 is an extraction layer only. Implementations should delegate to the
    existing validated QRWKV student paths unless a later phase explicitly
    changes behavior.
    """

    def init_params(self, key: jax.Array) -> Any:
        raise NotImplementedError

    def init_state(self, batch_size: int, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_full(
        self,
        params: Any,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        initial_state: Any | None = None,
        **kwargs: Any,
    ) -> tuple[StudentOutput, Any]:
        raise NotImplementedError

    def forward_step(
        self,
        params: Any,
        input_ids: jax.Array,
        state: Any,
        attention_mask: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[StudentOutput, Any]:
        raise NotImplementedError

    def export_state(self, state: Any) -> dict[str, Any]:
        raise NotImplementedError

    def import_state(
        self,
        payload: dict[str, Any],
        *,
        template: Any | None = None,
    ) -> Any:
        raise NotImplementedError

    def logits(self, output: StudentOutput) -> jax.Array:
        raise NotImplementedError
