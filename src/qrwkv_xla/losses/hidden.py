from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def hidden_mse_loss(
    student_hidden: Any,
    teacher_hidden: Any,
    attention_mask: Any | None = None,
) -> Any:
    student = jnp.asarray(student_hidden)
    teacher = jnp.asarray(teacher_hidden)
    if student.shape != teacher.shape:
        raise ValueError(
            "student_hidden.shape must match teacher_hidden.shape, "
            f"got {student.shape} and {teacher.shape}"
        )
    if student.ndim != 4:
        raise ValueError(
            f"hidden states must have shape [B,L,S,H], got {student.shape}"
        )

    squared_error = jnp.square(student - teacher)
    if attention_mask is None:
        return jnp.mean(squared_error)

    mask = jnp.asarray(attention_mask)
    if mask.shape == (student.shape[0], student.shape[2]):
        mask = mask[:, None, :, None]
    elif mask.shape != (student.shape[0], 1, student.shape[2], 1):
        raise ValueError(
            f"attention_mask must have shape [B,S] or [B,1,S,1], got {mask.shape}"
        )

    mask = mask.astype(squared_error.dtype)
    numerator = jnp.sum(squared_error * mask)
    denominator = jnp.maximum(
        jnp.sum(mask) * student.shape[1] * student.shape[3],
        jnp.asarray(1, dtype=squared_error.dtype),
    )
    return numerator / denominator
