from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def logits_kl_loss(
    student_logits: Any,
    teacher_logits: Any,
    attention_mask: Any | None = None,
    *,
    temperature: float = 1.0,
) -> Any:
    student = jnp.asarray(student_logits)
    teacher = jnp.asarray(teacher_logits)
    if student.shape != teacher.shape:
        raise ValueError(
            "student_logits.shape must match teacher_logits.shape, "
            f"got {student.shape} and {teacher.shape}"
        )
    if student.ndim != 3:
        raise ValueError(f"logits must have shape [B,S,V], got {student.shape}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    scaled_student = student / temperature
    scaled_teacher = teacher / temperature
    teacher_probs = jax.nn.softmax(scaled_teacher, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(scaled_teacher, axis=-1)
    student_log_probs = jax.nn.log_softmax(scaled_student, axis=-1)
    token_kl = jnp.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        axis=-1,
    )
    token_kl = token_kl * (temperature * temperature)

    if attention_mask is None:
        return jnp.mean(token_kl)

    mask = jnp.asarray(attention_mask)
    if mask.shape != student.shape[:2]:
        raise ValueError(f"attention_mask must have shape [B,S], got {mask.shape}")
    mask = mask.astype(token_kl.dtype)
    numerator = jnp.sum(token_kl * mask)
    denominator = jnp.maximum(jnp.sum(mask), jnp.asarray(1, dtype=token_kl.dtype))
    return numerator / denominator
