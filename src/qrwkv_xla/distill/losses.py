from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qrwkv_xla.distill.config import DistillLossConfig
from qrwkv_xla.losses.hidden import hidden_mse_loss
from qrwkv_xla.students.base import StudentOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: jax.Array
    hidden_mse: jax.Array | None = None
    logits_kl: jax.Array | None = None


def logits_kl_loss(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    attention_mask: jax.Array | None = None,
    temperature: float = 1.0,
) -> jax.Array:
    student = jnp.asarray(student_logits)
    teacher = jnp.asarray(teacher_logits)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if student.shape != teacher.shape:
        raise ValueError(
            "student_logits and teacher_logits must have matching shapes, "
            f"got {student.shape} and {teacher.shape}"
        )

    student_scaled = student / temperature
    teacher_scaled = teacher / temperature
    teacher_probs = jax.nn.softmax(teacher_scaled, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_scaled, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_scaled, axis=-1)
    token_kl = jnp.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        axis=-1,
    )

    if attention_mask is None:
        return jnp.mean(token_kl)

    mask = jnp.asarray(attention_mask)
    if mask.shape != student.shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {student.shape[:2]}, got {mask.shape}"
        )
    mask = mask.astype(token_kl.dtype)
    numerator = jnp.sum(token_kl * mask)
    denominator = jnp.maximum(jnp.sum(mask), jnp.asarray(1, dtype=token_kl.dtype))
    return numerator / denominator


def compute_distill_loss(
    *,
    student_output: StudentOutput,
    teacher_hidden_states: jax.Array,
    teacher_logits: jax.Array | None,
    attention_mask: jax.Array | None,
    loss_config: DistillLossConfig,
) -> LossBreakdown:
    if loss_config.attention_or_mixer.enabled:
        raise ValueError("attention_or_mixer distillation is not implemented yet")

    total = jnp.asarray(0.0, dtype=jnp.float32)
    hidden_value: jax.Array | None = None
    logits_value: jax.Array | None = None

    if loss_config.hidden_mse.enabled and loss_config.hidden_mse.weight > 0:
        hidden_value = hidden_mse_loss(
            student_output.hidden_states,
            teacher_hidden_states,
            attention_mask,
        )
        total = total + hidden_value * jnp.asarray(loss_config.hidden_mse.weight)

    if loss_config.logits_kl.enabled and loss_config.logits_kl.weight > 0:
        if student_output.logits is None:
            raise ValueError("logits_kl is enabled but student_output.logits is None")
        if teacher_logits is None:
            raise ValueError("logits_kl is enabled but teacher logits are missing")
        logits_value = logits_kl_loss(
            student_output.logits,
            teacher_logits,
            attention_mask,
        )
        total = total + logits_value * jnp.asarray(loss_config.logits_kl.weight)

    return LossBreakdown(
        total=total,
        hidden_mse=hidden_value,
        logits_kl=logits_value,
    )
