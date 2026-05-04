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
    attention_or_mixer: jax.Array | None = None


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
    teacher_attention_targets: jax.Array | None = None,
    loss_mask: jax.Array | None = None,
) -> LossBreakdown:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    hidden_value: jax.Array | None = None
    logits_value: jax.Array | None = None
    attention_value: jax.Array | None = None

    target_mask = loss_mask if loss_mask is not None else attention_mask

    if loss_config.hidden_mse.enabled and loss_config.hidden_mse.weight > 0:
        hidden_value = hidden_mse_loss(
            student_output.hidden_states,
            teacher_hidden_states,
            target_mask,
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
            target_mask,
        )
        total = total + logits_value * jnp.asarray(loss_config.logits_kl.weight)

    if (
        loss_config.attention_or_mixer.enabled
        and loss_config.attention_or_mixer.weight > 0
    ):
        if student_output.mixer_outputs is None:
            raise ValueError(
                "attention_or_mixer is enabled but student_output.mixer_outputs is None"
            )
        if teacher_attention_targets is None:
            raise ValueError(
                "attention_or_mixer is enabled but teacher attention_targets "
                "are missing"
            )
        attention_value = masked_attention_mixer_mse(
            student_output.mixer_outputs,
            teacher_attention_targets,
            target_mask,
        )
        total = total + attention_value * jnp.asarray(
            loss_config.attention_or_mixer.weight
        )

    return LossBreakdown(
        total=total,
        hidden_mse=hidden_value,
        logits_kl=logits_value,
        attention_or_mixer=attention_value,
    )


def masked_attention_mixer_mse(
    student_mixer_outputs: jax.Array,
    teacher_attention_targets: jax.Array,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
    student = jnp.asarray(student_mixer_outputs)
    teacher = jnp.asarray(teacher_attention_targets)
    if student.shape != teacher.shape:
        raise ValueError(
            "student mixer outputs and teacher attention targets must have matching "
            f"shapes, got {student.shape} and {teacher.shape}"
        )
    if student.ndim != 4:
        raise ValueError("attention/mixer tensors must be rank 4")
    squared = (student - teacher) ** 2
    per_token = jnp.mean(squared, axis=(1, 3))
    if attention_mask is None:
        return jnp.mean(per_token)
    mask = jnp.asarray(attention_mask)
    if mask.shape != student.shape[:1] + student.shape[2:3]:
        raise ValueError(
            f"attention_mask must have shape {student.shape[:1] + student.shape[2:3]}, "
            f"got {mask.shape}"
        )
    mask = mask.astype(per_token.dtype)
    numerator = jnp.sum(per_token * mask)
    denominator = jnp.maximum(jnp.sum(mask), jnp.asarray(1, dtype=per_token.dtype))
    return numerator / denominator


def attention_mixer_mse_loss(
    student_mixer_outputs: jax.Array,
    teacher_attention_targets: jax.Array,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
    return masked_attention_mixer_mse(
        student_mixer_outputs,
        teacher_attention_targets,
        attention_mask,
    )
