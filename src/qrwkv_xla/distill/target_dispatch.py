from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.distill.losses import (
    DistillationLossReport,
    logits_kl_loss,
    topk_tail_distillation_loss,
)

DENSE_LOGIT_TARGET_TYPES = frozenset({"dense_logits", "full_logits", "synthetic"})
TOPK_TAIL_TARGET_TYPE = "topk_with_tail_v0"
CASCADED_TARGET_TYPE = "cascaded_soft_labels_v1"


class UnsupportedTeacherTargetType(ValueError):
    pass


def dispatch_teacher_target_loss(
    *,
    student_logits: jax.Array,
    target_batch: Any,
    tail_loss_weight: float = 0.0,
    sparse_head_loss_weight: float = 1.0,
) -> DistillationLossReport:
    target_type = str(getattr(target_batch, "target_type", ""))
    attention_mask = getattr(target_batch, "attention_mask", None)
    if target_type in DENSE_LOGIT_TARGET_TYPES:
        teacher_logits = getattr(target_batch, "teacher_logits", None)
        if teacher_logits is None:
            raise ValueError(
                f"target_type {target_type!r} requires teacher_logits for dense loss"
            )
        dense_loss = logits_kl_loss(
            student_logits,
            teacher_logits,
            attention_mask,
        )
        token_count = _token_count(attention_mask, student_logits)
        return DistillationLossReport(
            total_loss=dense_loss,
            head_loss=dense_loss,
            tail_loss=jnp.asarray(0.0, dtype=jnp.float32),
            tail_loss_weight=0.0,
            token_count=token_count,
            target_type=target_type,
            distillation_loss_type="dense_logits_kl",
            top_k=None,
            sparse_head_loss_weight=0.0,
        )
    if target_type in {TOPK_TAIL_TARGET_TYPE, CASCADED_TARGET_TYPE}:
        missing = [
            name
            for name in ("top_token_ids", "top_log_probs", "attention_mask")
            if getattr(target_batch, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"target_type {target_type!r} missing compressed field(s): {missing}"
            )
        report = topk_tail_distillation_loss(
            student_logits,
            target_batch.top_token_ids,
            target_batch.top_log_probs,
            target_batch.attention_mask,
            tail_mass=getattr(target_batch, "tail_mass", None),
            top_mass=getattr(target_batch, "top_mass", None),
            teacher_entropy=getattr(target_batch, "teacher_entropy", None),
            tail_loss_weight=tail_loss_weight,
            sparse_head_loss_weight=sparse_head_loss_weight,
        )
        if target_type == CASCADED_TARGET_TYPE:
            return replace(
                report,
                target_type=CASCADED_TARGET_TYPE,
                distillation_loss_type="topk_head_only_until_p122",
                bucket_loss_weight=0.0,
            )
        return report
    expected = sorted(
        (*DENSE_LOGIT_TARGET_TYPES, TOPK_TAIL_TARGET_TYPE, CASCADED_TARGET_TYPE)
    )
    raise UnsupportedTeacherTargetType(
        f"unsupported teacher target_type {target_type!r}; expected one of {expected}"
    )


def _token_count(attention_mask: Any, student_logits: jax.Array) -> jax.Array:
    if attention_mask is None:
        return jnp.asarray(
            student_logits.shape[0] * student_logits.shape[1], dtype=jnp.float32
        )
    mask = jnp.asarray(attention_mask, dtype=jnp.float32)
    if mask.shape != jnp.asarray(student_logits).shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {jnp.asarray(student_logits).shape[:2]}, "
            f"got {mask.shape}"
        )
    return jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=jnp.float32))
