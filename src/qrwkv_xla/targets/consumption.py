from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.targets.store import TeacherTargetStore


@dataclass(frozen=True)
class OfflineTargetBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    teacher_logits: np.ndarray


def load_offline_target_batch(
    store: TeacherTargetStore,
    *,
    shard_id: int = 0,
) -> OfflineTargetBatch:
    """Load one stored teacher-target shard as a student-side offline batch."""
    store.validate()
    arrays = store.read_shard(shard_id)
    missing = [
        name for name in ("input_ids", "attention_mask", "logits") if name not in arrays
    ]
    if missing:
        raise ValueError(f"offline target shard missing required arrays: {missing}")

    input_ids = np.asarray(arrays["input_ids"])
    attention_mask = np.asarray(arrays["attention_mask"])
    teacher_logits = np.asarray(arrays["logits"])
    metadata = store.metadata

    if input_ids.ndim != 2:
        raise ValueError("offline input_ids must have shape [N,T]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("offline attention_mask shape must match input_ids")
    if teacher_logits.ndim != 3:
        raise ValueError("offline teacher_logits must have shape [N,T,V]")
    if teacher_logits.shape[:2] != input_ids.shape:
        raise ValueError("offline teacher_logits [N,T] must match input_ids")
    if input_ids.shape[1] != metadata.sequence_length:
        raise ValueError("offline input_ids sequence length must match metadata")
    if teacher_logits.shape[2] != metadata.vocab_size:
        raise ValueError("offline teacher_logits vocab size must match metadata")

    return OfflineTargetBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        teacher_logits=teacher_logits,
    )


def mse_logits_loss(student_logits: Any, teacher_logits: Any) -> jnp.ndarray:
    student = jnp.asarray(student_logits)
    teacher = jnp.asarray(teacher_logits)
    if student.shape != teacher.shape:
        raise ValueError(
            "student_logits and teacher_logits must have identical shapes, got "
            f"{student.shape} and {teacher.shape}"
        )
    return jnp.mean(jnp.square(student - teacher))
