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


@dataclass(frozen=True)
class TeacherTargetBatch:
    target_type: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    teacher_logits: np.ndarray | None = None
    top_token_ids: np.ndarray | None = None
    top_log_probs: np.ndarray | None = None
    top_mass: np.ndarray | None = None
    tail_mass: np.ndarray | None = None
    teacher_entropy: np.ndarray | None = None


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


def load_teacher_target_batch(
    store: TeacherTargetStore,
    *,
    shard_id: int = 0,
) -> TeacherTargetBatch:
    store.validate()
    arrays = store.read_shard(shard_id)
    metadata = store.metadata
    target_type = metadata.target_type
    if target_type in {"dense_logits", "full_logits", "synthetic"}:
        dense = load_offline_target_batch(store, shard_id=shard_id)
        return TeacherTargetBatch(
            target_type=target_type,
            input_ids=dense.input_ids,
            attention_mask=dense.attention_mask,
            teacher_logits=dense.teacher_logits,
        )
    if target_type == "topk_with_tail_v0":
        required = (
            "input_ids",
            "attention_mask",
            "top_token_ids",
            "top_log_probs",
            "top_mass",
            "tail_mass",
            "teacher_entropy",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(
                f"target_type {target_type!r} missing compressed field(s): {missing}"
            )
        input_ids = np.asarray(arrays["input_ids"])
        attention_mask = np.asarray(arrays["attention_mask"])
        top_token_ids = np.asarray(arrays["top_token_ids"])
        top_log_probs = np.asarray(arrays["top_log_probs"])
        top_mass = np.asarray(arrays["top_mass"])
        tail_mass = np.asarray(arrays["tail_mass"])
        teacher_entropy = np.asarray(arrays["teacher_entropy"])
        _validate_common_batch_shapes(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_length=metadata.sequence_length,
        )
        if top_token_ids.shape != top_log_probs.shape:
            raise ValueError(
                "top_token_ids shape must match top_log_probs shape, "
                f"got {top_token_ids.shape} and {top_log_probs.shape}"
            )
        if top_token_ids.shape[:2] != input_ids.shape:
            raise ValueError(
                "top_token_ids [N,T] must match input_ids, "
                f"got {top_token_ids.shape[:2]} and {input_ids.shape}"
            )
        for name, value in (
            ("top_mass", top_mass),
            ("tail_mass", tail_mass),
            ("teacher_entropy", teacher_entropy),
        ):
            if value.shape != input_ids.shape:
                raise ValueError(
                    f"{name} must have shape {input_ids.shape}, got {value.shape}"
                )
        if np.any(top_token_ids < 0) or np.any(top_token_ids >= metadata.vocab_size):
            raise ValueError(
                "top_token_ids contains ids outside teacher vocab range "
                f"[0, {metadata.vocab_size})"
            )
        return TeacherTargetBatch(
            target_type=target_type,
            input_ids=input_ids,
            attention_mask=attention_mask,
            top_token_ids=top_token_ids,
            top_log_probs=top_log_probs,
            top_mass=top_mass,
            tail_mass=tail_mass,
            teacher_entropy=teacher_entropy,
        )
    raise ValueError(
        f"unsupported target_type {target_type!r} for teacher target consumption"
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


def _validate_common_batch_shapes(
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    sequence_length: int,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("offline input_ids must have shape [N,T]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("offline attention_mask shape must match input_ids")
    if input_ids.shape[1] != sequence_length:
        raise ValueError("offline input_ids sequence length must match metadata")
