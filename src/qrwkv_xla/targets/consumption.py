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
    bucket_edges: np.ndarray | None = None
    bucket_mass: np.ndarray | None = None
    bucket_count: np.ndarray | None = None
    bucket_mean_logp: np.ndarray | None = None


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
    if target_type in {"topk_with_tail_v0", "cascaded_soft_labels_v1"}:
        required = [
            "input_ids",
            "attention_mask",
            "top_token_ids",
            "top_log_probs",
            "top_mass",
            "tail_mass",
            "teacher_entropy",
        ]
        if target_type == "cascaded_soft_labels_v1":
            required.extend(["bucket_mass", "bucket_count", "bucket_mean_logp"])
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
        kwargs: dict[str, np.ndarray | str | None] = {}
        if target_type == "cascaded_soft_labels_v1":
            bucket_mass = np.asarray(arrays["bucket_mass"])
            bucket_count = np.asarray(arrays["bucket_count"])
            bucket_mean_logp = np.asarray(arrays["bucket_mean_logp"])
            bucket_edges = _metadata_bucket_edges(metadata)
            expected_bucket_shape = bucket_mass.shape
            if bucket_count.shape != expected_bucket_shape:
                raise ValueError("bucket_count shape must match bucket_mass shape")
            if bucket_mean_logp.shape != expected_bucket_shape:
                raise ValueError("bucket_mean_logp shape must match bucket_mass shape")
            if bucket_mass.shape[:2] != input_ids.shape:
                raise ValueError(
                    "bucket arrays [N,T] must match input_ids, "
                    f"got {bucket_mass.shape[:2]} and {input_ids.shape}"
                )
            kwargs = {
                "bucket_edges": bucket_edges,
                "bucket_mass": bucket_mass,
                "bucket_count": bucket_count,
                "bucket_mean_logp": bucket_mean_logp,
            }
        return TeacherTargetBatch(
            target_type=target_type,
            input_ids=input_ids,
            attention_mask=attention_mask,
            top_token_ids=top_token_ids,
            top_log_probs=top_log_probs,
            top_mass=top_mass,
            tail_mass=tail_mass,
            teacher_entropy=teacher_entropy,
            **kwargs,
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


def _metadata_bucket_edges(metadata: Any) -> np.ndarray:
    raw_edges = metadata.target_params.get("bucket_edges", "")
    try:
        edges = np.asarray(
            [float(edge) for edge in raw_edges.split(",") if edge],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise ValueError("metadata target_params.bucket_edges must be numeric") from exc
    try:
        expected_bucket_count = int(metadata.target_params.get("bucket_count", "0"))
    except ValueError as exc:
        raise ValueError(
            "metadata target_params.bucket_count must be an integer"
        ) from exc
    if edges.shape != (expected_bucket_count + 1,):
        raise ValueError("bucket_edges length must equal bucket_count + 1")
    if edges.shape[0] < 2:
        raise ValueError("bucket_edges must contain at least two edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("bucket_edges must be finite")
    if not np.all(np.diff(edges) < 0):
        raise ValueError("bucket_edges must be strictly descending")
    if not np.isclose(edges[0], 1.0):
        raise ValueError("bucket_edges must start at 1.0")
    if not np.isclose(edges[-1], 0.0):
        raise ValueError("bucket_edges must end at 0.0")
    return edges
