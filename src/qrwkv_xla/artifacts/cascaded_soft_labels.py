from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CASCADED_BUCKET_EDGES = (1.0, 1e-3, 1e-6, 1e-9, 1e-12, 0.0)
DEFAULT_TOP_LOG_PROBS_DTYPE = "float16"
DEFAULT_BUCKET_MASS_DTYPE = "float32"
DEFAULT_BUCKET_MEAN_LOGP_DTYPE = "float32"


@dataclass(frozen=True)
class CascadedSoftLabelEncoding:
    top_token_ids: np.ndarray
    top_log_probs: np.ndarray
    top_mass: np.float32
    tail_mass: np.float32
    teacher_entropy: np.float32
    bucket_mass: np.ndarray
    bucket_count: np.ndarray
    bucket_mean_logp: np.ndarray


def encode_cascaded_soft_labels(
    logits: np.ndarray,
    *,
    top_k: int,
    bucket_edges: tuple[float, ...] = DEFAULT_CASCADED_BUCKET_EDGES,
    top_log_probs_dtype: str = DEFAULT_TOP_LOG_PROBS_DTYPE,
    bucket_mass_dtype: str = DEFAULT_BUCKET_MASS_DTYPE,
    bucket_mean_logp_dtype: str = DEFAULT_BUCKET_MEAN_LOGP_DTYPE,
) -> CascadedSoftLabelEncoding:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cascaded logits must be a non-empty rank-1 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("cascaded logits must be finite")
    if top_k <= 0 or top_k > values.size:
        raise ValueError("cascaded top_k must be within [1, vocab_size]")
    edges = validate_cascaded_bucket_edges(bucket_edges)
    shifted = values - np.max(values)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), dtype=np.float64))
    probs = np.exp(log_probs).astype(np.float32)
    top_token_ids = np.argsort(log_probs)[-top_k:][::-1].astype(np.int32)
    top_log_probs = log_probs[top_token_ids]
    top_probs = probs[top_token_ids]
    top_mass = np.float32(np.sum(top_probs, dtype=np.float32))
    tail_mass = np.float32(np.clip(1.0 - top_mass, 0.0, 1.0))
    teacher_entropy = np.float32(
        -np.sum(probs * log_probs, dtype=np.float32)
    )
    top_mask = np.zeros(values.shape, dtype=bool)
    top_mask[top_token_ids] = True
    bucket_mass = np.zeros(len(edges) - 1, dtype=np.float32)
    bucket_count = np.zeros(len(edges) - 1, dtype=np.int32)
    bucket_mean_logp = np.zeros(len(edges) - 1, dtype=np.float32)
    for bucket_id, (upper, lower) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        mask = (~top_mask) & (probs < upper) & (probs >= lower)
        bucket_mass[bucket_id] = np.sum(probs[mask], dtype=np.float32)
        bucket_count[bucket_id] = int(np.sum(mask))
        if bucket_count[bucket_id]:
            bucket_mean_logp[bucket_id] = np.mean(
                log_probs[mask], dtype=np.float32
            )
    return CascadedSoftLabelEncoding(
        top_token_ids=top_token_ids,
        top_log_probs=top_log_probs.astype(np.dtype(top_log_probs_dtype)),
        top_mass=top_mass,
        tail_mass=tail_mass,
        teacher_entropy=teacher_entropy,
        bucket_mass=bucket_mass.astype(np.dtype(bucket_mass_dtype)),
        bucket_count=bucket_count,
        bucket_mean_logp=bucket_mean_logp.astype(
            np.dtype(bucket_mean_logp_dtype)
        ),
    )


def validate_cascaded_bucket_edges(
    bucket_edges: tuple[float, ...],
) -> tuple[float, ...]:
    edges = np.asarray(bucket_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bucket_edges must contain at least two values")
    if not np.all(np.isfinite(edges)):
        raise ValueError("bucket_edges must be finite")
    if not np.all(np.diff(edges) < 0):
        raise ValueError("bucket_edges must be strictly descending")
    if edges[0] != 1.0 or edges[-1] != 0.0:
        raise ValueError("bucket_edges must start at 1.0 and end at 0.0")
    return tuple(float(value) for value in edges)
