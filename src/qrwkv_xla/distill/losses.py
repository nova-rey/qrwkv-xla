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


@dataclass(frozen=True)
class DistillationLossReport:
    total_loss: jax.Array
    head_loss: jax.Array
    tail_loss: jax.Array
    tail_loss_weight: float
    token_count: jax.Array
    target_type: str
    distillation_loss_type: str
    bucket_loss_weight: float = 0.0
    top_k: int | None = None
    sparse_head_loss_weight: float = 1.0
    mean_top_mass: jax.Array | None = None
    mean_tail_mass: jax.Array | None = None
    mean_teacher_entropy: jax.Array | None = None
    bucket_shape_loss: jax.Array | None = None
    bucket_shape_loss_weight: float = 0.0
    bucket_shape_loss_type: str | None = None
    bucket_count: int | None = None
    mean_teacher_top_mass: jax.Array | None = None
    mean_teacher_tail_mass: jax.Array | None = None
    mean_student_tail_mass: jax.Array | None = None
    mean_teacher_bucket_mass: jax.Array | None = None
    mean_student_bucket_mass: jax.Array | None = None
    mean_bucket_shape_kl: jax.Array | None = None


@dataclass(frozen=True)
class StudentBucketProjection:
    student_top_mass: jax.Array
    student_tail_mass: jax.Array
    student_bucket_mass: jax.Array
    student_bucket_count: jax.Array


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


def topk_tail_distillation_loss(
    student_logits: jax.Array,
    top_token_ids: jax.Array,
    top_log_probs: jax.Array,
    attention_mask: jax.Array,
    *,
    tail_mass: jax.Array | None = None,
    top_mass: jax.Array | None = None,
    teacher_entropy: jax.Array | None = None,
    tail_loss_weight: float = 0.0,
    sparse_head_loss_weight: float = 1.0,
    eps: float = 1e-8,
) -> DistillationLossReport:
    if tail_loss_weight < 0:
        raise ValueError(f"tail_loss_weight must be >= 0, got {tail_loss_weight}")
    if sparse_head_loss_weight <= 0:
        raise ValueError(
            f"sparse_head_loss_weight must be > 0, got {sparse_head_loss_weight}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    student = jnp.asarray(student_logits, dtype=jnp.float32)
    token_ids = jnp.asarray(top_token_ids, dtype=jnp.int32)
    teacher_top = jnp.asarray(top_log_probs, dtype=jnp.float32)
    mask = jnp.asarray(attention_mask, dtype=jnp.float32)

    if student.ndim != 3:
        raise ValueError(f"student_logits must have shape [B,T,V], got {student.shape}")
    if token_ids.ndim != 3:
        raise ValueError(
            f"top_token_ids must have shape [B,T,K], got {token_ids.shape}"
        )
    if teacher_top.shape != token_ids.shape:
        raise ValueError(
            "top_token_ids shape must match top_log_probs shape, "
            f"got {token_ids.shape} and {teacher_top.shape}"
        )
    if token_ids.shape[:2] != student.shape[:2]:
        raise ValueError(
            "top_token_ids [B,T] must match student_logits [B,T], "
            f"got {token_ids.shape[:2]} and {student.shape[:2]}"
        )
    if mask.shape != student.shape[:2]:
        raise ValueError(
            f"attention_mask must have shape {student.shape[:2]}, got {mask.shape}"
        )
    if student.shape[-1] <= 0:
        raise ValueError("student_logits vocab dimension must be > 0")
    if _array_has_any(token_ids < 0) or _array_has_any(token_ids >= student.shape[-1]):
        raise ValueError(
            "top_token_ids contains ids outside student vocab range "
            f"[0, {student.shape[-1]}) for target_type topk_with_tail_v0"
        )
    if tail_loss_weight > 0 and tail_mass is None:
        raise ValueError(
            "tail_loss_weight > 0 requires tail_mass for target_type topk_with_tail_v0"
        )

    student_top_logits = jnp.take_along_axis(student, token_ids, axis=-1)
    teacher_head_log_probs = jax.nn.log_softmax(teacher_top, axis=-1)
    student_head_log_probs = jax.nn.log_softmax(student_top_logits, axis=-1)
    teacher_head_probs = jnp.exp(teacher_head_log_probs)
    per_token_head_kl = jnp.sum(
        teacher_head_probs * (teacher_head_log_probs - student_head_log_probs),
        axis=-1,
    )
    token_count = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=jnp.float32))
    head_loss = jnp.sum(per_token_head_kl * mask) / token_count

    tail_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if tail_loss_weight > 0:
        teacher_tail = jnp.asarray(tail_mass, dtype=jnp.float32)
        if teacher_tail.shape != student.shape[:2]:
            raise ValueError(
                f"tail_mass must have shape {student.shape[:2]}, got "
                f"{teacher_tail.shape}"
            )
        student_head_logsumexp = jax.nn.logsumexp(student_top_logits, axis=-1)
        student_full_logsumexp = jax.nn.logsumexp(student, axis=-1)
        student_top_mass = jnp.exp(student_head_logsumexp - student_full_logsumexp)
        student_tail_mass = jnp.clip(1.0 - student_top_mass, eps, 1.0)
        teacher_tail_mass = jnp.clip(teacher_tail, eps, 1.0)
        per_token_tail_loss = jnp.square(
            jnp.log(student_tail_mass) - jnp.log(teacher_tail_mass)
        )
        tail_loss = jnp.sum(per_token_tail_loss * mask) / token_count

    total = head_loss * jnp.asarray(
        sparse_head_loss_weight, dtype=jnp.float32
    ) + tail_loss * jnp.asarray(tail_loss_weight, dtype=jnp.float32)
    return DistillationLossReport(
        total_loss=total,
        head_loss=head_loss,
        tail_loss=tail_loss,
        tail_loss_weight=tail_loss_weight,
        token_count=token_count,
        target_type="topk_with_tail_v0",
        distillation_loss_type="topk_tail_head_kl",
        top_k=int(token_ids.shape[-1]),
        sparse_head_loss_weight=sparse_head_loss_weight,
        mean_top_mass=_masked_mean(top_mass, mask) if top_mass is not None else None,
        mean_tail_mass=_masked_mean(tail_mass, mask) if tail_mass is not None else None,
        mean_teacher_top_mass=(
            _masked_mean(top_mass, mask) if top_mass is not None else None
        ),
        mean_teacher_tail_mass=(
            _masked_mean(tail_mass, mask) if tail_mass is not None else None
        ),
        mean_teacher_entropy=(
            _masked_mean(teacher_entropy, mask) if teacher_entropy is not None else None
        ),
    )


def project_student_tail_buckets(
    student_logits: jax.Array,
    top_token_ids: jax.Array,
    bucket_edges: jax.Array,
    *,
    eps: float = 1e-8,
) -> StudentBucketProjection:
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    student = jnp.asarray(student_logits, dtype=jnp.float32)
    token_ids = jnp.asarray(top_token_ids, dtype=jnp.int32)
    edges = jnp.asarray(bucket_edges, dtype=jnp.float32)

    if student.ndim != 3:
        raise ValueError(f"student_logits must have shape [B,T,V], got {student.shape}")
    if token_ids.ndim != 3:
        raise ValueError(
            f"top_token_ids must have shape [B,T,K], got {token_ids.shape}"
        )
    if token_ids.shape[:2] != student.shape[:2]:
        raise ValueError(
            "top_token_ids [B,T] must match student_logits [B,T], "
            f"got {token_ids.shape[:2]} and {student.shape[:2]}"
        )
    _validate_bucket_edges(edges)
    if _array_has_any(token_ids < 0) or _array_has_any(token_ids >= student.shape[-1]):
        raise ValueError(
            "top_token_ids contains ids outside student vocab range "
            f"[0, {student.shape[-1]}) for bucket projection"
        )

    probs = jax.nn.softmax(student, axis=-1)
    top_mask = jnp.sum(
        jax.nn.one_hot(token_ids, student.shape[-1], dtype=jnp.float32),
        axis=-2,
    )
    top_mask = jnp.clip(top_mask, 0.0, 1.0)
    tail_mask = 1.0 - top_mask
    student_top_mass = jnp.sum(probs * top_mask, axis=-1)
    student_tail_mass = jnp.clip(jnp.sum(probs * tail_mask, axis=-1), eps, 1.0)

    bucket_masks = []
    bucket_counts = []
    stopped_probs = jax.lax.stop_gradient(probs)
    for bucket_index in range(edges.shape[0] - 1):
        high = edges[bucket_index]
        low = edges[bucket_index + 1]
        if bucket_index == edges.shape[0] - 2:
            in_bucket = (stopped_probs <= high) & (stopped_probs >= low)
        else:
            in_bucket = (stopped_probs <= high) & (stopped_probs > low)
        bucket_mask = in_bucket.astype(jnp.float32) * tail_mask
        bucket_masks.append(bucket_mask)
        bucket_counts.append(jnp.sum(bucket_mask, axis=-1).astype(jnp.int32))

    stacked_masks = jnp.stack(bucket_masks, axis=-1)
    student_bucket_mass = jnp.sum(probs[..., None] * stacked_masks, axis=-2)
    student_bucket_count = jnp.stack(bucket_counts, axis=-1)
    return StudentBucketProjection(
        student_top_mass=student_top_mass,
        student_tail_mass=student_tail_mass,
        student_bucket_mass=student_bucket_mass,
        student_bucket_count=student_bucket_count,
    )


def cascaded_soft_labels_loss(
    student_logits: jax.Array,
    top_token_ids: jax.Array,
    top_log_probs: jax.Array,
    attention_mask: jax.Array,
    *,
    tail_mass: jax.Array,
    bucket_mass: jax.Array,
    bucket_edges: jax.Array,
    top_mass: jax.Array | None = None,
    teacher_entropy: jax.Array | None = None,
    head_loss_weight: float = 1.0,
    tail_mass_loss_weight: float = 0.0,
    bucket_shape_loss_weight: float = 0.0,
    bucket_shape_loss_type: str = "kl",
    eps: float = 1e-8,
) -> DistillationLossReport:
    if bucket_shape_loss_weight < 0:
        raise ValueError(
            f"bucket_shape_loss_weight must be >= 0, got {bucket_shape_loss_weight}"
        )
    if bucket_shape_loss_type not in {"kl", "log_mse"}:
        raise ValueError(
            "bucket_shape_loss_type must be 'kl' or 'log_mse', "
            f"got {bucket_shape_loss_type!r}"
        )

    topk_report = topk_tail_distillation_loss(
        student_logits,
        top_token_ids,
        top_log_probs,
        attention_mask,
        tail_mass=tail_mass,
        top_mass=top_mass,
        teacher_entropy=teacher_entropy,
        tail_loss_weight=tail_mass_loss_weight,
        sparse_head_loss_weight=head_loss_weight,
        eps=eps,
    )
    student = jnp.asarray(student_logits, dtype=jnp.float32)
    teacher_tail = jnp.asarray(tail_mass, dtype=jnp.float32)
    teacher_buckets = jnp.asarray(bucket_mass, dtype=jnp.float32)
    mask = jnp.asarray(attention_mask, dtype=jnp.float32)
    edges = jnp.asarray(bucket_edges, dtype=jnp.float32)
    _validate_bucket_edges(edges)

    if teacher_tail.shape != student.shape[:2]:
        raise ValueError(
            f"tail_mass must have shape {student.shape[:2]}, got {teacher_tail.shape}"
        )
    if teacher_buckets.shape[:2] != student.shape[:2]:
        raise ValueError(
            "bucket_mass [B,T] must match student_logits [B,T], "
            f"got {teacher_buckets.shape[:2]} and {student.shape[:2]}"
        )
    if teacher_buckets.ndim != 3:
        raise ValueError(
            f"bucket_mass must have shape [B,T,C], got {teacher_buckets.shape}"
        )
    if teacher_buckets.shape[-1] != edges.shape[0] - 1:
        raise ValueError(
            "bucket_mass bucket dimension must equal len(bucket_edges) - 1, "
            f"got {teacher_buckets.shape[-1]} and {edges.shape[0] - 1}"
        )

    projection = project_student_tail_buckets(
        student,
        top_token_ids,
        edges,
        eps=eps,
    )
    teacher_bucket_dist = _normalize_distribution(
        teacher_buckets,
        jnp.clip(teacher_tail, eps, 1.0),
        eps=eps,
    )
    student_bucket_dist = _normalize_distribution(
        projection.student_bucket_mass,
        projection.student_tail_mass,
        eps=eps,
    )
    per_token_bucket_kl = jnp.sum(
        teacher_bucket_dist
        * (jnp.log(teacher_bucket_dist) - jnp.log(student_bucket_dist)),
        axis=-1,
    )
    per_token_log_mse = jnp.mean(
        jnp.square(jnp.log(student_bucket_dist) - jnp.log(teacher_bucket_dist)),
        axis=-1,
    )
    token_count = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=jnp.float32))
    bucket_shape_kl = jnp.sum(per_token_bucket_kl * mask) / token_count
    bucket_shape_loss = (
        bucket_shape_kl
        if bucket_shape_loss_type == "kl"
        else jnp.sum(per_token_log_mse * mask) / token_count
    )
    total = topk_report.total_loss + bucket_shape_loss * jnp.asarray(
        bucket_shape_loss_weight, dtype=jnp.float32
    )
    return DistillationLossReport(
        total_loss=total,
        head_loss=topk_report.head_loss,
        tail_loss=topk_report.tail_loss,
        tail_loss_weight=tail_mass_loss_weight,
        token_count=topk_report.token_count,
        target_type="cascaded_soft_labels_v1",
        distillation_loss_type="cascaded_soft_labels",
        bucket_loss_weight=bucket_shape_loss_weight,
        top_k=topk_report.top_k,
        sparse_head_loss_weight=head_loss_weight,
        mean_top_mass=topk_report.mean_top_mass,
        mean_tail_mass=topk_report.mean_tail_mass,
        mean_teacher_entropy=topk_report.mean_teacher_entropy,
        bucket_shape_loss=bucket_shape_loss,
        bucket_shape_loss_weight=bucket_shape_loss_weight,
        bucket_shape_loss_type=bucket_shape_loss_type,
        bucket_count=int(teacher_buckets.shape[-1]),
        mean_teacher_top_mass=topk_report.mean_teacher_top_mass,
        mean_teacher_tail_mass=_masked_mean(teacher_tail, mask),
        mean_student_tail_mass=_masked_mean(projection.student_tail_mass, mask),
        mean_teacher_bucket_mass=_masked_mean(jnp.sum(teacher_buckets, axis=-1), mask),
        mean_student_bucket_mass=_masked_mean(
            jnp.sum(projection.student_bucket_mass, axis=-1),
            mask,
        ),
        mean_bucket_shape_kl=bucket_shape_kl,
    )


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


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    value_array = jnp.asarray(values, dtype=jnp.float32)
    if value_array.shape != mask.shape:
        raise ValueError(
            "metric array shape must match attention_mask, got "
            f"{value_array.shape} and {mask.shape}"
        )
    denominator = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=jnp.float32))
    return jnp.sum(value_array * mask) / denominator


def _array_has_any(value: jax.Array) -> bool:
    try:
        return bool(jnp.asarray(jnp.any(value)))
    except (jax.errors.TracerBoolConversionError, TypeError):
        return False


def _validate_bucket_edges(edges: jax.Array) -> None:
    if edges.ndim != 1:
        raise ValueError(f"bucket_edges must have shape [C+1], got {edges.shape}")
    if edges.shape[0] < 2:
        raise ValueError("bucket_edges must contain at least two edges")
    if _array_has_any(~jnp.isfinite(edges)):
        raise ValueError("bucket_edges must be finite")
    if _array_has_any(jnp.diff(edges) >= 0):
        raise ValueError("bucket_edges must be strictly descending")
    if bool(jnp.asarray(~jnp.isclose(edges[0], 1.0))):
        raise ValueError("bucket_edges must start at 1.0")
    if bool(jnp.asarray(~jnp.isclose(edges[-1], 0.0))):
        raise ValueError("bucket_edges must end at 0.0")


def _normalize_distribution(
    values: jax.Array,
    denominator: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    distribution = jnp.clip(values / denominator[..., None], eps, 1.0)
    return distribution / jnp.sum(distribution, axis=-1, keepdims=True)
