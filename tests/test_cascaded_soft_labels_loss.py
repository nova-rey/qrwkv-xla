from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.distill.losses import (
    cascaded_soft_labels_loss,
    project_student_tail_buckets,
)


def test_bucket_shape_loss_near_zero_when_bucket_distribution_matches() -> None:
    student = _student_logits()
    top_ids = _top_ids()
    edges = _bucket_edges()
    projection = project_student_tail_buckets(student, top_ids, edges)

    report = cascaded_soft_labels_loss(
        student,
        top_ids,
        _top_log_probs(),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=projection.student_tail_mass,
        bucket_mass=projection.student_bucket_mass,
        bucket_edges=edges,
        bucket_shape_loss_weight=0.01,
    )

    assert report.target_type == "cascaded_soft_labels_v1"
    assert report.distillation_loss_type == "cascaded_soft_labels"
    assert report.bucket_shape_loss is not None
    np.testing.assert_allclose(float(report.bucket_shape_loss), 0.0, atol=1e-5)


def test_bucket_shape_loss_positive_when_bucket_distribution_differs() -> None:
    student = _student_logits()
    top_ids = _top_ids()
    edges = _bucket_edges()
    projection = project_student_tail_buckets(student, top_ids, edges)
    flipped = projection.student_bucket_mass[..., ::-1]

    report = cascaded_soft_labels_loss(
        student,
        top_ids,
        _top_log_probs(),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=projection.student_tail_mass,
        bucket_mass=flipped,
        bucket_edges=edges,
        bucket_shape_loss_weight=0.01,
    )

    assert report.bucket_shape_loss is not None
    assert float(report.bucket_shape_loss) > 0.0


def test_bucket_shape_loss_weight_zero_does_not_change_total_loss() -> None:
    student = _student_logits()
    top_ids = _top_ids()
    edges = _bucket_edges()
    projection = project_student_tail_buckets(student, top_ids, edges)
    shifted = projection.student_bucket_mass[..., ::-1]

    without_bucket = cascaded_soft_labels_loss(
        student,
        top_ids,
        _top_log_probs(),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=projection.student_tail_mass,
        bucket_mass=shifted,
        bucket_edges=edges,
        bucket_shape_loss_weight=0.0,
    )
    with_bucket = cascaded_soft_labels_loss(
        student,
        top_ids,
        _top_log_probs(),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=projection.student_tail_mass,
        bucket_mass=shifted,
        bucket_edges=edges,
        bucket_shape_loss_weight=0.25,
    )

    np.testing.assert_allclose(
        float(without_bucket.total_loss),
        float(without_bucket.head_loss + without_bucket.tail_loss),
        rtol=1e-6,
    )
    assert float(with_bucket.total_loss) > float(without_bucket.total_loss)


def test_attention_mask_excludes_masked_positions() -> None:
    student = jnp.concatenate([_student_logits(), -_student_logits()], axis=1)
    top_ids = jnp.asarray([[[0, 1], [0, 1]]], dtype=jnp.int32)
    top_log_probs = jnp.asarray([[[0.0, -1.0], [-5.0, 0.0]]], dtype=jnp.float32)
    edges = _bucket_edges()
    projection = project_student_tail_buckets(student, top_ids, edges)

    masked = cascaded_soft_labels_loss(
        student,
        top_ids,
        top_log_probs,
        jnp.asarray([[1, 0]], dtype=jnp.int32),
        tail_mass=projection.student_tail_mass,
        bucket_mass=projection.student_bucket_mass,
        bucket_edges=edges,
        bucket_shape_loss_weight=0.01,
    )
    first_only = cascaded_soft_labels_loss(
        student[:, :1, :],
        top_ids[:, :1, :],
        top_log_probs[:, :1, :],
        jnp.asarray([[1]], dtype=jnp.int32),
        tail_mass=projection.student_tail_mass[:, :1],
        bucket_mass=projection.student_bucket_mass[:, :1, :],
        bucket_edges=edges,
        bucket_shape_loss_weight=0.01,
    )

    np.testing.assert_allclose(float(masked.total_loss), float(first_only.total_loss))


def test_tail_and_bucket_shape_losses_are_finite() -> None:
    student = _student_logits()
    top_ids = _top_ids()
    edges = _bucket_edges()
    projection = project_student_tail_buckets(student, top_ids, edges)

    report = cascaded_soft_labels_loss(
        student,
        top_ids,
        _top_log_probs(dtype=jnp.float16),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=jnp.clip(projection.student_tail_mass * 0.5, 1e-4, 1.0),
        bucket_mass=projection.student_bucket_mass,
        bucket_edges=edges,
        tail_mass_loss_weight=0.01,
        bucket_shape_loss_weight=0.01,
    )

    assert report.total_loss.dtype == jnp.float32
    assert bool(jnp.isfinite(report.tail_loss))
    assert report.bucket_shape_loss is not None
    assert bool(jnp.isfinite(report.bucket_shape_loss))


def _student_logits() -> jnp.ndarray:
    return jnp.asarray([[[4.0, 3.0, 0.0, -2.0, -5.0]]], dtype=jnp.float32)


def _top_ids() -> jnp.ndarray:
    return jnp.asarray([[[0, 1]]], dtype=jnp.int32)


def _top_log_probs(dtype=jnp.float32) -> jnp.ndarray:
    return jnp.asarray([[[0.0, -1.0]]], dtype=dtype)


def _bucket_edges() -> jnp.ndarray:
    return jnp.asarray([1.0, 0.1, 0.01, 0.0], dtype=jnp.float32)
