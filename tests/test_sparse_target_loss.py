from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.distill.losses import (
    cascaded_soft_labels_loss,
    topk_tail_distillation_loss,
)


def test_topk_tail_loss_is_near_zero_when_head_logits_match_teacher_shape() -> None:
    teacher_top = jnp.asarray([[[-0.2, -1.4, -2.0]]], dtype=jnp.float32)
    student = jnp.zeros((1, 1, 6), dtype=jnp.float32)
    student = student.at[:, :, [3, 1, 5]].set(teacher_top + 7.0)

    report = topk_tail_distillation_loss(
        student,
        jnp.asarray([[[3, 1, 5]]], dtype=jnp.int32),
        teacher_top,
        jnp.ones((1, 1), dtype=jnp.int32),
    )

    assert report.target_type == "topk_with_tail_v0"
    assert report.distillation_loss_type == "topk_tail_head_kl"
    assert report.top_k == 3
    np.testing.assert_allclose(float(report.head_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(report.total_loss), 0.0, atol=1e-6)


def test_topk_tail_loss_is_positive_when_head_ranking_differs() -> None:
    report = topk_tail_distillation_loss(
        jnp.asarray([[[0.0, 3.0, 1.0, -2.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 1, 2]]], dtype=jnp.int32),
        jnp.asarray([[[0.0, -2.0, -3.0]]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.int32),
    )

    assert float(report.head_loss) > 0.0
    assert float(report.total_loss) > 0.0


def test_topk_tail_loss_attention_mask_excludes_masked_positions() -> None:
    student = jnp.asarray(
        [[[4.0, 1.0, 0.0], [0.0, 5.0, -1.0]]],
        dtype=jnp.float32,
    )
    teacher_top = jnp.asarray(
        [[[0.0, -3.0], [-10.0, 0.0]]],
        dtype=jnp.float32,
    )

    masked = topk_tail_distillation_loss(
        student,
        jnp.asarray([[[0, 1], [0, 1]]], dtype=jnp.int32),
        teacher_top,
        jnp.asarray([[1, 0]], dtype=jnp.int32),
    )
    first_only = topk_tail_distillation_loss(
        student[:, :1, :],
        jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        teacher_top[:, :1, :],
        jnp.asarray([[1]], dtype=jnp.int32),
    )

    np.testing.assert_allclose(float(masked.head_loss), float(first_only.head_loss))
    assert float(masked.token_count) == 1.0


def test_topk_tail_loss_accepts_float16_teacher_log_probs_but_computes_float32() -> (
    None
):
    report = topk_tail_distillation_loss(
        jnp.asarray([[[2.0, 1.0, 0.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float16),
        jnp.ones((1, 1), dtype=jnp.int32),
    )

    assert report.total_loss.dtype == jnp.float32


def test_tail_loss_weight_zero_does_not_affect_total() -> None:
    report = topk_tail_distillation_loss(
        jnp.asarray([[[4.0, 3.0, -4.0, -5.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=jnp.asarray([[0.9]], dtype=jnp.float32),
        tail_loss_weight=0.0,
    )

    np.testing.assert_allclose(float(report.total_loss), float(report.head_loss))
    assert float(report.tail_loss) == 0.0


def test_tail_loss_weight_changes_total_when_tail_mass_differs() -> None:
    base = topk_tail_distillation_loss(
        jnp.asarray([[[5.0, 4.0, 3.0, 3.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=jnp.asarray([[0.01]], dtype=jnp.float32),
        tail_loss_weight=0.0,
    )
    with_tail = topk_tail_distillation_loss(
        jnp.asarray([[[5.0, 4.0, 3.0, 3.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=jnp.asarray([[0.01]], dtype=jnp.float32),
        tail_loss_weight=0.5,
    )

    assert float(with_tail.tail_loss) > 0.0
    assert float(with_tail.total_loss) > float(base.total_loss)


def test_cascaded_bucket_shape_weight_defaults_to_zero() -> None:
    report = cascaded_soft_labels_loss(
        jnp.asarray([[[4.0, 3.0, 0.0, -2.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.int32),
        tail_mass=jnp.asarray([[0.1]], dtype=jnp.float32),
        bucket_mass=jnp.asarray([[[0.08, 0.015, 0.005]]], dtype=jnp.float32),
        bucket_edges=jnp.asarray([1.0, 0.1, 0.01, 0.0], dtype=jnp.float32),
    )

    assert report.bucket_shape_loss_weight == 0.0
    np.testing.assert_allclose(float(report.total_loss), float(report.head_loss))
