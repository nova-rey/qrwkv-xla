from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from qrwkv_xla.distill.losses import logits_kl_loss
from qrwkv_xla.distill.target_dispatch import (
    UnsupportedTeacherTargetType,
    dispatch_teacher_target_loss,
)


def test_dense_logits_dispatch_calls_dense_kl_path() -> None:
    student = jnp.asarray([[[0.0, 1.0, 2.0]]], dtype=jnp.float32)
    teacher = jnp.asarray([[[0.1, 0.8, 2.2]]], dtype=jnp.float32)
    mask = jnp.ones((1, 1), dtype=jnp.int32)

    report = dispatch_teacher_target_loss(
        student_logits=student,
        target_batch=SimpleNamespace(
            target_type="dense_logits",
            teacher_logits=teacher,
            attention_mask=mask,
        ),
    )

    assert report.target_type == "dense_logits"
    assert report.distillation_loss_type == "dense_logits_kl"
    assert float(report.total_loss) == float(logits_kl_loss(student, teacher, mask))


def test_topk_with_tail_dispatch_calls_sparse_path() -> None:
    report = dispatch_teacher_target_loss(
        student_logits=jnp.asarray([[[3.0, 1.0, 0.0]]], dtype=jnp.float32),
        target_batch=SimpleNamespace(
            target_type="topk_with_tail_v0",
            attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
            top_token_ids=jnp.asarray([[[0, 1]]], dtype=jnp.int32),
            top_log_probs=jnp.asarray([[[0.0, -2.0]]], dtype=jnp.float32),
            top_mass=jnp.asarray([[0.9]], dtype=jnp.float32),
            tail_mass=jnp.asarray([[0.1]], dtype=jnp.float32),
            teacher_entropy=jnp.asarray([[0.5]], dtype=jnp.float32),
        ),
    )

    assert report.target_type == "topk_with_tail_v0"
    assert report.distillation_loss_type == "topk_tail_head_kl"
    assert report.top_k == 2
    assert report.mean_tail_mass is not None


def test_cascaded_soft_labels_dispatch_calls_cascaded_path() -> None:
    report = dispatch_teacher_target_loss(
        student_logits=jnp.asarray([[[4.0, 3.0, 0.0, -2.0]]], dtype=jnp.float32),
        target_batch=SimpleNamespace(
            target_type="cascaded_soft_labels_v1",
            attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
            top_token_ids=jnp.asarray([[[0, 1]]], dtype=jnp.int32),
            top_log_probs=jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
            top_mass=jnp.asarray([[0.9]], dtype=jnp.float32),
            tail_mass=jnp.asarray([[0.1]], dtype=jnp.float32),
            teacher_entropy=jnp.asarray([[0.5]], dtype=jnp.float32),
            bucket_edges=jnp.asarray([1.0, 0.1, 0.01, 0.0], dtype=jnp.float32),
            bucket_mass=jnp.asarray([[[0.08, 0.015, 0.005]]], dtype=jnp.float32),
        ),
        bucket_shape_loss_weight=0.01,
    )

    assert report.target_type == "cascaded_soft_labels_v1"
    assert report.distillation_loss_type == "cascaded_soft_labels"
    assert report.bucket_shape_loss is not None
    assert report.bucket_loss_weight == 0.01
    assert report.bucket_count == 3


def test_unsupported_target_type_fails_clearly() -> None:
    with pytest.raises(
        UnsupportedTeacherTargetType, match="unsupported teacher target_type"
    ):
        dispatch_teacher_target_loss(
            student_logits=jnp.zeros((1, 1, 3), dtype=jnp.float32),
            target_batch=SimpleNamespace(target_type="bucket_tail_v1"),
        )


def test_missing_compressed_fields_fail_clearly() -> None:
    with pytest.raises(ValueError, match="missing compressed field"):
        dispatch_teacher_target_loss(
            student_logits=jnp.zeros((1, 1, 3), dtype=jnp.float32),
            target_batch=SimpleNamespace(
                target_type="topk_with_tail_v0",
                attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                top_token_ids=jnp.asarray([[[0, 1]]], dtype=jnp.int32),
            ),
        )


def test_cascaded_missing_bucket_fields_fail_clearly() -> None:
    with pytest.raises(ValueError, match="bucket_mass"):
        dispatch_teacher_target_loss(
            student_logits=jnp.zeros((1, 1, 4), dtype=jnp.float32),
            target_batch=SimpleNamespace(
                target_type="cascaded_soft_labels_v1",
                attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                top_token_ids=jnp.asarray([[[0, 1]]], dtype=jnp.int32),
                top_log_probs=jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
                tail_mass=jnp.asarray([[0.1]], dtype=jnp.float32),
                bucket_edges=jnp.asarray([1.0, 0.1, 0.0], dtype=jnp.float32),
            ),
            bucket_shape_loss_weight=0.01,
        )


def test_top_token_ids_outside_student_vocab_fails_clearly() -> None:
    with pytest.raises(ValueError, match="outside student vocab range"):
        dispatch_teacher_target_loss(
            student_logits=jnp.zeros((1, 1, 3), dtype=jnp.float32),
            target_batch=SimpleNamespace(
                target_type="topk_with_tail_v0",
                attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                top_token_ids=jnp.asarray([[[0, 4]]], dtype=jnp.int32),
                top_log_probs=jnp.asarray([[[0.0, -1.0]]], dtype=jnp.float32),
                tail_mass=jnp.asarray([[0.1]], dtype=jnp.float32),
            ),
        )
