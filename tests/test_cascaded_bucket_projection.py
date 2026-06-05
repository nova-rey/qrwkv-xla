from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.distill.losses import project_student_tail_buckets


def test_student_bucket_projection_excludes_topk_ids() -> None:
    logits = jnp.asarray([[[6.0, 3.0, 1.0, -2.0, -4.0]]], dtype=jnp.float32)
    top_ids = jnp.asarray([[[0, 1]]], dtype=jnp.int32)
    edges = jnp.asarray([1.0, 0.1, 0.01, 0.0], dtype=jnp.float32)

    projection = project_student_tail_buckets(logits, top_ids, edges)
    probs = np.asarray(jnn.softmax(logits, axis=-1))[0, 0]

    np.testing.assert_allclose(
        float(projection.student_top_mass[0, 0]),
        float(probs[0] + probs[1]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(projection.student_bucket_mass)[0, 0].sum(),
        float(projection.student_tail_mass[0, 0]),
        rtol=1e-6,
    )


def test_student_bucket_count_sums_to_vocab_minus_topk() -> None:
    projection = project_student_tail_buckets(
        jnp.asarray([[[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]], dtype=jnp.float32),
        jnp.asarray([[[0, 2]]], dtype=jnp.int32),
        jnp.asarray([1.0, 0.2, 0.05, 0.0], dtype=jnp.float32),
    )

    assert int(jnp.sum(projection.student_bucket_count)) == 4


def test_bucket_edges_must_be_descending() -> None:
    with pytest.raises(ValueError, match="strictly descending"):
        project_student_tail_buckets(
            jnp.zeros((1, 1, 4), dtype=jnp.float32),
            jnp.asarray([[[0]]], dtype=jnp.int32),
            jnp.asarray([1.0, 0.1, 0.2, 0.0], dtype=jnp.float32),
        )


def test_projection_handles_empty_buckets() -> None:
    projection = project_student_tail_buckets(
        jnp.asarray([[[12.0, -12.0, -12.0, -12.0]]], dtype=jnp.float32),
        jnp.asarray([[[0]]], dtype=jnp.int32),
        jnp.asarray([1.0, 0.4, 0.2, 0.1, 0.0], dtype=jnp.float32),
    )

    assert np.any(np.asarray(projection.student_bucket_count)[0, 0] == 0)
    np.testing.assert_allclose(
        np.asarray(projection.student_bucket_mass)[0, 0].sum(),
        float(projection.student_tail_mass[0, 0]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_projection_handles_tiny_probabilities() -> None:
    projection = project_student_tail_buckets(
        jnp.asarray([[[30.0, 0.0, -20.0, -40.0]]], dtype=jnp.float32),
        jnp.asarray([[[0]]], dtype=jnp.int32),
        jnp.asarray([1.0, 1e-6, 1e-12, 0.0], dtype=jnp.float32),
    )

    assert bool(jnp.all(jnp.isfinite(projection.student_bucket_mass)))
    assert float(projection.student_tail_mass[0, 0]) > 0.0
