from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import FingerprintExemplarBatch
from qrwkv_xla.training import (
    FingerprintExemplarLossConfig,
    compute_fingerprint_exemplar_loss,
    compute_fingerprint_exemplar_loss_at_positions,
)


def test_exemplar_loss_zero_for_matching_uniform_distribution() -> None:
    batch = _batch(
        teacher_probs=np.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    )

    output = compute_fingerprint_exemplar_loss(jnp.zeros((1, 4)), batch)

    np.testing.assert_allclose(output.loss, 0.0, atol=1e-6)
    np.testing.assert_allclose(output.kl_loss, 0.0, atol=1e-6)
    np.testing.assert_allclose(output.cross_entropy, output.entropy, atol=1e-6)


def test_exemplar_loss_one_hot_matches_negative_log_probability() -> None:
    batch = _batch(
        teacher_probs=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    )
    logits = jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32)
    expected = -jax_log_softmax(logits)[0, 0]

    output = compute_fingerprint_exemplar_loss(logits, batch)

    np.testing.assert_allclose(output.loss, expected, rtol=1e-6)
    np.testing.assert_allclose(output.cross_entropy, expected, rtol=1e-6)
    np.testing.assert_allclose(output.entropy, 0.0, atol=1e-6)


def test_exemplar_loss_weights_records() -> None:
    teacher_probs = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    batch = _batch(teacher_probs=teacher_probs, weight=np.asarray([1.0, 3.0]))
    logits = jnp.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    per_record = jnp.asarray(
        [-jax_log_softmax(logits)[0, 0], -jax_log_softmax(logits)[1, 1]]
    )
    expected_weighted = (per_record[0] + 3.0 * per_record[1]) / 4.0
    expected_unweighted = jnp.mean(per_record)

    weighted = compute_fingerprint_exemplar_loss(logits, batch)
    unweighted = compute_fingerprint_exemplar_loss(
        logits,
        batch,
        FingerprintExemplarLossConfig(use_record_weights=False),
    )

    np.testing.assert_allclose(weighted.loss, expected_weighted, rtol=1e-6)
    np.testing.assert_allclose(weighted.mean_weight, 2.0, rtol=1e-6)
    np.testing.assert_allclose(unweighted.loss, expected_unweighted, rtol=1e-6)


def test_exemplar_loss_at_positions_selects_batch_positions() -> None:
    batch = _batch(
        teacher_probs=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        position=np.asarray([1, 0], dtype=np.int32),
    )
    logits = jnp.asarray(
        [
            [[0.0, 5.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    selected = jnp.asarray(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    wrapped = compute_fingerprint_exemplar_loss_at_positions(logits, batch)
    direct = compute_fingerprint_exemplar_loss(selected, batch)

    np.testing.assert_allclose(wrapped.loss, direct.loss, rtol=1e-6)


def jax_log_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))


def _batch(
    *,
    teacher_probs: np.ndarray,
    weight: np.ndarray | None = None,
    position: np.ndarray | None = None,
) -> FingerprintExemplarBatch:
    batch_size, vocab_size = teacher_probs.shape
    seq_len = 3
    return FingerprintExemplarBatch(
        input_ids=np.zeros((batch_size, seq_len), dtype=np.int32),
        position=(
            np.zeros((batch_size,), dtype=np.int32)
            if position is None
            else position.astype(np.int32)
        ),
        teacher_probs=teacher_probs.astype(np.float32),
        weight=(
            np.ones((batch_size,), dtype=np.float32)
            if weight is None
            else weight.astype(np.float32)
        ),
        mode_id=np.full((batch_size,), -1, dtype=np.int32),
        interestingness_score=np.full((batch_size,), np.nan, dtype=np.float32),
        reason_codes=tuple(() for _ in range(batch_size)),
        example_id=tuple(f"loss{index}" for index in range(batch_size)),
    )
