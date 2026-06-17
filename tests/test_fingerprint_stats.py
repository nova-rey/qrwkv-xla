from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.training import (
    compute_fingerprint_distribution_stats,
    compute_fingerprint_distribution_stats_at_positions,
    select_position_logits,
)


def test_uniform_distribution_stats() -> None:
    stats = compute_fingerprint_distribution_stats(jnp.zeros((2, 4)))

    np.testing.assert_allclose(np.asarray(stats.entropy), np.log(4.0), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top1_margin), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top8_mass), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top32_mass), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.tail_mass), 0.0, atol=1e-6)


def test_peaked_distribution_is_finite_and_near_deterministic() -> None:
    logits = jnp.asarray([[10.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    stats = compute_fingerprint_distribution_stats(logits)
    expected_probs = _softmax_np(np.asarray(logits))

    assert np.isfinite(np.asarray(stats.entropy)).all()
    assert float(stats.entropy[0]) < 0.01
    np.testing.assert_allclose(
        np.asarray(stats.top1_margin),
        expected_probs[:, 0] - expected_probs[:, 1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(stats.top8_mass), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top32_mass), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.tail_mass), 0.0, atol=1e-6)


def test_known_probability_distribution_stats() -> None:
    probs = np.asarray([[0.5, 0.25, 0.15, 0.10]], dtype=np.float32)
    logits = jnp.asarray(np.log(probs))
    stats = compute_fingerprint_distribution_stats(logits)

    expected_entropy = -np.sum(probs * np.log(probs), axis=-1)
    np.testing.assert_allclose(np.asarray(stats.entropy), expected_entropy, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top1_margin), 0.25, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top8_mass), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.top32_mass), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.tail_mass), 0.0, atol=1e-6)


def test_topk_mass_with_vocab_larger_than_32() -> None:
    probs = np.arange(40, 0, -1, dtype=np.float32)
    probs = probs / probs.sum()
    logits = jnp.asarray(np.log(probs[None, :]))
    stats = compute_fingerprint_distribution_stats(logits)
    sorted_probs = np.sort(probs)[::-1]

    np.testing.assert_allclose(
        np.asarray(stats.top8_mass),
        np.asarray([sorted_probs[:8].sum()]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(stats.top32_mass),
        np.asarray([sorted_probs[:32].sum()]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(stats.tail_mass),
        np.asarray([1.0 - sorted_probs[:32].sum()]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_select_position_logits() -> None:
    logits = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape((2, 3, 4))
    selected = select_position_logits(logits, jnp.asarray([0, 2]))

    np.testing.assert_array_equal(np.asarray(selected[0]), np.asarray(logits[0, 0, :]))
    np.testing.assert_array_equal(np.asarray(selected[1]), np.asarray(logits[1, 2, :]))


def test_position_stats_wrapper_matches_manual_selection() -> None:
    logits = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape((2, 3, 5))
    positions = jnp.asarray([1, 2])

    manual = compute_fingerprint_distribution_stats(
        select_position_logits(logits, positions)
    )
    wrapped = compute_fingerprint_distribution_stats_at_positions(logits, positions)

    np.testing.assert_allclose(np.asarray(wrapped.entropy), np.asarray(manual.entropy))
    np.testing.assert_allclose(
        np.asarray(wrapped.top1_margin),
        np.asarray(manual.top1_margin),
    )
    np.testing.assert_allclose(
        np.asarray(wrapped.top8_mass),
        np.asarray(manual.top8_mass),
    )
    np.testing.assert_allclose(
        np.asarray(wrapped.top32_mass),
        np.asarray(manual.top32_mass),
    )
    np.testing.assert_allclose(
        np.asarray(wrapped.tail_mass),
        np.asarray(manual.tail_mass),
    )


def test_invalid_shape_errors() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        compute_fingerprint_distribution_stats(jnp.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="rank 3"):
        select_position_logits(jnp.zeros((2, 4)), jnp.asarray([0, 1]))
    with pytest.raises(ValueError, match="rank 1"):
        select_position_logits(jnp.zeros((2, 3, 4)), jnp.asarray([[0], [1]]))
    with pytest.raises(ValueError, match="positions length"):
        select_position_logits(jnp.zeros((2, 3, 4)), jnp.asarray([0]))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
