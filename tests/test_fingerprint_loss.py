from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.artifacts import FingerprintBatch
from qrwkv_xla.training import (
    FingerprintCorridorLossConfig,
    FingerprintDistributionStats,
    compute_fingerprint_corridor_loss,
    inside_bounds,
    squared_hinge_bound_penalty,
)


def test_bound_penalty_is_zero_inside_inclusive_bounds() -> None:
    values = jnp.asarray([0.5, 1.0, 1.5])
    lower = jnp.asarray([0.0, 1.0, 1.0])
    upper = jnp.asarray([1.0, 2.0, 1.5])

    np.testing.assert_allclose(
        np.asarray(squared_hinge_bound_penalty(values, lower, upper)),
        np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(inside_bounds(values, lower, upper)),
        np.asarray([True, True, True]),
    )


def test_bound_penalty_outside_bounds() -> None:
    values = jnp.asarray([-1.0, 3.0, 1.5])
    lower = jnp.asarray([0.0, 0.0, 0.0])
    upper = jnp.asarray([1.0, 1.0, 1.0])

    np.testing.assert_allclose(
        np.asarray(squared_hinge_bound_penalty(values, lower, upper)),
        np.asarray([1.0, 4.0, 0.25], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(inside_bounds(values, lower, upper)),
        np.asarray([False, False, False]),
    )


def test_full_loss_zero_when_all_stats_inside() -> None:
    stats = _stats(
        entropy=[0.5, 0.7],
        top1_margin=[0.2, 0.3],
        top8_mass=[0.8, 0.9],
        top32_mass=[0.95, 0.97],
        tail_mass=[0.05, 0.03],
    )
    batch = _batch(size=2)

    output = compute_fingerprint_corridor_loss(stats, batch)

    np.testing.assert_allclose(np.asarray(output.loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.entropy_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.top1_margin_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.top8_mass_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.top32_mass_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.tail_mass_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.entropy_inside_rate), 1.0)
    np.testing.assert_allclose(np.asarray(output.top1_margin_inside_rate), 1.0)
    np.testing.assert_allclose(np.asarray(output.top8_mass_inside_rate), 1.0)
    np.testing.assert_allclose(np.asarray(output.top32_mass_inside_rate), 1.0)
    np.testing.assert_allclose(np.asarray(output.tail_mass_inside_rate), 1.0)
    np.testing.assert_allclose(np.asarray(output.all_inside_rate), 1.0)


def test_full_loss_nonzero_when_stats_violate_bounds() -> None:
    stats = _stats(
        entropy=[-0.5, 0.5, 0.5],
        top1_margin=[0.2, 0.2, 0.7],
        top8_mass=[0.8, 0.2, 1.2],
        top32_mass=[0.95, 0.95, 0.95],
        tail_mass=[0.05, 0.05, 0.5],
    )
    batch = _batch(size=3)

    output = compute_fingerprint_corridor_loss(stats, batch)

    assert float(output.loss) > 0.0
    assert float(output.entropy_loss) > 0.0
    assert float(output.top8_mass_loss) > 0.0
    assert float(output.top1_margin_loss) > 0.0
    assert float(output.tail_mass_loss) > 0.0
    np.testing.assert_allclose(np.asarray(output.top32_mass_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.entropy_inside_rate), 2.0 / 3.0)
    np.testing.assert_allclose(np.asarray(output.top8_mass_inside_rate), 1.0 / 3.0)
    np.testing.assert_allclose(np.asarray(output.all_inside_rate), 0.0)


def test_config_weights_disable_and_scale_stat_losses() -> None:
    stats = _stats(
        entropy=[-1.0, -1.0],
        top1_margin=[0.2, 0.2],
        top8_mass=[0.2, 0.2],
        top32_mass=[0.95, 0.95],
        tail_mass=[0.05, 0.05],
    )
    batch = _batch(size=2)

    output = compute_fingerprint_corridor_loss(
        stats,
        batch,
        FingerprintCorridorLossConfig(
            entropy_weight=0.0,
            top1_margin_weight=0.0,
            top8_mass_weight=2.0,
            top32_mass_weight=0.0,
            tail_mass_weight=0.0,
        ),
    )

    np.testing.assert_allclose(np.asarray(output.entropy_loss), 0.0)
    np.testing.assert_allclose(np.asarray(output.top8_mass_loss), 0.18, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(output.loss), 0.18, rtol=1e-6)


def test_record_weights_and_unweighted_mean() -> None:
    stats = _stats(
        entropy=[-1.0, -1.0],
        top1_margin=[0.2, 0.2],
        top8_mass=[0.8, 0.8],
        top32_mass=[0.95, 0.95],
        tail_mass=[0.05, 0.05],
    )
    weighted_batch = _batch(size=2, weight=[1.0, 3.0])
    unweighted_batch = _batch(size=2, weight=[1.0, 3.0])
    stats_with_unequal_penalty = _stats(
        entropy=[-1.0, -2.0],
        top1_margin=[0.2, 0.2],
        top8_mass=[0.8, 0.8],
        top32_mass=[0.95, 0.95],
        tail_mass=[0.05, 0.05],
    )

    weighted = compute_fingerprint_corridor_loss(
        stats_with_unequal_penalty,
        weighted_batch,
        FingerprintCorridorLossConfig(
            top1_margin_weight=0.0,
            top8_mass_weight=0.0,
            top32_mass_weight=0.0,
            tail_mass_weight=0.0,
        ),
    )
    unweighted = compute_fingerprint_corridor_loss(
        stats_with_unequal_penalty,
        unweighted_batch,
        FingerprintCorridorLossConfig(
            top1_margin_weight=0.0,
            top8_mass_weight=0.0,
            top32_mass_weight=0.0,
            tail_mass_weight=0.0,
            use_record_weights=False,
        ),
    )

    np.testing.assert_allclose(np.asarray(weighted.loss), (1.0 + 12.0) / 4.0)
    np.testing.assert_allclose(np.asarray(unweighted.loss), (1.0 + 4.0) / 2.0)
    np.testing.assert_allclose(np.asarray(weighted.mean_weight), 2.0)
    np.testing.assert_allclose(
        np.asarray(
            compute_fingerprint_corridor_loss(
                stats,
                weighted_batch,
                FingerprintCorridorLossConfig(
                    top1_margin_weight=0.0,
                    top8_mass_weight=0.0,
                    top32_mass_weight=0.0,
                    tail_mass_weight=0.0,
                ),
            ).loss
        ),
        1.0,
    )


def test_bound_primitives_are_jittable() -> None:
    fn = jax.jit(lambda x: squared_hinge_bound_penalty(x, x * 0.0, x * 0.0))
    np.testing.assert_allclose(
        np.asarray(fn(jnp.asarray([0.0, 2.0]))),
        np.asarray([0.0, 4.0], dtype=np.float32),
    )


def test_shape_errors() -> None:
    with pytest.raises(ValueError, match="rank 1"):
        squared_hinge_bound_penalty(
            jnp.zeros((2, 1)),
            jnp.zeros((2, 1)),
            jnp.ones((2, 1)),
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        inside_bounds(jnp.zeros((2,)), jnp.zeros((3,)), jnp.ones((2,)))
    with pytest.raises(ValueError, match="batch dimension mismatch"):
        compute_fingerprint_corridor_loss(_stats(entropy=[0.5, 0.5]), _batch(size=3))


def _stats(
    *,
    entropy,
    top1_margin=None,
    top8_mass=None,
    top32_mass=None,
    tail_mass=None,
) -> FingerprintDistributionStats:
    size = len(entropy)
    return FingerprintDistributionStats(
        entropy=jnp.asarray(entropy, dtype=jnp.float32),
        top1_margin=jnp.asarray(top1_margin or [0.2] * size, dtype=jnp.float32),
        top8_mass=jnp.asarray(top8_mass or [0.8] * size, dtype=jnp.float32),
        top32_mass=jnp.asarray(top32_mass or [0.95] * size, dtype=jnp.float32),
        tail_mass=jnp.asarray(tail_mass or [0.05] * size, dtype=jnp.float32),
    )


def _batch(*, size: int, weight=None) -> FingerprintBatch:
    return FingerprintBatch(
        input_ids=np.zeros((size, 4), dtype=np.int32),
        position=np.arange(size, dtype=np.int32),
        mode_id=np.zeros((size,), dtype=np.int32),
        entropy_min=np.zeros((size,), dtype=np.float32),
        entropy_max=np.ones((size,), dtype=np.float32),
        top1_margin_min=np.zeros((size,), dtype=np.float32),
        top1_margin_max=np.full((size,), 0.5, dtype=np.float32),
        top8_mass_min=np.full((size,), 0.5, dtype=np.float32),
        top8_mass_max=np.ones((size,), dtype=np.float32),
        top32_mass_min=np.full((size,), 0.9, dtype=np.float32),
        top32_mass_max=np.ones((size,), dtype=np.float32),
        tail_mass_min=np.zeros((size,), dtype=np.float32),
        tail_mass_max=np.full((size,), 0.1, dtype=np.float32),
        weight=np.asarray(weight or [1.0] * size, dtype=np.float32),
    )
