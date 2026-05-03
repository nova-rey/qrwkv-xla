from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from qrwkv_xla.distributed.reductions import metrics_pmean, tree_pmean


def test_tree_pmean_wraps_lax_pmean() -> None:
    with patch(
        "jax.lax.pmean", side_effect=lambda value, axis_name: value
    ) as mock_pmean:
        tree = {"w": jnp.ones((2, 2))}
        reduced = tree_pmean(tree, axis_name="data")

    assert reduced["w"].shape == (2, 2)
    mock_pmean.assert_called()


def test_metrics_pmean_wraps_lax_pmean() -> None:
    with patch(
        "jax.lax.pmean", side_effect=lambda value, axis_name: value
    ) as mock_pmean:
        metrics = metrics_pmean({"loss": jnp.asarray(1.0)}, axis_name="data")

    assert float(metrics["loss"]) == 1.0
    mock_pmean.assert_called_once()


@pytest.mark.skipif(
    jax.local_device_count() < 2, reason="requires >=2 local JAX devices"
)
def test_metrics_pmean_runs_inside_pmap() -> None:
    @jax.pmap(axis_name="data")
    def reduce_loss(x):
        return metrics_pmean({"loss": x}, axis_name="data")["loss"]

    values = jnp.arange(jax.local_device_count(), dtype=jnp.float32)
    reduced = reduce_loss(values)
    expected = jnp.mean(values)
    assert jnp.allclose(reduced, expected)
