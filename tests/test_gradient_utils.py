from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.training.gradients import (
    clip_gradients_by_global_norm,
    global_gradient_norm,
)


def test_global_gradient_norm_handles_nested_tree() -> None:
    grads = {
        "a": jnp.asarray([3.0, 4.0]),
        "b": (jnp.asarray([12.0]),),
    }

    assert float(global_gradient_norm(grads)) == pytest.approx(13.0)


def test_global_gradient_norm_skips_non_floating_leaves() -> None:
    grads = {
        "float": jnp.asarray([3.0, 4.0]),
        "int": jnp.asarray([100], dtype=jnp.int32),
    }

    assert float(global_gradient_norm(grads)) == pytest.approx(5.0)


def test_clip_gradients_by_global_norm_scales_to_threshold() -> None:
    grads = {"x": jnp.asarray([3.0, 4.0], dtype=jnp.float32)}

    result = clip_gradients_by_global_norm(
        grads,
        max_grad_norm=1.0,
        epsilon=1e-6,
    )

    assert float(result.global_norm) == pytest.approx(5.0)
    assert float(result.clip_scale) == pytest.approx(1.0 / (5.0 + 1e-6))
    assert bool(result.was_clipped)
    np.testing.assert_allclose(
        np.asarray(result.gradients["x"]),
        np.asarray(grads["x"]) * float(result.clip_scale),
    )
    assert float(result.clipped_global_norm) <= 1.0


def test_clip_gradients_disabled_preserves_tree_identity() -> None:
    grads = {"x": jnp.asarray([3.0, 4.0], dtype=jnp.float32)}

    result = clip_gradients_by_global_norm(grads, max_grad_norm=None)

    assert result.gradients is grads
    assert float(result.global_norm) == pytest.approx(5.0)
    assert float(result.clipped_global_norm) == pytest.approx(5.0)
    assert float(result.clip_scale) == pytest.approx(1.0)
    assert not bool(result.was_clipped)
