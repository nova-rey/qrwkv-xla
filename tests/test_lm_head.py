from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.students.lm_head import apply_lm_head, init_lm_head_params


def test_lm_head_init_shapes() -> None:
    params = init_lm_head_params(
        jax.random.PRNGKey(0),
        hidden_size=4,
        vocab_size=7,
    )

    assert params["weight"].shape == (4, 7)
    assert params["bias"].shape == (7,)


def test_lm_head_apply_shape() -> None:
    params = init_lm_head_params(
        jax.random.PRNGKey(0),
        hidden_size=4,
        vocab_size=7,
    )
    logits = apply_lm_head(jnp.ones((2, 3, 4)), params)

    assert logits.shape == (2, 3, 7)


def test_lm_head_same_seed_is_deterministic() -> None:
    first = init_lm_head_params(
        jax.random.PRNGKey(123),
        hidden_size=4,
        vocab_size=7,
    )
    second = init_lm_head_params(
        jax.random.PRNGKey(123),
        hidden_size=4,
        vocab_size=7,
    )

    np.testing.assert_array_equal(
        np.asarray(first["weight"]),
        np.asarray(second["weight"]),
    )
    np.testing.assert_array_equal(np.asarray(first["bias"]), np.asarray(second["bias"]))


def test_lm_head_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        init_lm_head_params(jax.random.PRNGKey(0), hidden_size=0, vocab_size=7)
    with pytest.raises(ValueError, match="vocab_size"):
        init_lm_head_params(jax.random.PRNGKey(0), hidden_size=4, vocab_size=0)
    with pytest.raises(ValueError, match=r"\[B,S,H\]"):
        apply_lm_head(
            jnp.ones((2, 4)),
            init_lm_head_params(jax.random.PRNGKey(0), hidden_size=4, vocab_size=7),
        )
