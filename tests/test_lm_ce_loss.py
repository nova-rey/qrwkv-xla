from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.lm import masked_next_token_cross_entropy


def test_masked_next_token_cross_entropy_matches_manual_nll() -> None:
    logits = jnp.asarray(
        [
            [
                [4.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        dtype=jnp.float32,
    )
    labels = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    label_mask = jnp.asarray([[True, True, False]])

    loss = masked_next_token_cross_entropy(
        logits=logits,
        labels=labels,
        label_mask=label_mask,
    )

    log_probs = np.asarray(jax.nn.log_softmax(logits, axis=-1))
    expected = -(log_probs[0, 0, 0] + log_probs[0, 1, 1]) / 2.0
    assert float(loss) == pytest.approx(float(expected))


def test_masked_next_token_cross_entropy_zero_for_empty_mask() -> None:
    loss = masked_next_token_cross_entropy(
        logits=jnp.zeros((1, 2, 3), dtype=jnp.float32),
        labels=jnp.zeros((1, 2), dtype=jnp.int32),
        label_mask=jnp.zeros((1, 2), dtype=bool),
    )

    assert float(loss) == pytest.approx(0.0)
