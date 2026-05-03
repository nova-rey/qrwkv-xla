from __future__ import annotations

import numpy as np


def numpy_rwkv7_reference_layer(
    inputs: np.ndarray,
    *,
    wr: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wg: np.ndarray,
    wo: np.ndarray,
    time_decay: np.ndarray,
    time_bias: np.ndarray,
    attention_mask: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy mirror of the current simplified JAX reference layer."""
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"inputs must have shape [B,S,H], got {x.shape}")

    batch_size, sequence_length, hidden_size = x.shape
    if attention_mask is None:
        mask = np.ones((batch_size, sequence_length, 1), dtype=np.float32)
    else:
        mask = np.asarray(attention_mask, dtype=np.float32)[:, :, None]

    state = (
        np.zeros((batch_size, hidden_size), dtype=np.float32)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float32).copy()
    )
    decay = _sigmoid(np.asarray(time_decay, dtype=np.float32))[None, :]
    bias = np.asarray(time_bias, dtype=np.float32)[None, :]
    outputs = np.zeros_like(x)

    for token_index in range(sequence_length):
        token = x[:, token_index, :]
        token_mask = mask[:, token_index, :]
        receptance = _sigmoid(token @ wr + bias)
        key = token @ wk
        value = token @ wv
        gate = _sigmoid(token @ wg)
        proposed_state = decay * state + key * value
        state = np.where(token_mask > 0, proposed_state, state)
        mixer = (receptance * state * gate) @ wo
        outputs[:, token_index, :] = np.tanh(token + mixer) * token_mask

    return outputs, state


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))
