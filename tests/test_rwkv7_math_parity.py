from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.optimizers import (
    OptimizerConfig,
    init_optimizer_state,
    optimizer_update,
)
from qrwkv_xla.students import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
    rwkv7_reference_layer,
)
from qrwkv_xla.students.rwkv7_reference_parity_harness import (
    numpy_rwkv7_reference_layer,
)


def test_jax_layer_matches_local_numpy_harness_with_mask_and_state() -> None:
    inputs = _inputs()
    params = _layer_params(3)
    attention_mask = jnp.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=jnp.int32)
    initial_state = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 50.0

    outputs, final_state = rwkv7_reference_layer(
        inputs,
        **params,
        attention_mask=attention_mask,
        initial_state=initial_state,
        return_state=True,
    )
    expected_outputs, expected_state = numpy_rwkv7_reference_layer(
        np.asarray(inputs),
        **{name: np.asarray(value) for name, value in params.items()},
        attention_mask=np.asarray(attention_mask),
        initial_state=np.asarray(initial_state),
    )

    np.testing.assert_allclose(np.asarray(outputs), expected_outputs, atol=1e-6)
    np.testing.assert_allclose(np.asarray(final_state), expected_state, atol=1e-6)


def test_all_at_once_matches_token_by_token_final_state() -> None:
    inputs = _inputs()
    params = _layer_params(3)
    attention_mask = jnp.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=jnp.int32)

    full_outputs, full_state = rwkv7_reference_layer(
        inputs,
        **params,
        attention_mask=attention_mask,
        return_state=True,
    )
    state = None
    token_outputs = []
    for token_index in range(inputs.shape[1]):
        token_output, state = rwkv7_reference_layer(
            inputs[:, token_index : token_index + 1, :],
            **params,
            attention_mask=attention_mask[:, token_index : token_index + 1],
            initial_state=state,
            return_state=True,
        )
        token_outputs.append(token_output)

    np.testing.assert_allclose(
        np.asarray(full_outputs),
        np.asarray(jnp.concatenate(token_outputs, axis=1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(full_state), np.asarray(state), atol=1e-6)


def test_batched_matches_unbatched_rows() -> None:
    inputs = _inputs()
    params = _layer_params(3)
    attention_mask = jnp.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=jnp.int32)

    batched_outputs, batched_state = rwkv7_reference_layer(
        inputs,
        **params,
        attention_mask=attention_mask,
        return_state=True,
    )

    row_outputs = []
    row_states = []
    for batch_index in range(inputs.shape[0]):
        output, state = rwkv7_reference_layer(
            inputs[batch_index : batch_index + 1],
            **params,
            attention_mask=attention_mask[batch_index : batch_index + 1],
            return_state=True,
        )
        row_outputs.append(output)
        row_states.append(state)

    np.testing.assert_allclose(
        np.asarray(batched_outputs),
        np.asarray(jnp.concatenate(row_outputs, axis=0)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(batched_state),
        np.asarray(jnp.concatenate(row_states, axis=0)),
        atol=1e-6,
    )


def test_eager_matches_jit_with_final_state() -> None:
    inputs = _inputs()
    params = _layer_params(3)
    attention_mask = jnp.array([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=jnp.int32)

    eager_outputs, eager_state = rwkv7_reference_layer(
        inputs,
        **params,
        attention_mask=attention_mask,
        return_state=True,
    )
    run_jit = jax.jit(
        lambda x, mask: rwkv7_reference_layer(
            x,
            **params,
            attention_mask=mask,
            return_state=True,
        )
    )
    jit_outputs, jit_state = run_jit(inputs, attention_mask)

    np.testing.assert_allclose(np.asarray(eager_outputs), np.asarray(jit_outputs))
    np.testing.assert_allclose(np.asarray(eager_state), np.asarray(jit_state))


def test_student_has_finite_grads_and_tiny_optimizer_step_has_no_nans() -> None:
    student = RWKV7ReferenceStudent(
        RWKV7ReferenceConfig(vocab_size=19, hidden_size=4, num_layers=2)
    )
    params = student.init_params(jax.random.PRNGKey(7))
    input_ids = jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32)
    target = jnp.zeros((2, 2, 3, 4), dtype=jnp.float32)

    def loss_fn(model_params: dict[str, jax.Array]) -> jax.Array:
        output = student.apply(model_params, input_ids)
        return jnp.mean((output.hidden_states - target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    opt_config = OptimizerConfig(type="sgd", learning_rate=1e-3)
    opt_state = init_optimizer_state(params, opt_config)
    updated_params, _, _ = optimizer_update(params, grads, opt_state, opt_config)
    updated_loss = loss_fn(updated_params)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(updated_loss)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(updated_params)
    )


def _inputs() -> jax.Array:
    return jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 20.0 - 0.3


def _layer_params(hidden_size: int) -> dict[str, jax.Array]:
    base = jnp.eye(hidden_size, dtype=jnp.float32)
    return {
        "wr": base * 0.2 + 0.01,
        "wk": base * 0.3 - 0.02,
        "wv": base * -0.25 + 0.04,
        "wg": base * 0.15 + 0.03,
        "wo": base * 0.4 - 0.01,
        "time_decay": jnp.linspace(-0.4, 0.2, hidden_size, dtype=jnp.float32),
        "time_bias": jnp.linspace(0.1, 0.3, hidden_size, dtype=jnp.float32),
    }
