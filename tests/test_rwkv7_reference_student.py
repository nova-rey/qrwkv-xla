from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.students import RWKV7ReferenceConfig, RWKV7ReferenceStudent


def test_rwkv7_reference_student_params_initialize() -> None:
    student = _student()
    params = student.init_params(jax.random.PRNGKey(0))

    assert params["embedding"].shape == (17, 5)
    assert params["wr"].shape == (3, 5, 5)
    assert params["wk"].shape == (3, 5, 5)
    assert params["wv"].shape == (3, 5, 5)
    assert params["wg"].shape == (3, 5, 5)
    assert params["wo"].shape == (3, 5, 5)
    assert params["time_decay"].shape == (3, 5)
    assert params["time_bias"].shape == (3, 5)


def test_rwkv7_reference_student_forward_shape() -> None:
    student = _student()
    params = student.init_params(jax.random.PRNGKey(0))

    output = student.apply(params, _input_ids())

    assert output.hidden_states.shape == (2, 3, 4, 5)
    assert output.logits is None


def test_rwkv7_reference_student_same_seed_has_same_params_and_output() -> None:
    student = _student()
    input_ids = _input_ids()
    first_params = student.init_params(jax.random.PRNGKey(123))
    second_params = student.init_params(jax.random.PRNGKey(123))

    first_output = student.apply(first_params, input_ids)
    second_output = student.apply(second_params, input_ids)

    _assert_trees_equal(first_params, second_params)
    np.testing.assert_allclose(
        np.asarray(first_output.hidden_states),
        np.asarray(second_output.hidden_states),
    )


def test_rwkv7_reference_student_different_seed_changes_params_and_output() -> None:
    student = _student()
    input_ids = _input_ids()
    first_params = student.init_params(jax.random.PRNGKey(123))
    second_params = student.init_params(jax.random.PRNGKey(456))

    first_output = student.apply(first_params, input_ids)
    second_output = student.apply(second_params, input_ids)

    assert _trees_differ(first_params, second_params)
    assert not np.allclose(
        np.asarray(first_output.hidden_states),
        np.asarray(second_output.hidden_states),
    )


def test_rwkv7_reference_student_attention_mask_affects_output() -> None:
    student = _student()
    params = student.init_params(jax.random.PRNGKey(0))
    input_ids = _input_ids()
    attention_mask = jnp.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=jnp.int32)

    unmasked = student.apply(params, input_ids)
    masked = student.apply(params, input_ids, attention_mask=attention_mask)

    assert masked.hidden_states.shape == unmasked.hidden_states.shape
    assert not np.allclose(
        np.asarray(unmasked.hidden_states),
        np.asarray(masked.hidden_states),
    )
    masked_positions = (1 - attention_mask)[:, None, :, None]
    np.testing.assert_allclose(
        np.asarray(masked.hidden_states * masked_positions),
        np.zeros((2, 3, 4, 5), dtype=np.float32),
        atol=1e-6,
    )


def test_grad_flows_through_simple_hidden_mse() -> None:
    student = _student()
    params = student.init_params(jax.random.PRNGKey(0))
    target = jnp.ones((2, 3, 4, 5), dtype=jnp.float32) * 0.1

    def loss_fn(model_params: dict[str, jax.Array]) -> jax.Array:
        output = student.apply(model_params, _input_ids())
        return jnp.mean((output.hidden_states - target) ** 2)

    grads = jax.grad(loss_fn)(params)

    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    assert any(bool(jnp.any(leaf != 0.0)) for leaf in jax.tree_util.tree_leaves(grads))


def _student() -> RWKV7ReferenceStudent:
    return RWKV7ReferenceStudent(
        RWKV7ReferenceConfig(vocab_size=17, hidden_size=5, num_layers=3)
    )


def _input_ids() -> jax.Array:
    return jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)


def _assert_trees_equal(
    first: dict[str, jax.Array],
    second: dict[str, jax.Array],
) -> None:
    for name in first:
        np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]))


def _trees_differ(first: dict[str, jax.Array], second: dict[str, jax.Array]) -> bool:
    return any(
        not np.array_equal(np.asarray(first[name]), np.asarray(second[name]))
        for name in first
    )
