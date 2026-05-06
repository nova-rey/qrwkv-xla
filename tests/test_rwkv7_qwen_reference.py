from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.students import (
    RWKV7QwenReferenceConfig,
    RWKV7QwenReferenceState,
    RWKV7QwenReferenceStudent,
    rwkv7_qwen_reference_group_kv,
    rwkv7_qwen_reference_rope,
)


def test_qwen_reference_param_shapes_are_nested_and_named() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(0))

    assert params["token_embedding"]["weight"].shape == (17, 4)
    layers = params["layers"]
    assert layers["input_layernorm"]["weight"].shape == (2, 4)
    assert layers["post_attention_layernorm"]["weight"].shape == (2, 4)
    attn = layers["self_attn"]
    assert attn["q_proj"]["weight"].shape == (2, 4, 4)
    assert attn["k_proj"]["weight"].shape == (2, 4, 2)
    assert attn["v_proj"]["weight"].shape == (2, 4, 2)
    assert attn["o_proj"]["weight"].shape == (2, 4, 4)
    assert attn["time_bias"].shape == (2, 4)
    assert attn["time_mix"].shape == (2, 4)
    mlp = layers["mlp"]
    assert mlp["gate_proj"]["weight"].shape == (2, 4, 16)
    assert mlp["up_proj"]["weight"].shape == (2, 4, 16)
    assert mlp["down_proj"]["weight"].shape == (2, 16, 4)
    assert params["final_layernorm"]["weight"].shape == (4,)
    assert params["lm_head"]["weight"].shape == (4, 17)
    assert _tree_is_finite(params)


def test_qwen_reference_invalid_head_config_raises() -> None:
    with pytest.raises(ValueError, match="num_heads must be divisible"):
        RWKV7QwenReferenceConfig(hidden_size=12, num_heads=3, num_kv_heads=2)
    with pytest.raises(ValueError, match="head_size must be even"):
        RWKV7QwenReferenceConfig(hidden_size=6, num_heads=2, num_kv_heads=1)


def test_qwen_reference_rope_shape_and_determinism() -> None:
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4)
    positions = jnp.array([0, 3], dtype=jnp.int32)

    first = rwkv7_qwen_reference_rope(x, positions)
    second = rwkv7_qwen_reference_rope(x, positions)

    assert first.shape == x.shape
    assert jnp.allclose(first, second)
    assert jnp.allclose(first[0], x[0])
    assert not jnp.allclose(first[1], x[1])


def test_qwen_reference_grouped_kv_equal_heads() -> None:
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4)
    grouped = rwkv7_qwen_reference_group_kv(x, num_heads=2)

    assert grouped.shape == (1, 2, 4)
    assert jnp.allclose(grouped, x)


def test_qwen_reference_grouped_kv_repeated_heads() -> None:
    x = jnp.array([[[1.0, 2.0]]], dtype=jnp.float32)
    grouped = rwkv7_qwen_reference_group_kv(x, num_heads=4)

    assert grouped.shape == (1, 4, 2)
    assert jnp.allclose(grouped[:, 0], x[:, 0])
    assert jnp.allclose(grouped[:, 3], x[:, 0])


def test_qwen_reference_state_shapes_and_position_tracking() -> None:
    student = _student()
    state = student.init_state(batch_size=3, next_position=5)

    assert isinstance(state, RWKV7QwenReferenceState)
    assert state.wkv_matrix_state.shape == (2, 3, 2, 2, 2)
    assert state.shift_state.shape == (2, 3, 4)
    assert int(state.next_position) == 5


def test_qwen_reference_forward_shape() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(1))
    output, state = student.apply_with_state(
        params,
        jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32),
        attention_mask=jnp.array([[1, 1, 1], [1, 1, 0]], dtype=jnp.int32),
    )

    assert output.hidden_states.shape == (2, 2, 3, 4)
    assert output.logits is not None
    assert output.logits.shape == (2, 3, 17)
    assert state.wkv_matrix_state.shape == (2, 2, 2, 2, 2)
    assert state.shift_state.shape == (2, 2, 4)
    assert int(state.next_position) == 3


def test_qwen_reference_full_vs_stepwise_equivalence_for_outputs_and_state() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(2))
    input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)

    full_output, full_state = student.apply_with_state(
        params,
        input_ids,
        attention_mask=attention_mask,
    )

    state = student.init_state(batch_size=1)
    logits_steps = []
    hidden_steps = []
    for index in range(input_ids.shape[1]):
        token_output, state = student.step(
            params,
            input_ids[:, index : index + 1],
            state,
            attention_mask=attention_mask[:, index : index + 1],
        )
        assert token_output.logits is not None
        logits_steps.append(token_output.logits)
        hidden_steps.append(token_output.hidden_states)

    assert full_output.logits is not None
    step_logits = jnp.concatenate(logits_steps, axis=1)
    step_hidden = jnp.concatenate(hidden_steps, axis=2)
    assert jnp.max(jnp.abs(full_output.logits - step_logits)) < 1e-5
    assert jnp.max(jnp.abs(full_output.hidden_states - step_hidden)) < 1e-5
    assert jnp.max(jnp.abs(full_state.wkv_matrix_state - state.wkv_matrix_state)) < 1e-5
    assert jnp.max(jnp.abs(full_state.shift_state - state.shift_state)) < 1e-5
    assert int(full_state.next_position) == int(state.next_position) == 4


def test_qwen_reference_jit_forward_runs() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(3))
    run = jax.jit(lambda ids: student.apply_with_state(params, ids)[0].logits)
    logits = run(jnp.array([[1, 2, 3]], dtype=jnp.int32))

    assert logits is not None
    assert logits.shape == (1, 3, 17)


def test_qwen_reference_gradients_are_finite() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(4))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    def loss_fn(model_params):
        output = student.apply(model_params, input_ids)
        assert output.logits is not None
        return jnp.mean(output.logits**2) + jnp.mean(output.hidden_states**2)

    grads = jax.grad(loss_fn)(params)
    assert _tree_is_finite(grads)


def test_qwen_reference_optimizer_step_keeps_params_finite() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(5))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    def loss_fn(model_params):
        output = student.apply(model_params, input_ids)
        assert output.logits is not None
        return jnp.mean(output.logits**2)

    grads = jax.grad(loss_fn)(params)
    optimizer_config = OptimizerConfig(type="sgd", learning_rate=0.01)
    optimizer_state = init_optimizer_state(params, optimizer_config)
    new_params, _optimizer_state, metrics = optimizer_update(
        params,
        grads,
        optimizer_state,
        optimizer_config,
    )

    assert math.isfinite(float(metrics["learning_rate"]))
    assert _tree_is_finite(new_params)


def _student(*, emit_logits: bool = False) -> RWKV7QwenReferenceStudent:
    return RWKV7QwenReferenceStudent(
        RWKV7QwenReferenceConfig(
            vocab_size=17,
            hidden_size=4,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            emit_logits=emit_logits,
        )
    )


def _tree_is_finite(tree: object) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
