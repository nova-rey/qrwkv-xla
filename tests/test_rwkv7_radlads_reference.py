from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from qrwkv_xla.distill import (
    DistillStageConfig,
    DistillStudentConfig,
    run_distill_stage,
)
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.students import (
    RWKV7RADLADSReferenceConfig,
    RWKV7RADLADSReferenceStudent,
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
    rwkv7_radlads_reference_initial_state,
    rwkv7_radlads_reference_layer,
)
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)


def test_radlads_reference_params_initialize_with_expected_shapes() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(0))

    assert params["embedding"].shape == (17, 4)
    assert params["wr"].shape == (2, 4, 4)
    assert params["ww"].shape == (2, 4, 4)
    assert params["wk"].shape == (2, 4, 4)
    assert params["wv"].shape == (2, 4, 4)
    assert params["wa"].shape == (2, 4, 4)
    assert params["wb"].shape == (2, 4, 4)
    assert params["wg"].shape == (2, 4, 4)
    assert params["wo"].shape == (2, 4, 4)
    assert params["time_bias"].shape == (2, 4)
    assert params["lm_head"]["weight"].shape == (4, 17)
    assert _tree_is_finite(params)


def test_radlads_reference_initial_state_shape() -> None:
    state = rwkv7_radlads_reference_initial_state(
        batch_size=3,
        num_layers=2,
        num_heads=2,
        head_size=2,
    )

    assert state.shape == (2, 3, 2, 2, 2)
    assert jnp.all(state == 0)


def test_radlads_reference_initial_state_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        rwkv7_radlads_reference_initial_state(
            batch_size=1,
            num_layers=1,
            num_heads=0,
            head_size=2,
        )


def test_radlads_reference_mask_zeroes_value_but_preserves_recurrence() -> None:
    hidden_size = 2
    zeros = jnp.zeros((hidden_size, hidden_size), dtype=jnp.float32)
    initial_state = jnp.ones((1, 1, hidden_size, hidden_size), dtype=jnp.float32)

    _outputs, final_state = rwkv7_radlads_reference_layer(
        jnp.ones((1, 1, hidden_size), dtype=jnp.float32),
        wr=zeros,
        ww=zeros,
        wk=zeros,
        wv=zeros,
        wa=zeros,
        wb=zeros,
        wg=zeros,
        wo=zeros,
        time_bias=jnp.zeros((hidden_size,), dtype=jnp.float32),
        num_heads=1,
        attention_mask=jnp.zeros((1, 1), dtype=jnp.float32),
        initial_state=initial_state,
        return_state=True,
    )

    decay = jnp.exp(-jnp.exp(jnp.asarray(-0.5)) * jax.nn.sigmoid(0.0))
    assert jnp.allclose(final_state, initial_state * decay, atol=1e-6)


def test_radlads_reference_forward_shape() -> None:
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
    assert state.shape == (2, 2, 2, 2, 2)


def test_radlads_reference_full_vs_stepwise_equivalence() -> None:
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

    step_logits = jnp.concatenate(logits_steps, axis=1)
    step_hidden = jnp.concatenate(hidden_steps, axis=2)
    assert full_output.logits is not None
    assert jnp.max(jnp.abs(full_output.logits - step_logits)) < 1e-5
    assert jnp.max(jnp.abs(full_output.hidden_states - step_hidden)) < 1e-5
    assert jnp.max(jnp.abs(full_state - state)) < 1e-5


def test_radlads_reference_jit_forward_runs() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(3))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    run = jax.jit(
        lambda ids, mask: student.apply_with_state(params, ids, mask)[0].logits
    )
    logits = run(input_ids, attention_mask)

    assert logits is not None
    assert logits.shape == (1, 3, 17)


def test_radlads_reference_gradients_are_finite() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(4))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    def loss_fn(model_params):
        output = student.apply(model_params, input_ids, attention_mask)
        assert output.logits is not None
        return jnp.mean(output.logits**2) + jnp.mean(output.hidden_states**2)

    grads = jax.grad(loss_fn)(params)
    assert _tree_is_finite(grads)


def test_radlads_reference_optimizer_step_keeps_params_finite() -> None:
    student = _student(emit_logits=True)
    params = student.init_params(jax.random.PRNGKey(5))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    def loss_fn(model_params):
        output = student.apply(model_params, input_ids, attention_mask)
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


def test_radlads_reference_distill_integration_runs_one_step(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(
                architecture="rwkv7_radlads_reference",
                vocab_size=32,
                num_heads=2,
            ),
            training=replace(DistillStageConfig().training, max_steps=1),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        )
    )

    assert result.steps == 1
    assert math.isfinite(result.final_loss)
    assert result.final_hidden_mse is not None


def test_placeholder_backend_still_supports_step_equivalence() -> None:
    student = RWKV7ReferenceStudent(
        RWKV7ReferenceConfig(
            vocab_size=17,
            hidden_size=4,
            num_layers=2,
            emit_logits=True,
        )
    )
    params = student.init_params(jax.random.PRNGKey(6))
    output = student.apply(
        params,
        jnp.array([[1, 2, 3]], dtype=jnp.int32),
        attention_mask=jnp.array([[1, 1, 1]], dtype=jnp.int32),
    )

    assert output.logits is not None
    assert output.logits.shape == (1, 3, 17)


def _student(*, emit_logits: bool = False) -> RWKV7RADLADSReferenceStudent:
    return RWKV7RADLADSReferenceStudent(
        RWKV7RADLADSReferenceConfig(
            vocab_size=17,
            hidden_size=4,
            num_layers=2,
            num_heads=2,
            emit_logits=emit_logits,
        )
    )


def _fake_bundle(tmp_path: Path) -> Path:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=4,
            num_layers=2,
            vocab_size=32,
        ),
        runtime=replace(
            config.runtime,
            output_dir=tmp_path / "bundle",
            batch_size=2,
            num_shards=1,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    return config.runtime.output_dir


def _tree_is_finite(tree: object) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
