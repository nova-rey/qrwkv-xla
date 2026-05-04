from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.datasets import TargetBatch
from qrwkv_xla.students import TinyStudent, TinyStudentConfig
from qrwkv_xla.targets import TargetFlags, TeacherTargetManifest, write_target_bundle
from qrwkv_xla.trainers import (
    SimpleTrainResult,
    TrainState,
    batch_to_jax,
    make_train_step,
    train_on_bundle_once,
)


def _target_batch() -> TargetBatch:
    return TargetBatch(
        input_ids=np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32),
        attention_mask=np.ones((2, 3), dtype=np.int32),
        loss_mask=np.ones((2, 3), dtype=np.int32),
        hidden_states=np.zeros((2, 2, 3, 4), dtype=np.float32),
    )


def test_batch_to_jax_returns_jax_arrays() -> None:
    converted = batch_to_jax(_target_batch())

    assert set(converted) == {"input_ids", "attention_mask", "hidden_states"}
    assert all(isinstance(value, jax.Array) for value in converted.values())


def test_jitted_train_step_updates_params_and_repeated_steps_do_not_crash() -> None:
    student = TinyStudent(TinyStudentConfig(vocab_size=8, hidden_size=4, num_layers=2))
    state = TrainState(
        params=student.init_params(jax.random.PRNGKey(0)),
        step=0,
        learning_rate=1e-2,
    )
    train_step = make_train_step(student.apply)
    batch = batch_to_jax(_target_batch())

    new_state, metrics = train_step(state, batch)

    assert jnp.isfinite(metrics["loss"])
    assert new_state.step == 1
    assert _params_changed(state.params, new_state.params)

    for _ in range(3):
        new_state, metrics = train_step(new_state, batch)
        assert jnp.isfinite(metrics["loss"])


def test_train_on_bundle_once_returns_loss_summary_only(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    batch = _target_batch()
    write_target_bundle(
        bundle_dir,
        _manifest(),
        [
            {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "loss_mask": batch.loss_mask,
                "hidden_states": batch.hidden_states,
            }
        ],
    )
    student = TinyStudent(TinyStudentConfig(vocab_size=8, hidden_size=4, num_layers=2))

    result = train_on_bundle_once(
        bundle_dir=bundle_dir,
        student=student,
        seed=0,
        learning_rate=1e-2,
        max_steps=1,
    )

    assert isinstance(result, SimpleTrainResult)
    assert [field.name for field in fields(result)] == [
        "initial_loss",
        "final_loss",
        "steps",
    ]
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    assert result.steps == 1


def _params_changed(
    before: dict[str, jax.Array],
    after: dict[str, jax.Array],
) -> bool:
    return any(
        not np.array_equal(np.asarray(before[name]), np.asarray(after[name]))
        for name in before
    )


def _manifest() -> TeacherTargetManifest:
    return TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="qwen",
        teacher_model_id=None,
        teacher_policy_label="Qwen3.latest",
        fallback_policy_label="Qwen3.0",
        tokenizer_id=None,
        sequence_length=3,
        hidden_size=4,
        num_layers=2,
        targets=TargetFlags(),
        dtype="fp32",
        created_by="test",
        notes=[],
    )
