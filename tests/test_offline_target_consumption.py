from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.students import CurrentQRWKVStudentBackend
from qrwkv_xla.targets import (
    OfflineTargetBatch,
    TeacherTargetStore,
    load_offline_target_batch,
    mse_logits_loss,
)
from qrwkv_xla.teachers import SyntheticTeacherBackend, emit_teacher_target_store


def test_load_offline_target_batch_round_trips_store_arrays(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, vocab_size=17)

    batch = load_offline_target_batch(store)

    assert isinstance(batch, OfflineTargetBatch)
    assert batch.input_ids.shape == (2, 3)
    assert batch.input_ids.dtype == np.int32
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.attention_mask.dtype == np.int32
    assert batch.teacher_logits.shape == (2, 3, store.metadata.vocab_size)
    assert batch.teacher_logits.dtype == np.float32
    assert batch.teacher_logits.shape[1] == store.metadata.sequence_length


def test_mse_logits_loss_returns_finite_scalar() -> None:
    teacher_logits = jnp.asarray(np.arange(12, dtype=np.float32).reshape(1, 3, 4))
    student_logits = teacher_logits + 0.5

    loss = mse_logits_loss(student_logits, teacher_logits)

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert float(loss) == pytest.approx(0.25)


def test_mse_logits_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        mse_logits_loss(jnp.zeros((1, 2, 3)), jnp.zeros((1, 2, 4)))


def test_offline_consumption_requires_no_live_teacher(tmp_path: Path) -> None:
    store = _emit_store(tmp_path, vocab_size=8)

    batch = load_offline_target_batch(TeacherTargetStore.open(store.root))

    assert store.metadata.source == {"kind": "synthetic"}
    assert batch.teacher_logits.shape == (2, 3, 8)
    assert bool(
        jnp.isfinite(mse_logits_loss(batch.teacher_logits, batch.teacher_logits))
    )


def test_current_student_backend_consumes_offline_target_shapes(
    tmp_path: Path,
) -> None:
    store = _emit_store(tmp_path, vocab_size=17)
    batch = load_offline_target_batch(store)
    student_backend = CurrentQRWKVStudentBackend.from_config(
        "rwkv7_qwen_reference",
        vocab_size=store.metadata.vocab_size,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        emit_logits=True,
    )
    params = student_backend.init_params(jax.random.PRNGKey(0))

    output, _state = student_backend.forward_full(
        params,
        jnp.asarray(batch.input_ids),
        attention_mask=jnp.asarray(batch.attention_mask),
    )
    student_logits = student_backend.logits(output)
    loss = mse_logits_loss(student_logits, batch.teacher_logits)

    assert student_logits.shape == batch.teacher_logits.shape
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def _emit_store(
    tmp_path: Path,
    *,
    vocab_size: int,
) -> TeacherTargetStore:
    return emit_teacher_target_store(
        SyntheticTeacherBackend(vocab_size=vocab_size),
        tmp_path / f"synthetic_targets_v{vocab_size}",
        num_examples=2,
        sequence_length=3,
    )
