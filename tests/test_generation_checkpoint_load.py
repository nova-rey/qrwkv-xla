from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from qrwkv_xla.generation import load_student_from_checkpoint
from tests.generation_test_utils import write_generation_checkpoint


def test_load_student_from_logits_checkpoint_emits_logits(tmp_path: Path) -> None:
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=True)

    loaded = load_student_from_checkpoint(checkpoint_dir)
    output = loaded.student.apply(
        loaded.params,
        jnp.asarray([[1, 2]], dtype=jnp.int32),
        jnp.ones((1, 2), dtype=jnp.int32),
    )

    assert output.logits is not None
    assert output.logits.shape == (1, 2, 512)


def test_load_student_from_hidden_only_checkpoint_fails(tmp_path: Path) -> None:
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=False)

    with pytest.raises(ValueError, match="emit_logits=true"):
        load_student_from_checkpoint(checkpoint_dir)
