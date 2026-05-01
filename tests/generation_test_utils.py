from __future__ import annotations

from pathlib import Path

import jax

from qrwkv_xla.checkpointing import save_checkpoint
from qrwkv_xla.students import create_student


def write_generation_checkpoint(tmp_path: Path, *, emit_logits: bool) -> Path:
    student = create_student(
        "tiny_student",
        vocab_size=512,
        hidden_size=4,
        num_layers=1,
        emit_logits=emit_logits,
        tie_embeddings=False,
    )
    params = student.init_params(jax.random.PRNGKey(0))
    checkpoint_dir = tmp_path / "checkpoints" / ("logits" if emit_logits else "hidden")
    save_checkpoint(
        checkpoint_dir,
        params,
        student_architecture="tiny_student",
        student_config={
            "architecture": "tiny_student",
            "vocab_size": 512,
            "hidden_size": 4,
            "num_layers": 1,
            "emit_logits": emit_logits,
            "tie_embeddings": False,
        },
        step=1,
        learning_rate=0.001,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"schema_version": "test"},
        overwrite=True,
    )
    return checkpoint_dir
