from __future__ import annotations

from qrwkv_xla.students.base import StudentModel
from qrwkv_xla.students.rwkv7_reference import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
)
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig


def create_student(
    architecture: str,
    *,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    emit_logits: bool = False,
    tie_embeddings: bool = False,
    emit_mixer_outputs: bool = False,
) -> StudentModel:
    if architecture == "tiny_student":
        return TinyStudent(
            TinyStudentConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                emit_logits=emit_logits,
                tie_embeddings=tie_embeddings,
                emit_mixer_outputs=emit_mixer_outputs,
            )
        )
    if architecture == "rwkv7_reference":
        return RWKV7ReferenceStudent(
            RWKV7ReferenceConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                emit_logits=emit_logits,
                tie_embeddings=tie_embeddings,
                emit_mixer_outputs=emit_mixer_outputs,
            )
        )
    raise ValueError(f"Unknown student architecture: {architecture}")
