from __future__ import annotations

from qrwkv_xla.students.base import StudentModel
from qrwkv_xla.students.rwkv7_radlads_reference import (
    RWKV7RADLADSReferenceConfig,
    RWKV7RADLADSReferenceStudent,
)
from qrwkv_xla.students.rwkv7_reference import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
)
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig

STUDENT_ARCHITECTURES = {
    "tiny_student",
    "rwkv7_reference",
    "rwkv7_radlads_reference",
}


def create_student(
    architecture: str,
    *,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int | None = None,
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
    if architecture == "rwkv7_radlads_reference":
        return RWKV7RADLADSReferenceStudent(
            RWKV7RADLADSReferenceConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=1 if num_heads is None else num_heads,
                emit_logits=emit_logits,
                tie_embeddings=tie_embeddings,
                emit_mixer_outputs=emit_mixer_outputs,
            )
        )
    raise ValueError(f"Unknown student architecture: {architecture}")
