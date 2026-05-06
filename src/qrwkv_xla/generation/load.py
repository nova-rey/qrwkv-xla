from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qrwkv_xla.checkpointing.simple import CheckpointManifest, load_checkpoint
from qrwkv_xla.students.factory import create_student


@dataclass(frozen=True)
class LoadedStudentForGeneration:
    student: Any
    params: dict
    checkpoint_dir: Path
    manifest: CheckpointManifest


def load_student_from_checkpoint(
    checkpoint_dir: str | Path,
) -> LoadedStudentForGeneration:
    loaded = load_checkpoint(checkpoint_dir)
    student_config = loaded.manifest.student_config
    if not bool(student_config.get("emit_logits", False)):
        raise ValueError(
            "generation requires checkpoint student_config.emit_logits=true; "
            "hidden-only checkpoints cannot be used for generation"
        )

    required = ("vocab_size", "hidden_size", "num_layers")
    missing = [name for name in required if name not in student_config]
    if missing:
        raise ValueError(
            "checkpoint student_config is missing required generation fields: "
            + ", ".join(missing)
        )

    student = create_student(
        loaded.manifest.student_architecture,
        vocab_size=int(student_config["vocab_size"]),
        hidden_size=int(student_config["hidden_size"]),
        num_layers=int(student_config["num_layers"]),
        num_heads=(
            None
            if student_config.get("num_heads") is None
            else int(student_config["num_heads"])
        ),
        emit_logits=True,
        tie_embeddings=bool(student_config.get("tie_embeddings", False)),
    )
    return LoadedStudentForGeneration(
        student=student,
        params=loaded.params,
        checkpoint_dir=loaded.checkpoint_dir,
        manifest=loaded.manifest,
    )
