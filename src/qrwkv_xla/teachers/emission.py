from __future__ import annotations

from pathlib import Path

from qrwkv_xla.targets import TeacherTargetStore
from qrwkv_xla.teachers.backend import TeacherBackend


def emit_teacher_target_store(
    backend: TeacherBackend,
    path: str | Path,
    *,
    num_examples: int,
    sequence_length: int,
    overwrite: bool = False,
) -> TeacherTargetStore:
    metadata = backend.build_metadata(
        num_examples=num_examples,
        sequence_length=sequence_length,
    )
    store = TeacherTargetStore.create(path, metadata, overwrite=overwrite)
    arrays = backend.emit_targets(
        num_examples=num_examples,
        sequence_length=sequence_length,
    )
    store.write_shard(0, arrays)
    store.validate()
    return TeacherTargetStore.open(path)
