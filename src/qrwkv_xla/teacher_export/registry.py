from __future__ import annotations

from qrwkv_xla.teacher_export.base import TeacherExporter
from qrwkv_xla.teacher_export.fake import FakeTeacherExporter


def get_teacher_exporter(name: str) -> TeacherExporter:
    normalized = name.strip().lower()
    if normalized == "fake":
        return FakeTeacherExporter()
    if normalized == "hf":
        from qrwkv_xla.teacher_export.hf import HFTeacherExporter

        return HFTeacherExporter()
    raise ValueError(f"Unknown teacher exporter backend: {name!r}")
