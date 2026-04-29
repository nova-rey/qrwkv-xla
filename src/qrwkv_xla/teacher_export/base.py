from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qrwkv_xla.targets.manifest import TeacherTargetManifest
from qrwkv_xla.teacher_export.config import TeacherExportConfig


@dataclass(frozen=True)
class ExportRequest:
    config: TeacherExportConfig
    output_dir: Path


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    manifest: TeacherTargetManifest
    shard_count: int
    total_examples: int


class TeacherExporter(Protocol):
    name: str

    def export(self, request: ExportRequest) -> ExportResult: ...
