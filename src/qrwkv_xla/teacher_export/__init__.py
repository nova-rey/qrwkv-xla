"""Teacher export interfaces for QRWKV-XLA."""

from qrwkv_xla.teacher_export.base import ExportRequest, ExportResult, TeacherExporter
from qrwkv_xla.teacher_export.config import (
    ExportRuntimeConfig,
    ExportTargetConfig,
    TeacherExportConfig,
    TeacherModelConfig,
    load_teacher_export_config,
    validate_teacher_export_config,
)
from qrwkv_xla.teacher_export.fake import FakeTeacherExporter
from qrwkv_xla.teacher_export.prompts import DEFAULT_TINY_PROMPTS, load_prompt_texts
from qrwkv_xla.teacher_export.qwen_policy import (
    QwenPolicyEntry,
    QwenPolicyMap,
    QwenResolution,
    load_qwen_policy,
    resolve_qwen_policy,
    resolve_qwen_policy_map,
    validate_qwen_policy,
)
from qrwkv_xla.teacher_export.registry import get_teacher_exporter

__all__ = [
    "TeacherModelConfig",
    "ExportTargetConfig",
    "ExportRuntimeConfig",
    "TeacherExportConfig",
    "validate_teacher_export_config",
    "load_teacher_export_config",
    "ExportRequest",
    "ExportResult",
    "TeacherExporter",
    "FakeTeacherExporter",
    "HFTeacherExporter",
    "HFTeacherExportError",
    "DEFAULT_TINY_PROMPTS",
    "load_prompt_texts",
    "QwenPolicyEntry",
    "QwenPolicyMap",
    "QwenResolution",
    "load_qwen_policy",
    "resolve_qwen_policy",
    "resolve_qwen_policy_map",
    "validate_qwen_policy",
    "get_teacher_exporter",
]


def __getattr__(name: str):
    if name in {"HFTeacherExporter", "HFTeacherExportError"}:
        from qrwkv_xla.teacher_export.hf import HFTeacherExporter, HFTeacherExportError

        exports = {
            "HFTeacherExporter": HFTeacherExporter,
            "HFTeacherExportError": HFTeacherExportError,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
