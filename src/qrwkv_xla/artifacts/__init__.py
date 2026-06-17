"""Artifact validation helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE",
    "BEHAVIORAL_FINGERPRINT_VERSION",
    "STUDENT_ARTIFACT_VERSION",
    "TEACHER_TEXTBOOK_VERSION",
    "FingerprintManifest",
    "StudentArtifactValidationReport",
    "TeacherTextbookValidationReport",
    "TeacherTextbookBuildConfig",
    "TinyTextExample",
    "ValidationResult",
    "build_fake_teacher_textbook",
    "build_hf_teacher_textbook",
    "build_teacher_textbook",
    "load_text_examples",
    "validate_fingerprint_artifact",
    "validate_student_artifact",
    "validate_teacher_textbook",
    "write_student_artifact_validation_report",
    "write_teacher_textbook_validation_report",
]

_FINGERPRINT_EXPORTS = {
    "BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE",
    "BEHAVIORAL_FINGERPRINT_VERSION",
    "FingerprintManifest",
    "ValidationResult",
    "validate_fingerprint_artifact",
}
_STUDENT_EXPORTS = {
    "STUDENT_ARTIFACT_VERSION",
    "StudentArtifactValidationReport",
    "validate_student_artifact",
    "write_student_artifact_validation_report",
}
_TEACHER_TEXTBOOK_EXPORTS = {
    "TEACHER_TEXTBOOK_VERSION",
    "TeacherTextbookValidationReport",
    "validate_teacher_textbook",
    "write_teacher_textbook_validation_report",
}
_TEACHER_TEXTBOOK_BUILDER_EXPORTS = {
    "TeacherTextbookBuildConfig",
    "TinyTextExample",
    "build_fake_teacher_textbook",
    "build_hf_teacher_textbook",
    "build_teacher_textbook",
    "load_text_examples",
}


def __getattr__(name: str) -> Any:
    if name in _FINGERPRINT_EXPORTS:
        from qrwkv_xla.artifacts import fingerprint

        return getattr(fingerprint, name)
    if name in _STUDENT_EXPORTS:
        from qrwkv_xla.artifacts import student_artifact

        return getattr(student_artifact, name)
    if name in _TEACHER_TEXTBOOK_EXPORTS:
        from qrwkv_xla.artifacts import teacher_textbook

        return getattr(teacher_textbook, name)
    if name in _TEACHER_TEXTBOOK_BUILDER_EXPORTS:
        from qrwkv_xla.artifacts import teacher_textbook_builder

        return getattr(teacher_textbook_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
