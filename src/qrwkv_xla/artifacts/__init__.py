"""Artifact validation helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE",
    "BEHAVIORAL_FINGERPRINT_VERSION",
    "STUDENT_ARTIFACT_VERSION",
    "TEACHER_TEXTBOOK_VERSION",
    "FingerprintManifest",
    "FingerprintBatch",
    "FingerprintExemplarBatch",
    "FingerprintExemplarDataset",
    "FingerprintExemplarLoaderConfig",
    "FingerprintExemplarRecord",
    "FingerprintArtifactSummary",
    "FingerprintLoaderConfig",
    "FingerprintTargetDataset",
    "FingerprintTargetRecord",
    "StudentArtifactValidationReport",
    "TeacherTextbookValidationReport",
    "TeacherTextbookBuildConfig",
    "TinyTextExample",
    "ValidationResult",
    "build_fake_teacher_textbook",
    "build_hf_teacher_textbook",
    "build_teacher_textbook",
    "load_text_examples",
    "load_fingerprint_exemplars",
    "load_fingerprint_targets",
    "summarize_fingerprint_artifact",
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
_FINGERPRINT_LOADER_EXPORTS = {
    "FingerprintBatch",
    "FingerprintLoaderConfig",
    "FingerprintTargetDataset",
    "FingerprintTargetRecord",
    "load_fingerprint_targets",
}
_FINGERPRINT_EXEMPLAR_EXPORTS = {
    "FingerprintExemplarBatch",
    "FingerprintExemplarDataset",
    "FingerprintExemplarLoaderConfig",
    "FingerprintExemplarRecord",
    "load_fingerprint_exemplars",
}
_FINGERPRINT_SUMMARY_EXPORTS = {
    "FingerprintArtifactSummary",
    "summarize_fingerprint_artifact",
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
    if name in _FINGERPRINT_LOADER_EXPORTS:
        from qrwkv_xla.artifacts import fingerprint_loader

        return getattr(fingerprint_loader, name)
    if name in _FINGERPRINT_EXEMPLAR_EXPORTS:
        from qrwkv_xla.artifacts import fingerprint_exemplars

        return getattr(fingerprint_exemplars, name)
    if name in _FINGERPRINT_SUMMARY_EXPORTS:
        from qrwkv_xla.artifacts import fingerprint_summary

        return getattr(fingerprint_summary, name)
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
