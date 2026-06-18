"""Teacher-side behavioral fingerprint capture utilities."""

from qrwkv_xla.fingerprint.capture import (
    FingerprintCaptureBudgetConfig,
    FingerprintCaptureConfig,
    FingerprintCaptureExample,
    FingerprintCaptureResult,
    FingerprintCorridorBoundsConfig,
    FingerprintExemplarReservoirCaptureConfig,
    FingerprintModeDiscoveryConfig,
    build_synthetic_capture_examples,
    capture_fingerprint_artifact,
)
from qrwkv_xla.fingerprint.real_teacher import (
    DEFAULT_TINY_REAL_TEACHER,
    TinyRealTeacherFingerprintCaptureConfig,
    TinyRealTeacherFingerprintCaptureResult,
    load_text_fixture,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.fingerprint.training_rehearsal import (
    DEFAULT_TINY_TEXTS,
    RealTeacherFingerprintTrainingRehearsalConfig,
    RealTeacherFingerprintTrainingRehearsalResult,
    run_real_teacher_fingerprint_training_rehearsal,
)

__all__ = [
    "DEFAULT_TINY_REAL_TEACHER",
    "FingerprintCaptureBudgetConfig",
    "FingerprintCaptureConfig",
    "FingerprintCaptureExample",
    "FingerprintCaptureResult",
    "FingerprintCorridorBoundsConfig",
    "FingerprintExemplarReservoirCaptureConfig",
    "FingerprintModeDiscoveryConfig",
    "DEFAULT_TINY_TEXTS",
    "RealTeacherFingerprintTrainingRehearsalConfig",
    "RealTeacherFingerprintTrainingRehearsalResult",
    "TinyRealTeacherFingerprintCaptureConfig",
    "TinyRealTeacherFingerprintCaptureResult",
    "build_synthetic_capture_examples",
    "capture_fingerprint_artifact",
    "load_text_fixture",
    "run_real_teacher_fingerprint_training_rehearsal",
    "run_tiny_real_teacher_fingerprint_capture",
]
