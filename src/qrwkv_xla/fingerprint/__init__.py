"""Teacher-side behavioral fingerprint capture utilities."""

from qrwkv_xla.fingerprint.baseline_comparison import (
    FingerprintBaselineComparisonConfig,
    FingerprintBaselineComparisonResult,
    run_fingerprint_baseline_comparison,
)
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
from qrwkv_xla.fingerprint.quality_per_byte import (
    FingerprintQualityPerByteExperimentConfig,
    FingerprintQualityPerByteExperimentResult,
    evaluate_student_corridor_adherence,
    run_fingerprint_quality_per_byte_experiment,
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
    "FingerprintBaselineComparisonConfig",
    "FingerprintBaselineComparisonResult",
    "FingerprintCorridorBoundsConfig",
    "FingerprintExemplarReservoirCaptureConfig",
    "FingerprintModeDiscoveryConfig",
    "FingerprintQualityPerByteExperimentConfig",
    "FingerprintQualityPerByteExperimentResult",
    "DEFAULT_TINY_TEXTS",
    "RealTeacherFingerprintTrainingRehearsalConfig",
    "RealTeacherFingerprintTrainingRehearsalResult",
    "TinyRealTeacherFingerprintCaptureConfig",
    "TinyRealTeacherFingerprintCaptureResult",
    "build_synthetic_capture_examples",
    "capture_fingerprint_artifact",
    "evaluate_student_corridor_adherence",
    "load_text_fixture",
    "run_fingerprint_baseline_comparison",
    "run_fingerprint_quality_per_byte_experiment",
    "run_real_teacher_fingerprint_training_rehearsal",
    "run_tiny_real_teacher_fingerprint_capture",
]
