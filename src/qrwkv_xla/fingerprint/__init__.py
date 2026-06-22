"""Teacher-side behavioral fingerprint capture utilities."""

from qrwkv_xla.fingerprint.arc_report import (
    FingerprintArc2ReportConfig,
    FingerprintArc2ReportResult,
    build_fingerprint_arc2_report,
    run_fingerprint_arc2_report,
)
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
from qrwkv_xla.fingerprint.held_out_evaluation import (
    HeldOutFingerprintEvaluationConfig,
    HeldOutFingerprintEvaluationResult,
    paired_bootstrap_interval,
    run_held_out_fingerprint_evaluation,
    select_held_out_winner,
    stable_hash,
    validate_fingerprint_provenance,
    write_fingerprint_provenance,
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
from qrwkv_xla.fingerprint.trained_baseline import (
    FingerprintTrainedBaselineConfig,
    FingerprintTrainedBaselineResult,
    masked_causal_lm_loss,
    parameter_fingerprint,
    run_fingerprint_trained_baseline_comparison,
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
    "FingerprintArc2ReportConfig",
    "FingerprintArc2ReportResult",
    "FingerprintCorridorBoundsConfig",
    "FingerprintExemplarReservoirCaptureConfig",
    "FingerprintModeDiscoveryConfig",
    "FingerprintQualityPerByteExperimentConfig",
    "FingerprintQualityPerByteExperimentResult",
    "FingerprintTrainedBaselineConfig",
    "FingerprintTrainedBaselineResult",
    "HeldOutFingerprintEvaluationConfig",
    "HeldOutFingerprintEvaluationResult",
    "DEFAULT_TINY_TEXTS",
    "RealTeacherFingerprintTrainingRehearsalConfig",
    "RealTeacherFingerprintTrainingRehearsalResult",
    "TinyRealTeacherFingerprintCaptureConfig",
    "TinyRealTeacherFingerprintCaptureResult",
    "build_synthetic_capture_examples",
    "build_fingerprint_arc2_report",
    "capture_fingerprint_artifact",
    "evaluate_student_corridor_adherence",
    "load_text_fixture",
    "masked_causal_lm_loss",
    "parameter_fingerprint",
    "paired_bootstrap_interval",
    "run_fingerprint_baseline_comparison",
    "run_fingerprint_arc2_report",
    "run_fingerprint_quality_per_byte_experiment",
    "run_fingerprint_trained_baseline_comparison",
    "run_held_out_fingerprint_evaluation",
    "run_real_teacher_fingerprint_training_rehearsal",
    "run_tiny_real_teacher_fingerprint_capture",
    "select_held_out_winner",
    "stable_hash",
    "validate_fingerprint_provenance",
    "write_fingerprint_provenance",
]
