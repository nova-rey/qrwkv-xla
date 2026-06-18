"""Training utilities for QRWKV-XLA."""

from qrwkv_xla.training.fingerprint_exemplar_loss import (
    FingerprintExemplarLossConfig,
    FingerprintExemplarLossOutput,
    compute_fingerprint_exemplar_loss,
    compute_fingerprint_exemplar_loss_at_positions,
)
from qrwkv_xla.training.fingerprint_loss import (
    FingerprintCorridorLossConfig,
    FingerprintCorridorLossOutput,
    compute_fingerprint_corridor_loss,
    inside_bounds,
    squared_hinge_bound_penalty,
)
from qrwkv_xla.training.fingerprint_real_student_forward import (
    REAL_STUDENT_FINGERPRINT_FORWARD_METRIC_KEYS,
    RealStudentFingerprintForwardConfig,
    RealStudentFingerprintForwardResult,
    classify_real_student_fingerprint_forward_status,
    run_real_student_fingerprint_forward_smoke,
)
from qrwkv_xla.training.fingerprint_reports import (
    render_corridor_fingerprint_smoke_summary,
    render_fingerprint_smoke_summary,
    render_mixed_fingerprint_smoke_summary,
    render_real_student_fingerprint_forward_summary,
    validate_fingerprint_smoke_report,
    write_fingerprint_smoke_summary,
)
from qrwkv_xla.training.fingerprint_smoke import (
    FINGERPRINT_MIXED_SMOKE_METRIC_KEYS,
    FINGERPRINT_SMOKE_METRIC_KEYS,
    FingerprintMixedLossOutput,
    FingerprintMixedSmokeConfig,
    FingerprintMixedSmokeResult,
    FingerprintTrainingSmokeConfig,
    FingerprintTrainingSmokeResult,
    classify_fingerprint_mixed_smoke_status,
    classify_fingerprint_smoke_status,
    run_mixed_fingerprint_training_smoke,
    run_tiny_fingerprint_training_smoke,
)
from qrwkv_xla.training.fingerprint_stats import (
    FingerprintDistributionStats,
    compute_fingerprint_distribution_stats,
    compute_fingerprint_distribution_stats_at_positions,
    select_position_logits,
)
from qrwkv_xla.training.gradients import (
    GradientClipResult,
    clip_gradients_by_global_norm,
    global_gradient_norm,
)
from qrwkv_xla.training.real_teacher_overfit import (
    RealTeacherOverfitResult,
    run_tiny_real_teacher_overfit_rehearsal,
)
from qrwkv_xla.training.tiny_overfit import (
    TinyOverfitResult,
    run_tiny_overfit_rehearsal,
)

__all__ = [
    "GradientClipResult",
    "FingerprintCorridorLossConfig",
    "FingerprintCorridorLossOutput",
    "FingerprintDistributionStats",
    "FingerprintExemplarLossConfig",
    "FingerprintExemplarLossOutput",
    "FingerprintMixedLossOutput",
    "FingerprintMixedSmokeConfig",
    "FingerprintMixedSmokeResult",
    "FingerprintTrainingSmokeConfig",
    "FingerprintTrainingSmokeResult",
    "FINGERPRINT_MIXED_SMOKE_METRIC_KEYS",
    "FINGERPRINT_SMOKE_METRIC_KEYS",
    "REAL_STUDENT_FINGERPRINT_FORWARD_METRIC_KEYS",
    "RealTeacherOverfitResult",
    "RealStudentFingerprintForwardConfig",
    "RealStudentFingerprintForwardResult",
    "TinyOverfitResult",
    "clip_gradients_by_global_norm",
    "classify_fingerprint_mixed_smoke_status",
    "classify_fingerprint_smoke_status",
    "classify_real_student_fingerprint_forward_status",
    "compute_fingerprint_corridor_loss",
    "compute_fingerprint_distribution_stats",
    "compute_fingerprint_distribution_stats_at_positions",
    "compute_fingerprint_exemplar_loss",
    "compute_fingerprint_exemplar_loss_at_positions",
    "global_gradient_norm",
    "inside_bounds",
    "render_corridor_fingerprint_smoke_summary",
    "render_fingerprint_smoke_summary",
    "render_mixed_fingerprint_smoke_summary",
    "render_real_student_fingerprint_forward_summary",
    "run_mixed_fingerprint_training_smoke",
    "run_real_student_fingerprint_forward_smoke",
    "run_tiny_fingerprint_training_smoke",
    "run_tiny_real_teacher_overfit_rehearsal",
    "run_tiny_overfit_rehearsal",
    "select_position_logits",
    "squared_hinge_bound_penalty",
    "validate_fingerprint_smoke_report",
    "write_fingerprint_smoke_summary",
]
