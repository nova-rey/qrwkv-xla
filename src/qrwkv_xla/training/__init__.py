"""Training utilities for QRWKV-XLA."""

from qrwkv_xla.training.fingerprint_loss import (
    FingerprintCorridorLossConfig,
    FingerprintCorridorLossOutput,
    compute_fingerprint_corridor_loss,
    inside_bounds,
    squared_hinge_bound_penalty,
)
from qrwkv_xla.training.fingerprint_smoke import (
    FINGERPRINT_SMOKE_METRIC_KEYS,
    FingerprintTrainingSmokeConfig,
    FingerprintTrainingSmokeResult,
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
    "FingerprintTrainingSmokeConfig",
    "FingerprintTrainingSmokeResult",
    "FINGERPRINT_SMOKE_METRIC_KEYS",
    "RealTeacherOverfitResult",
    "TinyOverfitResult",
    "clip_gradients_by_global_norm",
    "compute_fingerprint_corridor_loss",
    "compute_fingerprint_distribution_stats",
    "compute_fingerprint_distribution_stats_at_positions",
    "global_gradient_norm",
    "inside_bounds",
    "run_tiny_fingerprint_training_smoke",
    "run_tiny_real_teacher_overfit_rehearsal",
    "run_tiny_overfit_rehearsal",
    "select_position_logits",
    "squared_hinge_bound_penalty",
]
