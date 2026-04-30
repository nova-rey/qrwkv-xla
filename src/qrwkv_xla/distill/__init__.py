"""Distillation stage runtime for QRWKV-XLA."""

from qrwkv_xla.distill.config import (
    DistillationLossConfig,
    DistillationOptimizerConfig,
    DistillationStageConfig,
    DistillationStudentConfig,
    DistillationTrainingConfig,
    DistillLossConfig,
    DistillOptimizerConfig,
    DistillStageConfig,
    DistillStudentConfig,
    DistillTrainingConfig,
    LossWeightConfig,
    load_distill_stage_config,
    load_distillation_config,
    validate_distill_stage_config,
)

_LAZY_EXPORTS = {
    "LossBreakdown": ("qrwkv_xla.distill.losses", "LossBreakdown"),
    "compute_distill_loss": ("qrwkv_xla.distill.losses", "compute_distill_loss"),
    "logits_kl_loss": ("qrwkv_xla.distill.losses", "logits_kl_loss"),
    "DistillationStageResult": (
        "qrwkv_xla.distill.runner",
        "DistillationStageResult",
    ),
    "DistillStageResult": ("qrwkv_xla.distill.runner", "DistillStageResult"),
    "run_distill_stage": ("qrwkv_xla.distill.runner", "run_distill_stage"),
    "run_distillation_stage": (
        "qrwkv_xla.distill.runner",
        "run_distillation_stage",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "DistillLossConfig",
    "DistillOptimizerConfig",
    "DistillStageConfig",
    "DistillStageResult",
    "DistillStudentConfig",
    "DistillTrainingConfig",
    "DistillationLossConfig",
    "DistillationOptimizerConfig",
    "DistillationStageConfig",
    "DistillationStageResult",
    "DistillationStudentConfig",
    "DistillationTrainingConfig",
    "LossBreakdown",
    "LossWeightConfig",
    "compute_distill_loss",
    "load_distill_stage_config",
    "load_distillation_config",
    "logits_kl_loss",
    "run_distill_stage",
    "run_distillation_stage",
    "validate_distill_stage_config",
]
