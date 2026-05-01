"""Evaluation interfaces for QRWKV-XLA."""

from qrwkv_xla.eval.compare import (
    EvalComparisonResult,
    PromptComparison,
    compare_eval_snapshots,
    write_eval_comparison,
)
from qrwkv_xla.eval.config import (
    EvalConfig,
    EvalGenerationConfig,
    EvalPromptConfig,
    EvalSanityConfig,
    load_eval_config,
    replace_eval_overrides,
    validate_eval_config,
)
from qrwkv_xla.eval.harness import EvaluationResult, run_checkpoint_evaluation
from qrwkv_xla.eval.sanity import (
    SanityCheckResult,
    SanitySummary,
    run_generation_sanity_checks,
)

__all__ = [
    "EvalComparisonResult",
    "EvalConfig",
    "EvalGenerationConfig",
    "EvalPromptConfig",
    "EvalSanityConfig",
    "EvaluationResult",
    "PromptComparison",
    "SanityCheckResult",
    "SanitySummary",
    "compare_eval_snapshots",
    "load_eval_config",
    "replace_eval_overrides",
    "run_checkpoint_evaluation",
    "run_generation_sanity_checks",
    "validate_eval_config",
    "write_eval_comparison",
]
