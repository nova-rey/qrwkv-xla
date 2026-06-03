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
from qrwkv_xla.eval.exported_student import (
    ContinuationScore,
    ExportedStudentEvalAdapter,
    ToyContinuationExample,
    ToyEvalResult,
    load_toy_continuation_task,
    run_toy_exported_student_eval,
)
from qrwkv_xla.eval.harness import EvaluationResult, run_checkpoint_evaluation
from qrwkv_xla.eval.mini_eval import (
    MiniEvalResult,
    create_builtin_mini_eval_store,
    run_mini_eval_harness,
    write_mini_eval_report,
)
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
    "ContinuationScore",
    "ExportedStudentEvalAdapter",
    "MiniEvalResult",
    "PromptComparison",
    "SanityCheckResult",
    "SanitySummary",
    "ToyContinuationExample",
    "ToyEvalResult",
    "compare_eval_snapshots",
    "create_builtin_mini_eval_store",
    "load_toy_continuation_task",
    "load_eval_config",
    "replace_eval_overrides",
    "run_checkpoint_evaluation",
    "run_generation_sanity_checks",
    "run_mini_eval_harness",
    "run_toy_exported_student_eval",
    "validate_eval_config",
    "write_eval_comparison",
    "write_mini_eval_report",
]
