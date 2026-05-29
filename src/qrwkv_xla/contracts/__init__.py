"""Cross-boundary contracts for teacher, target, and student compatibility."""

from qrwkv_xla.contracts.compatibility import (
    CompatibilityResult,
    CompatibilityStatus,
    TeacherStudentCompatibility,
    validate_direct_logit_eligibility,
    validate_store_for_student_config,
    validate_vocab_compatibility,
)
from qrwkv_xla.contracts.vocab import (
    VocabContract,
    vocab_contract_from_metadata,
)

__all__ = [
    "CompatibilityResult",
    "CompatibilityStatus",
    "TeacherStudentCompatibility",
    "VocabContract",
    "validate_direct_logit_eligibility",
    "validate_store_for_student_config",
    "validate_vocab_compatibility",
    "vocab_contract_from_metadata",
]
