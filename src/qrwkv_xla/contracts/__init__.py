"""Cross-boundary contracts for teacher, target, and student compatibility."""

from qrwkv_xla.contracts.compatibility import (
    CompatibilityResult,
    validate_vocab_compatibility,
)
from qrwkv_xla.contracts.vocab import (
    VocabContract,
    vocab_contract_from_metadata,
)

__all__ = [
    "CompatibilityResult",
    "VocabContract",
    "validate_vocab_compatibility",
    "vocab_contract_from_metadata",
]
