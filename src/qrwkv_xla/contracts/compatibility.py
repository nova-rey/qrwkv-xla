from __future__ import annotations

from dataclasses import dataclass

from qrwkv_xla.contracts.vocab import VocabContract


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason: str
    target_contract: VocabContract
    student_contract: VocabContract | None = None


def validate_vocab_compatibility(
    target_contract: VocabContract,
    student_contract: VocabContract,
) -> CompatibilityResult:
    if target_contract.tokenizer_id != student_contract.tokenizer_id:
        return CompatibilityResult(
            compatible=False,
            reason=(
                "tokenizer_id mismatch: "
                f"target={target_contract.tokenizer_id!r} "
                f"student={student_contract.tokenizer_id!r}"
            ),
            target_contract=target_contract,
            student_contract=student_contract,
        )
    if target_contract.vocab_size != student_contract.vocab_size:
        return CompatibilityResult(
            compatible=False,
            reason=(
                "vocab_size mismatch: "
                f"target={target_contract.vocab_size} "
                f"student={student_contract.vocab_size}"
            ),
            target_contract=target_contract,
            student_contract=student_contract,
        )
    if (
        target_contract.tokenizer_hash is not None
        and student_contract.tokenizer_hash is not None
        and target_contract.tokenizer_hash != student_contract.tokenizer_hash
    ):
        return CompatibilityResult(
            compatible=False,
            reason=(
                "tokenizer_hash mismatch: "
                f"target={target_contract.tokenizer_hash!r} "
                f"student={student_contract.tokenizer_hash!r}"
            ),
            target_contract=target_contract,
            student_contract=student_contract,
        )
    return CompatibilityResult(
        compatible=True,
        reason="vocab contracts match",
        target_contract=target_contract,
        student_contract=student_contract,
    )
