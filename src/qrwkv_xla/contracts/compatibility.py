from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from qrwkv_xla.contracts.vocab import VocabContract, vocab_contract_from_metadata
from qrwkv_xla.targets.store import TeacherTargetStore

DIRECT_LOGIT_TARGET_TYPES = {"dense_logits", "full_logits", "synthetic"}
DIRECT_LOGIT_LOSS_MODES = {"direct_logits", "mse_logits"}
SPARSE_TOPK_TARGET_TYPES = {"topk_with_tail_v0"}
SPARSE_TOPK_LOSS_MODES = {"topk_tail_sparse", "sparse_targets"}


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason: str
    target_contract: VocabContract
    student_contract: VocabContract | None = None


@dataclass(frozen=True)
class TeacherStudentCompatibility:
    status: CompatibilityStatus
    reason: str
    target_type: str
    loss_mode: str
    teacher_contract: VocabContract | None
    student_contract: VocabContract | None

    @property
    def compatible(self) -> bool:
        return self.status is CompatibilityStatus.COMPATIBLE


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
    if target_contract.special_tokens and student_contract.special_tokens:
        target_special = dict(target_contract.special_tokens)
        student_special = dict(student_contract.special_tokens)
        if target_special != student_special:
            return CompatibilityResult(
                compatible=False,
                reason=(
                    "special token mismatch: "
                    f"target={target_special!r} student={student_special!r}"
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


def validate_direct_logit_eligibility(
    *,
    teacher_contract: VocabContract,
    student_contract: VocabContract,
    target_type: str,
    loss_mode: str = "direct_logits",
) -> TeacherStudentCompatibility:
    if loss_mode in SPARSE_TOPK_LOSS_MODES:
        if target_type not in SPARSE_TOPK_TARGET_TYPES:
            return TeacherStudentCompatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                reason=(
                    f"target_type {target_type!r} is not eligible for "
                    f"{loss_mode} in P120"
                ),
                target_type=target_type,
                loss_mode=loss_mode,
                teacher_contract=teacher_contract,
                student_contract=student_contract,
            )
        vocab_result = validate_vocab_compatibility(
            teacher_contract,
            student_contract,
        )
        if not vocab_result.compatible:
            return TeacherStudentCompatibility(
                status=CompatibilityStatus.INCOMPATIBLE,
                reason=vocab_result.reason,
                target_type=target_type,
                loss_mode=loss_mode,
                teacher_contract=teacher_contract,
                student_contract=student_contract,
            )
        return TeacherStudentCompatibility(
            status=CompatibilityStatus.COMPATIBLE,
            reason="compatible: topk_with_tail_v0 sparse target contracts match",
            target_type=target_type,
            loss_mode=loss_mode,
            teacher_contract=teacher_contract,
            student_contract=student_contract,
        )
    if loss_mode not in DIRECT_LOGIT_LOSS_MODES:
        return TeacherStudentCompatibility(
            status=CompatibilityStatus.UNSUPPORTED,
            reason=f"loss_mode {loss_mode!r} is not implemented for P99/P120",
            target_type=target_type,
            loss_mode=loss_mode,
            teacher_contract=teacher_contract,
            student_contract=student_contract,
        )
    if target_type not in DIRECT_LOGIT_TARGET_TYPES:
        return TeacherStudentCompatibility(
            status=CompatibilityStatus.UNSUPPORTED,
            reason=(
                f"target_type {target_type!r} is not eligible for {loss_mode} in P99"
            ),
            target_type=target_type,
            loss_mode=loss_mode,
            teacher_contract=teacher_contract,
            student_contract=student_contract,
        )
    vocab_result = validate_vocab_compatibility(
        teacher_contract,
        student_contract,
    )
    if not vocab_result.compatible:
        return TeacherStudentCompatibility(
            status=CompatibilityStatus.INCOMPATIBLE,
            reason=vocab_result.reason,
            target_type=target_type,
            loss_mode=loss_mode,
            teacher_contract=teacher_contract,
            student_contract=student_contract,
        )
    return TeacherStudentCompatibility(
        status=CompatibilityStatus.COMPATIBLE,
        reason="compatible: direct full_logits contracts match",
        target_type=target_type,
        loss_mode=loss_mode,
        teacher_contract=teacher_contract,
        student_contract=student_contract,
    )


def validate_store_for_student_config(
    store: TeacherTargetStore,
    student_config: Any,
    *,
    loss_mode: str = "direct_logits",
) -> TeacherStudentCompatibility:
    teacher_contract = vocab_contract_from_metadata(store.metadata)
    student_contract = _extract_student_contract(student_config)
    if student_contract is None:
        return TeacherStudentCompatibility(
            status=CompatibilityStatus.UNSUPPORTED,
            reason="student config does not expose a vocab_contract",
            target_type=store.metadata.target_type,
            loss_mode=loss_mode,
            teacher_contract=teacher_contract,
            student_contract=None,
        )
    return validate_direct_logit_eligibility(
        teacher_contract=teacher_contract,
        student_contract=student_contract,
        target_type=store.metadata.target_type,
        loss_mode=loss_mode,
    )


def _extract_student_contract(student_config: Any) -> VocabContract | None:
    if isinstance(student_config, VocabContract):
        return student_config
    contract = getattr(student_config, "vocab_contract", None)
    if isinstance(contract, VocabContract):
        return contract
    return None
