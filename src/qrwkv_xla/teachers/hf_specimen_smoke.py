from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qrwkv_xla.contracts import vocab_contract_from_metadata
from qrwkv_xla.teachers.emission import emit_teacher_target_store
from qrwkv_xla.teachers.hf import HFTeacherBackend, HFTeacherUnavailable

DEFAULT_HF_SPECIMEN_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
HF_SPECIMEN_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "qwen_specific_support",
    "gpt2_specific_architecture",
    "student_consumption_proven",
    "training_ready",
    "tokenizer_remapping_supported",
    "production_distillation_ready",
    "full_model_quality_proven",
)


@dataclass(frozen=True)
class HFTeacherSpecimenSmokeResult:
    status: str
    scope: str
    model_id: str
    local_files_only: bool
    allow_downloads: bool
    target_store_path: str
    target_store_validated: bool
    vocab_contract_extracted: bool
    tokenizer_id: str | None
    tokenizer_hash: str | None
    vocab_size: int | None
    sequence_length: int
    num_examples: int
    target_type: str | None
    logits_shape: tuple[int, ...] | None
    claims_not_made: tuple[str, ...]
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    phase: str = "P104"

    def to_report(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(self.to_report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report_path


@dataclass(frozen=True)
class HFTeacherSpecimenConfig:
    model_id: str
    prompts: tuple[str, ...] = ("hello",)
    sequence_length: int = 8
    local_files_only: bool = True
    allow_downloads: bool = False


@dataclass(frozen=True)
class HFTeacherSpecimenSwapReport:
    status: str
    scope: str
    specimen_count: int
    passed: int
    unavailable: int
    failed: int
    model_ids: tuple[str, ...]
    specimens: tuple[HFTeacherSpecimenSmokeResult, ...]
    claims_not_made: tuple[str, ...]
    phase: str = "P105"

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["specimens"] = [specimen.to_report() for specimen in self.specimens]
        return report

    def write_json(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(self.to_report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report_path


def run_hf_teacher_specimen_smoke(
    *,
    target_store: str | Path,
    model_id: str = DEFAULT_HF_SPECIMEN_MODEL_ID,
    prompts: tuple[str, ...] = ("hello",),
    sequence_length: int = 8,
    local_files_only: bool = True,
    allow_downloads: bool = False,
    backend: HFTeacherBackend | None = None,
) -> HFTeacherSpecimenSmokeResult:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")

    effective_local_files_only = False if allow_downloads else local_files_only
    target_store_path = Path(target_store)
    teacher = backend or HFTeacherBackend(
        model_id,
        local_files_only=effective_local_files_only,
        prompts=prompts,
    )
    try:
        store = emit_teacher_target_store(
            teacher,
            target_store_path,
            num_examples=len(prompts),
            sequence_length=sequence_length,
            overwrite=True,
        )
        store.validate()
        contract = vocab_contract_from_metadata(store.metadata)
        arrays = store.read_shard(0)
        logits_shape = tuple(arrays["logits"].shape)
        _validate_artifact_shapes(
            logits_shape=logits_shape,
            vocab_size=contract.vocab_size,
            sequence_length=store.metadata.sequence_length,
            num_examples=store.metadata.num_examples,
        )
        return HFTeacherSpecimenSmokeResult(
            status="pass",
            scope="tiny_hf_causal_lm_teacher_specimen_smoke",
            model_id=store.metadata.model_id,
            local_files_only=effective_local_files_only,
            allow_downloads=allow_downloads,
            target_store_path=str(store.root),
            target_store_validated=True,
            vocab_contract_extracted=True,
            tokenizer_id=contract.tokenizer_id,
            tokenizer_hash=contract.tokenizer_hash,
            vocab_size=contract.vocab_size,
            sequence_length=store.metadata.sequence_length,
            num_examples=store.metadata.num_examples,
            target_type=store.metadata.target_type,
            logits_shape=logits_shape,
            claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
        )
    except HFTeacherUnavailable as exc:
        return _unavailable_result(
            model_id=model_id,
            local_files_only=effective_local_files_only,
            allow_downloads=allow_downloads,
            target_store_path=target_store_path,
            sequence_length=sequence_length,
            num_examples=len(prompts),
            reason=_unavailable_reason(str(exc)),
            error=exc,
        )
    except Exception as exc:
        return HFTeacherSpecimenSmokeResult(
            status="fail",
            scope="tiny_hf_causal_lm_teacher_specimen_smoke",
            model_id=model_id,
            local_files_only=effective_local_files_only,
            allow_downloads=allow_downloads,
            target_store_path=str(target_store_path),
            target_store_validated=False,
            vocab_contract_extracted=False,
            tokenizer_id=None,
            tokenizer_hash=None,
            vocab_size=None,
            sequence_length=sequence_length,
            num_examples=len(prompts),
            target_type=None,
            logits_shape=None,
            claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _validate_artifact_shapes(
    *,
    logits_shape: tuple[int, ...],
    vocab_size: int,
    sequence_length: int,
    num_examples: int,
) -> None:
    expected = (num_examples, sequence_length, vocab_size)
    if logits_shape != expected:
        raise ValueError(
            f"HF specimen logits shape mismatch: actual={logits_shape} "
            f"expected={expected}"
        )


def _unavailable_result(
    *,
    model_id: str,
    local_files_only: bool,
    allow_downloads: bool,
    target_store_path: Path,
    sequence_length: int,
    num_examples: int,
    reason: str,
    error: HFTeacherUnavailable,
) -> HFTeacherSpecimenSmokeResult:
    return HFTeacherSpecimenSmokeResult(
        status="unavailable",
        scope="tiny_hf_causal_lm_teacher_specimen_smoke",
        model_id=model_id,
        local_files_only=local_files_only,
        allow_downloads=allow_downloads,
        target_store_path=str(target_store_path),
        target_store_validated=False,
        vocab_contract_extracted=False,
        tokenizer_id=None,
        tokenizer_hash=None,
        vocab_size=None,
        sequence_length=sequence_length,
        num_examples=num_examples,
        target_type=None,
        logits_shape=None,
        claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
        reason=reason,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _unavailable_reason(message: str) -> str:
    lowered = message.lower()
    if "transformers is not installed" in lowered:
        return "transformers_not_installed"
    if "local_files_only=true" in lowered or "not cached" in lowered:
        return "model_not_available_in_local_cache"
    return "optional_dependency_unavailable"


def run_hf_teacher_specimen_swap_smoke(
    specimens: Sequence[HFTeacherSpecimenConfig],
    *,
    target_store_root: str | Path,
    backends: Mapping[str, HFTeacherBackend] | None = None,
) -> HFTeacherSpecimenSwapReport:
    if not specimens:
        raise ValueError("specimens must contain at least one specimen")

    root = Path(target_store_root)
    results = []
    for index, specimen in enumerate(specimens):
        result = run_hf_teacher_specimen_smoke(
            target_store=root / _specimen_store_name(index, specimen.model_id),
            model_id=specimen.model_id,
            prompts=specimen.prompts,
            sequence_length=specimen.sequence_length,
            local_files_only=specimen.local_files_only,
            allow_downloads=specimen.allow_downloads,
            backend=(backends or {}).get(specimen.model_id),
        )
        results.append(result)

    passed = sum(result.status == "pass" for result in results)
    unavailable = sum(result.status == "unavailable" for result in results)
    failed = sum(result.status == "fail" for result in results)
    status = _swap_status(
        specimen_count=len(results),
        passed=passed,
        unavailable=unavailable,
        failed=failed,
    )
    return HFTeacherSpecimenSwapReport(
        status=status,
        scope="second_teacher_specimen_swap_smoke",
        specimen_count=len(results),
        passed=passed,
        unavailable=unavailable,
        failed=failed,
        model_ids=tuple(specimen.model_id for specimen in specimens),
        specimens=tuple(results),
        claims_not_made=HF_SPECIMEN_CLAIMS_NOT_MADE,
    )


def _swap_status(
    *,
    specimen_count: int,
    passed: int,
    unavailable: int,
    failed: int,
) -> str:
    if failed:
        return "fail"
    if passed == specimen_count:
        return "pass"
    if unavailable == specimen_count:
        return "unavailable"
    return "partial"


def _specimen_store_name(index: int, model_id: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in model_id.lower())
    return f"{index:02d}-{safe.strip('-') or 'specimen'}"
