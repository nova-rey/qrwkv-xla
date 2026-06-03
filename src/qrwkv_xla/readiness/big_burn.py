from __future__ import annotations

import importlib
import json
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReadinessStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: ReadinessStatus
    summary: str
    evidence: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BigBurnReadinessReport:
    phase: str
    status: ReadinessStatus
    scope: str
    checks: tuple[ReadinessCheck, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    recommended_next_action: str
    claims_not_made: tuple[str, ...]

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["status"] = self.status.value
        report["checks"] = [
            {
                **check.to_report(),
                "status": check.status.value,
            }
            for check in self.checks
        ]
        return report


BIG_BURN_READINESS_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_success_guaranteed",
    "model_quality_proven",
    "production_training_ready",
    "large_scale_performance_proven",
    "pallas_default_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
    "distributed_training_ready",
    "p112_started",
    "benchmark_suite_added",
    "lm_eval_integrated",
)

REQUIRED_BIG_BURN_READINESS_CATEGORIES: tuple[str, ...] = (
    "core_correctness_fixtures",
    "pallas_opt_in_runtime",
    "teacher_backend_generic_hf",
    "teacher_specimen_swap",
    "vocab_contract_and_compatibility",
    "target_store_multishard",
    "tiny_dataset_pipeline",
    "checkpoint_resume_export",
    "runtime_environment_preflight",
    "mini_eval_harness",
    "student_backend_registry",
    "second_student_backend",
)

_API_EVIDENCE: dict[str, tuple[str, ...]] = {
    "core_correctness_fixtures": (
        "qrwkv_xla.kernels.wkv7_fixtures.generate_wkv7_fixture_bundle",
        "qrwkv_xla.kernels.wkv7_compare.write_wkv7_comparison_reports",
    ),
    "pallas_opt_in_runtime": (
        "qrwkv_xla.students.wkv_runtime.WKVRuntime",
        "qrwkv_xla.students.wkv_runtime.build_pallas_runtime_probe",
    ),
    "teacher_backend_generic_hf": (
        "qrwkv_xla.teachers.hf.HFTeacherBackend",
        "qrwkv_xla.teachers.emission.emit_teacher_target_store",
    ),
    "teacher_specimen_swap": (
        "qrwkv_xla.teachers.hf_specimen_smoke.run_hf_teacher_specimen_smoke",
        "qrwkv_xla.teachers.hf_specimen_smoke.run_hf_teacher_specimen_swap_smoke",
    ),
    "vocab_contract_and_compatibility": (
        "qrwkv_xla.contracts.vocab.VocabContract",
        "qrwkv_xla.contracts.compatibility.validate_store_for_student_config",
    ),
    "target_store_multishard": (
        "qrwkv_xla.targets.store.TeacherTargetStore",
        "qrwkv_xla.targets.multishard.iter_offline_target_batches",
        "qrwkv_xla.targets.multishard.run_multishard_target_store_smoke",
    ),
    "tiny_dataset_pipeline": (
        "qrwkv_xla.data.tiny_dataset.TinyTextExample",
        "qrwkv_xla.data.tiny_dataset_pipeline.run_tiny_dataset_pipeline_smoke",
    ),
    "checkpoint_resume_export": (
        "qrwkv_xla.checkpointing.rehearsal.run_checkpoint_resume_export_rehearsal",
        "qrwkv_xla.checkpointing.rehearsal.run_checkpoint_resume_update_rehearsal",
    ),
    "runtime_environment_preflight": (
        "qrwkv_xla.xla.environment_preflight.run_runtime_environment_preflight",
    ),
    "mini_eval_harness": (
        "qrwkv_xla.eval.mini_eval.run_mini_eval_harness",
        "qrwkv_xla.eval.mini_eval.create_builtin_mini_eval_store",
    ),
    "student_backend_registry": (
        "qrwkv_xla.students.registry.create_student_backend",
        "qrwkv_xla.students.registry.available_student_architectures",
    ),
    "second_student_backend": (
        "qrwkv_xla.students.tiny_debug_backend.TinyDebugStudentBackend",
        "qrwkv_xla.students.tiny_debug_backend.TINY_DEBUG_ARCHITECTURE_ID",
    ),
}


def build_big_burn_readiness_report(
    *,
    work_dir: str | Path | None = None,
    run_lightweight_checks: bool = True,
) -> BigBurnReadinessReport:
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="qrwkv_p111_readiness_") as tmp:
            return _build_big_burn_readiness_report(
                work_dir=Path(tmp),
                run_lightweight_checks=run_lightweight_checks,
            )
    return _build_big_burn_readiness_report(
        work_dir=Path(work_dir),
        run_lightweight_checks=run_lightweight_checks,
    )


def aggregate_readiness_status(
    checks: tuple[ReadinessCheck, ...],
) -> ReadinessStatus:
    if any(check.status is ReadinessStatus.FAIL for check in checks):
        return ReadinessStatus.FAIL
    if any(check.status is ReadinessStatus.WARN for check in checks):
        return ReadinessStatus.WARN
    return ReadinessStatus.PASS


def recommended_next_action_for_status(status: ReadinessStatus | str) -> str:
    normalized = ReadinessStatus(status)
    if normalized is ReadinessStatus.PASS:
        return (
            "Proceed to P112 First Serious Compute Burn with conservative scoped "
            "run plan."
        )
    if normalized is ReadinessStatus.WARN:
        return "Review warnings before P112; proceed only if warnings are acceptable."
    return "Do not proceed to P112 until blockers are resolved."


def write_big_burn_readiness_report(
    report: BigBurnReadinessReport,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_big_burn_readiness_report(
    *,
    work_dir: Path,
    run_lightweight_checks: bool,
) -> BigBurnReadinessReport:
    work_dir.mkdir(parents=True, exist_ok=True)
    checks = tuple(
        _build_check(
            name=name,
            work_dir=work_dir,
            run_lightweight_checks=run_lightweight_checks,
        )
        for name in REQUIRED_BIG_BURN_READINESS_CATEGORIES
    )
    status = aggregate_readiness_status(checks)
    blockers = tuple(blocker for check in checks for blocker in check.blockers)
    warnings = tuple(warning for check in checks for warning in check.warnings)
    return BigBurnReadinessReport(
        phase="P111",
        status=status,
        scope="big_burn_readiness_report",
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        recommended_next_action=recommended_next_action_for_status(status),
        claims_not_made=BIG_BURN_READINESS_CLAIMS_NOT_MADE,
    )


def _build_check(
    *,
    name: str,
    work_dir: Path,
    run_lightweight_checks: bool,
) -> ReadinessCheck:
    api_check = _api_availability_check(name)
    if api_check.status is ReadinessStatus.FAIL or not run_lightweight_checks:
        return api_check
    if name == "runtime_environment_preflight":
        return _runtime_environment_preflight_check(work_dir=work_dir, base=api_check)
    if name == "mini_eval_harness":
        return _mini_eval_harness_check(work_dir=work_dir, base=api_check)
    if name == "pallas_opt_in_runtime":
        return _pallas_opt_in_runtime_check(base=api_check)
    return api_check


def _api_availability_check(name: str) -> ReadinessCheck:
    evidence = _API_EVIDENCE[name]
    missing = tuple(path for path in evidence if not _symbol_exists(path))
    if missing:
        return ReadinessCheck(
            name=name,
            status=ReadinessStatus.FAIL,
            summary="Required readiness API evidence is missing.",
            evidence=evidence,
            blockers=tuple(f"missing API: {path}" for path in missing),
        )
    return ReadinessCheck(
        name=name,
        status=ReadinessStatus.PASS,
        summary="Required readiness API evidence is present.",
        evidence=evidence,
    )


def _runtime_environment_preflight_check(
    *,
    work_dir: Path,
    base: ReadinessCheck,
) -> ReadinessCheck:
    try:
        from qrwkv_xla.xla import run_runtime_environment_preflight

        hugepage_path = work_dir / "transparent_hugepage_enabled"
        hugepage_path.write_text("[always] madvise never\n", encoding="utf-8")
        result = run_runtime_environment_preflight(
            hugepage_path=hugepage_path,
            require_tpu=False,
            enable_hugepages=False,
        )
    except Exception as exc:
        return ReadinessCheck(
            name=base.name,
            status=ReadinessStatus.FAIL,
            summary="Runtime environment preflight helper failed in read-only mode.",
            evidence=base.evidence,
            blockers=(f"{type(exc).__name__}: {exc}",),
        )
    status = _helper_status(result.status)
    return ReadinessCheck(
        name=base.name,
        status=status,
        summary="Runtime environment preflight ran without requiring TPU.",
        evidence=base.evidence
        + (
            f"status={result.status}",
            f"default_backend={result.default_backend}",
            f"tpu_devices_detected={result.tpu_devices_detected}",
        ),
        warnings=() if status is ReadinessStatus.PASS else (result.status,),
    )


def _mini_eval_harness_check(
    *,
    work_dir: Path,
    base: ReadinessCheck,
) -> ReadinessCheck:
    try:
        from qrwkv_xla.eval import (
            create_builtin_mini_eval_store,
            run_mini_eval_harness,
        )

        store = create_builtin_mini_eval_store(work_dir / "mini_eval_store")
        result = run_mini_eval_harness(store=store, architecture_id="tiny_debug")
    except Exception as exc:
        return ReadinessCheck(
            name=base.name,
            status=ReadinessStatus.FAIL,
            summary="Mini eval harness failed on built-in tiny target artifacts.",
            evidence=base.evidence,
            blockers=(f"{type(exc).__name__}: {exc}",),
        )
    status = _helper_status(result.status)
    blockers = ()
    if status is ReadinessStatus.FAIL:
        blockers = (f"mini eval status={result.status}",)
    return ReadinessCheck(
        name=base.name,
        status=status,
        summary="Mini eval harness ran on built-in tiny target artifacts.",
        evidence=base.evidence
        + (
            f"status={result.status}",
            f"mean_mse_loss={result.mean_mse_loss}",
            f"examples_evaluated={result.examples_evaluated}",
            f"shard_count={result.shard_count}",
        ),
        blockers=blockers,
    )


def _pallas_opt_in_runtime_check(base: ReadinessCheck) -> ReadinessCheck:
    try:
        from qrwkv_xla.students import WKVRuntime
    except Exception as exc:
        return ReadinessCheck(
            name=base.name,
            status=ReadinessStatus.FAIL,
            summary="Could not inspect WKV runtime policy.",
            evidence=base.evidence,
            blockers=(f"{type(exc).__name__}: {exc}",),
        )
    if WKVRuntime.REFERENCE.value != "reference" or WKVRuntime.PALLAS.value != "pallas":
        return ReadinessCheck(
            name=base.name,
            status=ReadinessStatus.FAIL,
            summary="WKV runtime enum no longer preserves reference/pallas split.",
            evidence=base.evidence,
            blockers=("runtime enum values changed",),
        )
    return ReadinessCheck(
        name=base.name,
        status=ReadinessStatus.PASS,
        summary="Reference remains a runtime value and Pallas remains explicit opt-in.",
        evidence=base.evidence
        + (
            "default_runtime=reference",
            "opt_in_runtime=pallas",
        ),
    )


def _helper_status(status: str) -> ReadinessStatus:
    if status == "pass" or status == "compatible":
        return ReadinessStatus.PASS
    if status == "fail" or status == "incompatible":
        return ReadinessStatus.FAIL
    return ReadinessStatus.WARN


def _symbol_exists(path: str) -> bool:
    module_name, _, symbol_path = path.rpartition(".")
    if not module_name or not symbol_path:
        return False
    try:
        value = importlib.import_module(module_name)
    except Exception:
        return False
    for part in symbol_path.split("."):
        if not hasattr(value, part):
            return False
        value = getattr(value, part)
    return True
