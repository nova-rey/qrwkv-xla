from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.contracts import (
    CompatibilityStatus,
    TeacherStudentCompatibility,
    validate_store_for_student_config,
    vocab_contract_from_metadata,
)
from qrwkv_xla.students import (
    CurrentQRWKVStudentBackend,
    SelectedStudentConfig,
    qrwkv_student_config_from_vocab_contract,
)
from qrwkv_xla.targets.consumption import (
    load_offline_target_batch,
    mse_logits_loss,
)
from qrwkv_xla.targets.store import TeacherTargetStore

REAL_TEACHER_CONSUMPTION_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
    "production_distillation_ready",
    "full_model_quality_proven",
)


@dataclass(frozen=True)
class RealTeacherConsumptionResult:
    status: str
    loss: float | None
    loss_finite: bool
    compatibility_status: str
    compatibility_reason: str
    target_type: str
    teacher_model_id: str
    tokenizer_id: str
    vocab_size: int
    student_backend: str
    student_runtime: str
    training_performed: bool
    teacher_logits_shape: tuple[int, ...] | None
    student_logits_shape: tuple[int, ...] | None
    claims_not_made: tuple[str, ...] = REAL_TEACHER_CONSUMPTION_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report.update(
            {
                "phase": "P100",
                "scope": "real_teacher_offline_consumption_smoke",
            }
        )
        return report


def run_real_teacher_offline_consumption_smoke(
    *,
    store: TeacherTargetStore,
    selected_student: SelectedStudentConfig | None = None,
    runtime: str | None = None,
    key: jax.Array | None = None,
) -> RealTeacherConsumptionResult:
    teacher_contract = vocab_contract_from_metadata(store.metadata)
    selected = selected_student or qrwkv_student_config_from_vocab_contract(
        teacher_contract,
        runtime=runtime,
    )
    compatibility = validate_store_for_student_config(store, selected)
    if not compatibility.compatible:
        return _result(
            status=compatibility.status.value,
            store=store,
            selected=selected,
            compatibility=compatibility,
            loss=None,
            loss_finite=False,
            teacher_logits_shape=None,
            student_logits_shape=None,
        )

    batch = load_offline_target_batch(store)
    backend = CurrentQRWKVStudentBackend.from_config(
        selected.architecture,
        vocab_size=selected.config.vocab_size,
        hidden_size=selected.config.hidden_size,
        num_layers=selected.config.num_layers,
        num_heads=selected.config.num_heads,
        num_kv_heads=selected.config.num_kv_heads,
        emit_logits=True,
        tie_embeddings=selected.config.tie_embeddings,
        emit_mixer_outputs=selected.config.emit_mixer_outputs,
        runtime=selected.runtime,
    )
    params = backend.init_params(jax.random.PRNGKey(0) if key is None else key)
    output, _state = backend.forward_full(
        params,
        jnp.asarray(batch.input_ids),
        attention_mask=jnp.asarray(batch.attention_mask),
    )
    student_logits = backend.logits(output)
    loss = mse_logits_loss(student_logits, batch.teacher_logits)
    loss_finite = bool(jnp.isfinite(loss))
    return _result(
        status="pass" if loss_finite else "fail",
        store=store,
        selected=selected,
        compatibility=compatibility,
        loss=float(loss),
        loss_finite=loss_finite,
        teacher_logits_shape=tuple(batch.teacher_logits.shape),
        student_logits_shape=tuple(student_logits.shape),
    )


def _result(
    *,
    status: str,
    store: TeacherTargetStore,
    selected: SelectedStudentConfig,
    compatibility: TeacherStudentCompatibility,
    loss: float | None,
    loss_finite: bool,
    teacher_logits_shape: tuple[int, ...] | None,
    student_logits_shape: tuple[int, ...] | None,
) -> RealTeacherConsumptionResult:
    return RealTeacherConsumptionResult(
        status=status,
        loss=loss,
        loss_finite=loss_finite,
        compatibility_status=compatibility.status.value,
        compatibility_reason=compatibility.reason,
        target_type=store.metadata.target_type,
        teacher_model_id=store.metadata.model_id,
        tokenizer_id=store.metadata.tokenizer_id,
        vocab_size=store.metadata.vocab_size,
        student_backend=selected.architecture,
        student_runtime=selected.runtime.value,
        training_performed=False,
        teacher_logits_shape=teacher_logits_shape,
        student_logits_shape=student_logits_shape,
    )


def compatibility_passed(result: RealTeacherConsumptionResult) -> bool:
    return result.compatibility_status == CompatibilityStatus.COMPATIBLE.value
