from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.contracts import (
    TeacherStudentCompatibility,
    VocabContract,
    validate_store_for_student_config,
    vocab_contract_from_metadata,
)
from qrwkv_xla.students import (
    CURRENT_QRWKV_ARCHITECTURE_ID,
    SelectedStudentConfig,
    StudentBackend,
    WKVRuntime,
    create_student_backend,
    qrwkv_student_config_from_vocab_contract,
)
from qrwkv_xla.targets import OfflineTargetBatch, TeacherTargetStore
from qrwkv_xla.targets.consumption import load_offline_target_batch, mse_logits_loss

REAL_TEACHER_OVERFIT_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
    "production_distillation_ready",
    "full_model_quality_proven",
    "large_scale_performance_proven",
)


@dataclass(frozen=True)
class RealTeacherOverfitResult:
    status: str
    initial_loss: float | None
    final_loss: float | None
    loss_moved: bool
    loss_finite: bool
    steps: int
    path_used: str
    compatibility_status: str
    compatibility_reason: str
    teacher_model_id: str
    tokenizer_id: str
    vocab_size: int
    student_architecture_id: str
    student_runtime: str
    training_kind: str
    teacher_created_by: str
    claims_not_made: tuple[str, ...] = REAL_TEACHER_OVERFIT_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report.update(
            {
                "phase": "P103",
                "scope": "tiny_real_teacher_overfit_rehearsal",
                "hf_required_for_baseline_ci": False,
                "internet_required": False,
                "gpu_or_tpu_required": False,
            }
        )
        return report


def run_tiny_real_teacher_overfit_rehearsal(
    *,
    store: TeacherTargetStore,
    architecture_id: str | None = None,
    runtime: str | WKVRuntime | None = None,
    student_vocab_contract: VocabContract | None = None,
    steps: int = 3,
    learning_rate: float = 0.3,
    key: jax.Array | None = None,
) -> RealTeacherOverfitResult:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    teacher_contract = vocab_contract_from_metadata(store.metadata)
    selected_contract = student_vocab_contract or teacher_contract
    selected = qrwkv_student_config_from_vocab_contract(
        selected_contract,
        runtime=runtime,
    )
    selected_architecture_id = architecture_id or CURRENT_QRWKV_ARCHITECTURE_ID
    backend = create_student_backend(
        vocab_contract=selected_contract,
        architecture_id=selected_architecture_id,
        runtime=runtime,
    )
    compatibility_target = selected
    if selected_architecture_id != CURRENT_QRWKV_ARCHITECTURE_ID:
        compatibility_target = backend
    compatibility = validate_store_for_student_config(store, compatibility_target)
    if not compatibility.compatible:
        return _blocked_result(
            store=store,
            selected=selected,
            selected_architecture_id=selected_architecture_id,
            compatibility=compatibility,
            steps=steps,
        )

    batch = load_offline_target_batch(store)
    student_logits = _student_logits(
        backend=backend,
        batch=batch,
        key=jax.random.PRNGKey(0) if key is None else key,
    )
    params = _init_adapter_params(batch)
    loss_and_grad = jax.value_and_grad(_adapter_loss)
    initial_loss = _adapter_loss(params, student_logits, batch)

    for _ in range(steps):
        _loss, grads = loss_and_grad(params, student_logits, batch)
        params = _sgd_update(params, grads, learning_rate=learning_rate)

    final_loss = _adapter_loss(params, student_logits, batch)
    loss_finite = bool(jnp.isfinite(initial_loss) & jnp.isfinite(final_loss))
    loss_moved = bool(final_loss < initial_loss)
    return RealTeacherOverfitResult(
        status="pass" if loss_finite and loss_moved else "fail",
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        loss_moved=loss_moved,
        loss_finite=loss_finite,
        steps=steps,
        path_used="tiny_trainable_logit_head",
        compatibility_status=compatibility.status.value,
        compatibility_reason=compatibility.reason,
        teacher_model_id=store.metadata.model_id,
        tokenizer_id=store.metadata.tokenizer_id,
        vocab_size=store.metadata.vocab_size,
        student_architecture_id=selected_architecture_id,
        student_runtime=selected.runtime.value,
        training_kind="tiny_controlled_rehearsal",
        teacher_created_by=store.metadata.created_by,
    )


def _student_logits(
    *,
    backend: StudentBackend,
    batch: OfflineTargetBatch,
    key: jax.Array,
) -> jax.Array:
    params = backend.init_params(key)
    output, _state = backend.forward_full(
        params,
        jnp.asarray(batch.input_ids),
        attention_mask=jnp.asarray(batch.attention_mask),
    )
    return backend.logits(output)


def _init_adapter_params(batch: OfflineTargetBatch) -> dict[str, jnp.ndarray]:
    input_ids = jnp.asarray(batch.input_ids)
    teacher_logits = jnp.asarray(batch.teacher_logits)
    if input_ids.ndim != 2 or teacher_logits.ndim != 3:
        raise ValueError(
            "real-teacher overfit batch must contain [N,T] inputs and [N,T,V] logits"
        )
    num_examples, sequence_length = input_ids.shape
    vocab_size = teacher_logits.shape[2]
    return {
        "row_bias": jnp.zeros((num_examples, 1, 1), dtype=jnp.float32),
        "position_bias": jnp.zeros((1, sequence_length, 1), dtype=jnp.float32),
        "token_bias": jnp.zeros((vocab_size, 1), dtype=jnp.float32),
        "vocab_bias": jnp.zeros((1, 1, vocab_size), dtype=jnp.float32),
        "bias": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _adapter_logits(
    params: dict[str, jnp.ndarray],
    student_logits: jax.Array,
    batch: OfflineTargetBatch,
) -> jnp.ndarray:
    input_ids = jnp.asarray(batch.input_ids, dtype=jnp.int32)
    return (
        jnp.asarray(student_logits)
        + params["row_bias"]
        + params["position_bias"]
        + params["token_bias"][input_ids]
        + params["vocab_bias"]
        + params["bias"]
    )


def _adapter_loss(
    params: dict[str, jnp.ndarray],
    student_logits: jax.Array,
    batch: OfflineTargetBatch,
) -> jnp.ndarray:
    return mse_logits_loss(
        _adapter_logits(params, student_logits, batch),
        batch.teacher_logits,
    )


def _sgd_update(
    params: dict[str, jnp.ndarray],
    grads: dict[str, jnp.ndarray],
    *,
    learning_rate: float,
) -> dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(
        lambda param, grad: param - learning_rate * grad,
        params,
        grads,
    )


def _blocked_result(
    *,
    store: TeacherTargetStore,
    selected: SelectedStudentConfig,
    selected_architecture_id: str,
    compatibility: TeacherStudentCompatibility,
    steps: int,
) -> RealTeacherOverfitResult:
    return RealTeacherOverfitResult(
        status=compatibility.status.value,
        initial_loss=None,
        final_loss=None,
        loss_moved=False,
        loss_finite=False,
        steps=steps,
        path_used="blocked_before_update",
        compatibility_status=compatibility.status.value,
        compatibility_reason=compatibility.reason,
        teacher_model_id=store.metadata.model_id,
        tokenizer_id=store.metadata.tokenizer_id,
        vocab_size=store.metadata.vocab_size,
        student_architecture_id=selected_architecture_id,
        student_runtime=selected.runtime.value,
        training_kind="none",
        teacher_created_by=store.metadata.created_by,
    )
