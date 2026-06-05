from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.contracts import (
    VocabContract,
    validate_store_for_student_config,
    vocab_contract_from_metadata,
)
from qrwkv_xla.distill.target_dispatch import dispatch_teacher_target_loss
from qrwkv_xla.students import WKVRuntime, create_student_backend
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
    iter_teacher_target_batches,
)

MINI_EVAL_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "benchmark_complete",
    "model_quality_proven",
    "training_ready",
    "lm_eval_integrated",
    "qwen_specific_support",
    "production_eval_ready",
    "tokenizer_remapping_supported",
)


@dataclass(frozen=True)
class MiniEvalResult:
    phase: str
    status: str
    scope: str
    mean_mse_loss: float | None
    loss_finite: bool
    shard_count: int
    examples_evaluated: int
    tokens_evaluated: int
    elements_evaluated: int
    compatibility_status: str
    compatibility_reason: str
    architecture_id: str
    runtime: str
    target_type: str
    teacher_target_type: str
    distillation_loss_type: str | None
    vocab_size: int
    top1_agreement: float | None
    mean_distillation_loss: float | None = None
    head_loss: float | None = None
    tail_loss: float | None = None
    tail_loss_weight: float = 0.0
    bucket_loss_weight: float = 0.0
    bucket_shape_loss: float | None = None
    bucket_shape_loss_weight: float = 0.0
    bucket_shape_loss_type: str | None = None
    bucket_count: int | None = None
    top_k: int | None = None
    mean_top_mass: float | None = None
    mean_tail_mass: float | None = None
    mean_teacher_top_mass: float | None = None
    mean_teacher_tail_mass: float | None = None
    mean_student_tail_mass: float | None = None
    mean_teacher_bucket_mass: float | None = None
    mean_student_bucket_mass: float | None = None
    mean_bucket_shape_kl: float | None = None
    mean_teacher_entropy: float | None = None
    claims_not_made: tuple[str, ...] = MINI_EVAL_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def run_mini_eval_harness(
    *,
    store: TeacherTargetStore,
    architecture_id: str | None = None,
    runtime: str | WKVRuntime | None = None,
    student_vocab_contract: VocabContract | None = None,
    tail_loss_weight: float = 0.0,
    sparse_head_loss_weight: float = 1.0,
    bucket_shape_loss_weight: float = 0.0,
    bucket_shape_loss_type: str = "kl",
) -> MiniEvalResult:
    if tail_loss_weight < 0:
        raise ValueError(f"tail_loss_weight must be >= 0, got {tail_loss_weight}")
    if sparse_head_loss_weight <= 0:
        raise ValueError(
            f"sparse_head_loss_weight must be > 0, got {sparse_head_loss_weight}"
        )
    if bucket_shape_loss_weight < 0:
        raise ValueError(
            f"bucket_shape_loss_weight must be >= 0, got {bucket_shape_loss_weight}"
        )
    if bucket_shape_loss_type not in {"kl", "log_mse"}:
        raise ValueError(
            "bucket_shape_loss_type must be 'kl' or 'log_mse', "
            f"got {bucket_shape_loss_type!r}"
        )
    teacher_contract = vocab_contract_from_metadata(store.metadata)
    selected_contract = student_vocab_contract or teacher_contract
    backend = create_student_backend(
        vocab_contract=selected_contract,
        architecture_id=architecture_id,
        runtime=runtime,
    )
    loss_mode = (
        "topk_tail_sparse"
        if store.metadata.target_type
        in {"topk_with_tail_v0", "cascaded_soft_labels_v1"}
        else "direct_logits"
    )
    compatibility = validate_store_for_student_config(
        store,
        backend,
        loss_mode=loss_mode,
    )
    selected_architecture = getattr(backend, "architecture_id", architecture_id)
    selected_runtime = getattr(backend, "runtime", runtime or WKVRuntime.REFERENCE)
    selected_runtime_value = getattr(selected_runtime, "value", str(selected_runtime))
    if not compatibility.compatible:
        return MiniEvalResult(
            phase="P110",
            status=compatibility.status.value,
            scope="mini_eval_harness_smoke",
            mean_mse_loss=None,
            loss_finite=False,
            shard_count=store.metadata.shard_count,
            examples_evaluated=0,
            tokens_evaluated=0,
            elements_evaluated=0,
            compatibility_status=compatibility.status.value,
            compatibility_reason=compatibility.reason,
            architecture_id=str(selected_architecture),
            runtime=str(selected_runtime_value),
            target_type=store.metadata.target_type,
            teacher_target_type=store.metadata.target_type,
            distillation_loss_type=None,
            vocab_size=store.metadata.vocab_size,
            top1_agreement=None,
        )

    params = backend.init_params(jax.random.PRNGKey(110))
    total_squared_error = 0.0
    total_distillation_loss = 0.0
    total_head_loss = 0.0
    total_tail_loss = 0.0
    total_bucket_shape_loss = 0.0
    total_metric_weight = 0.0
    total_top_mass = 0.0
    total_tail_mass = 0.0
    total_teacher_top_mass = 0.0
    total_teacher_tail_mass = 0.0
    total_student_tail_mass = 0.0
    total_teacher_bucket_mass = 0.0
    total_student_bucket_mass = 0.0
    total_bucket_shape_kl = 0.0
    total_teacher_entropy = 0.0
    top_k: int | None = None
    bucket_count: int | None = None
    distillation_loss_type: str | None = None
    total_examples = 0
    total_tokens = 0
    total_elements = 0
    top1_matches = 0

    for batch in iter_teacher_target_batches(store):
        output, _state = backend.forward_full(
            params,
            jnp.asarray(batch.input_ids),
            attention_mask=jnp.asarray(batch.attention_mask),
        )
        student_logits = np.asarray(backend.logits(output), dtype=np.float32)
        if student_logits.shape[-1] != store.metadata.vocab_size:
            raise ValueError(
                "student vocab size does not match teacher vocab contract: "
                f"student={student_logits.shape[-1]} "
                f"teacher={store.metadata.vocab_size}"
            )
        loss_report = dispatch_teacher_target_loss(
            student_logits=jnp.asarray(student_logits),
            target_batch=batch,
            tail_loss_weight=tail_loss_weight,
            sparse_head_loss_weight=sparse_head_loss_weight,
            bucket_shape_loss_weight=bucket_shape_loss_weight,
            bucket_shape_loss_type=bucket_shape_loss_type,
        )
        batch_weight = float(loss_report.token_count)
        total_distillation_loss += float(loss_report.total_loss) * batch_weight
        total_head_loss += float(loss_report.head_loss) * batch_weight
        total_tail_loss += float(loss_report.tail_loss) * batch_weight
        if loss_report.bucket_shape_loss is not None:
            total_bucket_shape_loss += (
                float(loss_report.bucket_shape_loss) * batch_weight
            )
        total_metric_weight += batch_weight
        top_k = loss_report.top_k if loss_report.top_k is not None else top_k
        bucket_count = (
            loss_report.bucket_count
            if loss_report.bucket_count is not None
            else bucket_count
        )
        distillation_loss_type = loss_report.distillation_loss_type
        if loss_report.mean_top_mass is not None:
            total_top_mass += float(loss_report.mean_top_mass) * batch_weight
        if loss_report.mean_tail_mass is not None:
            total_tail_mass += float(loss_report.mean_tail_mass) * batch_weight
        if loss_report.mean_teacher_top_mass is not None:
            total_teacher_top_mass += (
                float(loss_report.mean_teacher_top_mass) * batch_weight
            )
        if loss_report.mean_teacher_tail_mass is not None:
            total_teacher_tail_mass += (
                float(loss_report.mean_teacher_tail_mass) * batch_weight
            )
        if loss_report.mean_student_tail_mass is not None:
            total_student_tail_mass += (
                float(loss_report.mean_student_tail_mass) * batch_weight
            )
        if loss_report.mean_teacher_bucket_mass is not None:
            total_teacher_bucket_mass += (
                float(loss_report.mean_teacher_bucket_mass) * batch_weight
            )
        if loss_report.mean_student_bucket_mass is not None:
            total_student_bucket_mass += (
                float(loss_report.mean_student_bucket_mass) * batch_weight
            )
        if loss_report.mean_bucket_shape_kl is not None:
            total_bucket_shape_kl += (
                float(loss_report.mean_bucket_shape_kl) * batch_weight
            )
        if loss_report.mean_teacher_entropy is not None:
            total_teacher_entropy += (
                float(loss_report.mean_teacher_entropy) * batch_weight
            )

        if batch.teacher_logits is not None:
            teacher_logits = np.asarray(batch.teacher_logits, dtype=np.float32)
            if student_logits.shape != teacher_logits.shape:
                raise ValueError(
                    "student logits and teacher logits shape mismatch: "
                    f"{student_logits.shape} vs {teacher_logits.shape}"
                )

            diff = student_logits - teacher_logits
            total_squared_error += float(np.sum(np.square(diff)))
            total_elements += int(teacher_logits.size)
            top1_matches += int(
                np.sum(
                    np.argmax(student_logits, axis=-1)
                    == np.argmax(teacher_logits, axis=-1)
                )
            )
        elif batch.top_log_probs is not None:
            total_elements += int(batch.top_log_probs.size)
        total_examples += int(batch.input_ids.shape[0])
        total_tokens += int(batch.input_ids.size)

    mean_mse_loss = (
        total_squared_error / total_elements
        if total_elements > 0 and store.metadata.target_type != "topk_with_tail_v0"
        else None
    )
    mean_distillation_loss = (
        total_distillation_loss / total_metric_weight
        if total_metric_weight > 0
        else float("nan")
    )
    mean_head_loss = (
        total_head_loss / total_metric_weight if total_metric_weight > 0 else None
    )
    mean_tail_loss = (
        total_tail_loss / total_metric_weight if total_metric_weight > 0 else None
    )
    mean_bucket_shape_loss = (
        total_bucket_shape_loss / total_metric_weight
        if total_metric_weight > 0 and bucket_count is not None
        else None
    )
    loss_finite = bool(np.isfinite(mean_distillation_loss))
    top1_agreement = (
        top1_matches / total_tokens
        if total_tokens > 0 and store.metadata.target_type != "topk_with_tail_v0"
        else None
    )
    return MiniEvalResult(
        phase="P110",
        status="pass" if loss_finite else "fail",
        scope="mini_eval_harness_smoke",
        mean_mse_loss=None if mean_mse_loss is None else float(mean_mse_loss),
        loss_finite=loss_finite,
        shard_count=store.metadata.shard_count,
        examples_evaluated=total_examples,
        tokens_evaluated=total_tokens,
        elements_evaluated=total_elements,
        compatibility_status=compatibility.status.value,
        compatibility_reason=compatibility.reason,
        architecture_id=str(selected_architecture),
        runtime=str(selected_runtime_value),
        target_type=store.metadata.target_type,
        teacher_target_type=store.metadata.target_type,
        distillation_loss_type=distillation_loss_type,
        vocab_size=store.metadata.vocab_size,
        top1_agreement=None if top1_agreement is None else float(top1_agreement),
        mean_distillation_loss=float(mean_distillation_loss),
        head_loss=None if mean_head_loss is None else float(mean_head_loss),
        tail_loss=None if mean_tail_loss is None else float(mean_tail_loss),
        tail_loss_weight=tail_loss_weight,
        bucket_loss_weight=bucket_shape_loss_weight,
        bucket_shape_loss=(
            None if mean_bucket_shape_loss is None else float(mean_bucket_shape_loss)
        ),
        bucket_shape_loss_weight=bucket_shape_loss_weight,
        bucket_shape_loss_type=(
            bucket_shape_loss_type if bucket_count is not None else None
        ),
        bucket_count=bucket_count,
        top_k=top_k,
        mean_top_mass=(
            total_top_mass / total_metric_weight if total_top_mass else None
        ),
        mean_tail_mass=(
            total_tail_mass / total_metric_weight if total_tail_mass else None
        ),
        mean_teacher_top_mass=(
            total_teacher_top_mass / total_metric_weight
            if total_metric_weight > 0 and top_k is not None
            else None
        ),
        mean_teacher_tail_mass=(
            total_teacher_tail_mass / total_metric_weight
            if total_metric_weight > 0 and top_k is not None
            else None
        ),
        mean_student_tail_mass=(
            total_student_tail_mass / total_metric_weight
            if total_metric_weight > 0 and bucket_count is not None
            else None
        ),
        mean_teacher_bucket_mass=(
            total_teacher_bucket_mass / total_metric_weight
            if total_metric_weight > 0 and bucket_count is not None
            else None
        ),
        mean_student_bucket_mass=(
            total_student_bucket_mass / total_metric_weight
            if total_metric_weight > 0 and bucket_count is not None
            else None
        ),
        mean_bucket_shape_kl=(
            total_bucket_shape_kl / total_metric_weight
            if total_metric_weight > 0 and bucket_count is not None
            else None
        ),
        mean_teacher_entropy=(
            total_teacher_entropy / total_metric_weight
            if total_teacher_entropy
            else None
        ),
    )


def write_mini_eval_report(result: MiniEvalResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def create_builtin_mini_eval_store(path: str | Path) -> TeacherTargetStore:
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="synthetic-mini-eval-teacher",
        model_family="synthetic",
        tokenizer_id="mini-eval-tokenizer",
        tokenizer_hash="mini-eval-tokenizer",
        vocab_size=8,
        target_type="full_logits",
        dtype="float32",
        sequence_length=3,
        num_examples=4,
        shard_count=2,
        created_by="P110MiniEvalHarness",
        created_at="2026-06-03T00:00:00Z",
        source={"kind": "synthetic"},
        provenance={"phase": "P110"},
    )
    store = TeacherTargetStore.create(path, metadata, overwrite=True)
    store.write_shard(0, _mini_eval_arrays(offset=0))
    store.write_shard(1, _mini_eval_arrays(offset=3))
    return TeacherTargetStore.open(store.root)


def _mini_eval_arrays(*, offset: int) -> dict[str, np.ndarray]:
    input_ids = (np.arange(offset, offset + 6, dtype=np.int32) % 8).reshape(2, 3)
    vocab = np.arange(8, dtype=np.float32)
    logits = (
        input_ids[:, :, None].astype(np.float32) * 0.125 + vocab[None, None, :] * 0.25
    )
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids, dtype=np.int32),
        "logits": logits.astype(np.float32),
    }
