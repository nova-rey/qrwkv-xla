from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.targets import OfflineTargetBatch, load_offline_target_batch
from qrwkv_xla.targets.consumption import mse_logits_loss
from qrwkv_xla.teachers import SyntheticTeacherBackend, emit_teacher_target_store

TINY_OVERFIT_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "qwen_supported",
    "production_distillation_ready",
    "full_model_quality_proven",
    "large_scale_performance_proven",
)


@dataclass(frozen=True)
class TinyOverfitResult:
    initial_loss: float
    final_loss: float
    steps: int
    loss_moved: bool
    loss_finite: bool
    path_used: str
    claims_not_made: tuple[str, ...] = TINY_OVERFIT_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report.update(
            {
                "phase": "P96",
                "status": "pass" if self.loss_moved and self.loss_finite else "fail",
                "scope": "tiny_overfit_rehearsal",
                "live_teacher_required": False,
                "hf_or_qwen_required": False,
                "gpu_or_tpu_required": False,
                "training_kind": "tiny_controlled_rehearsal",
            }
        )
        return report


def run_tiny_overfit_rehearsal(
    *,
    num_examples: int = 2,
    sequence_length: int = 3,
    vocab_size: int = 8,
    steps: int = 3,
    learning_rate: float = 0.3,
) -> TinyOverfitResult:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    with tempfile.TemporaryDirectory(prefix="qrwkv_xla_p96_") as tmpdir:
        store = emit_teacher_target_store(
            SyntheticTeacherBackend(vocab_size=vocab_size),
            Path(tmpdir) / "synthetic_targets",
            num_examples=num_examples,
            sequence_length=sequence_length,
        )
        batch = load_offline_target_batch(store)

    params = _init_tiny_head_params(batch)
    loss_and_grad = jax.value_and_grad(_tiny_head_loss)
    initial_loss = _tiny_head_loss(params, batch)

    for _ in range(steps):
        _loss, grads = loss_and_grad(params, batch)
        params = _sgd_update(params, grads, learning_rate=learning_rate)

    final_loss = _tiny_head_loss(params, batch)
    initial_loss_float = float(initial_loss)
    final_loss_float = float(final_loss)
    loss_finite = bool(jnp.isfinite(initial_loss) & jnp.isfinite(final_loss))
    loss_moved = bool(final_loss < initial_loss)

    return TinyOverfitResult(
        initial_loss=initial_loss_float,
        final_loss=final_loss_float,
        steps=steps,
        loss_moved=loss_moved,
        loss_finite=loss_finite,
        path_used="tiny_trainable_logit_head",
    )


def _init_tiny_head_params(batch: OfflineTargetBatch) -> dict[str, jnp.ndarray]:
    input_ids = jnp.asarray(batch.input_ids)
    teacher_logits = jnp.asarray(batch.teacher_logits)
    if input_ids.ndim != 2 or teacher_logits.ndim != 3:
        raise ValueError(
            "tiny overfit batch must contain [N,T] inputs and [N,T,V] logits"
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


def _tiny_head_logits(
    params: dict[str, jnp.ndarray],
    batch: OfflineTargetBatch,
) -> jnp.ndarray:
    input_ids = jnp.asarray(batch.input_ids, dtype=jnp.int32)
    return (
        params["row_bias"]
        + params["position_bias"]
        + params["token_bias"][input_ids]
        + params["vocab_bias"]
        + params["bias"]
    )


def _tiny_head_loss(
    params: dict[str, jnp.ndarray],
    batch: OfflineTargetBatch,
) -> jnp.ndarray:
    return mse_logits_loss(_tiny_head_logits(params, batch), batch.teacher_logits)


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
