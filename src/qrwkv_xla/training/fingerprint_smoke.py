from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.artifacts import FingerprintBatch, load_fingerprint_targets
from qrwkv_xla.training.fingerprint_loss import (
    FingerprintCorridorLossConfig,
    FingerprintCorridorLossOutput,
    compute_fingerprint_corridor_loss,
)
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats_at_positions,
)

FINGERPRINT_SMOKE_METRIC_KEYS: tuple[str, ...] = (
    "fingerprint/loss_total",
    "fingerprint/loss_entropy",
    "fingerprint/loss_top1_margin",
    "fingerprint/loss_top8_mass",
    "fingerprint/loss_top32_mass",
    "fingerprint/loss_tail_mass",
    "fingerprint/inside_entropy_rate",
    "fingerprint/inside_top1_margin_rate",
    "fingerprint/inside_top8_mass_rate",
    "fingerprint/inside_top32_mass_rate",
    "fingerprint/inside_tail_mass_rate",
    "fingerprint/inside_all_rate",
)

FINGERPRINT_SMOKE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "teacher_generation",
    "real_teacher_required",
    "exemplar_reservoir",
    "mixed_csl_fingerprint_training",
    "model_quality_proven",
    "production_training_ready",
    "gpu_or_tpu_required",
)


@dataclass(frozen=True)
class FingerprintTrainingSmokeConfig:
    artifact_dir: Path
    output_dir: Path
    steps: int = 3
    batch_size: int = 2
    learning_rate: float = 0.5
    seed: int = 0
    shuffle: bool = False
    max_records: int | None = None
    drop_remainder: bool = False
    loss_config: FingerprintCorridorLossConfig = FingerprintCorridorLossConfig()


@dataclass(frozen=True)
class FingerprintTrainingSmokeResult:
    status: str
    initial_loss: float
    final_loss: float
    steps: int
    loss_finite: bool
    loss_non_negative: bool
    loss_non_increasing: bool
    metrics: dict[str, float]
    output_dir: str
    metrics_path: str
    checkpoint_path: str
    report_path: str
    claims_not_made: tuple[str, ...] = FINGERPRINT_SMOKE_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "phase": "P136",
            "scope": "tiny_fingerprint_training_smoke",
            "fingerprint_only": True,
            "teacher_required": False,
            "hf_download_required": False,
            "gpu_or_tpu_required": False,
        }


def run_tiny_fingerprint_training_smoke(
    config: FingerprintTrainingSmokeConfig,
) -> FingerprintTrainingSmokeResult:
    if config.steps <= 0:
        raise ValueError(f"steps must be > 0, got {config.steps}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")

    dataset = load_fingerprint_targets(
        config.artifact_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        seed=config.seed,
        drop_remainder=config.drop_remainder,
        max_records=config.max_records,
    )
    if dataset.num_records == 0:
        raise ValueError("fingerprint smoke requires at least one target record")

    params = _init_tiny_fingerprint_student(
        max_seq_len=dataset.max_seq_len,
        vocab_size=dataset.vocab_size,
        seed=config.seed,
    )
    full_batch = next(
        load_fingerprint_targets(
            config.artifact_dir,
            batch_size=dataset.num_records,
            max_records=dataset.num_records,
            shuffle=False,
            drop_remainder=False,
        ).iter_batches()
    )
    initial_output = _loss_output(params, full_batch, config.loss_config)
    initial_loss = initial_output.loss

    for _ in range(config.steps):
        for batch in dataset.iter_batches():
            loss, grads = jax.value_and_grad(_loss_value)(
                params,
                batch,
                config.loss_config,
            )
            del loss
            params = jax.tree_util.tree_map(
                lambda param, grad: param - config.learning_rate * grad,
                params,
                grads,
            )

    final_output = _loss_output(params, full_batch, config.loss_config)
    final_loss = final_output.loss
    metrics = _metrics_from_output(final_output)
    initial_loss_float = float(initial_loss)
    final_loss_float = float(final_loss)
    loss_finite = bool(jnp.isfinite(initial_loss) & jnp.isfinite(final_loss))
    loss_non_negative = bool((initial_loss >= 0.0) & (final_loss >= 0.0))
    loss_non_increasing = bool(final_loss <= initial_loss + 1e-6)
    status = (
        "pass" if loss_finite and loss_non_negative and loss_non_increasing else "fail"
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "fingerprint_smoke_report.json"

    metrics_payload = {
        "initial_loss": initial_loss_float,
        "final_loss": final_loss_float,
        "steps": config.steps,
        **metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "kind": "tiny_fingerprint_position_logit_head",
                "max_seq_len": dataset.max_seq_len,
                "vocab_size": dataset.vocab_size,
                "steps": config.steps,
                "position_logits_shape": list(params["position_logits"].shape),
                "position_logits_l2": float(jnp.linalg.norm(params["position_logits"])),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = FingerprintTrainingSmokeResult(
        status=status,
        initial_loss=initial_loss_float,
        final_loss=final_loss_float,
        steps=config.steps,
        loss_finite=loss_finite,
        loss_non_negative=loss_non_negative,
        loss_non_increasing=loss_non_increasing,
        metrics=metrics,
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
    )
    report_path.write_text(
        json.dumps(result.to_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _init_tiny_fingerprint_student(
    *,
    max_seq_len: int,
    vocab_size: int,
    seed: int,
) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    return {
        "position_logits": jax.random.normal(
            key,
            (max_seq_len, vocab_size),
            dtype=jnp.float32,
        )
        * 0.01
    }


def _tiny_fingerprint_logits(
    params: dict[str, jax.Array],
    batch: FingerprintBatch,
) -> jax.Array:
    batch_size = int(batch.input_ids.shape[0])
    del batch
    return jnp.broadcast_to(
        params["position_logits"][None, :, :],
        (
            batch_size,
            params["position_logits"].shape[0],
            params["position_logits"].shape[1],
        ),
    )


def _loss_value(
    params: dict[str, jax.Array],
    batch: FingerprintBatch,
    loss_config: FingerprintCorridorLossConfig,
) -> jax.Array:
    return _loss_output(params, batch, loss_config).loss


def _loss_output(
    params: dict[str, jax.Array],
    batch: FingerprintBatch,
    loss_config: FingerprintCorridorLossConfig,
) -> FingerprintCorridorLossOutput:
    logits = _tiny_fingerprint_logits(params, batch)
    stats = compute_fingerprint_distribution_stats_at_positions(logits, batch.position)
    return compute_fingerprint_corridor_loss(stats, batch, loss_config)


def _metrics_from_output(output: FingerprintCorridorLossOutput) -> dict[str, float]:
    return {
        "fingerprint/loss_total": float(output.loss),
        "fingerprint/loss_entropy": float(output.entropy_loss),
        "fingerprint/loss_top1_margin": float(output.top1_margin_loss),
        "fingerprint/loss_top8_mass": float(output.top8_mass_loss),
        "fingerprint/loss_top32_mass": float(output.top32_mass_loss),
        "fingerprint/loss_tail_mass": float(output.tail_mass_loss),
        "fingerprint/inside_entropy_rate": float(output.entropy_inside_rate),
        "fingerprint/inside_top1_margin_rate": float(output.top1_margin_inside_rate),
        "fingerprint/inside_top8_mass_rate": float(output.top8_mass_inside_rate),
        "fingerprint/inside_top32_mass_rate": float(output.top32_mass_inside_rate),
        "fingerprint/inside_tail_mass_rate": float(output.tail_mass_inside_rate),
        "fingerprint/inside_all_rate": float(output.all_inside_rate),
    }
