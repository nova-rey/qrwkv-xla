from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.artifacts import (
    FingerprintBatch,
    FingerprintExemplarBatch,
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.training.fingerprint_exemplar_loss import (
    FingerprintExemplarLossConfig,
    FingerprintExemplarLossOutput,
    compute_fingerprint_exemplar_loss_at_positions,
)
from qrwkv_xla.training.fingerprint_loss import (
    FingerprintCorridorLossConfig,
    FingerprintCorridorLossOutput,
    compute_fingerprint_corridor_loss,
)
from qrwkv_xla.training.fingerprint_reports import (
    validate_fingerprint_smoke_report,
    write_fingerprint_smoke_summary,
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
    "fingerprint/corridor/loss_total",
    "fingerprint/corridor/loss_entropy",
    "fingerprint/corridor/loss_top1_margin",
    "fingerprint/corridor/loss_top8_mass",
    "fingerprint/corridor/loss_top32_mass",
    "fingerprint/corridor/loss_tail_mass",
    "fingerprint/corridor/inside_entropy_rate",
    "fingerprint/corridor/inside_top1_margin_rate",
    "fingerprint/corridor/inside_top8_mass_rate",
    "fingerprint/corridor/inside_top32_mass_rate",
    "fingerprint/corridor/inside_tail_mass_rate",
    "fingerprint/corridor/inside_all_rate",
)

FINGERPRINT_MIXED_SMOKE_METRIC_KEYS: tuple[str, ...] = (
    "fingerprint/mixed_loss_total",
    "fingerprint/corridor_loss_weight",
    "fingerprint/exemplar_loss_weight",
    "fingerprint/corridor_loss_total",
    "fingerprint/corridor_loss_entropy",
    "fingerprint/corridor_loss_top1_margin",
    "fingerprint/corridor_loss_top8_mass",
    "fingerprint/corridor_loss_top32_mass",
    "fingerprint/corridor_loss_tail_mass",
    "fingerprint/corridor_inside_entropy_rate",
    "fingerprint/corridor_inside_top1_margin_rate",
    "fingerprint/corridor_inside_top8_mass_rate",
    "fingerprint/corridor_inside_top32_mass_rate",
    "fingerprint/corridor_inside_tail_mass_rate",
    "fingerprint/corridor_inside_all_rate",
    "fingerprint/exemplar_loss_total",
    "fingerprint/exemplar_kl_loss",
    "fingerprint/exemplar_cross_entropy",
    "fingerprint/exemplar_teacher_entropy",
    "fingerprint/corridor_batches_consumed",
    "fingerprint/exemplar_batches_consumed",
    "fingerprint/optimizer_steps_completed",
    "fingerprint/mixed/loss_total",
    "fingerprint/mixed/corridor_loss_weight",
    "fingerprint/mixed/exemplar_loss_weight",
    "fingerprint/mixed/optimizer_steps_completed",
    "fingerprint/mixed/corridor_batches_consumed",
    "fingerprint/mixed/exemplar_batches_consumed",
    "fingerprint/corridor/loss_total",
    "fingerprint/corridor/loss_entropy",
    "fingerprint/corridor/loss_top1_margin",
    "fingerprint/corridor/loss_top8_mass",
    "fingerprint/corridor/loss_top32_mass",
    "fingerprint/corridor/loss_tail_mass",
    "fingerprint/corridor/inside_entropy_rate",
    "fingerprint/corridor/inside_top1_margin_rate",
    "fingerprint/corridor/inside_top8_mass_rate",
    "fingerprint/corridor/inside_top32_mass_rate",
    "fingerprint/corridor/inside_tail_mass_rate",
    "fingerprint/corridor/inside_all_rate",
    "fingerprint/exemplar/loss_total",
    "fingerprint/exemplar/kl_loss",
    "fingerprint/exemplar/cross_entropy",
    "fingerprint/exemplar/teacher_entropy",
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
    loss_delta: float
    steps: int
    completed_steps: int
    train_batches_consumed: int
    loss_finite: bool
    loss_non_negative: bool
    loss_non_increasing: bool
    metrics_finite: bool
    metrics: dict[str, float]
    output_dir: str
    metrics_path: str
    checkpoint_path: str
    report_path: str
    summary_path: str
    artifact_dir: str = ""
    artifact_version: str = ""
    requested_steps: int = 0
    optimizer_steps_completed: int = 0
    seed: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0
    num_corridor_records: int = 0
    max_seq_len: int = 0
    vocab_size: int = 0
    tracked_stats: tuple[str, ...] = ()
    smoke_student_kind: str = "tiny_position_logit_head"
    smoke_student_uses_input_ids: bool = False
    main_runner_integrated: bool = False
    real_student_backend_integrated: bool = False
    teacher_required: bool = False
    exemplar_reservoir_enabled: bool = False
    artifact_kind: str = "behavioral_fingerprint"
    training_path_kind: str = "standalone_fingerprint_smoke"
    claims_not_made: tuple[str, ...] = FINGERPRINT_SMOKE_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        report = {
            **asdict(self),
            "phase": "P136.1",
            "report_schema_phase": "P139",
            "report_type": "corridor_only_smoke_report",
            "scope": "tiny_fingerprint_training_smoke",
            "fingerprint_only": True,
            "hf_required": False,
            "hf_download_required": False,
            "accelerator_required": False,
            "gpu_or_tpu_required": False,
        }
        report.update(_corridor_report_sections(self))
        return report


@dataclass(frozen=True)
class FingerprintMixedSmokeConfig:
    artifact_dir: Path
    output_dir: Path
    steps: int = 3
    corridor_batch_size: int = 2
    exemplar_batch_size: int = 2
    learning_rate: float = 0.5
    seed: int = 0
    corridor_shuffle: bool = False
    exemplar_shuffle: bool = False
    corridor_max_records: int | None = None
    exemplar_max_records: int | None = None
    corridor_drop_remainder: bool = False
    exemplar_drop_remainder: bool = False
    corridor_loss_weight: float = 1.0
    exemplar_loss_weight: float = 1.0
    corridor_loss_config: FingerprintCorridorLossConfig = (
        FingerprintCorridorLossConfig()
    )
    exemplar_loss_config: FingerprintExemplarLossConfig = (
        FingerprintExemplarLossConfig()
    )


@dataclass(frozen=True)
class FingerprintMixedLossOutput:
    loss: jax.Array
    corridor: FingerprintCorridorLossOutput
    exemplar: FingerprintExemplarLossOutput


@dataclass(frozen=True)
class FingerprintMixedSmokeResult:
    status: str
    initial_mixed_loss: float
    final_mixed_loss: float
    mixed_loss_delta: float
    corridor_loss_delta: float
    exemplar_loss_delta: float
    requested_steps: int
    optimizer_steps_completed: int
    corridor_batches_consumed: int
    exemplar_batches_consumed: int
    mixed_loss_finite: bool
    mixed_loss_non_negative: bool
    mixed_loss_non_increasing: bool
    metrics_finite: bool
    metrics: dict[str, float]
    output_dir: str
    metrics_path: str
    checkpoint_path: str
    report_path: str
    summary_path: str
    artifact_dir: str = ""
    artifact_version: str = ""
    seed: int = 0
    learning_rate: float = 0.0
    corridor_batch_size: int = 0
    exemplar_batch_size: int = 0
    num_corridor_records: int = 0
    num_exemplar_records: int = 0
    max_seq_len: int = 0
    vocab_size: int = 0
    tracked_stats: tuple[str, ...] = ()
    exemplar_payload_type: str | None = None
    corridor_loss_weight: float = 1.0
    exemplar_loss_weight: float = 1.0
    smoke_student_kind: str = "tiny_position_logit_head"
    smoke_student_uses_input_ids: bool = False
    main_runner_integrated: bool = False
    real_student_backend_integrated: bool = False
    teacher_required: bool = False
    exemplar_reservoir_enabled: bool = True
    artifact_kind: str = "behavioral_fingerprint"
    training_path_kind: str = "standalone_mixed_fingerprint_smoke"
    claims_not_made: tuple[str, ...] = (
        "teacher_generation",
        "real_teacher_required",
        "real_student_backend_integration",
        "main_runner_integration",
        "csl_exemplar_payloads",
        "model_quality_proven",
        "production_training_ready",
        "gpu_or_tpu_required",
    )

    def to_report(self) -> dict[str, Any]:
        report = {
            **asdict(self),
            "phase": "P138",
            "report_schema_phase": "P139",
            "report_type": "mixed_fingerprint_smoke_report",
            "scope": "tiny_mixed_fingerprint_smoke",
            "fingerprint_only": True,
            "hf_required": False,
            "hf_download_required": False,
            "accelerator_required": False,
            "gpu_or_tpu_required": False,
        }
        report.update(_mixed_report_sections(self))
        return report


def classify_fingerprint_smoke_status(
    *,
    completed_steps: int,
    requested_steps: int,
    train_batches_consumed: int,
    initial_loss: float,
    final_loss: float,
    metrics_finite: bool,
) -> str:
    loss_finite = bool(jnp.isfinite(initial_loss) & jnp.isfinite(final_loss))
    loss_non_negative = bool((initial_loss >= 0.0) & (final_loss >= 0.0))
    if (
        completed_steps == requested_steps
        and train_batches_consumed > 0
        and loss_finite
        and loss_non_negative
        and metrics_finite
    ):
        return "pass"
    return "fail"


def classify_fingerprint_mixed_smoke_status(
    *,
    optimizer_steps_completed: int,
    requested_steps: int,
    corridor_batches_consumed: int,
    exemplar_batches_consumed: int,
    initial_mixed_loss: float,
    final_mixed_loss: float,
    metrics_finite: bool,
) -> str:
    loss_finite = bool(
        jnp.isfinite(initial_mixed_loss) & jnp.isfinite(final_mixed_loss)
    )
    loss_non_negative = bool((initial_mixed_loss >= 0.0) & (final_mixed_loss >= 0.0))
    if (
        optimizer_steps_completed == requested_steps
        and corridor_batches_consumed > 0
        and exemplar_batches_consumed > 0
        and loss_finite
        and loss_non_negative
        and metrics_finite
    ):
        return "pass"
    return "fail"


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
    train_batches = tuple(dataset.iter_batches())
    if not train_batches:
        raise ValueError("Fingerprint smoke consumed zero optimizer batches.")
    artifact_summary = summarize_fingerprint_artifact(config.artifact_dir)

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

    train_batches_consumed = 0
    for update_index in range(config.steps):
        batch = train_batches[update_index % len(train_batches)]
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
        train_batches_consumed += 1

    final_output = _loss_output(params, full_batch, config.loss_config)
    final_loss = final_output.loss
    metrics = _metrics_from_output(final_output)
    initial_loss_float = float(initial_loss)
    final_loss_float = float(final_loss)
    loss_delta = final_loss_float - initial_loss_float
    loss_finite = bool(jnp.isfinite(initial_loss) & jnp.isfinite(final_loss))
    loss_non_negative = bool((initial_loss >= 0.0) & (final_loss >= 0.0))
    loss_non_increasing = bool(final_loss <= initial_loss + 1e-6)
    metrics_finite = all(bool(jnp.isfinite(value)) for value in metrics.values())
    status = classify_fingerprint_smoke_status(
        completed_steps=train_batches_consumed,
        requested_steps=config.steps,
        train_batches_consumed=train_batches_consumed,
        initial_loss=initial_loss_float,
        final_loss=final_loss_float,
        metrics_finite=metrics_finite,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "fingerprint_smoke_report.json"
    summary_path = output_dir / "fingerprint_run_summary.md"

    metrics_payload = {
        "initial_loss": initial_loss_float,
        "final_loss": final_loss_float,
        "loss_delta": loss_delta,
        "steps": config.steps,
        "completed_steps": train_batches_consumed,
        "train_batches_consumed": train_batches_consumed,
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
                "completed_steps": train_batches_consumed,
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
        loss_delta=loss_delta,
        steps=config.steps,
        completed_steps=train_batches_consumed,
        train_batches_consumed=train_batches_consumed,
        loss_finite=loss_finite,
        loss_non_negative=loss_non_negative,
        loss_non_increasing=loss_non_increasing,
        metrics_finite=metrics_finite,
        metrics=metrics,
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
        summary_path=str(summary_path),
        artifact_dir=str(config.artifact_dir),
        artifact_version=artifact_summary.artifact_version,
        requested_steps=config.steps,
        optimizer_steps_completed=train_batches_consumed,
        seed=config.seed,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_corridor_records=dataset.num_records,
        max_seq_len=dataset.max_seq_len,
        vocab_size=dataset.vocab_size,
        tracked_stats=dataset.tracked_stats,
    )
    report = result.to_report()
    _raise_if_invalid_report(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_fingerprint_smoke_summary(report, summary_path)
    return result


def run_mixed_fingerprint_training_smoke(
    config: FingerprintMixedSmokeConfig,
) -> FingerprintMixedSmokeResult:
    _validate_mixed_config(config)

    corridor_dataset = load_fingerprint_targets(
        config.artifact_dir,
        batch_size=config.corridor_batch_size,
        shuffle=config.corridor_shuffle,
        seed=config.seed,
        drop_remainder=config.corridor_drop_remainder,
        max_records=config.corridor_max_records,
    )
    exemplar_dataset = load_fingerprint_exemplars(
        config.artifact_dir,
        batch_size=config.exemplar_batch_size,
        shuffle=config.exemplar_shuffle,
        seed=config.seed,
        drop_remainder=config.exemplar_drop_remainder,
        max_records=config.exemplar_max_records,
        require_exemplars=True,
    )
    if corridor_dataset.num_records == 0:
        raise ValueError("mixed fingerprint smoke requires corridor target records")
    if exemplar_dataset.num_records == 0:
        raise ValueError("mixed fingerprint smoke requires exemplar records")
    if corridor_dataset.max_seq_len != exemplar_dataset.max_seq_len:
        raise ValueError(
            "corridor and exemplar max_seq_len mismatch: "
            f"{corridor_dataset.max_seq_len} vs {exemplar_dataset.max_seq_len}"
        )
    if corridor_dataset.vocab_size != exemplar_dataset.vocab_size:
        raise ValueError(
            "corridor and exemplar vocab_size mismatch: "
            f"{corridor_dataset.vocab_size} vs {exemplar_dataset.vocab_size}"
        )

    corridor_batches = tuple(corridor_dataset.iter_batches())
    exemplar_batches = tuple(exemplar_dataset.iter_batches())
    if not corridor_batches:
        raise ValueError("Mixed fingerprint smoke consumed zero corridor batches.")
    if not exemplar_batches:
        raise ValueError("Mixed fingerprint smoke consumed zero exemplar batches.")
    artifact_summary = summarize_fingerprint_artifact(config.artifact_dir)

    params = _init_tiny_fingerprint_student(
        max_seq_len=corridor_dataset.max_seq_len,
        vocab_size=corridor_dataset.vocab_size,
        seed=config.seed,
    )
    full_corridor_batch = next(
        load_fingerprint_targets(
            config.artifact_dir,
            batch_size=corridor_dataset.num_records,
            max_records=corridor_dataset.num_records,
            shuffle=False,
            drop_remainder=False,
        ).iter_batches()
    )
    full_exemplar_batch = next(
        load_fingerprint_exemplars(
            config.artifact_dir,
            batch_size=exemplar_dataset.num_records,
            max_records=exemplar_dataset.num_records,
            shuffle=False,
            drop_remainder=False,
            require_exemplars=True,
        ).iter_batches()
    )
    initial_output = _mixed_loss_output(
        params,
        full_corridor_batch,
        full_exemplar_batch,
        config,
    )

    corridor_batches_consumed = 0
    exemplar_batches_consumed = 0
    for update_index in range(config.steps):
        corridor_batch = corridor_batches[update_index % len(corridor_batches)]
        exemplar_batch = exemplar_batches[update_index % len(exemplar_batches)]
        loss, grads = jax.value_and_grad(_mixed_loss_value)(
            params,
            corridor_batch,
            exemplar_batch,
            config,
        )
        del loss
        params = jax.tree_util.tree_map(
            lambda param, grad: param - config.learning_rate * grad,
            params,
            grads,
        )
        corridor_batches_consumed += 1
        exemplar_batches_consumed += 1

    final_output = _mixed_loss_output(
        params,
        full_corridor_batch,
        full_exemplar_batch,
        config,
    )
    metrics = _mixed_metrics_from_output(
        final_output,
        corridor_loss_weight=config.corridor_loss_weight,
        exemplar_loss_weight=config.exemplar_loss_weight,
        corridor_batches_consumed=corridor_batches_consumed,
        exemplar_batches_consumed=exemplar_batches_consumed,
        optimizer_steps_completed=config.steps,
    )

    initial_mixed_loss = float(initial_output.loss)
    final_mixed_loss = float(final_output.loss)
    mixed_loss_delta = final_mixed_loss - initial_mixed_loss
    corridor_loss_delta = float(
        final_output.corridor.loss - initial_output.corridor.loss
    )
    exemplar_loss_delta = float(
        final_output.exemplar.loss - initial_output.exemplar.loss
    )
    mixed_loss_finite = bool(
        jnp.isfinite(initial_output.loss) & jnp.isfinite(final_output.loss)
    )
    mixed_loss_non_negative = bool(
        (initial_output.loss >= 0.0) & (final_output.loss >= 0.0)
    )
    mixed_loss_non_increasing = bool(final_output.loss <= initial_output.loss + 1e-6)
    metrics_finite = all(bool(jnp.isfinite(value)) for value in metrics.values())
    status = classify_fingerprint_mixed_smoke_status(
        optimizer_steps_completed=config.steps,
        requested_steps=config.steps,
        corridor_batches_consumed=corridor_batches_consumed,
        exemplar_batches_consumed=exemplar_batches_consumed,
        initial_mixed_loss=initial_mixed_loss,
        final_mixed_loss=final_mixed_loss,
        metrics_finite=metrics_finite,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "fingerprint_mixed_smoke_report.json"
    summary_path = output_dir / "fingerprint_run_summary.md"

    metrics_payload = {
        "initial_mixed_loss": initial_mixed_loss,
        "final_mixed_loss": final_mixed_loss,
        "mixed_loss_delta": mixed_loss_delta,
        "corridor_loss_delta": corridor_loss_delta,
        "exemplar_loss_delta": exemplar_loss_delta,
        "requested_steps": config.steps,
        "optimizer_steps_completed": config.steps,
        "corridor_batches_consumed": corridor_batches_consumed,
        "exemplar_batches_consumed": exemplar_batches_consumed,
        **metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "kind": "tiny_mixed_fingerprint_position_logit_head",
                "max_seq_len": corridor_dataset.max_seq_len,
                "vocab_size": corridor_dataset.vocab_size,
                "requested_steps": config.steps,
                "optimizer_steps_completed": config.steps,
                "corridor_batches_consumed": corridor_batches_consumed,
                "exemplar_batches_consumed": exemplar_batches_consumed,
                "position_logits_shape": list(params["position_logits"].shape),
                "position_logits_l2": float(jnp.linalg.norm(params["position_logits"])),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = FingerprintMixedSmokeResult(
        status=status,
        initial_mixed_loss=initial_mixed_loss,
        final_mixed_loss=final_mixed_loss,
        mixed_loss_delta=mixed_loss_delta,
        corridor_loss_delta=corridor_loss_delta,
        exemplar_loss_delta=exemplar_loss_delta,
        requested_steps=config.steps,
        optimizer_steps_completed=config.steps,
        corridor_batches_consumed=corridor_batches_consumed,
        exemplar_batches_consumed=exemplar_batches_consumed,
        mixed_loss_finite=mixed_loss_finite,
        mixed_loss_non_negative=mixed_loss_non_negative,
        mixed_loss_non_increasing=mixed_loss_non_increasing,
        metrics_finite=metrics_finite,
        metrics=metrics,
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
        summary_path=str(summary_path),
        artifact_dir=str(config.artifact_dir),
        artifact_version=artifact_summary.artifact_version,
        seed=config.seed,
        learning_rate=config.learning_rate,
        corridor_batch_size=config.corridor_batch_size,
        exemplar_batch_size=config.exemplar_batch_size,
        num_corridor_records=corridor_dataset.num_records,
        num_exemplar_records=exemplar_dataset.num_records,
        max_seq_len=corridor_dataset.max_seq_len,
        vocab_size=corridor_dataset.vocab_size,
        tracked_stats=corridor_dataset.tracked_stats,
        exemplar_payload_type=artifact_summary.exemplar_payload_type,
        corridor_loss_weight=config.corridor_loss_weight,
        exemplar_loss_weight=config.exemplar_loss_weight,
    )
    report = result.to_report()
    _raise_if_invalid_report(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_fingerprint_smoke_summary(report, summary_path)
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
    batch: FingerprintBatch | FingerprintExemplarBatch,
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


def _mixed_loss_value(
    params: dict[str, jax.Array],
    corridor_batch: FingerprintBatch,
    exemplar_batch: FingerprintExemplarBatch,
    config: FingerprintMixedSmokeConfig,
) -> jax.Array:
    return _mixed_loss_output(params, corridor_batch, exemplar_batch, config).loss


def _mixed_loss_output(
    params: dict[str, jax.Array],
    corridor_batch: FingerprintBatch,
    exemplar_batch: FingerprintExemplarBatch,
    config: FingerprintMixedSmokeConfig,
) -> FingerprintMixedLossOutput:
    corridor_logits = _tiny_fingerprint_logits(params, corridor_batch)
    corridor_stats = compute_fingerprint_distribution_stats_at_positions(
        corridor_logits,
        corridor_batch.position,
    )
    corridor = compute_fingerprint_corridor_loss(
        corridor_stats,
        corridor_batch,
        config.corridor_loss_config,
    )
    exemplar_logits = _tiny_fingerprint_logits(params, exemplar_batch)
    exemplar = compute_fingerprint_exemplar_loss_at_positions(
        exemplar_logits,
        exemplar_batch,
        config.exemplar_loss_config,
    )
    loss = (
        config.corridor_loss_weight * corridor.loss
        + config.exemplar_loss_weight * exemplar.loss
    )
    return FingerprintMixedLossOutput(loss=loss, corridor=corridor, exemplar=exemplar)


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
        "fingerprint/corridor/loss_total": float(output.loss),
        "fingerprint/corridor/loss_entropy": float(output.entropy_loss),
        "fingerprint/corridor/loss_top1_margin": float(output.top1_margin_loss),
        "fingerprint/corridor/loss_top8_mass": float(output.top8_mass_loss),
        "fingerprint/corridor/loss_top32_mass": float(output.top32_mass_loss),
        "fingerprint/corridor/loss_tail_mass": float(output.tail_mass_loss),
        "fingerprint/corridor/inside_entropy_rate": float(output.entropy_inside_rate),
        "fingerprint/corridor/inside_top1_margin_rate": float(
            output.top1_margin_inside_rate
        ),
        "fingerprint/corridor/inside_top8_mass_rate": float(
            output.top8_mass_inside_rate
        ),
        "fingerprint/corridor/inside_top32_mass_rate": float(
            output.top32_mass_inside_rate
        ),
        "fingerprint/corridor/inside_tail_mass_rate": float(
            output.tail_mass_inside_rate
        ),
        "fingerprint/corridor/inside_all_rate": float(output.all_inside_rate),
    }


def _mixed_metrics_from_output(
    output: FingerprintMixedLossOutput,
    *,
    corridor_loss_weight: float,
    exemplar_loss_weight: float,
    corridor_batches_consumed: int,
    exemplar_batches_consumed: int,
    optimizer_steps_completed: int,
) -> dict[str, float]:
    return {
        "fingerprint/mixed_loss_total": float(output.loss),
        "fingerprint/corridor_loss_weight": float(corridor_loss_weight),
        "fingerprint/exemplar_loss_weight": float(exemplar_loss_weight),
        "fingerprint/corridor_loss_total": float(output.corridor.loss),
        "fingerprint/corridor_loss_entropy": float(output.corridor.entropy_loss),
        "fingerprint/corridor_loss_top1_margin": float(
            output.corridor.top1_margin_loss
        ),
        "fingerprint/corridor_loss_top8_mass": float(output.corridor.top8_mass_loss),
        "fingerprint/corridor_loss_top32_mass": float(output.corridor.top32_mass_loss),
        "fingerprint/corridor_loss_tail_mass": float(output.corridor.tail_mass_loss),
        "fingerprint/corridor_inside_entropy_rate": float(
            output.corridor.entropy_inside_rate
        ),
        "fingerprint/corridor_inside_top1_margin_rate": float(
            output.corridor.top1_margin_inside_rate
        ),
        "fingerprint/corridor_inside_top8_mass_rate": float(
            output.corridor.top8_mass_inside_rate
        ),
        "fingerprint/corridor_inside_top32_mass_rate": float(
            output.corridor.top32_mass_inside_rate
        ),
        "fingerprint/corridor_inside_tail_mass_rate": float(
            output.corridor.tail_mass_inside_rate
        ),
        "fingerprint/corridor_inside_all_rate": float(output.corridor.all_inside_rate),
        "fingerprint/exemplar_loss_total": float(output.exemplar.loss),
        "fingerprint/exemplar_kl_loss": float(output.exemplar.kl_loss),
        "fingerprint/exemplar_cross_entropy": float(output.exemplar.cross_entropy),
        "fingerprint/exemplar_teacher_entropy": float(output.exemplar.entropy),
        "fingerprint/corridor_batches_consumed": float(corridor_batches_consumed),
        "fingerprint/exemplar_batches_consumed": float(exemplar_batches_consumed),
        "fingerprint/optimizer_steps_completed": float(optimizer_steps_completed),
        "fingerprint/mixed/loss_total": float(output.loss),
        "fingerprint/mixed/corridor_loss_weight": float(corridor_loss_weight),
        "fingerprint/mixed/exemplar_loss_weight": float(exemplar_loss_weight),
        "fingerprint/mixed/optimizer_steps_completed": float(optimizer_steps_completed),
        "fingerprint/mixed/corridor_batches_consumed": float(corridor_batches_consumed),
        "fingerprint/mixed/exemplar_batches_consumed": float(exemplar_batches_consumed),
        "fingerprint/corridor/loss_total": float(output.corridor.loss),
        "fingerprint/corridor/loss_entropy": float(output.corridor.entropy_loss),
        "fingerprint/corridor/loss_top1_margin": float(
            output.corridor.top1_margin_loss
        ),
        "fingerprint/corridor/loss_top8_mass": float(output.corridor.top8_mass_loss),
        "fingerprint/corridor/loss_top32_mass": float(output.corridor.top32_mass_loss),
        "fingerprint/corridor/loss_tail_mass": float(output.corridor.tail_mass_loss),
        "fingerprint/corridor/inside_entropy_rate": float(
            output.corridor.entropy_inside_rate
        ),
        "fingerprint/corridor/inside_top1_margin_rate": float(
            output.corridor.top1_margin_inside_rate
        ),
        "fingerprint/corridor/inside_top8_mass_rate": float(
            output.corridor.top8_mass_inside_rate
        ),
        "fingerprint/corridor/inside_top32_mass_rate": float(
            output.corridor.top32_mass_inside_rate
        ),
        "fingerprint/corridor/inside_tail_mass_rate": float(
            output.corridor.tail_mass_inside_rate
        ),
        "fingerprint/corridor/inside_all_rate": float(output.corridor.all_inside_rate),
        "fingerprint/exemplar/loss_total": float(output.exemplar.loss),
        "fingerprint/exemplar/kl_loss": float(output.exemplar.kl_loss),
        "fingerprint/exemplar/cross_entropy": float(output.exemplar.cross_entropy),
        "fingerprint/exemplar/teacher_entropy": float(output.exemplar.entropy),
    }


def _corridor_report_sections(result: FingerprintTrainingSmokeResult) -> dict[str, Any]:
    return {
        "artifact": {
            "artifact_type": result.artifact_kind,
            "artifact_version": result.artifact_version,
            "artifact_dir": result.artifact_dir,
            "vocab_size": result.vocab_size,
            "max_seq_len": result.max_seq_len,
        },
        "requested_steps": result.requested_steps or result.steps,
        "optimizer_steps_completed": (
            result.optimizer_steps_completed or result.completed_steps
        ),
        "seed": result.seed,
        "learning_rate": result.learning_rate,
        "corridor_targets": {
            "num_records": result.num_corridor_records,
            "batch_size": result.batch_size,
            "batches_consumed": result.train_batches_consumed,
            "max_seq_len": result.max_seq_len,
            "vocab_size": result.vocab_size,
            "tracked_stats": result.tracked_stats,
        },
        "loss": {
            "initial_total": result.initial_loss,
            "final_total": result.final_loss,
            "delta_total": result.loss_delta,
            "finite": result.loss_finite,
            "non_negative": result.loss_non_negative,
            "non_increasing": result.loss_non_increasing,
        },
        "corridor_metrics": _corridor_metric_section(result.metrics),
        "limitations": _common_limitations(),
    }


def _mixed_report_sections(result: FingerprintMixedSmokeResult) -> dict[str, Any]:
    return {
        "artifact": {
            "artifact_type": result.artifact_kind,
            "artifact_version": result.artifact_version,
            "artifact_dir": result.artifact_dir,
            "vocab_size": result.vocab_size,
            "max_seq_len": result.max_seq_len,
        },
        "corridor_targets": {
            "num_records": result.num_corridor_records,
            "batch_size": result.corridor_batch_size,
            "batches_consumed": result.corridor_batches_consumed,
            "max_seq_len": result.max_seq_len,
            "vocab_size": result.vocab_size,
            "tracked_stats": result.tracked_stats,
        },
        "exemplar_reservoir": {
            "enabled": result.exemplar_reservoir_enabled,
            "payload_type": result.exemplar_payload_type,
            "num_records": result.num_exemplar_records,
            "batch_size": result.exemplar_batch_size,
            "batches_consumed": result.exemplar_batches_consumed,
            "max_seq_len": result.max_seq_len,
            "vocab_size": result.vocab_size,
        },
        "loss_weights": {
            "corridor": result.corridor_loss_weight,
            "exemplar": result.exemplar_loss_weight,
        },
        "mixed_loss": {
            "initial_total": result.initial_mixed_loss,
            "final_total": result.final_mixed_loss,
            "delta_total": result.mixed_loss_delta,
            "finite": result.mixed_loss_finite,
            "non_negative": result.mixed_loss_non_negative,
            "non_increasing": result.mixed_loss_non_increasing,
        },
        "corridor_metrics": _corridor_metric_section(result.metrics),
        "exemplar_metrics": _exemplar_metric_section(result.metrics),
        "limitations": _common_limitations(),
    }


def _corridor_metric_section(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "loss_total": metrics["fingerprint/corridor/loss_total"],
        "loss_entropy": metrics["fingerprint/corridor/loss_entropy"],
        "loss_top1_margin": metrics["fingerprint/corridor/loss_top1_margin"],
        "loss_top8_mass": metrics["fingerprint/corridor/loss_top8_mass"],
        "loss_top32_mass": metrics["fingerprint/corridor/loss_top32_mass"],
        "loss_tail_mass": metrics["fingerprint/corridor/loss_tail_mass"],
        "inside_entropy_rate": metrics["fingerprint/corridor/inside_entropy_rate"],
        "inside_top1_margin_rate": metrics[
            "fingerprint/corridor/inside_top1_margin_rate"
        ],
        "inside_top8_mass_rate": metrics["fingerprint/corridor/inside_top8_mass_rate"],
        "inside_top32_mass_rate": metrics[
            "fingerprint/corridor/inside_top32_mass_rate"
        ],
        "inside_tail_mass_rate": metrics["fingerprint/corridor/inside_tail_mass_rate"],
        "inside_all_rate": metrics["fingerprint/corridor/inside_all_rate"],
    }


def _exemplar_metric_section(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "loss_total": metrics["fingerprint/exemplar/loss_total"],
        "kl_loss": metrics["fingerprint/exemplar/kl_loss"],
        "cross_entropy": metrics["fingerprint/exemplar/cross_entropy"],
        "teacher_entropy": metrics["fingerprint/exemplar/teacher_entropy"],
    }


def _common_limitations() -> tuple[str, ...]:
    return (
        "Standalone smoke only.",
        "Tiny position-logit head does not condition on input IDs.",
        "Main distillation runner is not integrated.",
        "Real QRWKV student backend is not exercised.",
        "No quality claims are made.",
    )


def _raise_if_invalid_report(report: dict[str, Any]) -> None:
    blockers = validate_fingerprint_smoke_report(report)
    if blockers:
        raise ValueError("fingerprint smoke report invalid: " + "; ".join(blockers))


def _validate_mixed_config(config: FingerprintMixedSmokeConfig) -> None:
    if config.steps <= 0:
        raise ValueError(f"steps must be > 0, got {config.steps}")
    if config.corridor_batch_size <= 0:
        raise ValueError(
            f"corridor_batch_size must be > 0, got {config.corridor_batch_size}"
        )
    if config.exemplar_batch_size <= 0:
        raise ValueError(
            f"exemplar_batch_size must be > 0, got {config.exemplar_batch_size}"
        )
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.corridor_loss_weight < 0.0:
        raise ValueError(
            "corridor_loss_weight must be non-negative, "
            f"got {config.corridor_loss_weight}"
        )
    if config.exemplar_loss_weight < 0.0:
        raise ValueError(
            "exemplar_loss_weight must be non-negative, "
            f"got {config.exemplar_loss_weight}"
        )
    if config.corridor_loss_weight == 0.0 and config.exemplar_loss_weight == 0.0:
        raise ValueError("at least one mixed smoke loss weight must be positive")
