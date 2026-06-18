from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import load_fingerprint_targets, summarize_fingerprint_artifact
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.students import CURRENT_QRWKV_ARCHITECTURE_ID, create_student_backend
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

REAL_STUDENT_FINGERPRINT_FORWARD_METRIC_KEYS: tuple[str, ...] = (
    "fingerprint/real_student_forward/logits_finite",
    "fingerprint/real_student_forward/logits_shape_batch",
    "fingerprint/real_student_forward/logits_shape_seq",
    "fingerprint/real_student_forward/logits_shape_vocab",
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


@dataclass(frozen=True)
class RealStudentFingerprintForwardConfig:
    artifact_dir: Path
    output_dir: Path
    batch_size: int = 2
    seed: int = 0
    shuffle: bool = False
    max_records: int | None = None
    drop_remainder: bool = False
    architecture_id: str = CURRENT_QRWKV_ARCHITECTURE_ID
    runtime: Any | None = None
    student_vocab_size: int | None = None
    student_max_seq_len: int | None = None
    loss_config: FingerprintCorridorLossConfig = FingerprintCorridorLossConfig()


@dataclass(frozen=True)
class RealStudentFingerprintForwardResult:
    status: str
    metrics_finite: bool
    logits_finite: bool
    corridor_loss_finite: bool
    corridor_loss_non_negative: bool
    corridor_batches_consumed: int
    requested_steps: int
    optimizer_steps_completed: int
    metrics: dict[str, float]
    output_dir: str
    metrics_path: str
    report_path: str
    summary_path: str
    artifact_dir: str
    artifact_version: str
    batch_size: int
    seed: int
    architecture_id: str
    student_backend_name: str
    student_vocab_size: int
    student_max_seq_len: int
    num_corridor_records: int
    max_seq_len: int
    vocab_size: int
    tracked_stats: tuple[str, ...]
    logits_shape: tuple[int, int, int]
    smoke_student_kind: str = "real_student_backend"
    smoke_student_uses_input_ids: bool = True
    main_runner_integrated: bool = False
    real_student_backend_integrated: bool = True
    teacher_required: bool = False
    exemplar_forward_enabled: bool = False
    artifact_kind: str = "behavioral_fingerprint"
    training_path_kind: str = "real_student_fingerprint_forward_smoke"

    def to_report(self) -> dict[str, Any]:
        report = {
            **asdict(self),
            "phase": "P140",
            "report_schema_phase": "P140",
            "report_type": "real_student_fingerprint_forward_smoke_report",
            "scope": "real_student_fingerprint_forward_smoke",
            "fingerprint_only": True,
            "hf_required": False,
            "hf_download_required": False,
            "accelerator_required": False,
            "gpu_or_tpu_required": False,
            "learning_rate": 0.0,
            "student": {
                "architecture_id": self.architecture_id,
                "backend_name": self.student_backend_name,
                "vocab_size": self.student_vocab_size,
                "max_seq_len": self.student_max_seq_len,
                "uses_input_ids": self.smoke_student_uses_input_ids,
            },
            "artifact": {
                "artifact_type": self.artifact_kind,
                "artifact_version": self.artifact_version,
                "artifact_dir": self.artifact_dir,
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
            },
            "corridor_targets": {
                "num_records": self.num_corridor_records,
                "batch_size": self.batch_size,
                "batches_consumed": self.corridor_batches_consumed,
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "tracked_stats": self.tracked_stats,
            },
            "forward": {
                "logits_shape": self.logits_shape,
                "logits_finite": self.logits_finite,
                "uses_input_ids": self.smoke_student_uses_input_ids,
            },
            "corridor": {
                "loss_finite": self.corridor_loss_finite,
                "loss_non_negative": self.corridor_loss_non_negative,
            },
            "corridor_metrics": _corridor_metric_section(self.metrics),
            "limitations": (
                "Forward-only smoke.",
                "No optimizer steps were run.",
                "Main distillation runner is not integrated.",
                "Teacher-side fingerprint capture remains future work.",
                "No model-quality claim is made.",
            ),
        }
        return report


def classify_real_student_fingerprint_forward_status(
    *,
    corridor_batches_consumed: int,
    logits_finite: bool,
    corridor_loss_finite: bool,
    corridor_loss_non_negative: bool,
    metrics_finite: bool,
) -> str:
    if (
        corridor_batches_consumed > 0
        and logits_finite
        and corridor_loss_finite
        and corridor_loss_non_negative
        and metrics_finite
    ):
        return "pass"
    return "fail"


def run_real_student_fingerprint_forward_smoke(
    config: RealStudentFingerprintForwardConfig,
) -> RealStudentFingerprintForwardResult:
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
    if config.max_records is not None and config.max_records < 0:
        raise ValueError(f"max_records must be >= 0, got {config.max_records}")

    artifact_summary = summarize_fingerprint_artifact(config.artifact_dir)
    dataset = load_fingerprint_targets(
        config.artifact_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        seed=config.seed,
        drop_remainder=config.drop_remainder,
        max_records=config.max_records,
    )
    if dataset.num_records == 0:
        raise ValueError("real student fingerprint smoke requires target records")
    batches = tuple(dataset.iter_batches())
    if not batches:
        raise ValueError("Real student fingerprint smoke consumed zero batches.")

    student_vocab_size = config.student_vocab_size or artifact_summary.vocab_size
    _validate_student_artifact_compatibility(
        artifact_vocab_size=artifact_summary.vocab_size,
        artifact_max_seq_len=artifact_summary.max_seq_len,
        student_vocab_size=student_vocab_size,
        student_max_seq_len=config.student_max_seq_len,
    )
    vocab_contract = VocabContract(
        tokenizer_id=artifact_summary.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=artifact_summary.tokenizer_name or None,
        vocab_size=student_vocab_size,
        model_id=artifact_summary.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=vocab_contract,
        architecture_id=config.architecture_id,
        runtime=config.runtime,
    )

    batch = batches[0]
    _validate_batch_token_ids(batch.input_ids, vocab_size=student_vocab_size)
    params = backend.init_params(jax.random.PRNGKey(config.seed))
    output, _state = backend.forward_full(
        params,
        jnp.asarray(batch.input_ids, dtype=jnp.int32),
    )
    logits = jnp.asarray(backend.logits(output))
    _validate_logits_shape(
        logits,
        expected_batch_size=int(batch.input_ids.shape[0]),
        expected_seq_len=int(batch.input_ids.shape[1]),
        expected_vocab_size=student_vocab_size,
    )
    _validate_positions_in_range(batch.position, seq_len=int(logits.shape[1]))

    stats = compute_fingerprint_distribution_stats_at_positions(
        logits,
        jnp.asarray(batch.position, dtype=jnp.int32),
    )
    corridor = compute_fingerprint_corridor_loss(stats, batch, config.loss_config)
    logits_finite = bool(jnp.all(jnp.isfinite(logits)))
    corridor_loss_finite = bool(jnp.isfinite(corridor.loss))
    corridor_loss_non_negative = bool(corridor.loss >= 0.0)
    metrics = {
        "fingerprint/real_student_forward/logits_finite": float(logits_finite),
        "fingerprint/real_student_forward/logits_shape_batch": float(logits.shape[0]),
        "fingerprint/real_student_forward/logits_shape_seq": float(logits.shape[1]),
        "fingerprint/real_student_forward/logits_shape_vocab": float(logits.shape[2]),
        **_metrics_from_corridor(corridor),
    }
    metrics_finite = all(bool(np.isfinite(value)) for value in metrics.values())
    status = classify_real_student_fingerprint_forward_status(
        corridor_batches_consumed=1,
        logits_finite=logits_finite,
        corridor_loss_finite=corridor_loss_finite,
        corridor_loss_non_negative=corridor_loss_non_negative,
        metrics_finite=metrics_finite,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "real_student_fingerprint_forward_report.json"
    summary_path = output_dir / "fingerprint_run_summary.md"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = RealStudentFingerprintForwardResult(
        status=status,
        metrics_finite=metrics_finite,
        logits_finite=logits_finite,
        corridor_loss_finite=corridor_loss_finite,
        corridor_loss_non_negative=corridor_loss_non_negative,
        corridor_batches_consumed=1,
        requested_steps=0,
        optimizer_steps_completed=0,
        metrics=metrics,
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        report_path=str(report_path),
        summary_path=str(summary_path),
        artifact_dir=str(config.artifact_dir),
        artifact_version=artifact_summary.artifact_version,
        batch_size=config.batch_size,
        seed=config.seed,
        architecture_id=config.architecture_id,
        student_backend_name=type(backend).__name__,
        student_vocab_size=student_vocab_size,
        student_max_seq_len=config.student_max_seq_len or artifact_summary.max_seq_len,
        num_corridor_records=dataset.num_records,
        max_seq_len=artifact_summary.max_seq_len,
        vocab_size=artifact_summary.vocab_size,
        tracked_stats=dataset.tracked_stats,
        logits_shape=(int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2])),
    )
    report = result.to_report()
    _raise_if_invalid_report(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_fingerprint_smoke_summary(report, summary_path)
    return result


def _validate_student_artifact_compatibility(
    *,
    artifact_vocab_size: int,
    artifact_max_seq_len: int,
    student_vocab_size: int,
    student_max_seq_len: int | None,
) -> None:
    if student_vocab_size != artifact_vocab_size:
        raise ValueError(
            "fingerprint artifact vocab_size="
            f"{artifact_vocab_size} but student vocab_size={student_vocab_size}"
        )
    if student_max_seq_len is not None and artifact_max_seq_len > student_max_seq_len:
        raise ValueError(
            "fingerprint artifact max_seq_len="
            f"{artifact_max_seq_len} exceeds student max_seq_len={student_max_seq_len}"
        )


def _validate_batch_token_ids(input_ids: np.ndarray, *, vocab_size: int) -> None:
    if input_ids.size == 0:
        raise ValueError("fingerprint batch input_ids must be non-empty")
    min_token = int(np.min(input_ids))
    max_token = int(np.max(input_ids))
    if min_token < 0 or max_token >= vocab_size:
        raise ValueError(
            "fingerprint batch input_ids outside student vocab: "
            f"min={min_token} max={max_token} vocab_size={vocab_size}"
        )


def _validate_logits_shape(
    logits: jax.Array,
    *,
    expected_batch_size: int,
    expected_seq_len: int,
    expected_vocab_size: int,
) -> None:
    if logits.ndim != 3:
        raise ValueError(
            f"student logits must be rank 3 [batch, seq, vocab], got {logits.shape}"
        )
    if logits.shape != (expected_batch_size, expected_seq_len, expected_vocab_size):
        raise ValueError(
            "student logits shape mismatch: expected "
            f"{(expected_batch_size, expected_seq_len, expected_vocab_size)}, "
            f"got {tuple(logits.shape)}"
        )


def _validate_positions_in_range(positions: np.ndarray, *, seq_len: int) -> None:
    if positions.size == 0:
        raise ValueError("fingerprint batch positions must be non-empty")
    min_position = int(np.min(positions))
    max_position = int(np.max(positions))
    if min_position < 0 or max_position >= seq_len:
        raise ValueError(
            "fingerprint target position outside student logits sequence: "
            f"min={min_position} max={max_position} seq_len={seq_len}"
        )


def _metrics_from_corridor(output: FingerprintCorridorLossOutput) -> dict[str, float]:
    return {
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


def _raise_if_invalid_report(report: dict[str, Any]) -> None:
    blockers = validate_fingerprint_smoke_report(report)
    if blockers:
        raise ValueError(
            "real student fingerprint report invalid: " + "; ".join(blockers)
        )
