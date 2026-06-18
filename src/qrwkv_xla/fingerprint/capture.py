from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
    BEHAVIORAL_FINGERPRINT_VERSION,
    validate_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import write_json
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats,
)

TRACKED_STATS: tuple[str, ...] = (
    "entropy",
    "top1_margin",
    "top8_mass",
    "top32_mass",
    "tail_mass",
)


@dataclass(frozen=True)
class FingerprintCaptureBudgetConfig:
    max_examples: int | None = None
    max_target_positions: int | None = None


@dataclass(frozen=True)
class FingerprintModeDiscoveryConfig:
    method: str = "stat_bands_v0"
    max_modes: int = 256
    min_mode_records: int = 1
    entropy_bins: tuple[float, ...] = (0.0, 1.0, 2.5, 4.0, 8.0, math.inf)
    top1_margin_bins: tuple[float, ...] = (0.0, 0.05, 0.15, 0.35, 1.0, math.inf)
    top32_mass_bins: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 1.0, math.inf)


@dataclass(frozen=True)
class FingerprintCorridorBoundsConfig:
    method: str = "minmax"
    min_width: float = 1.0e-6


@dataclass(frozen=True)
class FingerprintExemplarReservoirCaptureConfig:
    enabled: bool = True
    max_exemplars: int = 1000
    payload_type: str = "dense_probs"
    selection_policy: str = "top_interestingness_v0"
    per_mode_min: int = 0


@dataclass(frozen=True)
class FingerprintCaptureConfig:
    output_dir: Path
    artifact_version: str = BEHAVIORAL_FINGERPRINT_VERSION
    overwrite: bool = False
    teacher_model_name: str = "synthetic-p143-teacher"
    tokenizer_name: str = "synthetic-p143-tokenizer"
    dtype: str = "float32"
    capture_budget: FingerprintCaptureBudgetConfig = field(
        default_factory=FingerprintCaptureBudgetConfig
    )
    mode_discovery: FingerprintModeDiscoveryConfig = field(
        default_factory=FingerprintModeDiscoveryConfig
    )
    corridor_bounds: FingerprintCorridorBoundsConfig = field(
        default_factory=FingerprintCorridorBoundsConfig
    )
    exemplar_reservoir: FingerprintExemplarReservoirCaptureConfig = field(
        default_factory=FingerprintExemplarReservoirCaptureConfig
    )


@dataclass(frozen=True)
class FingerprintCaptureExample:
    example_id: str
    input_ids: tuple[int, ...]
    logits: np.ndarray


@dataclass(frozen=True)
class FingerprintCaptureResult:
    output_dir: Path
    manifest_path: Path
    modes_path: Path
    targets_path: Path
    exemplars_path: Path | None
    capture_summary_path: Path
    validation_ok: bool
    summary: dict[str, Any]


@dataclass
class _StatAggregate:
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.total += value

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass(frozen=True)
class _CapturedPosition:
    example_id: str
    input_ids: tuple[int, ...]
    position: int
    mode_id: int
    stats: dict[str, float]
    teacher_probs: tuple[float, ...]
    interestingness_score: float
    reason_codes: tuple[str, ...]


def capture_fingerprint_artifact(
    config: FingerprintCaptureConfig,
    examples: Any,
) -> FingerprintCaptureResult:
    _validate_config(config)
    selected_examples = _select_examples(tuple(examples), config.capture_budget)
    if not selected_examples:
        raise ValueError("fingerprint capture requires at least one example")

    max_seq_len, vocab_size = _validate_examples(selected_examples)
    if config.output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"output_dir already exists; pass overwrite=True: {config.output_dir}"
            )
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True)

    mode_ids: dict[tuple[int, int, int], int] = {}
    aggregates: dict[int, dict[str, _StatAggregate]] = {}
    captured: list[_CapturedPosition] = []

    for example in selected_examples:
        logits = np.asarray(example.logits, dtype=np.float32)
        stats = compute_fingerprint_distribution_stats(jnp.asarray(logits))
        stat_arrays = _stats_to_numpy(stats)
        probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        for position in range(logits.shape[0]):
            if (
                config.capture_budget.max_target_positions is not None
                and len(captured) >= config.capture_budget.max_target_positions
            ):
                break
            row_stats = {
                stat: float(stat_arrays[stat][position]) for stat in TRACKED_STATS
            }
            mode_key = _mode_key(row_stats, config.mode_discovery)
            if mode_key not in mode_ids:
                if len(mode_ids) >= config.mode_discovery.max_modes:
                    raise ValueError(
                        "stat_bands_v0 discovered more modes than max_modes="
                        f"{config.mode_discovery.max_modes}"
                    )
                mode_ids[mode_key] = len(mode_ids)
            mode_id = mode_ids[mode_key]
            aggregates.setdefault(
                mode_id, {stat: _StatAggregate() for stat in TRACKED_STATS}
            )
            for stat, value in row_stats.items():
                aggregates[mode_id][stat].update(value)
            captured.append(
                _CapturedPosition(
                    example_id=example.example_id,
                    input_ids=example.input_ids,
                    position=position,
                    mode_id=mode_id,
                    stats=row_stats,
                    teacher_probs=tuple(float(x) for x in probs[position]),
                    interestingness_score=_interestingness(row_stats),
                    reason_codes=_reason_codes(row_stats),
                )
            )
        if (
            config.capture_budget.max_target_positions is not None
            and len(captured) >= config.capture_budget.max_target_positions
        ):
            break

    if not captured:
        raise ValueError("fingerprint capture produced zero target positions")

    mode_bounds = _finalize_mode_bounds(aggregates, config.corridor_bounds)
    modes_payload = _modes_payload(mode_ids, aggregates, mode_bounds)
    target_rows = [
        _target_row(position, mode_bounds[position.mode_id]) for position in captured
    ]
    exemplar_rows = _select_exemplar_rows(captured, config.exemplar_reservoir)

    modes_path = config.output_dir / "modes.json"
    targets_path = config.output_dir / "targets" / "targets-00000.jsonl"
    exemplars_path = (
        config.output_dir / "exemplars" / "exemplars-00000.jsonl"
        if config.exemplar_reservoir.enabled
        else None
    )
    write_json(modes_path, modes_payload)
    _write_jsonl(targets_path, target_rows)
    if exemplars_path is not None:
        _write_jsonl(exemplars_path, exemplar_rows)

    manifest = _manifest_payload(
        config=config,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        target_count=len(target_rows),
        exemplar_count=len(exemplar_rows),
    )
    manifest_path = config.output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    validation = validate_fingerprint_artifact(config.output_dir)
    records_per_mode = Counter(str(position.mode_id) for position in captured)
    reason_distribution = Counter(
        reason for row in exemplar_rows for reason in row.get("reason_codes", ())
    )
    summary = _summary_payload(
        config=config,
        examples_processed=len(selected_examples),
        target_positions_processed=len(captured),
        modes_discovered=len(mode_ids),
        records_per_mode=dict(sorted(records_per_mode.items())),
        exemplars_retained=len(exemplar_rows),
        exemplar_reason_code_distribution=dict(sorted(reason_distribution.items())),
        artifact_validated=validation.ok,
        validation_blockers=validation.blockers,
        artifact_size_bytes=_artifact_size(config.output_dir),
    )
    capture_summary_path = config.output_dir / "capture_summary.json"
    write_json(capture_summary_path, summary)

    return FingerprintCaptureResult(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        modes_path=modes_path,
        targets_path=targets_path,
        exemplars_path=exemplars_path,
        capture_summary_path=capture_summary_path,
        validation_ok=validation.ok,
        summary=summary,
    )


def build_synthetic_capture_examples(
    *,
    num_examples: int,
    max_seq_len: int,
    vocab_size: int,
) -> tuple[FingerprintCaptureExample, ...]:
    if num_examples <= 0:
        raise ValueError(f"num_examples must be > 0, got {num_examples}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")

    examples: list[FingerprintCaptureExample] = []
    for example_index in range(num_examples):
        input_ids = tuple(
            int((example_index * 7 + position + 1) % vocab_size)
            for position in range(max_seq_len)
        )
        logits = np.zeros((max_seq_len, vocab_size), dtype=np.float32)
        for position in range(max_seq_len):
            pattern = (example_index + position) % 4
            primary = (example_index * 3 + position) % vocab_size
            secondary = (primary + 1) % vocab_size
            if pattern == 0:
                logits[position, primary] = 8.0
                logits[position, secondary] = 0.5
            elif pattern == 1:
                logits[position, primary] = 2.2
                logits[position, secondary] = 2.0
            elif pattern == 2:
                logits[position] = np.linspace(0.0, 1.0, vocab_size, dtype=np.float32)
            else:
                logits[position] = np.sin(
                    np.linspace(0.0, math.pi, vocab_size, dtype=np.float32)
                    + float(example_index)
                )
        examples.append(
            FingerprintCaptureExample(
                example_id=f"synthetic-{example_index:06d}",
                input_ids=input_ids,
                logits=logits,
            )
        )
    return tuple(examples)


def _validate_config(config: FingerprintCaptureConfig) -> None:
    if config.artifact_version != BEHAVIORAL_FINGERPRINT_VERSION:
        raise ValueError(
            "P143 supports artifact_version "
            f"{BEHAVIORAL_FINGERPRINT_VERSION!r}, got {config.artifact_version!r}"
        )
    if config.capture_budget.max_examples is not None:
        if config.capture_budget.max_examples <= 0:
            raise ValueError("capture_budget.max_examples must be > 0 when set")
    if config.capture_budget.max_target_positions is not None:
        if config.capture_budget.max_target_positions <= 0:
            raise ValueError("capture_budget.max_target_positions must be > 0 when set")
    if config.mode_discovery.method != "stat_bands_v0":
        raise ValueError("P143 supports only mode_discovery.method='stat_bands_v0'")
    if config.mode_discovery.max_modes <= 0:
        raise ValueError("mode_discovery.max_modes must be > 0")
    if config.mode_discovery.min_mode_records <= 0:
        raise ValueError("mode_discovery.min_mode_records must be > 0")
    _validate_bins(config.mode_discovery.entropy_bins, "entropy_bins")
    _validate_bins(config.mode_discovery.top1_margin_bins, "top1_margin_bins")
    _validate_bins(config.mode_discovery.top32_mass_bins, "top32_mass_bins")
    if config.corridor_bounds.method != "minmax":
        raise ValueError("P143 supports only corridor_bounds.method='minmax'")
    if config.corridor_bounds.min_width <= 0.0:
        raise ValueError("corridor_bounds.min_width must be > 0")
    if config.exemplar_reservoir.max_exemplars < 0:
        raise ValueError("exemplar_reservoir.max_exemplars must be >= 0")
    if config.exemplar_reservoir.payload_type != "dense_probs":
        raise ValueError("P143 supports only dense_probs exemplars")
    if config.exemplar_reservoir.selection_policy != "top_interestingness_v0":
        raise ValueError("P143 supports only selection_policy='top_interestingness_v0'")
    if config.exemplar_reservoir.per_mode_min < 0:
        raise ValueError("exemplar_reservoir.per_mode_min must be >= 0")


def _validate_bins(values: tuple[float, ...], name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two values")
    previous = -math.inf
    for value in values:
        if not math.isfinite(value) and value != math.inf:
            raise ValueError(f"{name} may only use finite values or +inf")
        if value <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        previous = value


def _select_examples(
    examples: tuple[FingerprintCaptureExample, ...],
    budget: FingerprintCaptureBudgetConfig,
) -> tuple[FingerprintCaptureExample, ...]:
    if budget.max_examples is None:
        return examples
    return examples[: budget.max_examples]


def _validate_examples(
    examples: tuple[FingerprintCaptureExample, ...],
) -> tuple[int, int]:
    first_logits = np.asarray(examples[0].logits)
    if first_logits.ndim != 2:
        raise ValueError("FingerprintCaptureExample.logits must be [seq, vocab]")
    max_seq_len, vocab_size = first_logits.shape
    if max_seq_len <= 0 or vocab_size <= 0:
        raise ValueError("FingerprintCaptureExample.logits must be non-empty")
    for example in examples:
        if not example.example_id.strip():
            raise ValueError("FingerprintCaptureExample.example_id must be non-empty")
        logits = np.asarray(example.logits)
        if logits.shape != (max_seq_len, vocab_size):
            raise ValueError(
                "all capture examples must share logits shape "
                f"{(max_seq_len, vocab_size)}, got {logits.shape}"
            )
        if len(example.input_ids) != max_seq_len:
            raise ValueError(
                "input_ids length must equal logits sequence length: "
                f"len(input_ids)={len(example.input_ids)} max_seq_len={max_seq_len}"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError(
                f"example {example.example_id!r} contains non-finite logits"
            )
        for token_id in example.input_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"example {example.example_id!r} has token id {token_id!r} "
                    f"outside [0, {vocab_size})"
                )
    return int(max_seq_len), int(vocab_size)


def _stats_to_numpy(stats: Any) -> dict[str, np.ndarray]:
    return {
        "entropy": np.asarray(stats.entropy, dtype=np.float64),
        "top1_margin": np.asarray(stats.top1_margin, dtype=np.float64),
        "top8_mass": np.asarray(stats.top8_mass, dtype=np.float64),
        "top32_mass": np.asarray(stats.top32_mass, dtype=np.float64),
        "tail_mass": np.asarray(stats.tail_mass, dtype=np.float64),
    }


def _mode_key(
    stats: dict[str, float],
    config: FingerprintModeDiscoveryConfig,
) -> tuple[int, int, int]:
    return (
        _bin_index(stats["entropy"], config.entropy_bins),
        _bin_index(stats["top1_margin"], config.top1_margin_bins),
        _bin_index(stats["top32_mass"], config.top32_mass_bins),
    )


def _bin_index(value: float, bins: tuple[float, ...]) -> int:
    index = int(np.searchsorted(np.asarray(bins), value, side="right") - 1)
    return max(0, min(index, len(bins) - 2))


def _interestingness(stats: dict[str, float]) -> float:
    return float(stats["entropy"] + stats["tail_mass"] - stats["top1_margin"])


def _reason_codes(stats: dict[str, float]) -> tuple[str, ...]:
    reasons: list[str] = []
    if stats["entropy"] >= 2.5:
        reasons.append("high_entropy")
    if stats["tail_mass"] >= 0.05:
        reasons.append("high_tail_mass")
    if stats["top1_margin"] <= 0.15:
        reasons.append("low_margin")
    if not reasons:
        reasons.append("mode_representative")
    return tuple(reasons)


def _finalize_mode_bounds(
    aggregates: dict[int, dict[str, _StatAggregate]],
    config: FingerprintCorridorBoundsConfig,
) -> dict[int, dict[str, dict[str, float]]]:
    return {
        mode_id: {
            stat: _bound_from_aggregate(stat, aggregate, config.min_width)
            for stat, aggregate in stat_aggregates.items()
        }
        for mode_id, stat_aggregates in aggregates.items()
    }


def _bound_from_aggregate(
    stat: str,
    aggregate: _StatAggregate,
    min_width: float,
) -> dict[str, float]:
    lower = aggregate.minimum
    upper = aggregate.maximum
    if upper - lower < min_width:
        center = (lower + upper) / 2.0
        lower = center - min_width / 2.0
        upper = center + min_width / 2.0
    if stat == "entropy":
        lower = max(0.0, lower)
        upper = max(lower, upper)
    else:
        lower = max(0.0, min(1.0, lower))
        upper = max(lower, min(1.0, upper))
    mean = aggregate.mean
    if stat == "entropy":
        mean = max(0.0, mean)
    else:
        mean = max(0.0, min(1.0, mean))
    mean = max(lower, min(upper, mean))
    return {"min": float(lower), "max": float(upper), "mean": float(mean)}


def _modes_payload(
    mode_ids: dict[tuple[int, int, int], int],
    aggregates: dict[int, dict[str, _StatAggregate]],
    mode_bounds: dict[int, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    by_id = {mode_id: mode_key for mode_key, mode_id in mode_ids.items()}
    modes: list[dict[str, Any]] = []
    for mode_id in sorted(by_id):
        entropy_bin, margin_bin, top32_bin = by_id[mode_id]
        modes.append(
            {
                "mode_id": mode_id,
                "name": f"stat_bands_v0/e{entropy_bin}_m{margin_bin}_t{top32_bin}",
                "description": "P143 stat_bands_v0 teacher-side capture mode.",
                "mode_key": {
                    "entropy_bin": entropy_bin,
                    "top1_margin_bin": margin_bin,
                    "top32_mass_bin": top32_bin,
                },
                "record_count": aggregates[mode_id]["entropy"].count,
                "bounds": mode_bounds[mode_id],
            }
        )
    return {"modes": modes}


def _target_row(
    position: _CapturedPosition,
    bounds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "example_id": position.example_id,
        "input_ids": list(position.input_ids),
        "position": position.position,
        "mode_id": position.mode_id,
        "bounds": bounds,
        "weight": 1.0,
    }


def _select_exemplar_rows(
    captured: list[_CapturedPosition],
    config: FingerprintExemplarReservoirCaptureConfig,
) -> list[dict[str, Any]]:
    if not config.enabled:
        return []
    if config.max_exemplars == 0:
        return []
    selected = sorted(
        captured,
        key=lambda row: (-row.interestingness_score, row.example_id, row.position),
    )[: config.max_exemplars]
    return [
        {
            "example_id": f"{row.example_id}:pos-{row.position}",
            "input_ids": list(row.input_ids),
            "position": row.position,
            "teacher_probs": list(row.teacher_probs),
            "mode_id": row.mode_id,
            "interestingness_score": row.interestingness_score,
            "reason_codes": list(row.reason_codes),
            "weight": 1.0,
        }
        for row in selected
    ]


def _manifest_payload(
    *,
    config: FingerprintCaptureConfig,
    max_seq_len: int,
    vocab_size: int,
    target_count: int,
    exemplar_count: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "artifact_type": BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
        "artifact_version": config.artifact_version,
        "created_by": "p143_teacher_side_capture_skeleton",
        "modes_file": "modes.json",
        "sequence": {
            "max_seq_len": max_seq_len,
            "target_positions": target_count,
        },
        "stats": {"tracked": list(TRACKED_STATS)},
        "target_shards": [
            {"path": "targets/targets-00000.jsonl", "num_records": target_count}
        ],
        "teacher": {
            "model_name": config.teacher_model_name,
            "tokenizer_name": config.tokenizer_name,
            "dtype": config.dtype,
            "vocab_size": vocab_size,
        },
        "capture": {
            "phase": "P143",
            "method": "teacher_side_capture_skeleton_v0",
            "mode_discovery_method": config.mode_discovery.method,
            "corridor_bounds_method": config.corridor_bounds.method,
            "exemplar_selection_policy": config.exemplar_reservoir.selection_policy,
        },
    }
    if config.exemplar_reservoir.enabled:
        manifest["exemplar_reservoir"] = {
            "enabled": True,
            "loss": "kl",
            "payload_type": config.exemplar_reservoir.payload_type,
            "num_records": exemplar_count,
            "max_exemplars": config.exemplar_reservoir.max_exemplars,
            "selection_policy": config.exemplar_reservoir.selection_policy,
            "shards": [
                {
                    "path": "exemplars/exemplars-00000.jsonl",
                    "num_records": exemplar_count,
                }
            ],
        }
    return manifest


def _summary_payload(
    *,
    config: FingerprintCaptureConfig,
    examples_processed: int,
    target_positions_processed: int,
    modes_discovered: int,
    records_per_mode: dict[str, int],
    exemplars_retained: int,
    exemplar_reason_code_distribution: dict[str, int],
    artifact_validated: bool,
    validation_blockers: tuple[str, ...],
    artifact_size_bytes: int,
) -> dict[str, Any]:
    return {
        "phase": "P143",
        "capture_method": "teacher_side_capture_skeleton_v0",
        "mode_discovery_method": config.mode_discovery.method,
        "corridor_bounds_method": config.corridor_bounds.method,
        "examples_processed": examples_processed,
        "target_positions_processed": target_positions_processed,
        "modes_discovered": modes_discovered,
        "records_per_mode": records_per_mode,
        "exemplar_reservoir_enabled": config.exemplar_reservoir.enabled,
        "max_exemplars": config.exemplar_reservoir.max_exemplars,
        "exemplars_retained": exemplars_retained,
        "exemplar_reason_code_distribution": exemplar_reason_code_distribution,
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_validated": artifact_validated,
        "validation_blockers": list(validation_blockers),
        "capture_config": _json_safe(asdict(config)),
        "teacher_required": False,
        "hf_required": False,
        "student_training_changed": False,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _artifact_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value
