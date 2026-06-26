from __future__ import annotations

import heapq
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import chain, islice
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
from qrwkv_xla.artifacts.cascaded_soft_labels import (
    DEFAULT_BUCKET_MASS_DTYPE,
    DEFAULT_BUCKET_MEAN_LOGP_DTYPE,
    DEFAULT_CASCADED_BUCKET_EDGES,
    DEFAULT_TOP_LOG_PROBS_DTYPE,
    encode_cascaded_soft_labels,
    validate_cascaded_bucket_edges,
)
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
TARGET_PAYLOAD_LEGACY_JSONL = "legacy_jsonl"
TARGET_PAYLOAD_PACKED_CORRIDOR_V1 = "packed_corridor_v1"
SUPPORTED_TARGET_PAYLOAD_TYPES = (
    TARGET_PAYLOAD_LEGACY_JSONL,
    TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
)


@dataclass(frozen=True)
class FingerprintCaptureBudgetConfig:
    max_examples: int | None = None
    max_target_positions: int | None = None
    packed_target_capacity: int | None = None


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
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    quantile_bins: int = 2048


@dataclass(frozen=True)
class FingerprintExemplarReservoirCaptureConfig:
    enabled: bool = True
    max_exemplars: int = 1000
    payload_type: str = "dense_probs"
    selection_policy: str = "top_interestingness_v0"
    per_mode_min: int = 0
    top_k: int = 256
    bucket_edges: tuple[float, ...] = DEFAULT_CASCADED_BUCKET_EDGES
    top_log_probs_dtype: str = DEFAULT_TOP_LOG_PROBS_DTYPE
    bucket_mass_dtype: str = DEFAULT_BUCKET_MASS_DTYPE
    bucket_mean_logp_dtype: str = DEFAULT_BUCKET_MEAN_LOGP_DTYPE
    shard_size: int = 256


@dataclass(frozen=True)
class FingerprintCaptureProgressConfig:
    enabled: bool = False
    progress_path: Path | None = None
    interval_seconds: float = 30.0
    interval_examples: int = 100
    teacher_batch_size: int | None = None


@dataclass(frozen=True)
class FingerprintCaptureConfig:
    output_dir: Path
    artifact_version: str = BEHAVIORAL_FINGERPRINT_VERSION
    overwrite: bool = False
    teacher_model_name: str = "synthetic-p143-teacher"
    tokenizer_name: str = "synthetic-p143-tokenizer"
    dtype: str = "float32"
    target_payload_type: str = TARGET_PAYLOAD_LEGACY_JSONL
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
    progress: FingerprintCaptureProgressConfig = field(
        default_factory=FingerprintCaptureProgressConfig
    )


@dataclass(frozen=True)
class FingerprintCaptureExample:
    example_id: str
    input_ids: tuple[int, ...]
    logits: np.ndarray


@dataclass(frozen=True)
class FingerprintCaptureBatch:
    example_ids: tuple[str, ...]
    input_ids: np.ndarray
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
    values: list[float] | None = field(default_factory=list)

    def update(self, value: float) -> None:
        self.count += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.total += value
        if self.values is not None:
            self.values.append(value)

    def update_many(self, values: np.ndarray) -> None:
        for value in values:
            self.update(float(value))

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def quantile(self, quantile: float, *, side: str) -> float:
        del side
        if self.values is None:
            raise ValueError("exact quantile values were not retained")
        return float(np.quantile(self.values, quantile))


@dataclass
class _HistogramStatAggregate:
    lower: float
    upper: float
    bins: int
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0
    histogram: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.histogram = np.zeros((self.bins,), dtype=np.int64)

    def update(self, value: float) -> None:
        self.count += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.total += value
        width = self.upper - self.lower
        if width <= 0.0:
            index = 0
        else:
            scaled = (value - self.lower) / width
            index = int(np.floor(scaled * self.bins))
            index = max(0, min(index, self.bins - 1))
        self.histogram[index] += 1

    def update_many(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.total += sum(float(value) for value in values)
        width = self.upper - self.lower
        if width <= 0.0:
            indexes = np.zeros(values.shape, dtype=np.int64)
        else:
            scaled = (values - self.lower) / width
            indexes = np.floor(scaled * self.bins).astype(np.int64)
            indexes = np.clip(indexes, 0, self.bins - 1)
        self.histogram += np.bincount(indexes, minlength=self.bins)[: self.bins]

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def quantile(self, quantile: float, *, side: str) -> float:
        if self.count == 0:
            return 0.0
        target = max(1, int(math.ceil(quantile * self.count)))
        index = int(np.searchsorted(np.cumsum(self.histogram), target, side="left"))
        index = max(0, min(index, self.bins - 1))
        edge_width = (self.upper - self.lower) / self.bins
        if side == "upper":
            return float(self.lower + edge_width * (index + 1))
        return float(self.lower + edge_width * index)


@dataclass(frozen=True)
class _CapturedPosition:
    example_id: str
    input_ids: tuple[int, ...]
    position: int
    mode_id: int
    stats: dict[str, float]
    exemplar_payload: dict[str, Any] | None
    interestingness_score: float
    reason_codes: tuple[str, ...]


class _BoundedExemplarReservoir:
    def __init__(self, config: FingerprintExemplarReservoirCaptureConfig) -> None:
        self._config = config
        self._global: list[tuple[Any, int, _CapturedPosition]] = []
        self._by_mode: dict[int, list[tuple[Any, int, _CapturedPosition]]] = {}
        self._sequence = 0

    def consider(
        self,
        position: _CapturedPosition,
        logits: np.ndarray,
        probs: np.ndarray | None,
    ) -> None:
        if not self._config.enabled or self._config.max_exemplars == 0:
            return
        quality = _reservoir_quality(position)
        global_eligible = _heap_would_accept(
            self._global, quality, self._config.max_exemplars
        )
        mode_heap = self._by_mode.setdefault(position.mode_id, [])
        mode_eligible = (
            self._config.selection_policy == "stratified_interestingness_v0"
            and self._config.per_mode_min > 0
            and _heap_would_accept(mode_heap, quality, self._config.per_mode_min)
        )
        if not global_eligible and not mode_eligible:
            return

        encoded = _CapturedPosition(
            example_id=position.example_id,
            input_ids=position.input_ids,
            position=position.position,
            mode_id=position.mode_id,
            stats=position.stats,
            exemplar_payload=_exemplar_payload(logits, probs, self._config),
            interestingness_score=position.interestingness_score,
            reason_codes=position.reason_codes,
        )
        if global_eligible:
            self._push(self._global, quality, encoded, self._config.max_exemplars)
        if mode_eligible:
            self._push(mode_heap, quality, encoded, self._config.per_mode_min)

    def selected(self) -> list[_CapturedPosition]:
        global_rows = [item[2] for item in self._global]
        if self._config.selection_policy != "stratified_interestingness_v0":
            return _top_interestingness(global_rows, self._config.max_exemplars)

        selected: list[_CapturedPosition] = []
        selected_keys: set[tuple[str, int]] = set()
        for mode_id in sorted(self._by_mode):
            rows = _top_interestingness(
                [item[2] for item in self._by_mode[mode_id]],
                self._config.per_mode_min,
            )
            for row in rows:
                if len(selected) >= self._config.max_exemplars:
                    return selected
                key = (row.example_id, row.position)
                if key not in selected_keys:
                    selected.append(row)
                    selected_keys.add(key)
        for row in _top_interestingness(global_rows, len(global_rows)):
            if len(selected) >= self._config.max_exemplars:
                break
            key = (row.example_id, row.position)
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)
        return selected

    def retained_count(self) -> int:
        return len(self.selected())

    def _push(
        self,
        heap: list[tuple[Any, int, _CapturedPosition]],
        quality: Any,
        row: _CapturedPosition,
        limit: int,
    ) -> None:
        item = (quality, self._sequence, row)
        self._sequence += 1
        if len(heap) < limit:
            heapq.heappush(heap, item)
        else:
            heapq.heapreplace(heap, item)


def _reservoir_quality(position: _CapturedPosition) -> tuple[Any, ...]:
    return (
        position.interestingness_score,
        tuple(-ord(char) for char in position.example_id),
        -position.position,
    )


def _heap_would_accept(
    heap: list[tuple[Any, int, _CapturedPosition]],
    quality: Any,
    limit: int,
) -> bool:
    return limit > 0 and (len(heap) < limit or quality > heap[0][0])


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _rate(count: int, elapsed_seconds: float) -> float | None:
    if count <= 0 or elapsed_seconds <= 0.0:
        return None
    return float(count / elapsed_seconds)


def _eta_seconds(
    *,
    examples_processed: int,
    examples_total: int | None,
    positions_processed: int,
    positions_total: int | None,
    elapsed_seconds: float,
) -> float | None:
    if elapsed_seconds <= 0.0:
        return None
    if positions_total is not None and positions_processed > 0:
        remaining = max(0, positions_total - positions_processed)
        return float(remaining / (positions_processed / elapsed_seconds))
    if examples_total is not None and examples_processed > 0:
        remaining = max(0, examples_total - examples_processed)
        return float(remaining / (examples_processed / elapsed_seconds))
    return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _print_progress_line(payload: dict[str, Any]) -> None:
    fields = (
        "status",
        "examples_processed",
        "examples_total",
        "target_positions_processed",
        "target_positions_total",
        "examples_per_second",
        "positions_per_second",
        "elapsed_seconds",
        "eta_seconds",
        "modes_discovered",
        "exemplars_retained",
        "exemplar_capacity",
    )
    print(
        " ".join(f"{field}={_progress_value(payload.get(field))}" for field in fields),
        flush=True,
    )


def _progress_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _wall_elapsed_seconds(payload: dict[str, Any]) -> float:
    started_at = payload.get("started_at")
    if not isinstance(started_at, str):
        return float(payload.get("elapsed_seconds") or 0.0)
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return float(payload.get("elapsed_seconds") or 0.0)
    return max(0.0, (datetime.now(UTC) - started).total_seconds())


class _CaptureProgressReporter:
    def __init__(
        self,
        *,
        config: FingerprintCaptureConfig,
        max_seq_len: int,
        examples_total: int | None,
        target_positions_total: int | None,
    ) -> None:
        self._config = config.progress
        self._capture_config = config
        self._max_seq_len = max_seq_len
        self._examples_total = examples_total
        self._target_positions_total = target_positions_total
        self._started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._last_emit_monotonic: float | None = None
        self._last_emit_examples = 0

    def emit_running(
        self,
        counters: dict[str, int],
        *,
        force: bool = False,
    ) -> None:
        if not self._config.enabled:
            return
        now = time.monotonic()
        if (
            not force
            and self._last_emit_monotonic is not None
            and now - self._last_emit_monotonic < self._config.interval_seconds
            and counters["examples_processed"] - self._last_emit_examples
            < self._config.interval_examples
        ):
            return
        self._emit("running", counters, now=now)

    def emit_complete(
        self,
        counters: dict[str, int],
        *,
        artifact_validated: bool,
        consumer_sanity: dict[str, Any] | None,
        artifact_size_bytes: int,
    ) -> None:
        if not self._config.enabled:
            return
        payload = self._payload("complete", counters, now=time.monotonic())
        payload.update(
            {
                "finished_at": _utc_now(),
                "artifact_validated": bool(artifact_validated),
                "validation_status": "pass" if artifact_validated else "fail",
                "consumer_sanity": consumer_sanity,
                "artifact_size_bytes": int(artifact_size_bytes),
            }
        )
        self._write(payload)

    def emit_failed(self, counters: dict[str, int], exc: Exception) -> None:
        if not self._config.enabled:
            return
        payload = self._payload("failed", counters, now=time.monotonic())
        payload.update(
            {
                "finished_at": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )
        self._write(payload)

    def _emit(
        self,
        status: str,
        counters: dict[str, int],
        *,
        now: float,
    ) -> None:
        payload = self._payload(status, counters, now=now)
        self._write(payload)
        self._last_emit_monotonic = now
        self._last_emit_examples = counters["examples_processed"]

    def _payload(
        self,
        status: str,
        counters: dict[str, int],
        *,
        now: float,
    ) -> dict[str, Any]:
        elapsed = max(0.0, now - self._started_monotonic)
        examples_processed = int(counters["examples_processed"])
        positions_processed = int(counters["target_positions_processed"])
        examples_per_second = _rate(examples_processed, elapsed)
        positions_per_second = _rate(positions_processed, elapsed)
        return {
            "schema_version": 1,
            "status": status,
            "phase": "teacher_capture",
            "started_at": self._started_at,
            "updated_at": _utc_now(),
            "elapsed_seconds": elapsed,
            "eta_seconds": _eta_seconds(
                examples_processed=examples_processed,
                examples_total=self._examples_total,
                positions_processed=positions_processed,
                positions_total=self._target_positions_total,
                elapsed_seconds=elapsed,
            ),
            "examples_processed": examples_processed,
            "examples_total": self._examples_total,
            "target_positions_processed": positions_processed,
            "target_positions_total": self._target_positions_total,
            "examples_per_second": examples_per_second,
            "positions_per_second": positions_per_second,
            "modes_discovered": int(counters["modes_discovered"]),
            "exemplars_retained": int(counters["exemplars_retained"]),
            "exemplar_capacity": self._capture_config.exemplar_reservoir.max_exemplars,
            "target_payload_type": self._capture_config.target_payload_type,
            "target_capture_memory_kind": _target_capture_memory_kind(
                self._capture_config
            ),
            "teacher_model": self._capture_config.teacher_model_name,
            "max_seq_len": self._max_seq_len,
            "pid": os.getpid(),
            "teacher_batch_size": self._config.teacher_batch_size,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        if self._config.progress_path is not None:
            _write_json_atomic(self._config.progress_path, payload)
        if payload["status"] == "running":
            _print_progress_line(payload)


def finalize_capture_progress(
    progress_path: Path,
    *,
    artifact_validated: bool,
    consumer_sanity: dict[str, Any] | None,
    artifact_size_bytes: int,
    examples_processed: int,
    target_positions_processed: int,
    modes_discovered: int,
    exemplars_retained: int,
    teacher_batch_size: int | None = None,
    batches_processed: int | None = None,
    effective_examples_per_batch: float | None = None,
) -> None:
    if not progress_path.exists():
        return
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    payload.update(
        {
            "status": "complete",
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
            "elapsed_seconds": _wall_elapsed_seconds(payload),
            "examples_processed": int(examples_processed),
            "target_positions_processed": int(target_positions_processed),
            "modes_discovered": int(modes_discovered),
            "exemplars_retained": int(exemplars_retained),
            "artifact_validated": bool(artifact_validated),
            "validation_status": "pass" if artifact_validated else "fail",
            "consumer_sanity": consumer_sanity,
            "artifact_size_bytes": int(artifact_size_bytes),
            "teacher_batch_size": teacher_batch_size,
            "batches_processed": batches_processed,
            "effective_examples_per_batch": effective_examples_per_batch,
        }
    )
    _write_json_atomic(progress_path, payload)


def mark_capture_progress_failed(progress_path: Path, exc: Exception) -> None:
    try:
        payload = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {}
        )
        if not isinstance(payload, dict):
            payload = {}
        payload.update(
            {
                "status": "failed",
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
                "elapsed_seconds": _wall_elapsed_seconds(payload),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "pid": os.getpid(),
            }
        )
        _write_json_atomic(progress_path, payload)
    except Exception:
        return


def capture_fingerprint_artifact(
    config: FingerprintCaptureConfig,
    examples: Any,
) -> FingerprintCaptureResult:
    _validate_config(config)
    selected_items = iter(examples)
    first_item = next(selected_items, None)
    if first_item is None:
        raise ValueError("fingerprint capture requires at least one example")
    first_batch = _coerce_capture_batch(first_item)
    _, max_seq_len, vocab_size = _validate_batch(first_batch)
    if config.output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"output_dir already exists; pass overwrite=True: {config.output_dir}"
            )
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True)

    packed_targets = config.target_payload_type == TARGET_PAYLOAD_PACKED_CORRIDOR_V1
    mode_ids: dict[tuple[int, int, int], int] = {}
    aggregates = {}
    captured: list[_CapturedPosition] = []
    target_writer = (
        _PackedTargetWriter.create(config, max_seq_len=max_seq_len)
        if packed_targets
        else None
    )
    exemplar_reservoir = _BoundedExemplarReservoir(config.exemplar_reservoir)

    examples_processed = 0
    examples_completed = 0
    batches_processed = 0
    target_positions_processed = 0
    records_per_mode: Counter[str] = Counter()
    progress = _CaptureProgressReporter(
        config=config,
        max_seq_len=max_seq_len,
        examples_total=config.capture_budget.max_examples,
        target_positions_total=_target_capacity(config, max_seq_len=max_seq_len),
    )

    def progress_counters() -> dict[str, int]:
        return {
            "examples_processed": examples_completed,
            "batches_processed": batches_processed,
            "target_positions_processed": target_positions_processed,
            "modes_discovered": len(mode_ids),
            "exemplars_retained": exemplar_reservoir.retained_count(),
        }

    progress.emit_running(progress_counters(), force=True)
    try:
        for batch in chain(
            (first_batch,), (_coerce_capture_batch(item) for item in selected_items)
        ):
            if config.capture_budget.max_examples is not None:
                examples_remaining = (
                    config.capture_budget.max_examples - examples_processed
                )
                if examples_remaining <= 0:
                    break
                batch = _trim_batch(batch, examples_remaining)
            batch_size, _, _ = _validate_batch(
                batch,
                expected_shape=(max_seq_len, vocab_size),
            )
            batches_processed += 1
            logits = np.asarray(batch.logits, dtype=np.float32)
            flat_logits = logits.reshape((batch_size * max_seq_len, vocab_size))
            stats = compute_fingerprint_distribution_stats(jnp.asarray(flat_logits))
            stat_arrays = {
                name: values.reshape((batch_size, max_seq_len))
                for name, values in _stats_to_numpy(stats).items()
            }
            probs = (
                np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
                if config.exemplar_reservoir.payload_type == "dense_probs"
                else None
            )
            for example_offset, example_id in enumerate(batch.example_ids):
                examples_processed += 1
                if (
                    config.capture_budget.max_target_positions is not None
                    and target_positions_processed
                    >= config.capture_budget.max_target_positions
                ):
                    break
                input_ids = tuple(
                    int(token) for token in batch.input_ids[example_offset]
                )
                positions_remaining = (
                    max_seq_len
                    if config.capture_budget.max_target_positions is None
                    else min(
                        max_seq_len,
                        config.capture_budget.max_target_positions
                        - target_positions_processed,
                    )
                )
                mode_id_rows: list[int] = []
                stat_values_by_mode: dict[int, dict[str, list[float]]] = {}
                for position in range(positions_remaining):
                    row_stats = {
                        stat: float(stat_arrays[stat][example_offset, position])
                        for stat in TRACKED_STATS
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
                    mode_id_rows.append(mode_id)
                    aggregates.setdefault(
                        mode_id,
                        {
                            stat: _new_stat_aggregate(
                                stat,
                                config=config,
                                vocab_size=vocab_size,
                                packed_targets=packed_targets,
                            )
                            for stat in TRACKED_STATS
                        },
                    )
                    mode_values = stat_values_by_mode.setdefault(
                        mode_id,
                        {stat: [] for stat in TRACKED_STATS},
                    )
                    for stat, value in row_stats.items():
                        mode_values[stat].append(value)
                    captured_position = _CapturedPosition(
                        example_id=example_id,
                        input_ids=input_ids,
                        position=position,
                        mode_id=mode_id,
                        stats=row_stats,
                        exemplar_payload=None,
                        interestingness_score=_interestingness(row_stats),
                        reason_codes=_reason_codes(row_stats),
                    )
                    if not packed_targets:
                        captured.append(captured_position)
                    exemplar_reservoir.consider(
                        captured_position,
                        logits[example_offset, position],
                        probs[example_offset, position] if probs is not None else None,
                    )
                    records_per_mode[str(mode_id)] += 1
                for mode_id, values_by_stat in stat_values_by_mode.items():
                    for stat, values in values_by_stat.items():
                        aggregates[mode_id][stat].update_many(
                            np.asarray(values, dtype=np.float64)
                        )
                if target_writer is not None and mode_id_rows:
                    target_writer.add_example_positions(
                        example_id=example_id,
                        input_ids=input_ids,
                        positions=np.arange(len(mode_id_rows), dtype=np.int32),
                        mode_ids=np.asarray(mode_id_rows, dtype=np.int32),
                    )
                target_positions_processed += len(mode_id_rows)
                if len(mode_id_rows) == max_seq_len:
                    examples_completed += 1
                progress.emit_running(progress_counters())
                if (
                    config.capture_budget.max_target_positions is not None
                    and target_positions_processed
                    >= config.capture_budget.max_target_positions
                ):
                    break
            if (
                config.capture_budget.max_target_positions is not None
                and target_positions_processed
                >= config.capture_budget.max_target_positions
            ):
                break
            if (
                config.capture_budget.max_examples is not None
                and examples_processed >= config.capture_budget.max_examples
            ):
                break
    except Exception as exc:
        progress.emit_failed(progress_counters(), exc)
        raise

    if target_positions_processed == 0:
        error = ValueError("fingerprint capture produced zero target positions")
        progress.emit_failed(progress_counters(), error)
        raise error

    try:
        mode_bounds = _finalize_mode_bounds(aggregates, config.corridor_bounds)
        modes_payload = _modes_payload(mode_ids, aggregates, mode_bounds)
        exemplar_rows = _exemplar_rows(exemplar_reservoir.selected())

        modes_path = config.output_dir / "modes.json"
        exemplar_shards = (
            _write_exemplar_shards(
                config.output_dir,
                exemplar_rows,
                config.exemplar_reservoir.shard_size,
            )
            if config.exemplar_reservoir.enabled
            else []
        )
        exemplars_path = (
            config.output_dir / exemplar_shards[0]["path"] if exemplar_shards else None
        )
        write_json(modes_path, modes_payload)
        if target_writer is None:
            target_payload, targets_path = _write_target_payload(
                config=config,
                captured=captured,
                mode_bounds=mode_bounds,
                max_seq_len=max_seq_len,
            )
        else:
            target_payload = target_writer.write(config.output_dir)
            targets_path = config.output_dir / "targets"
        manifest = _manifest_payload(
            config=config,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            target_count=target_positions_processed,
            target_payload=target_payload,
            target_capture_memory_kind=_target_capture_memory_kind(config),
            target_capacity=_target_capacity(config, max_seq_len=max_seq_len),
            quantile_aggregation_kind=_quantile_aggregation_kind(
                config, packed_targets
            ),
            bounded_quantile_state_size=_bounded_quantile_state_size(
                config,
                modes_discovered=len(mode_ids),
                packed_targets=packed_targets,
            ),
            exemplar_count=len(exemplar_rows),
            exemplar_shards=exemplar_shards,
        )
        manifest_path = config.output_dir / "manifest.json"
        write_json(manifest_path, manifest)

        validation = validate_fingerprint_artifact(config.output_dir)
        reason_distribution = Counter(
            reason for row in exemplar_rows for reason in row.get("reason_codes", ())
        )
        summary = _summary_payload(
            config=config,
            examples_processed=examples_processed,
            target_positions_processed=target_positions_processed,
            modes_discovered=len(mode_ids),
            records_per_mode=dict(sorted(records_per_mode.items())),
            exemplars_retained=len(exemplar_rows),
            exemplar_reason_code_distribution=dict(sorted(reason_distribution.items())),
            artifact_validated=validation.ok,
            validation_blockers=validation.blockers,
            artifact_size_bytes=_artifact_size(config.output_dir),
            target_payload_type=config.target_payload_type,
            target_capture_memory_kind=_target_capture_memory_kind(config),
            target_capacity=_target_capacity(config, max_seq_len=max_seq_len),
            quantile_aggregation_kind=_quantile_aggregation_kind(
                config, packed_targets
            ),
            bounded_quantile_state_size=_bounded_quantile_state_size(
                config,
                modes_discovered=len(mode_ids),
                packed_targets=packed_targets,
            ),
            target_payload_bytes=_target_payload_bytes(
                config.output_dir, target_payload
            ),
            exemplar_payload_bytes=sum(
                (config.output_dir / shard["path"]).stat().st_size
                for shard in exemplar_shards
            ),
            stored_top_k=[len(row.get("top_token_ids", ())) for row in exemplar_rows],
            dense_oracle_bytes=sum(
                len(
                    json.dumps(
                        {
                            **{
                                key: value
                                for key, value in row.items()
                                if key
                                not in {
                                    "top_token_ids",
                                    "top_log_probs",
                                    "top_mass",
                                    "tail_mass",
                                    "teacher_entropy",
                                    "bucket_edges",
                                    "bucket_mass",
                                    "bucket_count",
                                    "bucket_mean_logp",
                                    "original_vocab_size",
                                    "encoding_kind",
                                    "encoding_version",
                                }
                            },
                            "teacher_probs": [0.0] * vocab_size,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                for row in exemplar_rows
            ),
        )
        capture_summary_path = config.output_dir / "capture_summary.json"
        write_json(capture_summary_path, summary)
        progress.emit_complete(
            progress_counters(),
            artifact_validated=validation.ok,
            consumer_sanity=None,
            artifact_size_bytes=summary["artifact_size_bytes"],
        )
    except Exception as exc:
        progress.emit_failed(progress_counters(), exc)
        raise

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
    if config.target_payload_type not in SUPPORTED_TARGET_PAYLOAD_TYPES:
        raise ValueError(
            f"target_payload_type must be one of {SUPPORTED_TARGET_PAYLOAD_TYPES!r}"
        )
    if config.capture_budget.max_examples is not None:
        if config.capture_budget.max_examples <= 0:
            raise ValueError("capture_budget.max_examples must be > 0 when set")
    if config.capture_budget.max_target_positions is not None:
        if config.capture_budget.max_target_positions <= 0:
            raise ValueError("capture_budget.max_target_positions must be > 0 when set")
    if config.capture_budget.packed_target_capacity is not None:
        if config.capture_budget.packed_target_capacity <= 0:
            raise ValueError("capture_budget.packed_target_capacity must be > 0")
    if config.mode_discovery.method != "stat_bands_v0":
        raise ValueError("P143 supports only mode_discovery.method='stat_bands_v0'")
    if config.mode_discovery.max_modes <= 0:
        raise ValueError("mode_discovery.max_modes must be > 0")
    if config.mode_discovery.min_mode_records <= 0:
        raise ValueError("mode_discovery.min_mode_records must be > 0")
    _validate_bins(config.mode_discovery.entropy_bins, "entropy_bins")
    _validate_bins(config.mode_discovery.top1_margin_bins, "top1_margin_bins")
    _validate_bins(config.mode_discovery.top32_mass_bins, "top32_mass_bins")
    if config.corridor_bounds.method not in {"minmax", "quantile"}:
        raise ValueError("corridor_bounds.method must be 'minmax' or 'quantile'")
    if config.corridor_bounds.min_width <= 0.0:
        raise ValueError("corridor_bounds.min_width must be > 0")
    if not 0.0 <= config.corridor_bounds.lower_quantile < 1.0:
        raise ValueError("corridor_bounds.lower_quantile must be in [0.0, 1.0)")
    if not 0.0 < config.corridor_bounds.upper_quantile <= 1.0:
        raise ValueError("corridor_bounds.upper_quantile must be in (0.0, 1.0]")
    if config.corridor_bounds.lower_quantile >= config.corridor_bounds.upper_quantile:
        raise ValueError("corridor_bounds.lower_quantile must be < upper_quantile")
    if config.corridor_bounds.quantile_bins <= 0:
        raise ValueError("corridor_bounds.quantile_bins must be > 0")
    if config.exemplar_reservoir.max_exemplars < 0:
        raise ValueError("exemplar_reservoir.max_exemplars must be >= 0")
    if config.exemplar_reservoir.payload_type not in {
        "dense_probs",
        "cascaded_soft_labels_v1",
    }:
        raise ValueError("unsupported exemplar_reservoir.payload_type")
    if config.exemplar_reservoir.top_k <= 0:
        raise ValueError("exemplar_reservoir.top_k must be > 0")
    validate_cascaded_bucket_edges(config.exemplar_reservoir.bucket_edges)
    if config.exemplar_reservoir.shard_size <= 0:
        raise ValueError("exemplar_reservoir.shard_size must be > 0")
    if config.exemplar_reservoir.selection_policy not in {
        "top_interestingness_v0",
        "stratified_interestingness_v0",
    }:
        raise ValueError(
            "exemplar_reservoir.selection_policy must be "
            "'top_interestingness_v0' or 'stratified_interestingness_v0'"
        )
    if config.exemplar_reservoir.per_mode_min < 0:
        raise ValueError("exemplar_reservoir.per_mode_min must be >= 0")
    if config.progress.enabled:
        if config.progress.interval_seconds <= 0.0:
            raise ValueError("progress.interval_seconds must be > 0")
        if config.progress.interval_examples <= 0:
            raise ValueError("progress.interval_examples must be > 0")


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
    examples: Any,
    budget: FingerprintCaptureBudgetConfig,
) -> Any:
    if budget.max_examples is None:
        return examples
    return islice(examples, budget.max_examples)


def _coerce_capture_batch(item: Any) -> FingerprintCaptureBatch:
    if isinstance(item, FingerprintCaptureBatch):
        return item
    if isinstance(item, FingerprintCaptureExample):
        return FingerprintCaptureBatch(
            example_ids=(item.example_id,),
            input_ids=np.asarray([item.input_ids], dtype=np.int32),
            logits=np.asarray([item.logits], dtype=np.float32),
        )
    raise TypeError(
        "fingerprint capture inputs must be FingerprintCaptureExample "
        "or FingerprintCaptureBatch"
    )


def _trim_batch(
    batch: FingerprintCaptureBatch,
    max_examples: int,
) -> FingerprintCaptureBatch:
    if max_examples >= len(batch.example_ids):
        return batch
    return FingerprintCaptureBatch(
        example_ids=batch.example_ids[:max_examples],
        input_ids=np.asarray(batch.input_ids[:max_examples], dtype=np.int32),
        logits=np.asarray(batch.logits[:max_examples], dtype=np.float32),
    )


def _validate_example(
    example: FingerprintCaptureExample,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    logits = np.asarray(example.logits)
    if logits.ndim != 2:
        raise ValueError("FingerprintCaptureExample.logits must be [seq, vocab]")
    max_seq_len, vocab_size = logits.shape
    if max_seq_len <= 0 or vocab_size <= 0:
        raise ValueError("FingerprintCaptureExample.logits must be non-empty")
    if expected_shape is not None and logits.shape != expected_shape:
        raise ValueError(
            f"all capture examples must share logits shape {expected_shape}, "
            f"got {logits.shape}"
        )
    if not example.example_id.strip():
        raise ValueError("FingerprintCaptureExample.example_id must be non-empty")
    if len(example.input_ids) != max_seq_len:
        raise ValueError("input_ids length must equal logits sequence length")
    if not np.all(np.isfinite(logits)):
        raise ValueError(f"example {example.example_id!r} contains non-finite logits")
    for token_id in example.input_ids:
        if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            raise ValueError(
                f"example {example.example_id!r} has token id {token_id!r} "
                f"outside [0, {vocab_size})"
            )
    return int(max_seq_len), int(vocab_size)


def _validate_batch(
    batch: FingerprintCaptureBatch,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int]:
    logits = np.asarray(batch.logits)
    input_ids = np.asarray(batch.input_ids)
    if logits.ndim != 3:
        raise ValueError("FingerprintCaptureBatch.logits must be [batch, seq, vocab]")
    batch_size, max_seq_len, vocab_size = logits.shape
    if batch_size <= 0 or max_seq_len <= 0 or vocab_size <= 0:
        raise ValueError("FingerprintCaptureBatch.logits must be non-empty")
    if len(batch.example_ids) != batch_size:
        raise ValueError("FingerprintCaptureBatch.example_ids length must match batch")
    if input_ids.shape != (batch_size, max_seq_len):
        raise ValueError("FingerprintCaptureBatch.input_ids must be [batch, seq]")
    if expected_shape is not None and (max_seq_len, vocab_size) != expected_shape:
        raise ValueError(
            f"all capture examples must share logits shape {expected_shape}, "
            f"got {(max_seq_len, vocab_size)}"
        )
    if any(not example_id.strip() for example_id in batch.example_ids):
        raise ValueError("FingerprintCaptureBatch.example_ids must be non-empty")
    if not np.all(np.isfinite(logits)):
        raise ValueError("FingerprintCaptureBatch contains non-finite logits")
    if np.any(input_ids < 0) or np.any(input_ids >= vocab_size):
        raise ValueError("FingerprintCaptureBatch token ids outside vocabulary")
    return int(batch_size), int(max_seq_len), int(vocab_size)


def _stats_to_numpy(stats: Any) -> dict[str, np.ndarray]:
    return {
        "entropy": np.asarray(stats.entropy, dtype=np.float64),
        "top1_margin": np.asarray(stats.top1_margin, dtype=np.float64),
        "top8_mass": np.asarray(stats.top8_mass, dtype=np.float64),
        "top32_mass": np.asarray(stats.top32_mass, dtype=np.float64),
        "tail_mass": np.asarray(stats.tail_mass, dtype=np.float64),
    }


def _new_stat_aggregate(
    stat: str,
    *,
    config: FingerprintCaptureConfig,
    vocab_size: int,
    packed_targets: bool,
) -> Any:
    if packed_targets and config.corridor_bounds.method == "quantile":
        lower, upper = _histogram_range(stat, vocab_size)
        return _HistogramStatAggregate(
            lower=lower,
            upper=upper,
            bins=config.corridor_bounds.quantile_bins,
        )
    retain_values = config.corridor_bounds.method == "quantile"
    return _StatAggregate(values=[] if retain_values else None)


def _histogram_range(stat: str, vocab_size: int) -> tuple[float, float]:
    if stat == "entropy":
        return 0.0, max(float(math.log(vocab_size)), 1.0e-12)
    return 0.0, 1.0


def _target_capacity(
    config: FingerprintCaptureConfig,
    *,
    max_seq_len: int,
) -> int | None:
    if config.capture_budget.packed_target_capacity is not None:
        return config.capture_budget.packed_target_capacity
    if config.capture_budget.max_target_positions is not None:
        return config.capture_budget.max_target_positions
    if config.capture_budget.max_examples is not None:
        return config.capture_budget.max_examples * max_seq_len
    return None


def _target_capture_memory_kind(config: FingerprintCaptureConfig) -> str:
    if config.target_payload_type == TARGET_PAYLOAD_PACKED_CORRIDOR_V1:
        return "preallocated_typed_arrays"
    return "legacy_python_target_objects"


def _quantile_aggregation_kind(
    config: FingerprintCaptureConfig,
    packed_targets: bool,
) -> str:
    if config.corridor_bounds.method != "quantile":
        return "not_applicable_minmax"
    if packed_targets:
        return "bounded_fixed_histogram_v1"
    return "legacy_exact_values"


def _bounded_quantile_state_size(
    config: FingerprintCaptureConfig,
    *,
    modes_discovered: int,
    packed_targets: bool,
) -> int:
    if not packed_targets or config.corridor_bounds.method != "quantile":
        return 0
    return modes_discovered * len(TRACKED_STATS) * config.corridor_bounds.quantile_bins


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
    aggregates: dict[int, dict[str, Any]],
    config: FingerprintCorridorBoundsConfig,
) -> dict[int, dict[str, dict[str, float]]]:
    return {
        mode_id: {
            stat: _bound_from_aggregate(stat, aggregate, config)
            for stat, aggregate in stat_aggregates.items()
        }
        for mode_id, stat_aggregates in aggregates.items()
    }


def _bound_from_aggregate(
    stat: str,
    aggregate: Any,
    config: FingerprintCorridorBoundsConfig,
) -> dict[str, float]:
    if config.method == "quantile":
        lower = aggregate.quantile(config.lower_quantile, side="lower")
        upper = aggregate.quantile(config.upper_quantile, side="upper")
    else:
        lower = aggregate.minimum
        upper = aggregate.maximum
    if upper - lower < config.min_width:
        center = (lower + upper) / 2.0
        lower = center - config.min_width / 2.0
        upper = center + config.min_width / 2.0
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
    aggregates: dict[int, dict[str, Any]],
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


class _PackedTargetWriter:
    def __init__(self, *, capacity: int, max_seq_len: int) -> None:
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.count = 0
        self.example_indexes: dict[str, int] = {}
        self.example_ids: list[str] = []
        self.example_input_ids: list[tuple[int, ...]] = []
        self.position_example_index = np.empty((capacity,), dtype=np.int32)
        self.position = np.empty((capacity,), dtype=np.int32)
        self.mode_id = np.empty((capacity,), dtype=np.int32)
        self.weight = np.ones((capacity,), dtype=np.float32)

    @classmethod
    def create(
        cls,
        config: FingerprintCaptureConfig,
        *,
        max_seq_len: int,
    ) -> _PackedTargetWriter:
        capacity = _target_capacity(config, max_seq_len=max_seq_len)
        if capacity is None:
            raise ValueError(
                "packed_corridor_v1 capture requires max_target_positions, "
                "max_examples, or packed_target_capacity to bound target memory"
            )
        return cls(capacity=capacity, max_seq_len=max_seq_len)

    def add(
        self,
        *,
        example_id: str,
        input_ids: tuple[int, ...],
        position: int,
        mode_id: int,
    ) -> None:
        if self.count >= self.capacity:
            raise ValueError(
                f"packed_corridor_v1 target capacity exceeded: capacity={self.capacity}"
            )
        example_index = self.example_indexes.get(example_id)
        if example_index is None:
            example_index = len(self.example_ids)
            self.example_indexes[example_id] = example_index
            self.example_ids.append(example_id)
            self.example_input_ids.append(input_ids)
        elif self.example_input_ids[example_index] != input_ids:
            raise ValueError(f"example_id has inconsistent input_ids: {example_id}")
        self.position_example_index[self.count] = example_index
        self.position[self.count] = position
        self.mode_id[self.count] = mode_id
        self.count += 1

    def add_example_positions(
        self,
        *,
        example_id: str,
        input_ids: tuple[int, ...],
        positions: np.ndarray,
        mode_ids: np.ndarray,
    ) -> None:
        if positions.shape != mode_ids.shape:
            raise ValueError("packed positions and mode_ids must have matching shape")
        count = int(positions.size)
        if count == 0:
            return
        if self.count + count > self.capacity:
            raise ValueError(
                f"packed_corridor_v1 target capacity exceeded: capacity={self.capacity}"
            )
        example_index = self.example_indexes.get(example_id)
        if example_index is None:
            example_index = len(self.example_ids)
            self.example_indexes[example_id] = example_index
            self.example_ids.append(example_id)
            self.example_input_ids.append(input_ids)
        elif self.example_input_ids[example_index] != input_ids:
            raise ValueError(f"example_id has inconsistent input_ids: {example_id}")
        target_slice = slice(self.count, self.count + count)
        self.position_example_index[target_slice] = example_index
        self.position[target_slice] = positions.astype(np.int32, copy=False)
        self.mode_id[target_slice] = mode_ids.astype(np.int32, copy=False)
        self.count += count

    def write(self, output_dir: Path) -> dict[str, Any]:
        targets_dir = output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        examples_input_ids = np.asarray(
            self.example_input_ids,
            dtype=np.int32,
        ).reshape((len(self.example_ids), self.max_seq_len))
        arrays = {
            "examples_input_ids": (
                "targets/examples_input_ids.npy",
                examples_input_ids,
            ),
            "position_example_index": (
                "targets/position_example_index.npy",
                self.position_example_index[: self.count].copy(),
            ),
            "position": ("targets/position.npy", self.position[: self.count].copy()),
            "mode_id": ("targets/mode_id.npy", self.mode_id[: self.count].copy()),
            "weight": ("targets/weight.npy", self.weight[: self.count].copy()),
        }
        for relative_path, array in arrays.values():
            np.save(output_dir / relative_path, array)

        metadata_path = targets_dir / "examples_metadata.jsonl"
        metadata_path.write_text(
            "".join(
                json.dumps(
                    {"example_index": index, "example_id": example_id},
                    sort_keys=True,
                )
                + "\n"
                for index, example_id in enumerate(self.example_ids)
            ),
            encoding="utf-8",
        )

        return _packed_target_payload(
            arrays=arrays,
            num_records=self.count,
            num_examples=len(self.example_ids),
            max_seq_len=self.max_seq_len,
            target_capacity=self.capacity,
        )


def _write_target_payload(
    *,
    config: FingerprintCaptureConfig,
    captured: list[_CapturedPosition],
    mode_bounds: dict[int, dict[str, dict[str, float]]],
    max_seq_len: int,
) -> tuple[dict[str, Any], Path]:
    if config.target_payload_type == TARGET_PAYLOAD_LEGACY_JSONL:
        target_rows = [
            _target_row(position, mode_bounds[position.mode_id])
            for position in captured
        ]
        targets_path = config.output_dir / "targets" / "targets-00000.jsonl"
        _write_jsonl(targets_path, target_rows)
        return (
            {
                "kind": TARGET_PAYLOAD_LEGACY_JSONL,
                "num_records": len(target_rows),
                "shards": [
                    {
                        "path": "targets/targets-00000.jsonl",
                        "num_records": len(target_rows),
                    }
                ],
            },
            targets_path,
        )
    raise ValueError("packed target payloads must use the streaming writer")


def _packed_target_payload(
    *,
    arrays: dict[str, tuple[str, np.ndarray]],
    num_records: int,
    num_examples: int,
    max_seq_len: int,
    target_capacity: int,
) -> dict[str, Any]:
    return {
        "kind": TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
        "num_records": num_records,
        "num_examples": num_examples,
        "max_seq_len": max_seq_len,
        "target_capacity": target_capacity,
        "target_capture_memory_kind": "preallocated_typed_arrays",
        "mode_table_path": "modes.json",
        "ordering": "capture_position_order_v1",
        "example_index_contract": "zero_based_row_index_into_examples_input_ids",
        "arrays": {
            name: {
                "path": relative_path,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, (relative_path, array) in arrays.items()
        },
        "examples_metadata": {
            "path": "targets/examples_metadata.jsonl",
            "num_records": num_examples,
        },
    }


def _exemplar_rows(selected: list[_CapturedPosition]) -> list[dict[str, Any]]:
    return [
        {
            "example_id": f"{row.example_id}:pos-{row.position}",
            "input_ids": list(row.input_ids),
            "position": row.position,
            **(row.exemplar_payload or {}),
            "mode_id": row.mode_id,
            "interestingness_score": row.interestingness_score,
            "reason_codes": list(row.reason_codes),
            "weight": 1.0,
        }
        for row in selected
    ]


def _top_interestingness(
    captured: list[_CapturedPosition],
    max_exemplars: int,
) -> list[_CapturedPosition]:
    return sorted(
        captured,
        key=lambda row: (-row.interestingness_score, row.example_id, row.position),
    )[:max_exemplars]


def _manifest_payload(
    *,
    config: FingerprintCaptureConfig,
    max_seq_len: int,
    vocab_size: int,
    target_count: int,
    target_payload: dict[str, Any],
    target_capture_memory_kind: str,
    target_capacity: int | None,
    quantile_aggregation_kind: str,
    bounded_quantile_state_size: int,
    exemplar_count: int,
    exemplar_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    legacy_shards = (
        target_payload.get("shards", [])
        if target_payload.get("kind") == TARGET_PAYLOAD_LEGACY_JSONL
        else []
    )
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
        "target_payload": target_payload,
        "target_shards": legacy_shards,
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
            "lower_quantile": config.corridor_bounds.lower_quantile,
            "upper_quantile": config.corridor_bounds.upper_quantile,
            "quantile_bins": config.corridor_bounds.quantile_bins,
            "quantile_aggregation_kind": quantile_aggregation_kind,
            "bounded_quantile_state_size": bounded_quantile_state_size,
            "target_capture_memory_kind": target_capture_memory_kind,
            "target_capacity": target_capacity,
            "actual_target_count": target_count,
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
            "encoding_contract": {
                "kind": config.exemplar_reservoir.payload_type,
                "version": 1,
                "top_k": config.exemplar_reservoir.top_k,
                "bucket_edges": list(config.exemplar_reservoir.bucket_edges),
                "top_log_probs_dtype": config.exemplar_reservoir.top_log_probs_dtype,
                "bucket_mass_dtype": config.exemplar_reservoir.bucket_mass_dtype,
                "bucket_mean_logp_dtype": (
                    config.exemplar_reservoir.bucket_mean_logp_dtype
                ),
            },
            "shards": exemplar_shards,
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
    target_payload_type: str,
    target_capture_memory_kind: str,
    target_capacity: int | None,
    quantile_aggregation_kind: str,
    bounded_quantile_state_size: int,
    target_payload_bytes: int,
    exemplar_payload_bytes: int,
    stored_top_k: list[int],
    dense_oracle_bytes: int,
) -> dict[str, Any]:
    mean_k = float(np.mean(stored_top_k)) if stored_top_k else 0.0
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
        "exemplar_selection_policy": config.exemplar_reservoir.selection_policy,
        "per_mode_min": config.exemplar_reservoir.per_mode_min,
        "artifact_size_bytes": artifact_size_bytes,
        "target_payload_type": target_payload_type,
        "target_capture_memory_kind": target_capture_memory_kind,
        "target_capacity": target_capacity,
        "actual_target_count": target_positions_processed,
        "quantile_aggregation_kind": quantile_aggregation_kind,
        "bounded_quantile_state_size": bounded_quantile_state_size,
        "quantile_bins": config.corridor_bounds.quantile_bins,
        "target_payload_bytes": target_payload_bytes,
        "bytes_per_target_position": (
            target_payload_bytes / target_positions_processed
            if target_positions_processed
            else 0.0
        ),
        "exemplar_target_type": config.exemplar_reservoir.payload_type,
        "configured_top_k": config.exemplar_reservoir.top_k,
        "actual_stored_k": {
            "min": min(stored_top_k, default=0),
            "mean": mean_k,
            "max": max(stored_top_k, default=0),
        },
        "exemplar_payload_bytes": exemplar_payload_bytes,
        "bytes_per_exemplar": (
            exemplar_payload_bytes / exemplars_retained if exemplars_retained else 0.0
        ),
        "dense_json_oracle_bytes": dense_oracle_bytes,
        "compression_ratio_vs_dense_json_oracle": (
            dense_oracle_bytes / exemplar_payload_bytes
            if exemplar_payload_bytes
            else None
        ),
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


def _target_payload_bytes(output_dir: Path, target_payload: dict[str, Any]) -> int:
    if target_payload.get("kind") == TARGET_PAYLOAD_LEGACY_JSONL:
        return sum(
            (output_dir / shard["path"]).stat().st_size
            for shard in target_payload.get("shards", ())
        )
    paths = [
        array["path"]
        for array in target_payload.get("arrays", {}).values()
        if isinstance(array, dict) and isinstance(array.get("path"), str)
    ]
    metadata = target_payload.get("examples_metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
        paths.append(metadata["path"])
    return sum((output_dir / path).stat().st_size for path in paths)


def _write_exemplar_shards(
    output_dir: Path, rows: list[dict[str, Any]], shard_size: int
) -> list[dict[str, Any]]:
    if not rows:
        relative = Path("exemplars") / "exemplars-00000.jsonl"
        _write_jsonl(output_dir / relative, [])
        return [{"path": str(relative), "num_records": 0}]
    shards = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        relative = Path("exemplars") / f"exemplars-{shard_index:05d}.jsonl"
        _write_jsonl(output_dir / relative, shard_rows)
        shards.append({"path": str(relative), "num_records": len(shard_rows)})
    return shards


def _exemplar_payload(
    logits: np.ndarray,
    probs: np.ndarray | None,
    config: FingerprintExemplarReservoirCaptureConfig,
) -> dict[str, Any]:
    if config.payload_type == "dense_probs":
        if probs is None:
            raise ValueError("dense exemplar probabilities are missing")
        return {"teacher_probs": [float(value) for value in probs]}
    encoded = encode_cascaded_soft_labels(
        logits,
        top_k=min(config.top_k, int(np.asarray(logits).size)),
        bucket_edges=config.bucket_edges,
        top_log_probs_dtype=config.top_log_probs_dtype,
        bucket_mass_dtype=config.bucket_mass_dtype,
        bucket_mean_logp_dtype=config.bucket_mean_logp_dtype,
    )
    return {
        "encoding_kind": "cascaded_soft_labels_v1",
        "encoding_version": 1,
        "original_vocab_size": int(np.asarray(logits).size),
        "top_token_ids": encoded.top_token_ids.tolist(),
        "top_log_probs": encoded.top_log_probs.astype(np.float32).tolist(),
        "top_mass": float(encoded.top_mass),
        "tail_mass": float(encoded.tail_mass),
        "teacher_entropy": float(encoded.teacher_entropy),
        "bucket_edges": list(config.bucket_edges),
        "bucket_mass": encoded.bucket_mass.astype(np.float32).tolist(),
        "bucket_count": encoded.bucket_count.tolist(),
        "bucket_mean_logp": encoded.bucket_mean_logp.astype(np.float32).tolist(),
    }


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
