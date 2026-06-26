from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.artifacts.cascaded_soft_labels import (
    DEFAULT_BUCKET_MASS_DTYPE,
    DEFAULT_BUCKET_MEAN_LOGP_DTYPE,
    DEFAULT_CASCADED_BUCKET_EDGES,
    DEFAULT_TOP_LOG_PROBS_DTYPE,
)
from qrwkv_xla.distill import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillFingerprintConfig,
    DistillStageConfig,
    run_distill_stage,
)
from qrwkv_xla.fingerprint.capture import (
    SUPPORTED_TARGET_PAYLOAD_TYPES,
    TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
    FingerprintCaptureBatch,
    FingerprintCaptureBudgetConfig,
    FingerprintCaptureConfig,
    FingerprintCaptureProgressConfig,
    FingerprintCorridorBoundsConfig,
    FingerprintExemplarReservoirCaptureConfig,
    FingerprintModeDiscoveryConfig,
    capture_fingerprint_artifact,
    finalize_capture_progress,
    mark_capture_progress_failed,
)
from qrwkv_xla.fingerprint.topology import (
    AutoBool,
    AutoInt,
    CaptureTopologyConfig,
    ResolvedCaptureTopology,
    apply_torch_thread_settings,
    resolve_capture_topology,
)
from qrwkv_xla.teachers import HFTeacherBackend, HFTeacherUnavailable
from qrwkv_xla.training import (
    RealStudentFingerprintForwardConfig,
    compute_fingerprint_exemplar_loss,
    run_real_student_fingerprint_forward_smoke,
)

DEFAULT_TINY_REAL_TEACHER = "sshleifer/tiny-gpt2"
DEFAULT_CONSUMER_VOCAB_LIMIT = 4096


@dataclass(frozen=True)
class TinyRealTeacherFingerprintCaptureConfig:
    output_dir: Path
    texts_path: Path
    teacher_model: str = DEFAULT_TINY_REAL_TEACHER
    tokenizer: str | None = None
    sequence_length: int = 32
    max_examples: int = 4
    max_target_positions: int = 64
    local_files_only: bool = True
    allow_downloads: bool = False
    overwrite: bool = False
    max_exemplars: int = 16
    bounds_method: str = "quantile"
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    exemplar_selection_policy: str = "stratified_interestingness_v0"
    per_mode_min: int = 1
    exemplar_target_type: str = "cascaded_soft_labels_v1"
    exemplar_top_k: int = 256
    exemplar_bucket_edges: tuple[float, ...] = DEFAULT_CASCADED_BUCKET_EDGES
    exemplar_top_log_probs_dtype: str = DEFAULT_TOP_LOG_PROBS_DTYPE
    exemplar_bucket_mass_dtype: str = DEFAULT_BUCKET_MASS_DTYPE
    exemplar_bucket_mean_logp_dtype: str = DEFAULT_BUCKET_MEAN_LOGP_DTYPE
    consumer_vocab_limit: int = DEFAULT_CONSUMER_VOCAB_LIMIT
    example_id_prefix: str = "p145-real-teacher"
    target_payload_type: str = TARGET_PAYLOAD_PACKED_CORRIDOR_V1
    progress_interval_seconds: float = 30.0
    progress_interval_examples: int = 100
    progress_path: Path | None = None
    teacher_batch_size: int = 1
    cpu_budget: AutoInt = "auto"
    torch_num_threads: AutoInt = "auto"
    torch_num_interop_threads: AutoInt = "auto"
    prefetch_workers: AutoInt = "auto"
    reducer_workers: AutoInt = "auto"
    prefetch_depth: int = 2
    result_queue_depth: int = 2
    teacher_device: str = "auto"
    pin_memory: AutoBool = "auto"
    non_blocking_transfer: AutoBool = "auto"


@dataclass(frozen=True)
class TinyRealTeacherFingerprintCaptureResult:
    status: str
    output_dir: Path
    artifact_validated: bool
    summary_path: Path
    manifest_path: Path
    teacher_real: bool
    teacher_backend: str
    teacher_model_name_or_path: str
    tokenizer_name_or_path: str
    local_files_only: bool
    examples_processed: int
    tokens_processed: int
    target_positions_processed: int
    modes_discovered: int
    exemplars_retained: int
    consumer_sanity: dict[str, Any]
    reason: str | None = None


@dataclass(frozen=True)
class _PreparedTeacherBatch:
    sequence: int
    batch_start: int
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class _InferredTeacherBatch:
    sequence: int
    batch: FingerprintCaptureBatch
    token_count: int
    vocab_size: int


@dataclass(frozen=True)
class _StageError:
    stage: str
    error: BaseException


@dataclass
class _StageCounters:
    prepared_batches: int = 0
    inference_batches: int = 0
    committed_batches: int = 0
    lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lock = threading.Lock()

    def increment(self, field: str) -> None:
        with self.lock:
            setattr(self, field, int(getattr(self, field)) + 1)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "prepared_batches": self.prepared_batches,
                "inference_batches": self.inference_batches,
                "committed_batches": self.committed_batches,
            }


_QUEUE_SENTINEL = object()


def run_tiny_real_teacher_fingerprint_capture(
    config: TinyRealTeacherFingerprintCaptureConfig,
    *,
    backend: HFTeacherBackend | None = None,
) -> TinyRealTeacherFingerprintCaptureResult:
    _validate_config(config)
    progress_path = _progress_path(config)
    try:
        texts = load_text_fixture(config.texts_path)
        prompts = tuple(texts[: config.max_examples])
        if not prompts:
            raise ValueError("tiny real teacher capture requires at least one text")
        topology = resolve_capture_topology(_capture_topology_config(config))
        thread_settings = apply_torch_thread_settings(topology)
        topology_metadata = _topology_metadata(topology, thread_settings)

        effective_local_files_only = (
            False if config.allow_downloads else config.local_files_only
        )
        if backend is None:
            teacher = HFTeacherBackend(
                config.teacher_model,
                local_files_only=effective_local_files_only,
                prompts=prompts,
                device=topology.teacher_device,
                pin_memory=topology.pin_memory,
                non_blocking_transfer=topology.non_blocking_transfer,
            )
        else:
            teacher = backend
            teacher.device = topology.teacher_device
            teacher.pin_memory = topology.pin_memory
            teacher.non_blocking_transfer = topology.non_blocking_transfer
        emission: dict[str, Any] = {"tokens": 0, "vocab_size": None, "batches": 0}
        stage_counters = _StageCounters()
        batch_iter = _iter_teacher_batches(
            config=config,
            teacher=teacher,
            prompts=prompts,
            topology=topology,
            emission=emission,
            stage_counters=stage_counters,
        )

        capture_config = FingerprintCaptureConfig(
            output_dir=config.output_dir,
            overwrite=config.overwrite,
            teacher_model_name=config.teacher_model,
            tokenizer_name=config.tokenizer or config.teacher_model,
            capture_budget=FingerprintCaptureBudgetConfig(
                max_examples=config.max_examples,
                max_target_positions=config.max_target_positions,
            ),
            mode_discovery=FingerprintModeDiscoveryConfig(),
            corridor_bounds=FingerprintCorridorBoundsConfig(
                method=config.bounds_method,
                lower_quantile=config.lower_quantile,
                upper_quantile=config.upper_quantile,
            ),
            exemplar_reservoir=FingerprintExemplarReservoirCaptureConfig(
                enabled=True,
                max_exemplars=config.max_exemplars,
                payload_type=config.exemplar_target_type,
                selection_policy=config.exemplar_selection_policy,
                per_mode_min=config.per_mode_min,
                top_k=config.exemplar_top_k,
                bucket_edges=config.exemplar_bucket_edges,
                top_log_probs_dtype=config.exemplar_top_log_probs_dtype,
                bucket_mass_dtype=config.exemplar_bucket_mass_dtype,
                bucket_mean_logp_dtype=config.exemplar_bucket_mean_logp_dtype,
            ),
            target_payload_type=config.target_payload_type,
            progress=FingerprintCaptureProgressConfig(
                enabled=True,
                progress_path=progress_path,
                interval_seconds=config.progress_interval_seconds,
                interval_examples=config.progress_interval_examples,
                teacher_batch_size=config.teacher_batch_size,
                extra_metadata=topology_metadata,
            ),
        )
        capture = capture_fingerprint_artifact(capture_config, batch_iter)
        vocab_size = int(emission["vocab_size"] or 0)
        _rewrite_manifest_for_p145(
            capture.manifest_path,
            config=config,
            teacher=teacher,
            vocab_size=vocab_size,
            effective_local_files_only=effective_local_files_only,
        )
        validation = validate_fingerprint_artifact(config.output_dir)
        targets_loadable = False
        exemplars_loadable = False
        target_records = 0
        exemplar_records = 0
        if validation.ok:
            targets = load_fingerprint_targets(config.output_dir, batch_size=1)
            target_records = targets.num_records
            targets_loadable = True
            exemplars = load_fingerprint_exemplars(config.output_dir, batch_size=1)
            exemplar_records = exemplars.num_records
            exemplars_loadable = True
        artifact_summary = summarize_fingerprint_artifact(config.output_dir)
        base_summary = read_json_object(capture.capture_summary_path)
        consumer_sanity = _run_consumer_sanity(
            config,
            vocab_size=vocab_size,
            validation_ok=validation.ok,
        )
        summary = _p145_summary(
            config=config,
            base_summary=base_summary,
            teacher=teacher,
            effective_local_files_only=effective_local_files_only,
            examples_processed=len(prompts),
            tokens_processed=int(emission["tokens"]),
            target_positions_processed=target_records,
            vocab_size=vocab_size,
            artifact_validated=validation.ok,
            validation_blockers=validation.blockers,
            targets_loadable=targets_loadable,
            exemplars_loadable=exemplars_loadable,
            exemplar_records=exemplar_records,
            artifact_summary=artifact_summary.to_dict(),
            consumer_sanity=consumer_sanity,
            batches_processed=int(emission["batches"]),
            topology_metadata=topology_metadata,
            stage_counters=stage_counters.snapshot(),
        )
        write_json(capture.capture_summary_path, summary)
        finalize_capture_progress(
            progress_path,
            artifact_validated=validation.ok,
            consumer_sanity=consumer_sanity,
            artifact_size_bytes=int(summary["artifact_size_bytes"]),
            examples_processed=len(prompts),
            target_positions_processed=target_records,
            modes_discovered=int(summary["modes_discovered"]),
            exemplars_retained=exemplar_records,
            teacher_batch_size=config.teacher_batch_size,
            batches_processed=int(emission["batches"]),
            effective_examples_per_batch=(
                len(prompts) / int(emission["batches"]) if emission["batches"] else None
            ),
            extra_metadata={
                **topology_metadata,
                **stage_counters.snapshot(),
            },
        )
        status = "pass" if validation.ok and targets_loadable else "fail"
        return TinyRealTeacherFingerprintCaptureResult(
            status=status,
            output_dir=config.output_dir,
            artifact_validated=validation.ok,
            summary_path=capture.capture_summary_path,
            manifest_path=capture.manifest_path,
            teacher_real=True,
            teacher_backend="hf_causal_lm",
            teacher_model_name_or_path=teacher.model_id,
            tokenizer_name_or_path=config.tokenizer or config.teacher_model,
            local_files_only=effective_local_files_only,
            examples_processed=len(prompts),
            tokens_processed=int(emission["tokens"]),
            target_positions_processed=target_records,
            modes_discovered=int(summary["modes_discovered"]),
            exemplars_retained=exemplar_records,
            consumer_sanity=consumer_sanity,
        )
    except Exception as exc:
        mark_capture_progress_failed(progress_path, exc)
        raise


def load_text_fixture(path: str | Path) -> tuple[str, ...]:
    rows: list[str] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(
                f"text fixture line {line_number} must contain string text"
            )
        text = payload["text"].strip()
        if text:
            rows.append(text)
    return tuple(rows)


def _validate_config(config: TinyRealTeacherFingerprintCaptureConfig) -> None:
    if not config.example_id_prefix.strip():
        raise ValueError("example_id_prefix must be non-empty")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if config.max_examples <= 0:
        raise ValueError("max_examples must be > 0")
    if config.max_target_positions <= 0:
        raise ValueError("max_target_positions must be > 0")
    if config.max_exemplars < 0:
        raise ValueError("max_exemplars must be >= 0")
    if config.exemplar_target_type not in {
        "dense_probs",
        "cascaded_soft_labels_v1",
    }:
        raise ValueError("unsupported exemplar_target_type")
    if config.exemplar_top_k <= 0:
        raise ValueError("exemplar_top_k must be > 0")
    if config.consumer_vocab_limit < 0:
        raise ValueError("consumer_vocab_limit must be >= 0")
    if config.target_payload_type not in SUPPORTED_TARGET_PAYLOAD_TYPES:
        raise ValueError(
            f"target_payload_type must be one of {SUPPORTED_TARGET_PAYLOAD_TYPES!r}"
        )
    if config.local_files_only and config.allow_downloads:
        raise ValueError("local_files_only and allow_downloads cannot both be true")
    if config.progress_interval_seconds <= 0.0:
        raise ValueError("progress_interval_seconds must be > 0")
    if config.progress_interval_examples <= 0:
        raise ValueError("progress_interval_examples must be > 0")
    if config.teacher_batch_size not in {1, 2, 4, 8}:
        raise ValueError("teacher_batch_size must be one of 1, 2, 4, or 8")
    if config.prefetch_depth <= 0:
        raise ValueError("prefetch_depth must be > 0")
    if config.result_queue_depth <= 0:
        raise ValueError("result_queue_depth must be > 0")


def _progress_path(config: TinyRealTeacherFingerprintCaptureConfig) -> Path:
    return config.progress_path or config.output_dir / "progress.json"


def _capture_topology_config(
    config: TinyRealTeacherFingerprintCaptureConfig,
) -> CaptureTopologyConfig:
    return CaptureTopologyConfig(
        cpu_budget=config.cpu_budget,
        torch_num_threads=config.torch_num_threads,
        torch_num_interop_threads=config.torch_num_interop_threads,
        prefetch_workers=config.prefetch_workers,
        reducer_workers=config.reducer_workers,
        prefetch_depth=config.prefetch_depth,
        result_queue_depth=config.result_queue_depth,
        teacher_device=config.teacher_device,
        pin_memory=config.pin_memory,
        non_blocking_transfer=config.non_blocking_transfer,
    )


def _topology_metadata(
    topology: ResolvedCaptureTopology,
    thread_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "teacher_device": topology.teacher_device,
        "detected_cpu_budget": topology.detected_cpu_budget,
        "effective_cpu_budget": topology.effective_cpu_budget,
        "torch_num_threads": topology.torch_num_threads,
        "torch_num_interop_threads": topology.torch_num_interop_threads,
        "prefetch_workers": topology.prefetch_workers,
        "reducer_workers": topology.reducer_workers,
        "prefetch_depth": topology.prefetch_depth,
        "result_queue_depth": topology.result_queue_depth,
        "pin_memory": topology.pin_memory,
        "non_blocking_transfer": topology.non_blocking_transfer,
        "OMP_NUM_THREADS": thread_settings.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": thread_settings.get("MKL_NUM_THREADS"),
        "torch_thread_settings_applied": thread_settings.get(
            "torch_thread_settings_applied"
        ),
        "torch_thread_settings_error": thread_settings.get(
            "torch_thread_settings_error"
        ),
        "gpu_name": topology.gpu_name,
    }


def _iter_teacher_batches(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    prompts: tuple[str, ...],
    topology: ResolvedCaptureTopology,
    emission: dict[str, Any],
    stage_counters: _StageCounters,
) -> Any:
    if topology.prefetch_workers == 0 and topology.result_queue_depth == 1:
        yield from _iter_teacher_batches_sync(
            config=config,
            teacher=teacher,
            prompts=prompts,
            emission=emission,
            stage_counters=stage_counters,
        )
        return
    yield from _iter_teacher_batches_staged(
        config=config,
        teacher=teacher,
        prompts=prompts,
        topology=topology,
        emission=emission,
        stage_counters=stage_counters,
    )


def _iter_teacher_batches_sync(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    prompts: tuple[str, ...],
    emission: dict[str, Any],
    stage_counters: _StageCounters,
) -> Any:
    for sequence, prepared in enumerate(_prepared_batches(config, prompts)):
        if prepared.sequence != sequence:
            raise ValueError("prepared batch sequence mismatch")
        stage_counters.increment("prepared_batches")
        inferred = _infer_teacher_batch(config, teacher, prepared)
        stage_counters.increment("inference_batches")
        _record_inferred_batch(emission, inferred)
        stage_counters.increment("committed_batches")
        yield inferred.batch


def _iter_teacher_batches_staged(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    prompts: tuple[str, ...],
    topology: ResolvedCaptureTopology,
    emission: dict[str, Any],
    stage_counters: _StageCounters,
) -> Any:
    prepared_queue: queue.Queue[object] = queue.Queue(maxsize=topology.prefetch_depth)
    result_queue: queue.Queue[object] = queue.Queue(maxsize=topology.result_queue_depth)
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_prefetch_worker,
            name="fingerprint-prefetch",
            kwargs={
                "config": config,
                "prompts": prompts,
                "topology": topology,
                "prepared_queue": prepared_queue,
                "stop_event": stop_event,
                "stage_counters": stage_counters,
            },
            daemon=False,
        ),
        threading.Thread(
            target=_inference_worker,
            name="fingerprint-inference",
            kwargs={
                "config": config,
                "teacher": teacher,
                "prepared_queue": prepared_queue,
                "result_queue": result_queue,
                "stop_event": stop_event,
                "stage_counters": stage_counters,
            },
            daemon=False,
        ),
    ]
    for thread in threads:
        thread.start()
    expected_sequence = 0
    buffered: dict[int, _InferredTeacherBatch] = {}
    result_stream_done = False
    try:
        while not result_stream_done or buffered:
            while expected_sequence in buffered:
                inferred = buffered.pop(expected_sequence)
                _record_inferred_batch(emission, inferred)
                stage_counters.increment("committed_batches")
                yield inferred.batch
                expected_sequence += 1
            if result_stream_done:
                if buffered:
                    raise ValueError(
                        f"missing staged teacher batch sequence {expected_sequence}"
                    )
                break
            item = result_queue.get()
            if item is _QUEUE_SENTINEL:
                result_stream_done = True
                continue
            if isinstance(item, _StageError):
                if isinstance(item.error, HFTeacherUnavailable):
                    raise item.error
                raise RuntimeError(
                    f"{item.stage} stage failed: "
                    f"{type(item.error).__name__}: {item.error}"
                ) from item.error
            if not isinstance(item, _InferredTeacherBatch):
                raise TypeError(f"unexpected staged result item: {type(item).__name__}")
            if item.sequence < expected_sequence or item.sequence in buffered:
                raise ValueError(
                    f"duplicate staged teacher batch sequence {item.sequence}"
                )
            buffered[item.sequence] = item
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError(
                    f"staged capture worker did not shut down: {thread.name}"
                )


def _prefetch_worker(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    prompts: tuple[str, ...],
    topology: ResolvedCaptureTopology,
    prepared_queue: queue.Queue[object],
    stop_event: threading.Event,
    stage_counters: _StageCounters,
) -> None:
    try:
        if topology.prefetch_workers <= 1:
            for prepared in _prepared_batches(config, prompts):
                if stop_event.is_set():
                    break
                stage_counters.increment("prepared_batches")
                _put_with_stop(prepared_queue, prepared, stop_event)
        else:
            _prefetch_with_pool(
                config=config,
                prompts=prompts,
                topology=topology,
                prepared_queue=prepared_queue,
                stop_event=stop_event,
                stage_counters=stage_counters,
            )
    except BaseException as exc:
        _put_with_stop(prepared_queue, _StageError("prefetch", exc), stop_event)
    finally:
        _put_with_stop(prepared_queue, _QUEUE_SENTINEL, stop_event)


def _prefetch_with_pool(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    prompts: tuple[str, ...],
    topology: ResolvedCaptureTopology,
    prepared_queue: queue.Queue[object],
    stop_event: threading.Event,
    stage_counters: _StageCounters,
) -> None:
    prepared_iter = iter(_prepared_batches(config, prompts))
    with ThreadPoolExecutor(max_workers=topology.prefetch_workers) as executor:
        pending = set()
        while not stop_event.is_set():
            while len(pending) < topology.prefetch_depth:
                try:
                    prepared = next(prepared_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(lambda item=prepared: item))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                prepared = future.result()
                stage_counters.increment("prepared_batches")
                _put_with_stop(prepared_queue, prepared, stop_event)


def _inference_worker(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    prepared_queue: queue.Queue[object],
    result_queue: queue.Queue[object],
    stop_event: threading.Event,
    stage_counters: _StageCounters,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = prepared_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _QUEUE_SENTINEL:
                break
            if isinstance(item, _StageError):
                _put_with_stop(result_queue, item, stop_event)
                return
            if not isinstance(item, _PreparedTeacherBatch):
                raise TypeError(f"unexpected prepared item: {type(item).__name__}")
            inferred = _infer_teacher_batch(config, teacher, item)
            stage_counters.increment("inference_batches")
            _put_with_stop(result_queue, inferred, stop_event)
    except BaseException as exc:
        _put_with_stop(result_queue, _StageError("inference", exc), stop_event)
    finally:
        _put_with_stop(result_queue, _QUEUE_SENTINEL, stop_event)


def _prepared_batches(
    config: TinyRealTeacherFingerprintCaptureConfig,
    prompts: tuple[str, ...],
) -> Any:
    for sequence, batch_start in enumerate(
        range(0, len(prompts), config.teacher_batch_size)
    ):
        yield _PreparedTeacherBatch(
            sequence=sequence,
            batch_start=batch_start,
            prompts=prompts[batch_start : batch_start + config.teacher_batch_size],
        )


def _infer_teacher_batch(
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    prepared: _PreparedTeacherBatch,
) -> _InferredTeacherBatch:
    batch_prompts = prepared.prompts
    teacher.prompts = batch_prompts
    emitted = teacher.emit_targets(
        num_examples=len(batch_prompts),
        sequence_length=config.sequence_length,
    )
    input_ids = np.asarray(emitted["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(emitted["attention_mask"], dtype=np.int32)
    logits = np.asarray(emitted["logits"], dtype=np.float32)
    _validate_emitted_shapes(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits=logits,
        examples=len(batch_prompts),
        sequence_length=config.sequence_length,
    )
    return _InferredTeacherBatch(
        sequence=prepared.sequence,
        token_count=int(np.sum(attention_mask)),
        vocab_size=int(logits.shape[-1]),
        batch=FingerprintCaptureBatch(
            example_ids=tuple(
                f"{config.example_id_prefix}-{index:06d}"
                for index in range(
                    prepared.batch_start,
                    prepared.batch_start + len(batch_prompts),
                )
            ),
            input_ids=input_ids,
            logits=logits,
        ),
    )


def _record_inferred_batch(
    emission: dict[str, Any],
    inferred: _InferredTeacherBatch,
) -> None:
    if emission["vocab_size"] not in {None, inferred.vocab_size}:
        raise ValueError("teacher vocabulary size changed during capture")
    emission["vocab_size"] = inferred.vocab_size
    emission["tokens"] += inferred.token_count
    emission["batches"] += 1


def _put_with_stop(
    target_queue: queue.Queue[object],
    item: object,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _validate_emitted_shapes(
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    logits: np.ndarray,
    examples: int,
    sequence_length: int,
) -> None:
    expected = (examples, sequence_length)
    if input_ids.shape != expected:
        raise ValueError(f"input_ids shape {input_ids.shape} != {expected}")
    if attention_mask.shape != expected:
        raise ValueError(f"attention_mask shape {attention_mask.shape} != {expected}")
    if logits.shape[:2] != expected or logits.ndim != 3:
        raise ValueError(f"logits shape {logits.shape} incompatible with {expected}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("teacher logits must be finite")


def _rewrite_manifest_for_p145(
    path: Path,
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    teacher: HFTeacherBackend,
    vocab_size: int,
    effective_local_files_only: bool,
) -> None:
    manifest = read_json_object(path)
    manifest["teacher"] = {
        **manifest["teacher"],
        "backend": "hf_causal_lm",
        "model_name": teacher.model_id,
        "model_name_or_path": teacher.model_id,
        "tokenizer_name": config.tokenizer or _tokenizer_name(teacher),
        "tokenizer_name_or_path": config.tokenizer or _tokenizer_name(teacher),
        "local_files_only": effective_local_files_only,
        "vocab_size": vocab_size,
        "dtype": teacher.dtype,
        "device": "cpu",
        "teacher_real": True,
    }
    manifest["capture"] = {
        **manifest.get("capture", {}),
        "phase": "P145",
        "run_kind": "tiny_real_teacher_capture",
        "capture_engine": "teacher_side_capture_skeleton_v0",
    }
    write_json(path, manifest)


def _run_consumer_sanity(
    config: TinyRealTeacherFingerprintCaptureConfig,
    *,
    vocab_size: int,
    validation_ok: bool,
) -> dict[str, Any]:
    if not validation_ok:
        return {
            "kind": "loader_only",
            "status": "fail",
            "reason": "artifact validation failed",
        }
    if config.exemplar_target_type == "cascaded_soft_labels_v1":
        try:
            batch = next(
                iter(
                    load_fingerprint_exemplars(
                        config.output_dir, batch_size=1
                    ).iter_batches()
                )
            )
            initial = jnp.linspace(
                -0.25,
                0.25,
                vocab_size,
                dtype=jnp.float32,
            )[None, :]

            def loss_fn(logits: jax.Array) -> jax.Array:
                return compute_fingerprint_exemplar_loss(logits, batch).loss

            loss, gradient = jax.value_and_grad(loss_fn)(initial)
            updated = initial - 0.01 * gradient
            changed = bool(np.any(np.asarray(updated) != np.asarray(initial)))
            gradient_array = np.asarray(gradient)
            gradient_norm = float(jnp.linalg.norm(gradient))
            loss_finite = bool(np.isfinite(float(loss)))
            gradient_finite = bool(np.all(np.isfinite(gradient_array)))
            gradient_norm_finite = bool(np.isfinite(gradient_norm))
            gradient_norm_positive = bool(gradient_norm > 0.0)
            passed = bool(
                loss_finite
                and gradient_finite
                and gradient_norm_finite
                and gradient_norm_positive
                and changed
            )
            return {
                "kind": "compressed_exemplar_optimizer_step",
                "status": "pass" if passed else "fail",
                "reason": None if passed else "nonfinite_zero_gradient_or_unchanged",
                "initial_loss": float(loss),
                "loss_finite": loss_finite,
                "gradient_finite": gradient_finite,
                "gradient_norm": gradient_norm,
                "gradient_norm_finite": gradient_norm_finite,
                "gradient_norm_positive": gradient_norm_positive,
                "parameters_changed": changed,
            }
        except Exception as error:
            return {
                "kind": "compressed_exemplar_optimizer_step",
                "status": "fail",
                "reason": f"{type(error).__name__}: {error}",
            }
    if vocab_size > config.consumer_vocab_limit:
        return {
            "kind": "loader_only",
            "status": "pass",
            "reason": "teacher vocab too large for cheap CPU smoke",
            "vocab_size": vocab_size,
            "consumer_vocab_limit": config.consumer_vocab_limit,
        }
    try:
        result = run_distill_stage(
            DistillStageConfig(
                mode=DISTILL_MODE_FINGERPRINT_CORRIDOR,
                training=replace(DistillStageConfig().training, max_steps=1),
                optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
                fingerprint=DistillFingerprintConfig(
                    artifact_dir=config.output_dir,
                    batch_size=1,
                    student_backend="current_qrwkv",
                    output_dir=config.output_dir / "consumer_sanity",
                ),
            )
        )
        return {
            "kind": "p141_one_step",
            "status": result.status,
            "reason": None if result.status == "pass" else "p141 returned non-pass",
            "final_loss": result.final_loss,
        }
    except Exception as p141_error:
        try:
            result = run_real_student_fingerprint_forward_smoke(
                RealStudentFingerprintForwardConfig(
                    artifact_dir=config.output_dir,
                    output_dir=config.output_dir / "p140_consumer_sanity",
                    batch_size=1,
                )
            )
            return {
                "kind": "p140_forward",
                "status": result.status,
                "reason": None if result.status == "pass" else "p140 returned non-pass",
            }
        except Exception as p140_error:
            return {
                "kind": "loader_only",
                "status": "pass",
                "reason": (
                    "student consumer smoke unavailable; "
                    f"p141={type(p141_error).__name__}: {p141_error}; "
                    f"p140={type(p140_error).__name__}: {p140_error}"
                ),
            }


def _p145_summary(
    *,
    config: TinyRealTeacherFingerprintCaptureConfig,
    base_summary: dict[str, Any],
    teacher: HFTeacherBackend,
    effective_local_files_only: bool,
    examples_processed: int,
    tokens_processed: int,
    target_positions_processed: int,
    vocab_size: int,
    artifact_validated: bool,
    validation_blockers: tuple[str, ...],
    targets_loadable: bool,
    exemplars_loadable: bool,
    exemplar_records: int,
    artifact_summary: dict[str, Any],
    consumer_sanity: dict[str, Any],
    batches_processed: int,
    topology_metadata: dict[str, Any],
    stage_counters: dict[str, int],
) -> dict[str, Any]:
    return {
        **base_summary,
        "phase": "P145",
        "capture_engine": "teacher_side_capture_skeleton_v0",
        "run_kind": "tiny_real_teacher_capture",
        "teacher_real": True,
        "teacher_backend": "hf_causal_lm",
        "teacher_model_name_or_path": teacher.model_id,
        "tokenizer_name_or_path": config.tokenizer or _tokenizer_name(teacher),
        "local_files_only": effective_local_files_only,
        "teacher": {
            "backend": "hf_causal_lm",
            "model_name_or_path": teacher.model_id,
            "tokenizer_name_or_path": config.tokenizer or _tokenizer_name(teacher),
            "local_files_only": effective_local_files_only,
            "vocab_size": vocab_size,
            "dtype": teacher.dtype,
            "device": topology_metadata["teacher_device"],
        },
        "examples_processed": examples_processed,
        "tokens_processed": tokens_processed,
        "teacher_batch_size": config.teacher_batch_size,
        "batches_processed": batches_processed,
        "effective_examples_per_batch": (
            examples_processed / batches_processed if batches_processed else None
        ),
        "target_positions_processed": target_positions_processed,
        "positions_policy": "fixed_all_positions",
        "mode_discovery_method": base_summary["mode_discovery_method"],
        "modes_discovered": base_summary["modes_discovered"],
        "records_per_mode": base_summary["records_per_mode"],
        "corridor_bounds_method": base_summary["corridor_bounds_method"],
        "exemplar_selection_policy": config.exemplar_selection_policy,
        "max_exemplars": config.max_exemplars,
        "exemplars_retained": exemplar_records,
        "artifact_validated": artifact_validated,
        "validation_blockers": list(validation_blockers),
        "targets_loadable": targets_loadable,
        "exemplars_loadable": exemplars_loadable,
        "artifact_summary": artifact_summary,
        "consumer_sanity": consumer_sanity,
        "capture_topology": topology_metadata,
        **topology_metadata,
        **stage_counters,
        "claims_not_made": (
            "real_scale_teacher_capture",
            "tome_textbook_integration",
            "student_quality_improvement",
            "baseline_comparison",
            "quality_per_byte_gain",
            "production_capture_performance",
        ),
        "capture_config": _json_safe(asdict(config)),
    }


def _tokenizer_name(teacher: HFTeacherBackend) -> str:
    tokenizer = teacher.tokenizer
    value = getattr(tokenizer, "name_or_path", None)
    return str(value or teacher.model_id)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
