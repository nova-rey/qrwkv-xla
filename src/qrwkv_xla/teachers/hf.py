from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from qrwkv_xla.artifacts.cascaded_soft_labels import validate_cascaded_bucket_edges
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)

HF_CREATED_AT: Final = "2026-05-29T00:00:00Z"
GpuVocabChunkSize = int | Literal["auto"]
GPU_VOCAB_CHUNK_AUTO_CANDIDATES: Final = (32768, 16384, 8192, 4096, 2048, 1024)
GPU_REDUCTION_SAFETY_RESERVE_BYTES: Final = 512 * 1024 * 1024
GPU_REDUCTION_WORKSPACE_TENSORS: Final = 5


class HFTeacherUnavailable(RuntimeError):
    """Raised when optional HF dependencies or local model files are unavailable."""


@dataclass(frozen=True)
class HFCompactTeacherTargets:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    entropy: np.ndarray
    max_log_prob: np.ndarray
    top1_margin: np.ndarray
    top8_mass: np.ndarray
    top32_mass: np.ndarray
    tail_mass: np.ndarray
    top_token_ids: np.ndarray
    top_log_probs: np.ndarray
    top_mass: np.ndarray
    bucket_masses: np.ndarray
    bucket_counts: np.ndarray
    bucket_mean_log_probs: np.ndarray
    original_vocab_size: int
    logits_dtype: str
    estimated_raw_logits_bytes: int
    gpu_memory_allocated_bytes: int | None
    gpu_memory_reserved_bytes: int | None
    gpu_vocab_chunk_size_requested: GpuVocabChunkSize
    gpu_vocab_chunk_size_effective: int
    gpu_vocab_chunks_per_batch: int
    estimated_reduction_workspace_bytes: int
    peak_gpu_memory_allocated_bytes: int | None
    peak_gpu_memory_reserved_bytes: int | None
    gpu_vocab_chunk_auto_policy: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "entropy": self.entropy,
            "max_log_prob": self.max_log_prob,
            "top1_margin": self.top1_margin,
            "top8_mass": self.top8_mass,
            "top32_mass": self.top32_mass,
            "tail_mass": self.tail_mass,
            "top_token_ids": self.top_token_ids,
            "top_log_probs": self.top_log_probs,
            "top_mass": self.top_mass,
            "bucket_masses": self.bucket_masses,
            "bucket_counts": self.bucket_counts,
            "bucket_mean_log_probs": self.bucket_mean_log_probs,
            "original_vocab_size": self.original_vocab_size,
            "logits_dtype": self.logits_dtype,
            "estimated_raw_logits_bytes": self.estimated_raw_logits_bytes,
            "gpu_memory_allocated_bytes": self.gpu_memory_allocated_bytes,
            "gpu_memory_reserved_bytes": self.gpu_memory_reserved_bytes,
            "gpu_vocab_chunk_size_requested": self.gpu_vocab_chunk_size_requested,
            "gpu_vocab_chunk_size_effective": self.gpu_vocab_chunk_size_effective,
            "gpu_vocab_chunks_per_batch": self.gpu_vocab_chunks_per_batch,
            "estimated_reduction_workspace_bytes": (
                self.estimated_reduction_workspace_bytes
            ),
            "peak_gpu_memory_allocated_bytes": self.peak_gpu_memory_allocated_bytes,
            "peak_gpu_memory_reserved_bytes": self.peak_gpu_memory_reserved_bytes,
            "gpu_vocab_chunk_auto_policy": self.gpu_vocab_chunk_auto_policy,
        }


@dataclass
class HFTeacherBackend:
    model_id: str
    revision: str | None = None
    local_files_only: bool = True
    dtype: str = "float32"
    prompts: Sequence[str] = ("hello", "goodbye")
    tokenizer: Any | None = None
    model: Any | None = None
    name: str = "hf"
    model_family: str = "hf"
    device: str = "cpu"
    pin_memory: bool = False
    non_blocking_transfer: bool = False
    _loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.prompts:
            raise ValueError("prompts must contain at least one prompt")

    def load(self) -> None:
        if self._loaded:
            return
        if self.tokenizer is None or self.model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise HFTeacherUnavailable(
                    "transformers is not installed; install optional HF teacher "
                    "dependencies to use HFTeacherBackend"
                ) from exc

            kwargs = {
                "revision": self.revision,
                "local_files_only": self.local_files_only,
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            try:
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id,
                        **kwargs,
                    )
                if self.model is None:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        **kwargs,
                    )
            except Exception as exc:  # pragma: no cover - exercised via fakes.
                local_note = (
                    " with local_files_only=True" if self.local_files_only else ""
                )
                raise HFTeacherUnavailable(
                    f"HF model_id {self.model_id!r} is unavailable{local_note}: {exc}"
                ) from exc
        _ensure_pad_token(self.tokenizer)
        if self.device != "cpu" and hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True

    def vocab_contract(self) -> VocabContract:
        self.load()
        vocab_size = _tokenizer_vocab_size(self.tokenizer)
        return VocabContract(
            tokenizer_id=_tokenizer_id(self.tokenizer, fallback=self.model_id),
            tokenizer_hash=None,
            vocab_size=vocab_size,
            model_id=self.model_id,
            model_family=self.model_family,
            special_tokens=_special_tokens(self.tokenizer, vocab_size=vocab_size),
        )

    def build_metadata(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> TargetStoreMetadata:
        _validate_shape(num_examples=num_examples, sequence_length=sequence_length)
        contract = self.vocab_contract()
        return TargetStoreMetadata(
            schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
            target_store_version=TEACHER_TARGET_STORE_VERSION,
            model_id=self.model_id,
            model_family=self.model_family,
            tokenizer_id=contract.tokenizer_id,
            tokenizer_hash=contract.tokenizer_hash,
            vocab_size=contract.vocab_size,
            target_type="full_logits",
            dtype=self.dtype,
            sequence_length=sequence_length,
            num_examples=num_examples,
            shard_count=1,
            created_by="HFTeacherBackend",
            created_at=HF_CREATED_AT,
            source={"kind": "hf", "local_files_only": str(self.local_files_only)},
            provenance={"phase": "P98", "backend": self.name},
        )

    def emit_targets(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> dict[str, np.ndarray]:
        _validate_shape(num_examples=num_examples, sequence_length=sequence_length)
        prompts = _select_prompts(self.prompts, num_examples=num_examples)
        encoded = self.encode_prompts(prompts, sequence_length=sequence_length)
        return self.emit_targets_from_encoded(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    def encode_prompts(
        self,
        prompts: Sequence[str],
        *,
        sequence_length: int,
    ) -> dict[str, Any]:
        _validate_shape(num_examples=len(prompts), sequence_length=sequence_length)
        self.load()
        encoded = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt",
        )
        if self.device != "cpu":
            encoded = {
                key: _move_to_device(
                    value,
                    device=self.device,
                    non_blocking=self.non_blocking_transfer,
                )
                for key, value in encoded.items()
            }
        input_ids = _to_numpy(encoded["input_ids"]).astype(np.int32, copy=False)
        attention_mask = _to_numpy(
            encoded.get("attention_mask", np.ones_like(input_ids))
        ).astype(np.int32, copy=False)
        _validate_encoded_shapes(
            input_ids=input_ids,
            attention_mask=attention_mask,
            examples=len(prompts),
            sequence_length=sequence_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
            "input_ids_np": input_ids,
            "attention_mask_np": attention_mask,
        }

    def emit_targets_from_encoded(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
    ) -> dict[str, np.ndarray]:
        self.load()
        input_ids_np = _to_numpy(input_ids).astype(np.int32, copy=False)
        attention_mask_np = _to_numpy(
            attention_mask if attention_mask is not None else np.ones_like(input_ids_np)
        ).astype(np.int32, copy=False)
        outputs = _model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = _to_numpy(outputs.logits).astype(np.float32, copy=False)
        if attention_mask_np.shape != input_ids_np.shape:
            raise ValueError("HF encoded attention_mask shape must match input_ids")
        if logits.ndim != 3 or logits.shape[:2] != input_ids_np.shape:
            raise ValueError("HF logits [N,T] must match encoded input_ids")
        return {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np,
            "logits": logits,
        }

    def emit_compact_targets_from_encoded(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        top_k: int,
        bucket_edges: tuple[float, ...],
        gpu_vocab_chunk_size: GpuVocabChunkSize = "auto",
    ) -> HFCompactTeacherTargets:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        _validate_gpu_vocab_chunk_size(gpu_vocab_chunk_size)
        edges = validate_cascaded_bucket_edges(bucket_edges)
        self.load()
        outputs = _model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and hasattr(logits, "detach"):
            return _compact_torch_targets(
                logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_k=top_k,
                bucket_edges=edges,
                gpu_vocab_chunk_size=gpu_vocab_chunk_size,
                non_blocking_transfer=self.non_blocking_transfer,
            )
        return _compact_numpy_targets(
            np.asarray(logits, dtype=np.float32),
            input_ids=np.asarray(input_ids, dtype=np.int32),
            attention_mask=(
                None
                if attention_mask is None
                else np.asarray(attention_mask, dtype=np.int32)
            ),
            top_k=top_k,
            bucket_edges=edges,
            gpu_vocab_chunk_size=gpu_vocab_chunk_size,
        )


def _model_forward(model: Any, *, input_ids: Any, attention_mask: Any | None) -> Any:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "output_hidden_states": False,
        "output_attentions": False,
        "return_dict": True,
    }
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        try:
            return model(**kwargs)
        except TypeError:
            return model(input_ids=input_ids, attention_mask=attention_mask)
    inference_mode = getattr(torch, "inference_mode", torch.no_grad)
    with inference_mode():
        try:
            return model(**kwargs)
        except TypeError:
            return model(input_ids=input_ids, attention_mask=attention_mask)


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None and hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = eos_token


def _tokenizer_id(tokenizer: Any, *, fallback: str) -> str:
    value = getattr(tokenizer, "name_or_path", None) or fallback
    return str(value)


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    try:
        return int(len(tokenizer))
    except TypeError as exc:
        raise ValueError("HF tokenizer does not expose vocab_size or __len__") from exc


def _special_tokens(tokenizer: Any, *, vocab_size: int) -> dict[str, int]:
    tokens: dict[str, int] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and 0 <= value < vocab_size:
            tokens[name.removesuffix("_token_id")] = value
    return tokens


def _select_prompts(prompts: Sequence[str], *, num_examples: int) -> list[str]:
    return [str(prompts[index % len(prompts)]) for index in range(num_examples)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _move_to_device(value: Any, *, device: str, non_blocking: bool) -> Any:
    if not hasattr(value, "to"):
        return value
    try:
        return value.to(device, non_blocking=non_blocking)
    except TypeError:
        return value.to(device)


def _compact_torch_targets(
    logits: Any,
    *,
    input_ids: Any,
    attention_mask: Any | None,
    top_k: int,
    bucket_edges: tuple[float, ...],
    gpu_vocab_chunk_size: GpuVocabChunkSize,
    non_blocking_transfer: bool,
) -> HFCompactTeacherTargets:
    logits_dtype = str(logits.dtype)
    batch_size, sequence_length, vocab_size = _logits_shape(logits)
    effective_top_k = min(int(top_k), vocab_size)
    chunk_plan = _resolve_torch_vocab_chunk_plan(
        logits,
        requested=gpu_vocab_chunk_size,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        logits_dtype=logits_dtype,
    )
    estimated_bytes = (
        batch_size * sequence_length * vocab_size * _dtype_itemsize(logits_dtype)
    )
    _reset_torch_peak_memory_stats(logits)
    try:
        reduced = _chunked_torch_compact_reduce(
            logits,
            top_k=effective_top_k,
            bucket_edges=bucket_edges,
            chunk_size=chunk_plan.effective_chunk_size,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            peak_allocated, peak_reserved = _torch_peak_gpu_memory(logits)
            raise RuntimeError(
                "compact GPU reduction ran out of memory: "
                f"teacher_batch_size={batch_size} batch_size={batch_size} "
                f"sequence_length={sequence_length} "
                f"vocab_size={vocab_size} "
                f"gpu_vocab_chunk_size_requested={gpu_vocab_chunk_size} "
                "gpu_vocab_chunk_size_effective="
                f"{chunk_plan.effective_chunk_size} "
                f"estimated_raw_logits_bytes={estimated_bytes} "
                "estimated_reduction_workspace_bytes="
                f"{chunk_plan.estimated_workspace_bytes} "
                f"peak_gpu_memory_allocated_bytes={peak_allocated} "
                f"peak_gpu_memory_reserved_bytes={peak_reserved}; "
                "retry with a smaller --gpu-vocab-chunk-size or reduce "
                "--teacher-batch-size"
            ) from exc
        raise

    input_ids_np = _tensor_to_numpy(input_ids, non_blocking=non_blocking_transfer)
    attention_mask_np = (
        np.ones_like(input_ids_np, dtype=np.int32)
        if attention_mask is None
        else _tensor_to_numpy(attention_mask, non_blocking=non_blocking_transfer)
    )
    gpu_allocated, gpu_reserved = _torch_gpu_memory(logits)
    peak_allocated, peak_reserved = _torch_peak_gpu_memory(logits)
    return HFCompactTeacherTargets(
        input_ids=input_ids_np.astype(np.int32, copy=False),
        attention_mask=attention_mask_np.astype(np.int32, copy=False),
        entropy=_tensor_to_numpy(
            reduced.entropy, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        max_log_prob=_tensor_to_numpy(
            reduced.top_log_probs[..., 0], non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top1_margin=_tensor_to_numpy(
            reduced.top1_margin, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top8_mass=_tensor_to_numpy(
            reduced.top8_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top32_mass=_tensor_to_numpy(
            reduced.top32_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        tail_mass=_tensor_to_numpy(
            reduced.tail_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top_token_ids=_tensor_to_numpy(
            reduced.top_token_ids, non_blocking=non_blocking_transfer
        ).astype(np.int32, copy=False),
        top_log_probs=_tensor_to_numpy(
            reduced.top_log_probs, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top_mass=_tensor_to_numpy(
            reduced.top_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        bucket_masses=_tensor_to_numpy(
            reduced.bucket_masses, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        bucket_counts=_tensor_to_numpy(
            reduced.bucket_counts, non_blocking=non_blocking_transfer
        ).astype(np.int32, copy=False),
        bucket_mean_log_probs=_tensor_to_numpy(
            reduced.bucket_mean_log_probs, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        original_vocab_size=vocab_size,
        logits_dtype=logits_dtype,
        estimated_raw_logits_bytes=estimated_bytes,
        gpu_memory_allocated_bytes=gpu_allocated,
        gpu_memory_reserved_bytes=gpu_reserved,
        gpu_vocab_chunk_size_requested=gpu_vocab_chunk_size,
        gpu_vocab_chunk_size_effective=chunk_plan.effective_chunk_size,
        gpu_vocab_chunks_per_batch=chunk_plan.chunks_per_batch,
        estimated_reduction_workspace_bytes=chunk_plan.estimated_workspace_bytes,
        peak_gpu_memory_allocated_bytes=peak_allocated,
        peak_gpu_memory_reserved_bytes=peak_reserved,
        gpu_vocab_chunk_auto_policy=chunk_plan.auto_policy,
    )


@dataclass(frozen=True)
class _TorchVocabChunkPlan:
    requested_chunk_size: GpuVocabChunkSize
    effective_chunk_size: int
    chunks_per_batch: int
    estimated_workspace_bytes: int
    auto_policy: dict[str, Any] | None


@dataclass(frozen=True)
class _TorchCompactReduction:
    entropy: Any
    max_log_prob: Any
    top1_margin: Any
    top8_mass: Any
    top32_mass: Any
    tail_mass: Any
    top_token_ids: Any
    top_log_probs: Any
    top_mass: Any
    bucket_masses: Any
    bucket_counts: Any
    bucket_mean_log_probs: Any


def _resolve_torch_vocab_chunk_plan(
    logits: Any,
    *,
    requested: GpuVocabChunkSize,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    logits_dtype: str,
) -> _TorchVocabChunkPlan:
    _validate_gpu_vocab_chunk_size(requested)
    if requested != "auto":
        effective = min(int(requested), vocab_size)
        return _TorchVocabChunkPlan(
            requested_chunk_size=requested,
            effective_chunk_size=effective,
            chunks_per_batch=_ceil_div(vocab_size, effective),
            estimated_workspace_bytes=_estimate_reduction_workspace_bytes(
                batch_size=batch_size,
                sequence_length=sequence_length,
                chunk_size=effective,
            ),
            auto_policy=None,
        )

    available_bytes, total_bytes = _torch_available_gpu_memory(logits)
    useful_candidates = [
        min(candidate, vocab_size)
        for candidate in GPU_VOCAB_CHUNK_AUTO_CANDIDATES
        if min(candidate, vocab_size) > 0
    ]
    useful_candidates = list(dict.fromkeys(useful_candidates))
    evaluated = []
    for candidate in useful_candidates:
        workspace_bytes = _estimate_reduction_workspace_bytes(
            batch_size=batch_size,
            sequence_length=sequence_length,
            chunk_size=candidate,
        )
        safe = (
            available_bytes is None
            or workspace_bytes + GPU_REDUCTION_SAFETY_RESERVE_BYTES <= available_bytes
        )
        evaluated.append(
            {
                "candidate": candidate,
                "estimated_reduction_workspace_bytes": workspace_bytes,
                "safe": safe,
            }
        )
        if safe:
            return _TorchVocabChunkPlan(
                requested_chunk_size=requested,
                effective_chunk_size=candidate,
                chunks_per_batch=_ceil_div(vocab_size, candidate),
                estimated_workspace_bytes=workspace_bytes,
                auto_policy={
                    "candidates_descending": list(GPU_VOCAB_CHUNK_AUTO_CANDIDATES),
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "vocab_size": vocab_size,
                    "logits_dtype": logits_dtype,
                    "available_gpu_memory_bytes": available_bytes,
                    "total_gpu_memory_bytes": total_bytes,
                    "safety_reserve_bytes": GPU_REDUCTION_SAFETY_RESERVE_BYTES,
                    "workspace_tensors": GPU_REDUCTION_WORKSPACE_TENSORS,
                    "evaluated": evaluated,
                    "selected": candidate,
                },
            )
    raise RuntimeError(
        "compact GPU reduction could not select a safe vocab chunk size: "
        f"batch_size={batch_size} sequence_length={sequence_length} "
        f"vocab_size={vocab_size} logits_dtype={logits_dtype} "
        f"available_gpu_memory_bytes={available_bytes} "
        f"safety_reserve_bytes={GPU_REDUCTION_SAFETY_RESERVE_BYTES} "
        f"candidates={list(GPU_VOCAB_CHUNK_AUTO_CANDIDATES)}"
    )


def _chunked_torch_compact_reduce(
    logits: Any,
    *,
    top_k: int,
    bucket_edges: tuple[float, ...],
    chunk_size: int,
) -> _TorchCompactReduction:
    import torch

    batch_size, sequence_length, vocab_size = _logits_shape(logits)
    effective_top_k = min(int(top_k), vocab_size)
    stat_k = min(max(32, effective_top_k, 2), vocab_size)
    global_max = torch.full(
        (batch_size, sequence_length),
        -torch.inf,
        dtype=torch.float32,
        device=logits.device,
    )
    global_sum = torch.zeros(
        (batch_size, sequence_length), dtype=torch.float32, device=logits.device
    )
    stat_logits = None
    stat_token_ids = None

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = logits[..., start:end].float()
        chunk_max = torch.amax(chunk, dim=-1)
        chunk_shifted_exp_sum = torch.sum(
            torch.exp(chunk - chunk_max.unsqueeze(-1)),
            dim=-1,
        )
        new_max = torch.maximum(global_max, chunk_max)
        global_sum = (
            torch.exp(global_max - new_max) * global_sum
            + torch.exp(chunk_max - new_max) * chunk_shifted_exp_sum
        )
        global_max = new_max

        local_k = min(stat_k, end - start)
        local_logits, local_offsets = torch.topk(
            chunk,
            k=local_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        local_token_ids = local_offsets + start
        if stat_logits is None:
            stat_logits = local_logits
            stat_token_ids = local_token_ids
        else:
            merged_logits = torch.cat((stat_logits, local_logits), dim=-1)
            merged_token_ids = torch.cat((stat_token_ids, local_token_ids), dim=-1)
            merge_k = min(stat_k, int(merged_logits.shape[-1]))
            stat_logits, merged_offsets = torch.topk(
                merged_logits,
                k=merge_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            stat_token_ids = torch.gather(merged_token_ids, -1, merged_offsets)

    if stat_logits is None or stat_token_ids is None:
        raise ValueError("compact reduction requires a non-empty vocabulary")

    global_logsumexp = global_max + torch.log(global_sum)
    stat_log_probs = stat_logits - global_logsumexp.unsqueeze(-1)
    stat_probs = torch.exp(stat_log_probs)
    top_token_ids = stat_token_ids[..., :effective_top_k]
    top_log_probs = stat_log_probs[..., :effective_top_k]
    top_probs = stat_probs[..., :effective_top_k]
    top_mass = torch.sum(top_probs, dim=-1)
    top1 = stat_probs[..., 0]
    top2 = stat_probs[..., 1] if vocab_size >= 2 else torch.zeros_like(top1)
    top8_mass = torch.sum(stat_probs[..., : min(8, stat_k)], dim=-1)
    top32_mass = torch.sum(stat_probs[..., : min(32, stat_k)], dim=-1)

    bucket_count = len(bucket_edges) - 1
    entropy = torch.zeros(
        (batch_size, sequence_length), dtype=torch.float32, device=logits.device
    )
    bucket_masses = torch.zeros(
        (batch_size, sequence_length, bucket_count),
        dtype=torch.float32,
        device=logits.device,
    )
    bucket_counts = torch.zeros(
        (batch_size, sequence_length, bucket_count),
        dtype=torch.int32,
        device=logits.device,
    )
    bucket_logp_sums = torch.zeros(
        (batch_size, sequence_length, bucket_count),
        dtype=torch.float32,
        device=logits.device,
    )

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = logits[..., start:end].float()
        chunk_log_probs = chunk - global_logsumexp.unsqueeze(-1)
        chunk_probs = torch.exp(chunk_log_probs)
        entropy += -torch.sum(chunk_probs * chunk_log_probs, dim=-1)
        excluded = _torch_chunk_top_token_mask(
            top_token_ids=top_token_ids,
            chunk_width=end - start,
            chunk_start=start,
            device=logits.device,
        )
        for bucket_id, (upper, lower) in enumerate(
            zip(bucket_edges[:-1], bucket_edges[1:], strict=True)
        ):
            mask = (
                (~excluded)
                & (chunk_probs < float(upper))
                & (chunk_probs >= float(lower))
            )
            bucket_masses[..., bucket_id] += torch.sum(
                chunk_probs.masked_fill(~mask, 0.0),
                dim=-1,
            )
            counts = torch.sum(mask.to(torch.int32), dim=-1)
            bucket_counts[..., bucket_id] += counts
            bucket_logp_sums[..., bucket_id] += torch.sum(
                chunk_log_probs.masked_fill(~mask, 0.0),
                dim=-1,
            )

    bucket_mean_log_probs = torch.where(
        bucket_counts > 0,
        bucket_logp_sums / bucket_counts.to(bucket_logp_sums.dtype),
        torch.zeros_like(bucket_logp_sums),
    )
    tail_mass = torch.clamp(1.0 - top32_mass, min=0.0)
    return _TorchCompactReduction(
        entropy=entropy,
        max_log_prob=top_log_probs[..., 0],
        top1_margin=top1 - top2,
        top8_mass=top8_mass,
        top32_mass=top32_mass,
        tail_mass=tail_mass,
        top_token_ids=top_token_ids,
        top_log_probs=top_log_probs,
        top_mass=top_mass,
        bucket_masses=bucket_masses,
        bucket_counts=bucket_counts,
        bucket_mean_log_probs=bucket_mean_log_probs,
    )


def _torch_chunk_top_token_mask(
    *,
    top_token_ids: Any,
    chunk_width: int,
    chunk_start: int,
    device: Any,
) -> Any:
    import torch

    relative_ids = top_token_ids - int(chunk_start)
    in_chunk = (relative_ids >= 0) & (relative_ids < int(chunk_width))
    mask = torch.zeros(
        (*top_token_ids.shape[:2], int(chunk_width)),
        dtype=torch.bool,
        device=device,
    )
    for top_index in range(int(top_token_ids.shape[-1])):
        valid = in_chunk[..., top_index]
        update = torch.zeros_like(mask)
        update.scatter_(
            -1,
            torch.clamp(
                relative_ids[..., top_index],
                min=0,
                max=chunk_width - 1,
            ).unsqueeze(-1),
            valid.unsqueeze(-1),
        )
        mask |= update
    return mask


def _compact_numpy_targets(
    logits: np.ndarray,
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray | None,
    top_k: int,
    bucket_edges: tuple[float, ...],
    gpu_vocab_chunk_size: GpuVocabChunkSize,
) -> HFCompactTeacherTargets:
    _validate_gpu_vocab_chunk_size(gpu_vocab_chunk_size)
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"logits must be rank 3 [B,T,V], got {values.shape}")
    batch_size, sequence_length, vocab_size = values.shape
    effective_top_k = min(int(top_k), vocab_size)
    chunk_size = _resolve_numpy_vocab_chunk_size(
        gpu_vocab_chunk_size,
        vocab_size=vocab_size,
    )
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True, dtype=np.float64)
    )
    probs = np.exp(log_probs).astype(np.float32)
    sorted_ids = np.argsort(log_probs, axis=-1)[..., ::-1]
    top_token_ids = sorted_ids[..., :effective_top_k].astype(np.int32)
    top_log_probs = np.take_along_axis(log_probs, top_token_ids, axis=-1).astype(
        np.float32
    )
    top_probs = np.exp(top_log_probs).astype(np.float32)
    top_mass = np.sum(top_probs, axis=-1, dtype=np.float32)
    stat_probs = np.take_along_axis(
        probs, sorted_ids[..., : min(32, vocab_size)], axis=-1
    )
    top1 = stat_probs[..., 0]
    top2 = stat_probs[..., 1] if vocab_size >= 2 else np.zeros_like(top1)
    top8_mass = np.sum(stat_probs[..., : min(8, vocab_size)], axis=-1, dtype=np.float32)
    top32_mass = np.sum(stat_probs, axis=-1, dtype=np.float32)
    top_mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(top_mask, top_token_ids, True, axis=-1)
    bucket_masses = np.zeros(
        (*values.shape[:2], len(bucket_edges) - 1), dtype=np.float32
    )
    bucket_counts = np.zeros(bucket_masses.shape, dtype=np.int32)
    bucket_mean_log_probs = np.zeros(bucket_masses.shape, dtype=np.float32)
    for bucket_id, (upper, lower) in enumerate(
        zip(bucket_edges[:-1], bucket_edges[1:], strict=True)
    ):
        mask = (~top_mask) & (probs < upper) & (probs >= lower)
        bucket_masses[..., bucket_id] = np.sum(
            np.where(mask, probs, 0.0), axis=-1, dtype=np.float32
        )
        bucket_counts[..., bucket_id] = np.sum(mask, axis=-1, dtype=np.int32)
        logp_sum = np.sum(np.where(mask, log_probs, 0.0), axis=-1, dtype=np.float32)
        np.divide(
            logp_sum,
            bucket_counts[..., bucket_id],
            out=bucket_mean_log_probs[..., bucket_id],
            where=bucket_counts[..., bucket_id] > 0,
        )
    estimated_bytes = int(values.size * values.dtype.itemsize)
    return HFCompactTeacherTargets(
        input_ids=np.asarray(input_ids, dtype=np.int32),
        attention_mask=(
            np.ones(input_ids.shape, dtype=np.int32)
            if attention_mask is None
            else np.asarray(attention_mask, dtype=np.int32)
        ),
        entropy=(-np.sum(probs * log_probs, axis=-1, dtype=np.float32)).astype(
            np.float32
        ),
        max_log_prob=top_log_probs[..., 0].astype(np.float32),
        top1_margin=(top1 - top2).astype(np.float32),
        top8_mass=top8_mass.astype(np.float32),
        top32_mass=top32_mass.astype(np.float32),
        tail_mass=np.maximum(1.0 - top32_mass, 0.0).astype(np.float32),
        top_token_ids=top_token_ids,
        top_log_probs=top_log_probs,
        top_mass=top_mass.astype(np.float32),
        bucket_masses=bucket_masses,
        bucket_counts=bucket_counts,
        bucket_mean_log_probs=bucket_mean_log_probs,
        original_vocab_size=vocab_size,
        logits_dtype=str(values.dtype),
        estimated_raw_logits_bytes=estimated_bytes,
        gpu_memory_allocated_bytes=None,
        gpu_memory_reserved_bytes=None,
        gpu_vocab_chunk_size_requested=gpu_vocab_chunk_size,
        gpu_vocab_chunk_size_effective=chunk_size,
        gpu_vocab_chunks_per_batch=_ceil_div(vocab_size, chunk_size),
        estimated_reduction_workspace_bytes=_estimate_reduction_workspace_bytes(
            batch_size=batch_size,
            sequence_length=sequence_length,
            chunk_size=chunk_size,
        ),
        peak_gpu_memory_allocated_bytes=None,
        peak_gpu_memory_reserved_bytes=None,
        gpu_vocab_chunk_auto_policy=(
            {
                "candidates_descending": list(GPU_VOCAB_CHUNK_AUTO_CANDIDATES),
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "vocab_size": vocab_size,
                "logits_dtype": str(values.dtype),
                "available_gpu_memory_bytes": None,
                "total_gpu_memory_bytes": None,
                "safety_reserve_bytes": GPU_REDUCTION_SAFETY_RESERVE_BYTES,
                "workspace_tensors": GPU_REDUCTION_WORKSPACE_TENSORS,
                "selected": chunk_size,
            }
            if gpu_vocab_chunk_size == "auto"
            else None
        ),
    )


def _logits_shape(logits: Any) -> tuple[int, int, int]:
    shape = tuple(int(dim) for dim in logits.shape)
    if len(shape) != 3:
        raise ValueError(f"logits must be rank 3 [B,T,V], got {shape}")
    return shape


def _dtype_itemsize(dtype_name: str) -> int:
    lowered = dtype_name.lower()
    if "float16" in lowered or "bfloat16" in lowered or "half" in lowered:
        return 2
    if "float64" in lowered or "double" in lowered:
        return 8
    return 4


def _validate_gpu_vocab_chunk_size(value: GpuVocabChunkSize) -> None:
    if value == "auto":
        return
    if int(value) <= 0:
        raise ValueError("gpu_vocab_chunk_size must be > 0 or 'auto'")


def _resolve_numpy_vocab_chunk_size(
    requested: GpuVocabChunkSize,
    *,
    vocab_size: int,
) -> int:
    _validate_gpu_vocab_chunk_size(requested)
    if requested == "auto":
        return min(vocab_size, GPU_VOCAB_CHUNK_AUTO_CANDIDATES[0])
    return min(int(requested), vocab_size)


def _ceil_div(numerator: int, denominator: int) -> int:
    return int((int(numerator) + int(denominator) - 1) // int(denominator))


def _estimate_reduction_workspace_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    chunk_size: int,
) -> int:
    return int(
        batch_size
        * sequence_length
        * chunk_size
        * GPU_REDUCTION_WORKSPACE_TENSORS
        * np.dtype(np.float32).itemsize
    )


def _tensor_to_numpy(value: Any, *, non_blocking: bool) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        device = getattr(value, "device", None)
        if str(device) != "cpu" and hasattr(value, "to"):
            try:
                value = value.to("cpu", non_blocking=non_blocking)
            except TypeError:
                value = value.to("cpu")
        else:
            value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _torch_gpu_memory(logits: Any) -> tuple[int | None, int | None]:
    device = getattr(logits, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return None, None
    try:
        import torch

        return (
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
        )
    except Exception:
        return None, None


def _torch_peak_gpu_memory(logits: Any) -> tuple[int | None, int | None]:
    device = getattr(logits, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return None, None
    try:
        import torch

        return (
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        )
    except Exception:
        return None, None


def _reset_torch_peak_memory_stats(logits: Any) -> None:
    device = getattr(logits, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _torch_available_gpu_memory(logits: Any) -> tuple[int | None, int | None]:
    device = getattr(logits, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return None, None
    try:
        import torch

        free, total = torch.cuda.mem_get_info(device)
        return int(free), int(total)
    except Exception:
        return None, None


def _validate_shape(*, num_examples: int, sequence_length: int) -> None:
    if num_examples <= 0:
        raise ValueError(f"num_examples must be > 0, got {num_examples}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")


def _validate_encoded_shapes(
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    examples: int,
    sequence_length: int,
) -> None:
    expected = (examples, sequence_length)
    if input_ids.shape != expected:
        raise ValueError(
            "HF tokenizer emitted input_ids with shape "
            f"{input_ids.shape}, expected {expected}"
        )
    if attention_mask.shape != expected:
        raise ValueError("HF tokenizer attention_mask shape must match input_ids")
