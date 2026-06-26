from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from qrwkv_xla.artifacts.cascaded_soft_labels import validate_cascaded_bucket_edges
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)

HF_CREATED_AT: Final = "2026-05-29T00:00:00Z"


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
    ) -> HFCompactTeacherTargets:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
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
    non_blocking_transfer: bool,
) -> HFCompactTeacherTargets:
    import torch

    logits_dtype = str(logits.dtype)
    batch_size, sequence_length, vocab_size = _logits_shape(logits)
    effective_top_k = min(int(top_k), vocab_size)
    estimated_bytes = (
        batch_size * sequence_length * vocab_size * _dtype_itemsize(logits_dtype)
    )
    try:
        compute_logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(compute_logits, dim=-1)
        probs = torch.exp(log_probs)
        stat_k = min(max(32, effective_top_k, 2), vocab_size)
        stat_probs, _ = torch.topk(probs, k=stat_k, dim=-1, largest=True, sorted=True)
        top1 = stat_probs[..., 0]
        top2 = stat_probs[..., 1] if vocab_size >= 2 else torch.zeros_like(top1)
        top8_mass = torch.sum(stat_probs[..., : min(8, stat_k)], dim=-1)
        top32_mass = torch.sum(stat_probs[..., : min(32, stat_k)], dim=-1)
        top_log_probs, top_token_ids = torch.topk(
            log_probs,
            k=effective_top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        top_probs = torch.exp(top_log_probs)
        top_mass = torch.sum(top_probs, dim=-1)
        top_mask = torch.zeros_like(probs, dtype=torch.bool)
        top_mask.scatter_(-1, top_token_ids, True)
        bucket_mass_rows = []
        bucket_count_rows = []
        bucket_mean_rows = []
        for upper, lower in zip(bucket_edges[:-1], bucket_edges[1:], strict=True):
            mask = (~top_mask) & (probs < float(upper)) & (probs >= float(lower))
            bucket_mass = torch.sum(
                torch.where(mask, probs, torch.zeros_like(probs)),
                dim=-1,
            )
            bucket_count = torch.sum(mask.to(torch.int32), dim=-1)
            bucket_logp_sum = torch.sum(
                torch.where(mask, log_probs, torch.zeros_like(log_probs)), dim=-1
            )
            bucket_mean = torch.where(
                bucket_count > 0,
                bucket_logp_sum / bucket_count.to(bucket_logp_sum.dtype),
                torch.zeros_like(bucket_logp_sum),
            )
            bucket_mass_rows.append(bucket_mass)
            bucket_count_rows.append(bucket_count)
            bucket_mean_rows.append(bucket_mean)
        bucket_masses = torch.stack(bucket_mass_rows, dim=-1)
        bucket_counts = torch.stack(bucket_count_rows, dim=-1)
        bucket_mean_log_probs = torch.stack(bucket_mean_rows, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        tail_mass = torch.clamp(1.0 - top32_mass, min=0.0)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "compact GPU reduction ran out of memory: "
                f"batch_size={batch_size} sequence_length={sequence_length} "
                f"vocab_size={vocab_size} "
                f"estimated_raw_logits_bytes={estimated_bytes}; "
                "try a smaller teacher_batch_size"
            ) from exc
        raise

    input_ids_np = _tensor_to_numpy(input_ids, non_blocking=non_blocking_transfer)
    attention_mask_np = (
        np.ones_like(input_ids_np, dtype=np.int32)
        if attention_mask is None
        else _tensor_to_numpy(attention_mask, non_blocking=non_blocking_transfer)
    )
    gpu_allocated, gpu_reserved = _torch_gpu_memory(logits)
    return HFCompactTeacherTargets(
        input_ids=input_ids_np.astype(np.int32, copy=False),
        attention_mask=attention_mask_np.astype(np.int32, copy=False),
        entropy=_tensor_to_numpy(entropy, non_blocking=non_blocking_transfer).astype(
            np.float32,
            copy=False,
        ),
        max_log_prob=_tensor_to_numpy(
            top_log_probs[..., 0], non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top1_margin=_tensor_to_numpy(
            top1 - top2, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top8_mass=_tensor_to_numpy(
            top8_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top32_mass=_tensor_to_numpy(
            top32_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        tail_mass=_tensor_to_numpy(
            tail_mass, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top_token_ids=_tensor_to_numpy(
            top_token_ids, non_blocking=non_blocking_transfer
        ).astype(np.int32, copy=False),
        top_log_probs=_tensor_to_numpy(
            top_log_probs, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        top_mass=_tensor_to_numpy(top_mass, non_blocking=non_blocking_transfer).astype(
            np.float32,
            copy=False,
        ),
        bucket_masses=_tensor_to_numpy(
            bucket_masses, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        bucket_counts=_tensor_to_numpy(
            bucket_counts, non_blocking=non_blocking_transfer
        ).astype(np.int32, copy=False),
        bucket_mean_log_probs=_tensor_to_numpy(
            bucket_mean_log_probs, non_blocking=non_blocking_transfer
        ).astype(np.float32, copy=False),
        original_vocab_size=vocab_size,
        logits_dtype=logits_dtype,
        estimated_raw_logits_bytes=estimated_bytes,
        gpu_memory_allocated_bytes=gpu_allocated,
        gpu_memory_reserved_bytes=gpu_reserved,
    )


def _compact_numpy_targets(
    logits: np.ndarray,
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray | None,
    top_k: int,
    bucket_edges: tuple[float, ...],
) -> HFCompactTeacherTargets:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"logits must be rank 3 [B,T,V], got {values.shape}")
    batch_size, sequence_length, vocab_size = values.shape
    effective_top_k = min(int(top_k), vocab_size)
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
