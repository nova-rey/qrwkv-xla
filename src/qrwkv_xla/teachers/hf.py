from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)

HF_CREATED_AT: Final = "2026-05-29T00:00:00Z"


class HFTeacherUnavailable(RuntimeError):
    """Raised when optional HF dependencies or local model files are unavailable."""


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
