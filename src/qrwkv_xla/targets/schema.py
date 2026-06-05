from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TEACHER_TARGET_STORE_SCHEMA_VERSION = "qrwkv_xla.teacher_target_store.v1"
TEACHER_TARGET_STORE_VERSION = "p93"
SUPPORTED_TARGET_TYPES = {
    "dense_logits",
    "full_logits",
    "full_logprobs",
    "top_k_logprobs",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
    "hidden_states",
    "attention_derived",
    "synthetic",
}
P93_ARRAY_TARGET_TYPES = {
    "dense_logits",
    "full_logits",
    "synthetic",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
}
FLOAT_DTYPES = {"float32", "fp32", "bfloat16", "bf16", "float16", "fp16"}


@dataclass(frozen=True)
class TargetStoreMetadata:
    schema_version: str
    target_store_version: str
    model_id: str
    model_family: str | None
    tokenizer_id: str
    tokenizer_hash: str | None
    vocab_size: int
    target_type: str
    dtype: str
    sequence_length: int
    num_examples: int
    shard_count: int
    created_by: str
    created_at: str
    source: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    target_params: dict[str, str] = field(default_factory=dict)


def target_store_metadata_to_dict(metadata: TargetStoreMetadata) -> dict[str, Any]:
    return asdict(metadata)


def target_store_metadata_from_dict(payload: dict[str, Any]) -> TargetStoreMetadata:
    if not isinstance(payload, dict):
        raise ValueError("target store metadata must be a JSON object")
    metadata = TargetStoreMetadata(
        schema_version=str(payload.get("schema_version", "")),
        target_store_version=str(payload.get("target_store_version", "")),
        model_id=str(payload.get("model_id", "")),
        model_family=_optional_str(payload.get("model_family")),
        tokenizer_id=str(payload.get("tokenizer_id", "")),
        tokenizer_hash=_optional_str(payload.get("tokenizer_hash")),
        vocab_size=int(payload.get("vocab_size", 0)),
        target_type=str(payload.get("target_type", "")),
        dtype=str(payload.get("dtype", "")),
        sequence_length=int(payload.get("sequence_length", 0)),
        num_examples=int(payload.get("num_examples", 0)),
        shard_count=int(payload.get("shard_count", 0)),
        created_by=str(payload.get("created_by", "")),
        created_at=str(payload.get("created_at", "")),
        source=_string_mapping(payload.get("source", {}), "source"),
        provenance=_string_mapping(payload.get("provenance", {}), "provenance"),
        target_params=_string_mapping(
            payload.get("target_params", {}),
            "target_params",
        ),
    )
    validate_target_store_metadata(metadata)
    return metadata


def validate_target_store_metadata(metadata: TargetStoreMetadata) -> None:
    if metadata.schema_version != TEACHER_TARGET_STORE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported teacher target store schema_version: "
            f"{metadata.schema_version!r}"
        )
    if metadata.target_store_version != TEACHER_TARGET_STORE_VERSION:
        raise ValueError(
            "unsupported teacher target_store_version: "
            f"{metadata.target_store_version!r}"
        )
    if not metadata.model_id.strip():
        raise ValueError("model_id must be non-empty")
    if not metadata.tokenizer_id.strip():
        raise ValueError("tokenizer_id must be non-empty")
    if metadata.vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {metadata.vocab_size}")
    if metadata.target_type not in SUPPORTED_TARGET_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_TARGET_TYPES))
        raise ValueError(
            f"unsupported target_type {metadata.target_type!r}; "
            f"expected one of {{{allowed}}}"
        )
    if metadata.dtype not in FLOAT_DTYPES:
        allowed = ", ".join(sorted(FLOAT_DTYPES))
        raise ValueError(f"dtype must be one of {{{allowed}}}, got {metadata.dtype!r}")
    if metadata.sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {metadata.sequence_length}")
    if metadata.num_examples <= 0:
        raise ValueError(f"num_examples must be > 0, got {metadata.num_examples}")
    if metadata.shard_count <= 0:
        raise ValueError(f"shard_count must be > 0, got {metadata.shard_count}")
    if not metadata.created_by.strip():
        raise ValueError("created_by must be non-empty")
    if not metadata.created_at.strip():
        raise ValueError("created_at must be non-empty")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}
