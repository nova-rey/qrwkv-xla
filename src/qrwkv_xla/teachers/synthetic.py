from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)

SYNTHETIC_CREATED_AT: Final = "2026-05-29T00:00:00Z"


@dataclass(frozen=True)
class SyntheticTeacherBackend:
    """Deterministic tiny teacher backend for P94 emission smoke."""

    name: str = "synthetic"
    model_id: str = "synthetic-teacher-v0"
    model_family: str = "synthetic"
    tokenizer_id: str = "synthetic-tokenizer-v0"
    tokenizer_hash: str = "synthetic"
    vocab_size: int = 8
    dtype: str = "float32"

    def build_metadata(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> TargetStoreMetadata:
        _validate_shape(num_examples=num_examples, sequence_length=sequence_length)
        return TargetStoreMetadata(
            schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
            target_store_version=TEACHER_TARGET_STORE_VERSION,
            model_id=self.model_id,
            model_family=self.model_family,
            tokenizer_id=self.tokenizer_id,
            tokenizer_hash=self.tokenizer_hash,
            vocab_size=self.vocab_size,
            target_type="synthetic",
            dtype=self.dtype,
            sequence_length=sequence_length,
            num_examples=num_examples,
            shard_count=1,
            created_by="SyntheticTeacherBackend",
            created_at=SYNTHETIC_CREATED_AT,
            source={"kind": "synthetic"},
            provenance={"phase": "P94", "backend": self.name},
        )

    def emit_targets(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> dict[str, np.ndarray]:
        _validate_shape(num_examples=num_examples, sequence_length=sequence_length)
        n = np.arange(num_examples, dtype=np.int32)[:, None]
        t = np.arange(sequence_length, dtype=np.int32)[None, :]
        v = np.arange(self.vocab_size, dtype=np.float32)[None, None, :]
        input_ids = ((n + t) % self.vocab_size).astype(np.int32)
        attention_mask = np.ones((num_examples, sequence_length), dtype=np.int32)
        logits = (
            n.astype(np.float32)[:, :, None] * 0.25
            + t.astype(np.float32)[:, :, None] * 0.5
            + v * 0.125
        ).astype(np.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logits": logits,
        }


def _validate_shape(*, num_examples: int, sequence_length: int) -> None:
    if num_examples <= 0:
        raise ValueError(f"num_examples must be > 0, got {num_examples}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
