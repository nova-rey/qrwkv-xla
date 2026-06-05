from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qrwkv_xla.targets.schema import (
    P93_ARRAY_TARGET_TYPES,
    TargetStoreMetadata,
    target_store_metadata_from_dict,
    target_store_metadata_to_dict,
    validate_target_store_metadata,
)


def target_bundle_dir(root: str | Path, bundle_id: str) -> Path:
    if not bundle_id.strip():
        raise ValueError("bundle_id must be non-empty")
    return Path(root) / bundle_id


def manifest_path(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "manifest.json"


def shards_dir(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "shards"


def shard_path(bundle_dir: str | Path, index: int) -> Path:
    if index < 0:
        raise ValueError("shard index must be >= 0")
    return shards_dir(bundle_dir) / f"shard_{index:06d}.npz"


def ensure_bundle_layout(bundle_dir: str | Path) -> None:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    shards_dir(bundle_path).mkdir(parents=True, exist_ok=True)


def list_shard_paths(bundle_dir: str | Path) -> list[Path]:
    return sorted(shards_dir(bundle_dir).glob("shard_*.npz"))


def target_store_metadata_path(store_dir: str | Path) -> Path:
    return Path(store_dir) / "metadata.json"


def target_store_shards_dir(store_dir: str | Path) -> Path:
    return Path(store_dir) / "shards"


def target_store_shard_path(store_dir: str | Path, shard_id: int) -> Path:
    if shard_id < 0:
        raise ValueError("shard_id must be >= 0")
    return target_store_shards_dir(store_dir) / f"shard-{shard_id:05d}.npz"


@dataclass(frozen=True)
class TeacherTargetStore:
    root: Path
    metadata: TargetStoreMetadata

    @classmethod
    def create(
        cls,
        path: str | Path,
        metadata: TargetStoreMetadata,
        *,
        overwrite: bool = False,
    ) -> TeacherTargetStore:
        validate_target_store_metadata(metadata)
        root = Path(path)
        metadata_file = target_store_metadata_path(root)
        if metadata_file.exists() and not overwrite:
            raise ValueError(f"teacher target store already exists: {root}")
        root.mkdir(parents=True, exist_ok=True)
        target_store_shards_dir(root).mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(
            json.dumps(
                target_store_metadata_to_dict(metadata),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return cls(root=root, metadata=metadata)

    @classmethod
    def open(cls, path: str | Path) -> TeacherTargetStore:
        root = Path(path)
        metadata_file = target_store_metadata_path(root)
        if not metadata_file.is_file():
            raise ValueError(f"missing metadata.json: {metadata_file}")
        metadata = target_store_metadata_from_dict(
            json.loads(metadata_file.read_text(encoding="utf-8"))
        )
        return cls(root=root, metadata=metadata)

    def write_shard(
        self,
        shard_id: int,
        arrays: Mapping[str, np.ndarray],
    ) -> Path:
        self._validate_shard_arrays(arrays)
        path = target_store_shard_path(self.root, shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return path

    def read_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        path = target_store_shard_path(self.root, shard_id)
        if not path.is_file():
            raise ValueError(f"missing target shard: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}

    def list_shards(self) -> list[Path]:
        return sorted(target_store_shards_dir(self.root).glob("shard-*.npz"))

    def validate(self) -> None:
        validate_target_store_metadata(self.metadata)
        shard_paths = self.list_shards()
        if len(shard_paths) != self.metadata.shard_count:
            raise ValueError(
                "teacher target store shard_count mismatch: "
                f"metadata={self.metadata.shard_count} actual={len(shard_paths)}"
            )
        total_examples = 0
        for shard_id in range(self.metadata.shard_count):
            arrays = self.read_shard(shard_id)
            self._validate_shard_arrays(arrays)
            total_examples += int(np.asarray(arrays["input_ids"]).shape[0])
        if total_examples != self.metadata.num_examples:
            raise ValueError(
                "teacher target store num_examples mismatch: "
                f"metadata={self.metadata.num_examples} actual={total_examples}"
            )

    def _validate_shard_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        if self.metadata.target_type not in P93_ARRAY_TARGET_TYPES:
            raise ValueError(
                "P93 TeacherTargetStore shard validation supports only target types "
                f"{sorted(P93_ARRAY_TARGET_TYPES)}, got "
                f"{self.metadata.target_type!r}"
            )
        if self.metadata.target_type == "topk_with_tail_v0":
            _validate_topk_tail_arrays(arrays, self.metadata)
            return
        missing = [
            name
            for name in ("input_ids", "attention_mask", "logits")
            if name not in arrays
        ]
        if missing:
            raise ValueError(f"teacher target shard missing required arrays: {missing}")
        input_ids = np.asarray(arrays["input_ids"])
        attention_mask = np.asarray(arrays["attention_mask"])
        logits = np.asarray(arrays["logits"])
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [N,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        if logits.ndim != 3:
            raise ValueError("logits must have shape [N,T,V]")
        if logits.shape[0] != input_ids.shape[0]:
            raise ValueError("logits batch dimension must match input_ids")
        if input_ids.shape[1] != self.metadata.sequence_length:
            raise ValueError(
                "input_ids sequence_length must match metadata.sequence_length"
            )
        if logits.shape[1] != self.metadata.sequence_length:
            raise ValueError("logits sequence_length must match metadata")
        if logits.shape[2] != self.metadata.vocab_size:
            raise ValueError("logits vocab_size must match metadata")
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise ValueError("input_ids dtype must be integer")
        if not (
            np.issubdtype(attention_mask.dtype, np.integer)
            or np.issubdtype(attention_mask.dtype, np.bool_)
        ):
            raise ValueError("attention_mask dtype must be integer or bool")
        if not np.issubdtype(logits.dtype, np.floating):
            raise ValueError("logits dtype must be floating")
        if _canonical_dtype(logits.dtype) != _canonical_dtype(self.metadata.dtype):
            raise ValueError("logits dtype must match metadata.dtype")


def _validate_topk_tail_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: TargetStoreMetadata,
) -> None:
    required = (
        "input_ids",
        "attention_mask",
        "top_token_ids",
        "top_log_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"topk_with_tail_v0 shard missing required arrays: {missing}")
    input_ids = np.asarray(arrays["input_ids"])
    attention_mask = np.asarray(arrays["attention_mask"])
    top_token_ids = np.asarray(arrays["top_token_ids"])
    top_log_probs = np.asarray(arrays["top_log_probs"])
    top_mass = np.asarray(arrays["top_mass"])
    tail_mass = np.asarray(arrays["tail_mass"])
    teacher_entropy = np.asarray(arrays["teacher_entropy"])
    top_k = _metadata_top_k(metadata)

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [N,T]")
    expected_nt = input_ids.shape
    if input_ids.shape[1] != metadata.sequence_length:
        raise ValueError(
            "input_ids sequence_length must match metadata.sequence_length"
        )
    if attention_mask.shape != expected_nt:
        raise ValueError("attention_mask shape must match input_ids")
    if top_token_ids.shape != (*expected_nt, top_k):
        raise ValueError("top_token_ids shape must be [N,T,K] and match metadata top_k")
    if top_log_probs.shape != (*expected_nt, top_k):
        raise ValueError("top_log_probs shape must be [N,T,K] and match metadata top_k")
    for name, value in (
        ("top_mass", top_mass),
        ("tail_mass", tail_mass),
        ("teacher_entropy", teacher_entropy),
    ):
        if value.shape != expected_nt:
            raise ValueError(f"{name} shape must be [N,T]")
    if not np.issubdtype(input_ids.dtype, np.integer):
        raise ValueError("input_ids dtype must be integer")
    if not (
        np.issubdtype(attention_mask.dtype, np.integer)
        or np.issubdtype(attention_mask.dtype, np.bool_)
    ):
        raise ValueError("attention_mask dtype must be integer or bool")
    if not np.issubdtype(top_token_ids.dtype, np.integer):
        raise ValueError("top_token_ids dtype must be integer")
    if not np.issubdtype(top_log_probs.dtype, np.floating):
        raise ValueError("top_log_probs dtype must be floating")
    expected_top_log_probs_dtype = metadata.target_params.get(
        "top_log_probs_dtype",
        metadata.dtype,
    )
    if _canonical_dtype(top_log_probs.dtype) != _canonical_dtype(
        expected_top_log_probs_dtype
    ):
        raise ValueError("top_log_probs dtype must match metadata target_params")
    for name, value in (
        ("top_mass", top_mass),
        ("tail_mass", tail_mass),
        ("teacher_entropy", teacher_entropy),
    ):
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} dtype must be floating")
    if np.any(top_token_ids < 0) or np.any(top_token_ids >= metadata.vocab_size):
        raise ValueError("top_token_ids must be within [0, vocab_size)")
    if not np.all(np.isfinite(top_log_probs)):
        raise ValueError("top_log_probs must be finite")
    if np.any(np.diff(top_log_probs, axis=-1) > 0):
        raise ValueError("top_log_probs must be sorted descending")
    sorted_ids = np.sort(top_token_ids, axis=-1)
    if np.any(np.diff(sorted_ids, axis=-1) == 0):
        raise ValueError("top_token_ids must not contain duplicates per position")
    if not np.all(np.isfinite(top_mass)):
        raise ValueError("top_mass must be finite")
    if not np.all(np.isfinite(tail_mass)):
        raise ValueError("tail_mass must be finite")
    if np.any(top_mass < 0.0) or np.any(top_mass > 1.0 + 1e-3):
        raise ValueError("top_mass must be in [0, 1]")
    if np.any(tail_mass < 0.0) or np.any(tail_mass > 1.0 + 1e-3):
        raise ValueError("tail_mass must be in [0, 1]")
    if not np.allclose(top_mass + tail_mass, 1.0, atol=_mass_tolerance(metadata)):
        raise ValueError("top_mass + tail_mass must be approximately 1")
    if not np.all(np.isfinite(teacher_entropy)):
        raise ValueError("teacher_entropy must be finite")
    if np.any(teacher_entropy < 0.0):
        raise ValueError("teacher_entropy must be non-negative")


def _metadata_top_k(metadata: TargetStoreMetadata) -> int:
    try:
        top_k = int(metadata.target_params.get("top_k", "0"))
    except ValueError as exc:
        raise ValueError("metadata target_params.top_k must be an integer") from exc
    if top_k <= 0:
        raise ValueError("metadata target_params.top_k must be > 0")
    if top_k > metadata.vocab_size:
        raise ValueError("metadata target_params.top_k must be <= vocab_size")
    return top_k


def _mass_tolerance(metadata: TargetStoreMetadata) -> float:
    dtype = metadata.target_params.get("top_log_probs_dtype", metadata.dtype)
    return 1e-3 if _canonical_dtype(dtype) == "float16" else 1e-5


def _canonical_dtype(dtype: object) -> str:
    value = str(np.dtype(dtype)) if not isinstance(dtype, str) else dtype
    return {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }.get(value, value)
