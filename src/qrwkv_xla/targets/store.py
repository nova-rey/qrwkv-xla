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


def _canonical_dtype(dtype: object) -> str:
    value = str(np.dtype(dtype)) if not isinstance(dtype, str) else dtype
    return {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }.get(value, value)
