from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qrwkv_xla.targets.bundle import validate_target_bundle
from qrwkv_xla.targets.shards import read_shard
from qrwkv_xla.targets.store import list_shard_paths


@dataclass(frozen=True)
class TargetBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    loss_mask: np.ndarray
    hidden_states: np.ndarray | None
    logits: np.ndarray | None = None
    attention_targets: np.ndarray | None = None


@dataclass(frozen=True)
class TargetBundleDataset:
    bundle_dir: Path

    @classmethod
    def from_path(cls, bundle_dir: str | Path) -> TargetBundleDataset:
        bundle_path = Path(bundle_dir)
        validate_target_bundle(bundle_path)
        return cls(bundle_path)

    def iter_shards(self) -> Iterator[TargetBatch]:
        for shard_file in list_shard_paths(self.bundle_dir):
            arrays = read_shard(shard_file)
            yield TargetBatch(
                input_ids=np.asarray(arrays["input_ids"]),
                attention_mask=np.asarray(arrays["attention_mask"]),
                loss_mask=np.asarray(arrays["loss_mask"]),
                hidden_states=(
                    np.asarray(arrays["hidden_states"])
                    if "hidden_states" in arrays
                    else None
                ),
                logits=np.asarray(arrays["logits"]) if "logits" in arrays else None,
                attention_targets=(
                    np.asarray(arrays["attention_targets"])
                    if "attention_targets" in arrays
                    else None
                ),
            )

    def first_batch(self) -> TargetBatch:
        for batch in self.iter_shards():
            return batch
        raise ValueError(f"Target bundle contains no shards: {self.bundle_dir}")
