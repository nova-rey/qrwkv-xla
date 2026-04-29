from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from qrwkv_xla.targets.manifest import TeacherTargetManifest
from qrwkv_xla.targets.shards import read_shard, validate_shard_arrays, write_shard
from qrwkv_xla.targets.store import (
    ensure_bundle_layout,
    list_shard_paths,
    manifest_path,
    shard_path,
    shards_dir,
)
from qrwkv_xla.targets.validate import (
    manifest_from_dict,
    manifest_to_dict,
    validate_manifest,
)


def write_manifest(path: str | Path, manifest: TeacherTargetManifest) -> None:
    validate_manifest(manifest)
    payload = manifest_to_dict(manifest)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> TeacherTargetManifest:
    manifest_file = Path(path)
    if not manifest_file.exists():
        raise ValueError(f"manifest.json does not exist: {manifest_file}")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    return manifest_from_dict(data)


def write_target_bundle(
    bundle_dir: str | Path,
    manifest: TeacherTargetManifest,
    shards: Iterable[Mapping[str, np.ndarray]],
) -> None:
    bundle_path = Path(bundle_dir)
    ensure_bundle_layout(bundle_path)
    write_manifest(manifest_path(bundle_path), manifest)

    wrote_any = False
    for index, arrays in enumerate(shards):
        validate_shard_arrays(
            arrays,
            sequence_length=manifest.sequence_length,
            hidden_size=manifest.hidden_size,
            num_layers=manifest.num_layers,
        )
        write_shard(shard_path(bundle_path, index), arrays)
        wrote_any = True

    if not wrote_any:
        raise ValueError("Target bundle must contain at least one shard")


def inspect_target_bundle(bundle_dir: str | Path) -> dict[str, object]:
    bundle_path = Path(bundle_dir)
    validate_target_bundle(bundle_path)
    manifest = read_manifest(manifest_path(bundle_path))
    shard_paths = list_shard_paths(bundle_path)

    total_examples = 0
    target_keys: set[str] = set()
    for shard_file in shard_paths:
        arrays = read_shard(shard_file)
        total_examples += int(np.asarray(arrays["input_ids"]).shape[0])
        target_keys.update(arrays.keys())

    return {
        "schema_version": manifest.schema_version,
        "teacher_family": manifest.teacher_family,
        "teacher_model_id": manifest.teacher_model_id,
        "teacher_policy_label": manifest.teacher_policy_label,
        "sequence_length": manifest.sequence_length,
        "hidden_size": manifest.hidden_size,
        "num_layers": manifest.num_layers,
        "dtype": manifest.dtype,
        "shard_count": len(shard_paths),
        "total_examples": total_examples,
        "target_keys": sorted(target_keys),
    }


def validate_target_bundle(bundle_dir: str | Path) -> None:
    bundle_path = Path(bundle_dir)
    manifest_file = manifest_path(bundle_path)
    shard_dir = shards_dir(bundle_path)

    if not manifest_file.exists():
        raise ValueError(f"Missing manifest.json: {manifest_file}")
    if not shard_dir.exists():
        raise ValueError(f"Missing shards directory: {shard_dir}")

    manifest = read_manifest(manifest_file)
    shard_files = list_shard_paths(bundle_path)
    if not shard_files:
        raise ValueError("Target bundle must contain at least one shard")

    for shard_file in shard_files:
        arrays = read_shard(shard_file)
        validate_shard_arrays(
            arrays,
            sequence_length=manifest.sequence_length,
            hidden_size=manifest.hidden_size,
            num_layers=manifest.num_layers,
        )
