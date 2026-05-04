"""Target artifact interfaces for QRWKV-XLA."""

from qrwkv_xla.targets.bundle import (
    LoadedTeacherTargetBundle,
    inspect_target_bundle,
    load_teacher_target_bundle,
    read_manifest,
    validate_target_bundle,
    write_manifest,
    write_target_bundle,
)
from qrwkv_xla.targets.manifest import (
    TargetFlags,
    TargetShardInfo,
    TeacherTargetManifest,
)
from qrwkv_xla.targets.shards import (
    REQUIRED_SHARD_KEYS,
    hash_shard_arrays,
    read_shard,
    validate_shard_arrays,
    write_shard,
)
from qrwkv_xla.targets.validate import (
    manifest_from_dict,
    manifest_to_dict,
    validate_manifest,
)

__all__ = [
    "REQUIRED_SHARD_KEYS",
    "LoadedTeacherTargetBundle",
    "TargetFlags",
    "TargetShardInfo",
    "TeacherTargetManifest",
    "hash_shard_arrays",
    "inspect_target_bundle",
    "load_teacher_target_bundle",
    "manifest_from_dict",
    "manifest_to_dict",
    "read_manifest",
    "read_shard",
    "validate_manifest",
    "validate_shard_arrays",
    "validate_target_bundle",
    "write_manifest",
    "write_shard",
    "write_target_bundle",
]
