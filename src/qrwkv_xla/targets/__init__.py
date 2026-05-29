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
from qrwkv_xla.targets.consumption import (
    OfflineTargetBatch,
    load_offline_target_batch,
    mse_logits_loss,
)
from qrwkv_xla.targets.manifest import (
    TargetFlags,
    TargetShardInfo,
    TeacherTargetManifest,
)
from qrwkv_xla.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    target_store_metadata_from_dict,
    target_store_metadata_to_dict,
    validate_target_store_metadata,
)
from qrwkv_xla.targets.shards import (
    REQUIRED_SHARD_KEYS,
    hash_shard_arrays,
    read_shard,
    validate_shard_arrays,
    write_shard,
)
from qrwkv_xla.targets.store import TeacherTargetStore
from qrwkv_xla.targets.validate import (
    manifest_from_dict,
    manifest_to_dict,
    validate_manifest,
)

__all__ = [
    "REQUIRED_SHARD_KEYS",
    "LoadedTeacherTargetBundle",
    "OfflineTargetBatch",
    "TargetFlags",
    "TargetShardInfo",
    "TargetStoreMetadata",
    "TEACHER_TARGET_STORE_SCHEMA_VERSION",
    "TEACHER_TARGET_STORE_VERSION",
    "TeacherTargetStore",
    "TeacherTargetManifest",
    "hash_shard_arrays",
    "inspect_target_bundle",
    "load_offline_target_batch",
    "load_teacher_target_bundle",
    "manifest_from_dict",
    "manifest_to_dict",
    "mse_logits_loss",
    "read_manifest",
    "read_shard",
    "target_store_metadata_from_dict",
    "target_store_metadata_to_dict",
    "validate_manifest",
    "validate_shard_arrays",
    "validate_target_bundle",
    "validate_target_store_metadata",
    "write_manifest",
    "write_shard",
    "write_target_bundle",
]
