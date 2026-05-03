"""Minimal pmap data-parallel utilities for QRWKV-XLA."""

from qrwkv_xla.distributed.config import (
    DistributedConfig,
    load_distributed_config,
    validate_distributed_config,
)
from qrwkv_xla.distributed.devices import (
    DeviceTopology,
    format_device_topology,
    get_device_topology,
    require_device_count,
)
from qrwkv_xla.distributed.reductions import metrics_pmean, tree_pmean
from qrwkv_xla.distributed.replication import (
    replicate_to_devices,
    unreplicate_from_devices,
)
from qrwkv_xla.distributed.sharding import (
    can_shard_batch,
    shard_array_for_devices,
    shard_batch_for_devices,
    unshard_first_device,
)

__all__ = [
    "DeviceTopology",
    "DistributedConfig",
    "can_shard_batch",
    "format_device_topology",
    "get_device_topology",
    "load_distributed_config",
    "metrics_pmean",
    "replicate_to_devices",
    "require_device_count",
    "shard_array_for_devices",
    "shard_batch_for_devices",
    "tree_pmean",
    "unreplicate_from_devices",
    "unshard_first_device",
    "validate_distributed_config",
]
