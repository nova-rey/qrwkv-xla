"""First serious compute burn launchpad helpers."""

from qrwkv_xla.burn.config import (
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    load_first_serious_burn_config,
    write_first_serious_burn_config,
)
from qrwkv_xla.burn.example_sharding import (
    CONTIGUOUS_BY_PROCESS,
    ROUND_ROBIN_BY_PROCESS,
    ExampleShard,
    build_example_shard,
    contiguous_example_shard,
    round_robin_example_shard,
    verify_global_example_shards,
)
from qrwkv_xla.burn.fingerprints import fingerprint_bytes, fingerprint_pytree
from qrwkv_xla.burn.first_serious_burn import (
    FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE,
    FirstSeriousBurnResult,
    run_first_serious_burn,
    write_first_serious_burn_report,
)
from qrwkv_xla.burn.sync_diagnostics import (
    CollectiveSyncProbe,
    SyncReadiness,
    evaluate_distributed_training_readiness,
    run_collective_sync_probe,
)

__all__ = [
    "FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE",
    "CONTIGUOUS_BY_PROCESS",
    "CollectiveSyncProbe",
    "ROUND_ROBIN_BY_PROCESS",
    "ExampleShard",
    "FirstSeriousBurnConfig",
    "FirstSeriousBurnResult",
    "SyncReadiness",
    "build_example_shard",
    "contiguous_example_shard",
    "default_first_serious_burn_config",
    "evaluate_distributed_training_readiness",
    "fingerprint_bytes",
    "fingerprint_pytree",
    "load_first_serious_burn_config",
    "round_robin_example_shard",
    "run_collective_sync_probe",
    "run_first_serious_burn",
    "verify_global_example_shards",
    "write_first_serious_burn_config",
    "write_first_serious_burn_report",
]
