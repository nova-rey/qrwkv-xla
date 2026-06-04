"""First serious compute burn launchpad helpers."""

from qrwkv_xla.burn.config import (
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    load_first_serious_burn_config,
    write_first_serious_burn_config,
)
from qrwkv_xla.burn.first_serious_burn import (
    FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE,
    FirstSeriousBurnResult,
    run_first_serious_burn,
    write_first_serious_burn_report,
)

__all__ = [
    "FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE",
    "FirstSeriousBurnConfig",
    "FirstSeriousBurnResult",
    "default_first_serious_burn_config",
    "load_first_serious_burn_config",
    "run_first_serious_burn",
    "write_first_serious_burn_config",
    "write_first_serious_burn_report",
]
