"""XLA/JAX runtime helpers for QRWKV-XLA."""

from qrwkv_xla.xla.environment_preflight import (
    JaxDeviceInfo,
    JaxEnvironmentInfo,
    RuntimeEnvironmentReport,
    TransparentHugepageInfo,
    TransparentHugepageMutationInfo,
    enable_transparent_hugepages,
    inspect_jax_environment,
    read_transparent_hugepage_status,
    run_runtime_environment_preflight,
)
from qrwkv_xla.xla.inspect import (
    JaxRuntimeInfo,
    format_jax_runtime_info,
    get_jax_runtime_info,
)
from qrwkv_xla.xla.static_checks import XlaSmokeResult, run_xla_distill_smoke

__all__ = [
    "JaxDeviceInfo",
    "JaxEnvironmentInfo",
    "JaxRuntimeInfo",
    "RuntimeEnvironmentReport",
    "TransparentHugepageInfo",
    "TransparentHugepageMutationInfo",
    "XlaSmokeResult",
    "enable_transparent_hugepages",
    "format_jax_runtime_info",
    "get_jax_runtime_info",
    "inspect_jax_environment",
    "read_transparent_hugepage_status",
    "run_runtime_environment_preflight",
    "run_xla_distill_smoke",
]
