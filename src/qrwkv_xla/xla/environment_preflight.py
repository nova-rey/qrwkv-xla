from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRANSPARENT_HUGEPAGE_PATH = Path("/sys/kernel/mm/transparent_hugepage/enabled")
TRANSPARENT_HUGEPAGE_ENABLE_COMMAND = (
    'sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"'
)
RUNTIME_ENVIRONMENT_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "performance_benchmark_complete",
    "pjit_ready",
    "pallas_default_ready",
    "big_burn_ready",
)


@dataclass(frozen=True)
class JaxDeviceInfo:
    id: str
    platform: str
    device_kind: str


@dataclass(frozen=True)
class TransparentHugepageInfo:
    path: str
    raw_status: str | None
    status: str
    enabled: bool | None
    recommended_enable_command: str = TRANSPARENT_HUGEPAGE_ENABLE_COMMAND
    error_message: str | None = None


@dataclass(frozen=True)
class TransparentHugepageMutationInfo:
    attempted: bool
    ok: bool | None
    error_message: str | None = None


@dataclass(frozen=True)
class RuntimeEnvironmentReport:
    phase: str
    status: str
    scope: str
    python_version: str
    jax_available: bool
    jax_version: str | None
    jaxlib_version: str | None
    default_backend: str | None
    devices: tuple[JaxDeviceInfo, ...]
    tpu_devices_detected: bool
    transparent_hugepages: TransparentHugepageInfo
    mutation_attempted: bool
    mutation_ok: bool | None
    jax_error_message: str | None
    claims_not_made: tuple[str, ...] = RUNTIME_ENVIRONMENT_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JaxEnvironmentInfo:
    jax_available: bool
    jax_version: str | None
    jaxlib_version: str | None
    default_backend: str | None
    devices: tuple[JaxDeviceInfo, ...]
    error_message: str | None = None

    @property
    def tpu_devices_detected(self) -> bool:
        return any(device.platform == "tpu" for device in self.devices)


def inspect_jax_environment(jax_module: Any | None = None) -> JaxEnvironmentInfo:
    try:
        jax = jax_module if jax_module is not None else importlib.import_module("jax")
    except ImportError as exc:
        return JaxEnvironmentInfo(
            jax_available=False,
            jax_version=None,
            jaxlib_version=None,
            default_backend=None,
            devices=(),
            error_message=str(exc),
        )

    errors: list[str] = []
    try:
        default_backend = str(jax.default_backend())
    except Exception as exc:  # pragma: no cover - exercised with fake modules.
        default_backend = None
        errors.append(f"default_backend: {exc}")

    try:
        raw_devices = tuple(jax.devices())
        devices = tuple(_device_info(device) for device in raw_devices)
    except Exception as exc:  # pragma: no cover - exercised with fake modules.
        devices = ()
        errors.append(f"devices: {exc}")

    return JaxEnvironmentInfo(
        jax_available=True,
        jax_version=str(getattr(jax, "__version__", "unknown")),
        jaxlib_version=_module_version("jaxlib"),
        default_backend=default_backend,
        devices=devices,
        error_message="; ".join(errors) if errors else None,
    )


def read_transparent_hugepage_status(
    path: str | Path = TRANSPARENT_HUGEPAGE_PATH,
) -> TransparentHugepageInfo:
    hugepage_path = Path(path)
    try:
        raw_status = hugepage_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return TransparentHugepageInfo(
            path=str(hugepage_path),
            raw_status=None,
            status="unavailable",
            enabled=None,
            error_message="transparent hugepage status file is unavailable",
        )
    except OSError as exc:
        return TransparentHugepageInfo(
            path=str(hugepage_path),
            raw_status=None,
            status="unavailable",
            enabled=None,
            error_message=str(exc),
        )

    active = _active_hugepage_value(raw_status)
    if active == "always":
        return TransparentHugepageInfo(
            path=str(hugepage_path),
            raw_status=raw_status,
            status="enabled",
            enabled=True,
        )
    if active in {"madvise", "never"}:
        return TransparentHugepageInfo(
            path=str(hugepage_path),
            raw_status=raw_status,
            status="disabled",
            enabled=False,
        )
    return TransparentHugepageInfo(
        path=str(hugepage_path),
        raw_status=raw_status,
        status="unknown",
        enabled=None,
        error_message="could not identify active transparent hugepage mode",
    )


def enable_transparent_hugepages(
    path: str | Path = TRANSPARENT_HUGEPAGE_PATH,
) -> TransparentHugepageMutationInfo:
    try:
        Path(path).write_text("always\n", encoding="utf-8")
    except OSError as exc:
        return TransparentHugepageMutationInfo(
            attempted=True,
            ok=False,
            error_message=str(exc),
        )
    return TransparentHugepageMutationInfo(attempted=True, ok=True)


def run_runtime_environment_preflight(
    *,
    hugepage_path: str | Path = TRANSPARENT_HUGEPAGE_PATH,
    require_tpu: bool = False,
    enable_hugepages: bool = False,
) -> RuntimeEnvironmentReport:
    jax_info = inspect_jax_environment()
    hugepage_info = read_transparent_hugepage_status(hugepage_path)
    mutation = (
        enable_transparent_hugepages(hugepage_path)
        if enable_hugepages
        else TransparentHugepageMutationInfo(attempted=False, ok=None)
    )
    status = _preflight_status(
        jax_info=jax_info,
        hugepage_info=hugepage_info,
        mutation=mutation,
        require_tpu=require_tpu,
    )
    return RuntimeEnvironmentReport(
        phase="P109",
        status=status,
        scope="tpu_environment_hygiene_runtime_readiness",
        python_version=sys.version.split()[0],
        jax_available=jax_info.jax_available,
        jax_version=jax_info.jax_version,
        jaxlib_version=jax_info.jaxlib_version,
        default_backend=jax_info.default_backend,
        devices=jax_info.devices,
        tpu_devices_detected=jax_info.tpu_devices_detected,
        transparent_hugepages=hugepage_info,
        mutation_attempted=mutation.attempted,
        mutation_ok=mutation.ok,
        jax_error_message=jax_info.error_message,
    )


def _preflight_status(
    *,
    jax_info: JaxEnvironmentInfo,
    hugepage_info: TransparentHugepageInfo,
    mutation: TransparentHugepageMutationInfo,
    require_tpu: bool,
) -> str:
    if require_tpu and not jax_info.tpu_devices_detected:
        return "fail"
    if mutation.attempted and mutation.ok is False:
        return "warn"
    if not jax_info.jax_available:
        return "warn"
    if jax_info.tpu_devices_detected and hugepage_info.status != "enabled":
        return "warn"
    return "pass"


def _device_info(device: object) -> JaxDeviceInfo:
    return JaxDeviceInfo(
        id=str(getattr(device, "id", "?")),
        platform=str(getattr(device, "platform", "unknown")),
        device_kind=str(getattr(device, "device_kind", type(device).__name__)),
    )


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _active_hugepage_value(raw_status: str) -> str | None:
    for token in raw_status.split():
        if token.startswith("[") and token.endswith("]"):
            value = token.strip("[]")
            return value or None
    return None
