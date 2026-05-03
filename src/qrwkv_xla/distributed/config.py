from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = False
    mode: str = "single_device"
    axis_name: str = "data"
    require_multiple_devices: bool = False
    min_device_count: int = 2


def load_distributed_config(data: Any, *, section: str) -> DistributedConfig:
    if data is None:
        return DistributedConfig()
    if not isinstance(data, dict):
        raise ValueError(f"{section}.distributed must be a mapping")
    config = DistributedConfig(
        enabled=bool(data.get("enabled", False)),
        mode=str(data.get("mode", "single_device")),
        axis_name=str(data.get("axis_name", "data")),
        require_multiple_devices=bool(data.get("require_multiple_devices", False)),
        min_device_count=int(data.get("min_device_count", 2)),
    )
    validate_distributed_config(config)
    return config


def validate_distributed_config(config: DistributedConfig) -> None:
    if config.mode not in {"single_device", "pmap_data_parallel"}:
        raise ValueError(
            "distributed.mode must be one of {'single_device', 'pmap_data_parallel'}"
        )
    if not config.axis_name:
        raise ValueError("distributed.axis_name must be non-empty")
    if config.min_device_count < 1:
        raise ValueError("distributed.min_device_count must be >= 1")
    if config.require_multiple_devices and config.min_device_count < 2:
        raise ValueError(
            "distributed.require_multiple_devices requires min_device_count >= 2"
        )
    if not config.enabled and config.mode != "single_device":
        raise ValueError(
            "distributed.mode must be single_device when distributed is disabled"
        )
