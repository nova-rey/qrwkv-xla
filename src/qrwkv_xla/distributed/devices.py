from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class DeviceTopology:
    backend: str
    device_count: int
    local_device_count: int
    devices: tuple[str, ...]
    has_multiple_devices: bool


def get_device_topology() -> DeviceTopology:
    devices = tuple(str(device) for device in jax.devices())
    device_count = len(devices)
    local_device_count = int(jax.local_device_count())
    return DeviceTopology(
        backend=str(jax.default_backend()),
        device_count=device_count,
        local_device_count=local_device_count,
        devices=devices,
        has_multiple_devices=local_device_count >= 2,
    )


def require_device_count(min_count: int) -> DeviceTopology:
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    topology = get_device_topology()
    if topology.local_device_count < min_count:
        raise RuntimeError(
            "not enough local JAX devices for pmap smoke: "
            f"required {min_count}, found {topology.local_device_count} "
            f"(backend={topology.backend})"
        )
    return topology


def format_device_topology(topology: DeviceTopology) -> str:
    device_text = ", ".join(topology.devices) if topology.devices else "<none>"
    return "\n".join(
        [
            f"backend: {topology.backend}",
            f"device_count: {topology.device_count}",
            f"local_device_count: {topology.local_device_count}",
            f"has_multiple_devices: {topology.has_multiple_devices}",
            f"devices: {device_text}",
        ]
    )
