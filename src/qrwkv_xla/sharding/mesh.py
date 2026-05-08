from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh


@dataclass(frozen=True)
class MeshInfo:
    backend: str
    platform: str
    device_count: int
    local_device_count: int
    device_kinds: tuple[str, ...]
    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    multi_device: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["device_kinds"] = list(self.device_kinds)
        payload["mesh_shape"] = list(self.mesh_shape)
        payload["mesh_axis_names"] = list(self.mesh_axis_names)
        return payload


def create_named_mesh(
    *,
    mesh_axis: str = "data",
    require_multi_device: bool = False,
) -> tuple[Mesh, MeshInfo]:
    if not mesh_axis:
        raise ValueError("mesh_axis must be non-empty")
    devices = tuple(jax.devices())
    if not devices:
        raise RuntimeError("JAX reported no devices")
    device_count = len(devices)
    local_device_count = int(jax.local_device_count())
    if require_multi_device and device_count < 2:
        raise RuntimeError(
            "P46 pjit sharding smoke requires at least 2 JAX devices, "
            f"found {device_count} on backend={jax.default_backend()}"
        )

    selected_devices = np.asarray(devices)
    mesh = Mesh(selected_devices, (mesh_axis,))
    fallback_reason = None
    if device_count < 2:
        fallback_reason = "single_device_fallback"
    first_platform = str(getattr(devices[0], "platform", jax.default_backend()))
    device_kinds = tuple(
        sorted({str(getattr(device, "device_kind", device)) for device in devices})
    )
    info = MeshInfo(
        backend=str(jax.default_backend()),
        platform=first_platform,
        device_count=device_count,
        local_device_count=local_device_count,
        device_kinds=device_kinds,
        mesh_shape=tuple(int(dim) for dim in mesh.devices.shape),
        mesh_axis_names=tuple(str(axis) for axis in mesh.axis_names),
        multi_device=device_count >= 2,
        fallback_reason=fallback_reason,
    )
    return mesh, info
