from __future__ import annotations

import pytest

from qrwkv_xla.distributed.devices import (
    DeviceTopology,
    format_device_topology,
    get_device_topology,
    require_device_count,
)


def test_get_device_topology_returns_sane_fields() -> None:
    topology = get_device_topology()

    assert isinstance(topology, DeviceTopology)
    assert topology.backend
    assert topology.device_count >= 1
    assert topology.local_device_count >= 1
    assert len(topology.devices) == topology.device_count
    assert topology.has_multiple_devices is (topology.local_device_count >= 2)


def test_require_device_count_one_passes() -> None:
    topology = require_device_count(1)
    assert topology.local_device_count >= 1


def test_require_device_count_large_number_raises() -> None:
    with pytest.raises(RuntimeError, match="not enough local JAX devices"):
        require_device_count(999)


def test_format_device_topology_includes_backend_and_device_count() -> None:
    formatted = format_device_topology(get_device_topology())
    assert "backend:" in formatted
    assert "device_count:" in formatted
    assert "local_device_count:" in formatted
