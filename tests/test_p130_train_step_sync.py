from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.burn.sync_diagnostics import (
    average_pytree_across_processes,
    global_mean_scalar,
    verify_checksum_sync_across_processes,
)


def test_average_pytree_across_processes_averages_each_leaf(monkeypatch) -> None:
    _patch_process_allgather(monkeypatch, offsets=(0.0, 2.0, 4.0, 6.0))

    averaged = average_pytree_across_processes(
        {
            "w": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            "b": jnp.asarray(3.0, dtype=jnp.float32),
        },
        process_count=4,
    )

    np.testing.assert_allclose(averaged["w"], np.asarray([4.0, 5.0]))
    np.testing.assert_allclose(averaged["b"], np.asarray(6.0))


def test_global_mean_scalar_uses_same_collective(monkeypatch) -> None:
    _patch_process_allgather(monkeypatch, offsets=(0.0, 1.0, 2.0, 3.0))

    value = global_mean_scalar(jnp.asarray(2.0, dtype=jnp.float32), process_count=4)

    np.testing.assert_allclose(value, np.asarray(3.5))


def test_checksum_sync_verification_passes_when_all_values_match(monkeypatch) -> None:
    _patch_process_allgather(monkeypatch, values=(7.0, 7.0, 7.0, 7.0))

    report = verify_checksum_sync_across_processes(7.0, process_count=4)

    assert report.verified is True
    assert report.global_min == 7.0
    assert report.global_max == 7.0


def test_checksum_sync_verification_fails_when_values_differ(monkeypatch) -> None:
    _patch_process_allgather(monkeypatch, values=(7.0, 7.0, 8.0, 7.0))

    report = verify_checksum_sync_across_processes(7.0, process_count=4)

    assert report.verified is False
    assert report.global_min == 7.0
    assert report.global_max == 8.0


def _patch_process_allgather(
    monkeypatch,
    *,
    offsets: tuple[float, ...] | None = None,
    values: tuple[float, ...] | None = None,
) -> None:
    import jax.experimental.multihost_utils as multihost_utils

    def fake_process_allgather(value, *, tiled=False):
        del tiled
        arr = np.asarray(value)
        if values is not None:
            return np.asarray(values, dtype=arr.dtype)
        assert offsets is not None
        return np.stack([arr + offset for offset in offsets], axis=0)

    monkeypatch.setattr(multihost_utils, "process_allgather", fake_process_allgather)
