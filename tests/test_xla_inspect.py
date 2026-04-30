from __future__ import annotations

from qrwkv_xla.xla import JaxRuntimeInfo, format_jax_runtime_info, get_jax_runtime_info


def test_get_jax_runtime_info_returns_runtime_info() -> None:
    info = get_jax_runtime_info()
    assert isinstance(info, JaxRuntimeInfo)
    assert info.jax_available is True
    assert info.default_backend
    assert len(info.devices) >= 1


def test_format_jax_runtime_info_is_non_empty() -> None:
    info = get_jax_runtime_info()
    formatted = format_jax_runtime_info(info)
    assert formatted
    assert "jax_available:" in formatted
