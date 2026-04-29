from __future__ import annotations

import qrwkv_xla


def test_package_imports() -> None:
    assert qrwkv_xla is not None


def test_package_version() -> None:
    assert isinstance(qrwkv_xla.__version__, str)
    assert qrwkv_xla.__version__
